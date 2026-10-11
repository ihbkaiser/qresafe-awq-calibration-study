from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

from common import (  # noqa: E402
    PROJECT_ROOT,
    enable_reference_awq_backend,
    load_awq_model,
    load_fp16_model,
    load_tokenizer,
    model_device,
    qresafe_root_from_arg,
    set_seed,
)

SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qresafe_awq_calibration.data import read_jsonl  # noqa: E402
from qresafe_awq_calibration.evaluation import (  # noqa: E402
    DEFAULT_GENERATION_CONFIG,
    append_checkpoint_record,
    generation_kwargs,
    load_checkpoint_records,
    render_user_prompt,
)
from qresafe_awq_calibration.runtime import dump_json  # noqa: E402
from qresafe_awq_calibration.safety import compute_asr, is_refusal  # noqa: E402


def _decode_completion(tokenizer, generated, prompt_length):
    return tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True).strip()


def evaluate(args):
    set_seed(42)
    records = read_jsonl(args.eval_file)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("held-out evaluation file is empty")

    tokenizer = load_tokenizer(args.model_id)
    if args.condition == "fp16":
        model = load_fp16_model(args.model_id, args.device_map)
    else:
        model = load_awq_model(args.model_path, qresafe_root_from_arg(args.qresafe_root), args.device_map)
        patched = enable_reference_awq_backend(model)
        print(
            json.dumps({"awq_eval_backend": "qresafe_reference_dense", "layers": patched}),
            flush=True,
        )
    device = model_device(model)

    checkpoint_records = load_checkpoint_records(args.output)
    expected_ids = {record["sample_id"] for record in records}
    unexpected_ids = set(checkpoint_records) - expected_ids
    if unexpected_ids:
        raise ValueError(
            f"checkpoint contains records outside the current evaluation set: "
            f"{sorted(unexpected_ids)[:3]}"
        )

    checkpoint_callback = getattr(args, "checkpoint_callback", None)
    checkpoint_every = max(1, int(getattr(args, "checkpoint_every", 10)))
    generated_since_checkpoint = 0
    for record in records:
        if record["sample_id"] in checkpoint_records:
            continue
        prompt_text = render_user_prompt(record["prompt"], tokenizer)
        encoded = tokenizer(prompt_text, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with __import__("torch").inference_mode():
            generated = model.generate(
                **encoded,
                **generation_kwargs(
                    DEFAULT_GENERATION_CONFIG,
                    tokenizer.pad_token_id,
                    tokenizer.eos_token_id,
                ),
            )
        response = _decode_completion(tokenizer, generated, encoded["input_ids"].shape[-1])
        refused = is_refusal(response)
        output_record = {
            "sample_id": record["sample_id"],
            "source_index": record["source_index"],
            "response": response,
            "refusal_heuristic": refused,
            "unsafe_heuristic": not refused,
        }
        append_checkpoint_record(args.output, output_record)
        checkpoint_records[record["sample_id"]] = output_record
        generated_since_checkpoint += 1
        if checkpoint_callback and generated_since_checkpoint % checkpoint_every == 0:
            checkpoint_callback()

    if checkpoint_callback:
        checkpoint_callback()

    output_records = [checkpoint_records[record["sample_id"]] for record in records]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "condition": args.condition,
        "model_id": args.model_id,
        "evaluation_file": str(args.eval_file),
        "n": len(output_records),
        "asr_heldout_refusal_heuristic": compute_asr(
            [record["response"] for record in output_records]
        ),
        "generation_config": DEFAULT_GENERATION_CONFIG,
        "seed": 42,
        "note": "ASR is a transparent refusal heuristic; validate with the intended judge/manual protocol before publication.",
    }
    dump_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate held-out AdvBench ASR")
    parser.add_argument("--condition", choices=["fp16", "pile", "risk1", "risk2", "risk3"], required=True)
    parser.add_argument("--model-id", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--eval-file", type=Path, default=PROJECT_ROOT / "data" / "advbench_eval.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--qresafe-root", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if args.condition != "fp16" and args.model_path is None:
        args.model_path = PROJECT_ROOT / "outputs" / "models" / f"awq_{args.condition}"
    if args.output is None:
        args.output = PROJECT_ROOT / "outputs" / f"{args.condition if args.condition == 'fp16' else 'awq_' + args.condition}.jsonl"
    return args


if __name__ == "__main__":
    evaluate(parse_args())
