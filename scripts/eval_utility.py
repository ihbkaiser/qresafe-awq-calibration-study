from __future__ import annotations

import argparse
import json
import math
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
)

SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qresafe_awq_calibration.runtime import dump_json  # noqa: E402


def utility_forward_model(model, condition):
    """Return the causal-LM object whose forward pass produces logits.

    Transformers causal-LM wrappers expose ``.model`` as the decoder-only
    backbone, which returns ``BaseModelOutputWithPast`` without logits.  The
    official Q-resafe AWQ wrapper instead exposes the causal-LM as ``.model``.
    """
    if condition == "fp16":
        return model
    return getattr(model, "model", model)


def evaluate(args):
    import torch
    from datasets import load_dataset

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
    model_for_forward = utility_forward_model(model, args.condition)
    device = model_device(model_for_forward)
    model_for_forward.eval()

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenized = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
    if args.max_tokens is not None:
        tokenized = tokenized[:, : args.max_tokens]
    n_samples = tokenized.numel() // args.seqlen
    if n_samples == 0:
        raise ValueError("utility corpus has fewer tokens than one evaluation sequence")

    total_nll = 0.0
    total_tokens = 0
    for index in range(n_samples):
        start = index * args.seqlen
        batch = tokenized[:, start : start + args.seqlen].to(device)
        with torch.inference_mode():
            logits = model_for_forward(batch).logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss.item())
        total_tokens += int(shift_labels.numel())

    ppl = math.exp(total_nll / total_tokens)
    result = {
        "condition": args.condition,
        "model_id": args.model_id,
        "dataset": "wikitext-2-raw-v1/test",
        "ppl": ppl,
        "seqlen": args.seqlen,
        "sequences": n_samples,
    }
    dump_json(args.output, result)
    print(json.dumps(result, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Q-resafe WikiText utility sanity check")
    parser.add_argument("--condition", choices=["fp16", "pile", "risk1", "risk2", "risk3"], required=True)
    parser.add_argument("--model-id", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--qresafe-root", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.condition != "fp16" and args.model_path is None:
        args.model_path = PROJECT_ROOT / "outputs" / "models" / f"awq_{args.condition}"
    if args.output is None:
        args.output = PROJECT_ROOT / "outputs" / f"{args.condition if args.condition == 'fp16' else 'awq_' + args.condition}.ppl.json"
    return args


if __name__ == "__main__":
    evaluate(parse_args())
