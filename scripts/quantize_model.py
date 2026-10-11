from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from common import PROJECT_ROOT, load_tokenizer, qresafe_root_from_arg, set_seed  # noqa: E402

import sys

SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qresafe_awq_calibration.calibration import (  # noqa: E402
    calibration_metrics,
    require_complete_block,
    token_match_calibration,
    tokenize_records_to_chunks,
)
from qresafe_awq_calibration.config import ExperimentConfig  # noqa: E402
from qresafe_awq_calibration.data import read_jsonl  # noqa: E402
from qresafe_awq_calibration.runtime import dump_json, import_qresafe_auto, qresafe_commit  # noqa: E402


def _pile_records(tokenizer, config: ExperimentConfig):
    from datasets import load_dataset

    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation").shuffle(
        seed=config.seed
    )
    records = []
    for source_index, row in enumerate(dataset):
        text = str(row["text"]).strip()
        token_count = len(tokenizer.encode(text))
        if token_count > config.max_calib_seq_len or token_count == 0:
            continue
        records.append(
            {
                "sample_id": f"pileval-{source_index}",
                "source": "mit-han-lab/pile-val-backup",
                "source_index": source_index,
                "text": text,
            }
        )
        if len(records) == config.max_calib_samples:
            break
    if len(records) != config.max_calib_samples:
        raise RuntimeError(
            "Pile-val produced fewer eligible examples than Q-resafe's "
            f"max_calib_samples={config.max_calib_samples}"
        )
    return records


def _load_risk_records(condition: str, data_dir: Path):
    if condition == "risk1":
        return read_jsonl(data_dir / "risk1.jsonl")
    if condition == "risk2":
        return read_jsonl(data_dir / "risk2.jsonl")
    if condition == "risk3":
        return read_jsonl(data_dir / "risk3.jsonl")
    raise ValueError(f"unsupported custom calibration condition: {condition}")


def quantize_condition(
    condition: str,
    model_id: str,
    output_dir: Path,
    data_dir: Path,
    qresafe_root: Path,
    device_map: str,
    token_matched: bool = False,
) -> Path:
    if condition not in {"pile", "risk1", "risk2", "risk3"}:
        raise ValueError("condition must be one of pile, risk1, risk2, risk3")
    if token_matched and condition == "pile":
        raise ValueError("token-matched confirmation only applies to Risk conditions")

    config = ExperimentConfig(model_id=model_id)
    set_seed(config.seed)
    tokenizer = load_tokenizer(model_id)

    metric_records = None
    sample_ids = None
    source = None
    calib_texts = None
    if condition == "pile":
        calib_seq_len = config.max_calib_seq_len
    else:
        calib_seq_len = config.risk_max_calib_seq_len
        records = _load_risk_records(condition, data_dir)
        if len(records) != config.risk_samples:
            raise ValueError(
                f"{condition} must contain exactly {config.risk_samples} records, got {len(records)}"
            )
        if token_matched:
            calib_texts, metric_records = token_match_calibration(
                records,
                tokenizer,
                config.token_matched_target_tokens,
                calib_seq_len,
            )
        else:
            calib_texts = tokenize_records_to_chunks(records, tokenizer, calib_seq_len)
            metric_records = calibration_metrics(
                records, tokenizer, calib_seq_len, chunk_long_records=True
            )
        require_complete_block(metric_records, condition, calib_seq_len)
        sample_ids = [record["sample_id"] for record in records]
        source = records[0].get("source", condition)

    AutoAWQForCausalLM = import_qresafe_auto(qresafe_root)
    model = AutoAWQForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map=device_map,
        low_cpu_mem_usage=True,
        use_cache=False,
    )

    if condition == "pile":
        # Deliberately omit calib_data: this is the unmodified Q-resafe
        # default path, which resolves to pileval internally.
        model.quantize(tokenizer, quant_config=config.quant_config)
        metric_records = _pile_records(tokenizer, config)
        metric = calibration_metrics(metric_records, tokenizer, config.max_calib_seq_len)
        sample_ids = [record["sample_id"] for record in metric_records]
        source = "pileval (Q-resafe default)"
    else:
        model.quantize(
            tokenizer,
            quant_config=config.quant_config,
            calib_data=calib_texts,
            max_calib_samples=config.max_calib_samples,
            max_calib_seq_len=calib_seq_len,
        )
        metric = metric_records

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "condition": condition,
        "model_id": model_id,
        "bits": 4,
        "quant_config": config.quant_config,
        "seed": config.seed,
        "max_calib_samples": (
            config.max_calib_samples if condition == "pile" else config.risk_samples
        ),
        "max_calib_seq_len": calib_seq_len,
        "calibration_block_length_policy": (
            "qresafe_default_512"
            if condition == "pile"
            else "token_matched_common_384"
            if token_matched
            else "common_risk_128"
        ),
        "calibration_source": source,
        "calibration_sample_ids": sample_ids,
        "calibration_metrics": metric,
        "qresafe_quant_without_ft_commit": qresafe_commit(qresafe_root),
    }
    if token_matched:
        metadata["token_match_target_tokens"] = config.token_matched_target_tokens
    dump_json(output_dir / "quantization_metadata.json", metadata)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run one locked Q-resafe AWQ condition")
    parser.add_argument("--condition", choices=["pile", "risk1", "risk2", "risk3"], required=True)
    parser.add_argument("--model-id", default=ExperimentConfig().model_id)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--qresafe-root", default=None)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or (PROJECT_ROOT / "outputs" / "models" / f"awq_{args.condition}")
    result = quantize_condition(
        args.condition,
        args.model_id,
        output_dir,
        args.data_dir,
        qresafe_root_from_arg(args.qresafe_root),
        args.device_map,
    )
    print(f"Saved quantized model to {result}")


if __name__ == "__main__":
    main()
