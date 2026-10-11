from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common import PROJECT_ROOT  # noqa: E402

import sys

SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qresafe_awq_calibration.config import ExperimentConfig  # noqa: E402
from qresafe_awq_calibration.data import (  # noqa: E402
    build_risk1_records,
    build_risk2_records,
    build_risk3_records,
    extract_advbench_prompt,
    split_advbench_calibration_and_eval,
    write_jsonl,
)


def _load_dataset(dataset_id: str, split: str, config_name: str | None = None):
    from datasets import load_dataset

    kwargs = {"path": dataset_id, "split": split}
    if config_name:
        kwargs["name"] = config_name
    return load_dataset(**kwargs)


def _sample_indices(length: int, count: int, seed: int):
    if length < count:
        raise ValueError(f"dataset has {length} rows, cannot sample {count}")
    return random.Random(seed).sample(range(length), count)


def _row_with_source_index(row, source_index):
    copied = dict(row)
    copied["source_index"] = source_index
    return copied


def _merge_response_rows(rows, response_file: Path | None, response_column: str):
    if response_file is None:
        return rows
    responses = []
    with response_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                responses.append(json.loads(line))
    by_index = {row["source_index"]: row[response_column] for row in responses if response_column in row}
    by_prompt = {row["prompt"]: row[response_column] for row in responses if response_column in row and "prompt" in row}
    merged = []
    for row in rows:
        copied = dict(row)
        if row["source_index"] in by_index:
            copied[response_column] = by_index[row["source_index"]]
        elif row.get("prompt") in by_prompt:
            copied[response_column] = by_prompt[row["prompt"]]
        merged.append(copied)
    return merged


def _eval_records(rows):
    records = []
    for position, row in enumerate(rows):
        source_index = row.get("source_index", position)
        prompt = extract_advbench_prompt(row)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"AdvBench evaluation row {source_index} has no prompt")
        records.append(
            {
                "sample_id": f"advbench-{source_index}",
                "source": "AdvBench held-out evaluation",
                "source_index": source_index,
                "prompt": prompt,
            }
        )
    return records


def prepare(args):
    config = ExperimentConfig()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ultra = _load_dataset(args.ultrachat_dataset, args.ultrachat_split, args.ultrachat_config)
    ultra_rows = [_row_with_source_index(row, index) for index, row in enumerate(ultra)]
    risk1_indices = _sample_indices(len(ultra_rows), config.risk_samples, config.seed)
    risk1 = build_risk1_records(ultra_rows, risk1_indices)

    risk2 = build_risk2_records()
    if len(risk2) != config.risk_samples:
        raise AssertionError("Risk-II reconstruction must contain exactly 10 examples")

    advbench = _load_dataset(args.advbench_dataset, args.advbench_split, args.advbench_config)
    schema_probe = {"columns": list(getattr(advbench, "column_names", []))}
    if len(advbench) > 114:
        probe_row = advbench[114]
        schema_probe["row_114_shapes"] = {
            key: {
                "type": type(value).__name__,
                "length": len(value) if hasattr(value, "__len__") and not isinstance(value, str) else None,
            }
            for key, value in probe_row.items()
        }
    print("ADVBENCH_SCHEMA_JSON=" + json.dumps(schema_probe, ensure_ascii=False), flush=True)
    adv_rows = [_row_with_source_index(row, index) for index, row in enumerate(advbench)]
    adv_rows = _merge_response_rows(adv_rows, args.risk3_response_file, args.risk3_response_column)
    risk3_indices = _sample_indices(len(adv_rows), config.risk_samples, config.seed)
    risk3_rows, heldout_rows = split_advbench_calibration_and_eval(adv_rows, risk3_indices)
    risk3 = build_risk3_records(risk3_rows, risk3_indices, args.risk3_response_column)
    heldout = _eval_records(heldout_rows)

    write_jsonl(output_dir / "risk1.jsonl", risk1)
    write_jsonl(output_dir / "risk2.jsonl", risk2)
    write_jsonl(output_dir / "risk3.jsonl", risk3)
    write_jsonl(output_dir / "advbench_eval.jsonl", heldout)

    indices = {
        "seed": config.seed,
        "risk1": {"dataset": args.ultrachat_dataset, "split": args.ultrachat_split, "source_indices": risk1_indices},
        "risk2": {
            "source": "Q-resafe description reconstruction",
            "exact_dataset_available": False,
            "source_indices": list(range(config.risk_samples)),
        },
        "risk3": {
            "dataset": args.advbench_dataset,
            "split": args.advbench_split,
            "source_indices": risk3_indices,
            "response_column": args.risk3_response_column,
            "reconstruction": True,
            "exact_qresafe_selection_available": False,
            "heldout_eval_count": len(heldout),
        },
        "leakage_check": {
            "risk3_calibration_indices": risk3_indices,
            "risk3_eval_indices": [row["source_index"] for row in heldout],
            "intersection": sorted(
                set(risk3_indices).intersection(row["source_index"] for row in heldout)
            ),
        },
    }
    (output_dir / "data_indices.json").write_text(
        json.dumps(indices, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote Risk-I/Risk-II/Risk-III and held-out AdvBench data to {output_dir}")
    print(f"Risk-I indices: {risk1_indices}")
    print(f"Risk-III indices: {risk3_indices}; held-out rows: {len(heldout)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare locked calibration and held-out data")
    parser.add_argument("--ultrachat-dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--ultrachat-config", default=None)
    parser.add_argument("--ultrachat-split", default="train_sft")
    parser.add_argument(
        "--advbench-dataset",
        default="AlignmentResearch/AdvBench",
        help="Public AdvBench export with paired harmful response targets",
    )
    parser.add_argument("--advbench-config", default=None)
    parser.add_argument("--advbench-split", default="train")
    parser.add_argument(
        "--risk3-response-file",
        type=Path,
        default=None,
        help="JSONL keyed by source_index or prompt, containing the corresponding response column",
    )
    parser.add_argument(
        "--risk3-response-column",
        default="proxy_gen_target",
        help="Paired harmful response field; AlignmentResearch/AdvBench uses proxy_gen_target",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
