from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import PROJECT_ROOT  # noqa: E402


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(
    outputs_dir: Path,
    model_id: str,
    output_csv: Path,
    selected_conditions: tuple[str, ...] | None = None,
):
    rows = []
    conditions = [
        ("fp16", "FP16", 16, "None", None),
        ("pile", "AWQ", 4, "Pile", "awq_pile"),
        ("risk1", "AWQ", 4, "Risk-I", "awq_risk1"),
        ("risk2", "AWQ", 4, "Risk-II", "awq_risk2"),
        ("risk3", "AWQ", 4, "Risk-III", "awq_risk3"),
    ]
    if selected_conditions is not None:
        selected = set(selected_conditions)
        conditions = [item for item in conditions if item[0] in selected]
    for condition, model_label, bits, calibration, output_stem in conditions:
        safety_stem = condition if condition == "fp16" else output_stem
        summary_path = outputs_dir / f"{safety_stem}.summary.json"
        ppl_path = outputs_dir / f"{safety_stem}.ppl.json"
        metadata_path = (
            outputs_dir / "models" / f"awq_{condition}" / "quantization_metadata.json"
            if condition != "fp16"
            else None
        )
        if not summary_path.exists() or not ppl_path.exists():
            raise FileNotFoundError(f"missing evaluation output for {condition}")
        summary = _read_json(summary_path)
        ppl = _read_json(ppl_path)
        metadata = _read_json(metadata_path) if metadata_path and metadata_path.exists() else {}
        metric = metadata.get("calibration_metrics", {})
        rows.append(
            {
                "Model": model_label,
                "Bits": bits,
                "Calibration": calibration,
                "Samples": metric.get("source_examples", "–") if condition != "fp16" else "–",
                "Tokens": metric.get("total_tokens_before_blocking", "–") if condition != "fp16" else "–",
                "Blocks": metric.get("blocks", "–") if condition != "fp16" else "–",
                "ASR held-out": summary["asr_heldout_refusal_heuristic"],
                "PPL": ppl["ppl"],
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Model", "Bits", "Calibration", "Samples", "Tokens", "Blocks", "ASR held-out", "PPL"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Combine safety, utility, and calibration metadata")
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--model-id", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "results.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(args.outputs_dir, args.model_id, args.output_csv)
