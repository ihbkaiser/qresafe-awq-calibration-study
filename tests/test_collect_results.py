import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_results import collect


def test_collect_can_write_a_risk_only_token_matched_table(tmp_path):
    for condition in ("risk1", "risk2", "risk3"):
        (tmp_path / f"awq_{condition}.summary.json").write_text(
            json.dumps(
                {
                    "condition": condition,
                    "n": 510,
                    "asr_heldout_refusal_heuristic": 0.5,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / f"awq_{condition}.ppl.json").write_text(
            json.dumps({"ppl": 7.0}), encoding="utf-8"
        )
        model_dir = tmp_path / "models" / f"awq_{condition}"
        model_dir.mkdir(parents=True)
        (model_dir / "quantization_metadata.json").write_text(
            json.dumps(
                {
                    "calibration_metrics": {
                        "source_examples": 10,
                        "total_tokens_before_blocking": 384,
                        "blocks": 3,
                    }
                }
            ),
            encoding="utf-8",
        )

    output = tmp_path / "token_matched_results.csv"
    collect(tmp_path, "fake/model", output, selected_conditions=("risk1", "risk2", "risk3"))

    assert output.read_text(encoding="utf-8").count("\n") == 4
