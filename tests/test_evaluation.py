import json
from pathlib import Path

from qresafe_awq_calibration.evaluation import (
    DEFAULT_GENERATION_CONFIG,
    append_checkpoint_record,
    render_user_prompt,
    generation_kwargs,
    load_checkpoint_records,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"{messages[0]['role']}:{messages[0]['content']}|GEN"


def test_eval_prompt_uses_user_only_chat_turn_and_generation_prompt():
    assert render_user_prompt("test prompt", FakeTokenizer()) == "user:test prompt|GEN"


def test_generation_config_is_greedy_and_serializable():
    assert DEFAULT_GENERATION_CONFIG["do_sample"] is False
    assert DEFAULT_GENERATION_CONFIG["max_new_tokens"] == 128
    kwargs = generation_kwargs(DEFAULT_GENERATION_CONFIG, pad_token_id=2, eos_token_id=3)
    assert kwargs == {
        "do_sample": False,
        "max_new_tokens": 128,
        "pad_token_id": 2,
        "eos_token_id": 3,
    }
    json.dumps(DEFAULT_GENERATION_CONFIG)


def test_evaluation_checkpoint_is_append_only_and_resume_safe(tmp_path: Path):
    checkpoint = tmp_path / "risk2.jsonl"
    first = {"sample_id": "advbench-1", "response": "first"}
    second = {"sample_id": "advbench-2", "response": "second"}

    append_checkpoint_record(checkpoint, first)
    append_checkpoint_record(checkpoint, second)

    loaded = load_checkpoint_records(checkpoint)

    assert list(loaded) == ["advbench-1", "advbench-2"]
    assert loaded["advbench-1"] == first
    assert loaded["advbench-2"] == second
