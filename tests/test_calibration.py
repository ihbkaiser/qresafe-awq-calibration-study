import pytest

from qresafe_awq_calibration.calibration import (
    calibration_metrics,
    render_chat_messages,
    token_match_calibration,
    tokenize_records_to_chunks,
)
from qresafe_awq_calibration.config import ExperimentConfig


def _risk_records():
    return [
        {
            "sample_id": f"risk2-{index}",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        }
        for index in range(10)
    ]


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        return "\n".join(f"{m['role']}:{m['content']}" for m in messages)

    def encode(self, text):
        return list(range(len(text)))


def test_render_chat_messages_uses_a_complete_conversation_without_generation_prompt():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    assert render_chat_messages(messages, tokenizer) == "user:hello\nassistant:hi"


def test_calibration_metrics_match_qresafe_concat_and_floor_blocking():
    tokenizer = FakeTokenizer()
    records = [
        {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
        {"messages": [{"role": "user", "content": "cc"}, {"role": "assistant", "content": "d"}]},
    ]

    metrics = calibration_metrics(records, tokenizer, max_seq_len=32)

    # Rendered lengths are 18 and 19; Q-resafe discards no sample, then floors
    # the concatenated stream to complete 32-token blocks.
    assert metrics["source_examples"] == 2
    assert metrics["used_examples"] == 2
    assert metrics["total_tokens_before_blocking"] == 37
    assert metrics["blocks"] == 1
    assert metrics["mean_tokens_per_example"] == 18.5


def test_calibration_metrics_reports_samples_longer_than_qresafe_limit():
    tokenizer = FakeTokenizer()
    records = [
        {"messages": [{"role": "user", "content": "x" * 20}, {"role": "assistant", "content": "y"}]},
        {"messages": [{"role": "user", "content": "ok"}, {"role": "assistant", "content": "yes"}]},
    ]

    metrics = calibration_metrics(records, tokenizer, max_seq_len=32)

    assert metrics["source_examples"] == 2
    assert metrics["discarded_examples"] == 1
    assert metrics["used_examples"] == 1
    assert metrics["blocks"] == 0


def test_render_chat_messages_requires_user_then_assistant_pair():
    with pytest.raises(ValueError, match="user.*assistant"):
        render_chat_messages(
            [{"role": "user", "content": "only one message"}], FakeTokenizer()
        )


def test_risk_conditions_use_a_common_short_block_length_when_corpus_is_small():
    config = ExperimentConfig()

    assert config.max_calib_seq_len == 512
    assert config.risk_max_calib_seq_len == 128


def test_risk_calibration_budget_must_contain_one_complete_qresafe_block():
    tokenizer = FakeTokenizer()
    records = _risk_records()
    metrics = calibration_metrics(records, tokenizer, max_seq_len=512)

    assert metrics["blocks"] == 0
    with pytest.raises(ValueError, match="no complete calibration block"):
        from qresafe_awq_calibration.calibration import require_complete_block

        require_complete_block(metrics, condition="risk2", max_seq_len=512)


def test_token_chunks_preserve_all_calibration_tokens_without_duplication():
    tokenizer = FakeTokenizer()
    records = [
        {"messages": [{"role": "user", "content": "a" * 70}, {"role": "assistant", "content": "b" * 70}]},
        {"messages": [{"role": "user", "content": "c" * 30}, {"role": "assistant", "content": "d" * 30}]},
    ]

    chunks = tokenize_records_to_chunks(records, tokenizer, max_seq_len=32)
    original = calibration_metrics(records, tokenizer, max_seq_len=512)

    assert all(0 < len(chunk) <= 32 for chunk in chunks)
    assert sum(len(chunk) for chunk in chunks) == original["total_tokens_before_blocking"]
    assert len(chunks) == 8


def test_chunked_metrics_count_long_source_records_instead_of_discarding_them():
    tokenizer = FakeTokenizer()
    records = _risk_records()

    metrics = calibration_metrics(
        records, tokenizer, max_seq_len=32, chunk_long_records=True
    )

    assert metrics["used_examples"] == 10
    assert metrics["discarded_examples"] == 0
    assert metrics["blocks"] >= 1


def test_token_match_calibration_uses_exact_common_prefix_budget_and_full_blocks():
    tokenizer = FakeTokenizer()
    records = [
        {
            "messages": [
                {"role": "user", "content": "a" * 20},
                {"role": "assistant", "content": "b" * 20},
            ],
        },
        {
            "messages": [
                {"role": "user", "content": "c" * 20},
                {"role": "assistant", "content": "d" * 20},
            ],
        },
    ]

    chunks, metrics = token_match_calibration(
        records, tokenizer, target_tokens=32, max_seq_len=8
    )

    assert [token for chunk in chunks for token in chunk] == list(range(32))
    assert all(len(chunk) == 8 for chunk in chunks)
    assert metrics["source_examples"] == 2
    assert metrics["total_tokens_before_blocking"] == 32
    assert metrics["original_total_tokens_before_matching"] == 112
    assert metrics["discarded_tokens"] == 80
    assert metrics["blocks"] == 4
