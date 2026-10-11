import pytest

from qresafe_awq_calibration.data import (
    build_risk1_records,
    build_risk2_records,
    build_risk3_records,
)


def test_risk3_requires_a_corresponding_response_column():
    rows = [{"prompt": "a harmful behavior"}]

    with pytest.raises(ValueError, match="response"):
        build_risk3_records(rows, sample_indices=[0], response_column="response")


def test_risk3_keeps_prompt_and_corresponding_response_in_one_pair():
    rows = [{"source_index": 4, "prompt": "a benchmark prompt", "response": "corresponding response"}]

    record = build_risk3_records(rows, sample_indices=[4])[0]

    assert record["source_index"] == 4
    assert record["messages"] == [
        {"role": "user", "content": "a benchmark prompt"},
        {"role": "assistant", "content": "corresponding response"},
    ]


def test_risk3_reconstructs_alignmentresearch_advbench_schema():
    rows = [
        {
            "source_index": 7,
            "instructions": ["benchmark instruction"],
            "proxy_gen_target": "paired harmful target",
        }
    ]

    record = build_risk3_records(
        rows,
        sample_indices=[7],
        response_column="proxy_gen_target",
    )[0]

    assert record["response_column"] == "proxy_gen_target"
    assert record["messages"] == [
        {"role": "user", "content": "benchmark instruction"},
        {"role": "assistant", "content": "paired harmful target"},
    ]


def test_risk3_accepts_alignmentresearch_content_field():
    rows = [
        {
            "source_index": 8,
            "instructions": "",
            "content": ["content instruction"],
            "proxy_gen_target": "content paired target",
        }
    ]

    record = build_risk3_records(rows, sample_indices=[8], response_column="proxy_gen_target")[0]

    assert record["messages"][0]["content"] == "content instruction"


def test_risk1_extracts_the_first_user_assistant_pair():
    rows = [
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "benign question"},
                {"role": "assistant", "content": "benign answer"},
            ]
        }
    ]

    record = build_risk1_records(rows, sample_indices=[0])[0]

    assert [message["role"] for message in record["messages"]] == ["user", "assistant"]
    assert record["source"] == "UltraChat"


def test_risk2_reconstruction_has_ten_traceable_examples():
    records = build_risk2_records()

    assert len(records) == 10
    assert all(record["source"] == "Q-resafe description reconstruction" for record in records)
    assert [record["source_index"] for record in records] == list(range(10))
