import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import quantize_model


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        return "\n".join(message["content"] for message in messages)

    def encode(self, text):
        return list(range(len(text)))

    def save_pretrained(self, output_dir):
        Path(output_dir, "tokenizer.json").write_text("{}", encoding="utf-8")


class _FakeModel:
    def __init__(self):
        self.quantize_kwargs = None

    def quantize(self, tokenizer, **kwargs):
        self.quantize_kwargs = kwargs

    def save_quantized(self, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "model.safetensors").write_text("saved", encoding="utf-8")


def _write_risk2(path: Path, content: str = "question"):
    records = [
        {
            "sample_id": f"risk2-{index}",
            "source": "reconstruction",
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": content},
            ],
        }
        for index in range(10)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_risk_quantization_uses_the_common_128_token_block(monkeypatch, tmp_path):
    fake_model = _FakeModel()
    monkeypatch.setattr(quantize_model, "load_tokenizer", lambda _: _FakeTokenizer())
    monkeypatch.setattr(
        quantize_model,
        "import_qresafe_auto",
        lambda _: type("FakeAuto", (), {"from_pretrained": lambda *args, **kwargs: fake_model}),
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_risk2(data_dir / "risk2.jsonl", content="x" * 20)
    output_dir = tmp_path / "model"

    quantize_model.quantize_condition(
        "risk2", "fake/model", output_dir, data_dir, tmp_path / "Qresafe", "cpu"
    )

    assert fake_model.quantize_kwargs["max_calib_samples"] == 128
    assert fake_model.quantize_kwargs["max_calib_seq_len"] == 128
    assert all(
        0 < len(chunk) <= 128 for chunk in fake_model.quantize_kwargs["calib_data"]
    )
    metadata = json.loads((output_dir / "quantization_metadata.json").read_text())
    assert metadata["max_calib_seq_len"] == 128
    assert metadata["max_calib_samples"] == 10


def test_risk_quantization_fails_before_model_load_without_a_complete_block(monkeypatch, tmp_path):
    model_loaded = False

    def fail_if_loaded(*args, **kwargs):
        nonlocal model_loaded
        model_loaded = True
        raise AssertionError("model must not load when calibration budget is invalid")

    monkeypatch.setattr(quantize_model, "load_tokenizer", lambda _: _FakeTokenizer())
    monkeypatch.setattr(
        quantize_model,
        "import_qresafe_auto",
        lambda _: type("FakeAuto", (), {"from_pretrained": fail_if_loaded}),
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_risk2(data_dir / "risk2.jsonl", content="x")

    with pytest.raises(ValueError, match="no complete calibration block"):
        quantize_model.quantize_condition(
            "risk2", "fake/model", tmp_path / "model", data_dir, tmp_path / "Qresafe", "cpu"
        )

    assert model_loaded is False


def test_token_matched_risk_quantization_uses_common_384_token_budget(monkeypatch, tmp_path):
    fake_model = _FakeModel()
    monkeypatch.setattr(quantize_model, "load_tokenizer", lambda _: _FakeTokenizer())
    monkeypatch.setattr(
        quantize_model,
        "import_qresafe_auto",
        lambda _: type("FakeAuto", (), {"from_pretrained": lambda *args, **kwargs: fake_model}),
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_risk2(data_dir / "risk2.jsonl", content="x" * 20)
    output_dir = tmp_path / "model"

    quantize_model.quantize_condition(
        "risk2",
        "fake/model",
        output_dir,
        data_dir,
        tmp_path / "Qresafe",
        "cpu",
        token_matched=True,
    )

    flattened = [token for chunk in fake_model.quantize_kwargs["calib_data"] for token in chunk]
    assert len(flattened) == 384
    assert all(len(chunk) == 128 for chunk in fake_model.quantize_kwargs["calib_data"])
    metadata = json.loads((output_dir / "quantization_metadata.json").read_text())
    assert metadata["calibration_block_length_policy"] == "token_matched_common_384"
    assert metadata["calibration_metrics"]["total_tokens_before_blocking"] == 384
