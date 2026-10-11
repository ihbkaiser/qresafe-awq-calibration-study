from modal_app import (
    DATASET_RUNTIME_PACKAGES,
    GPU_TYPE,
    PROJECT_REMOTE_ROOT,
    QRESAFE_REMOTE_ROOT,
    _condition_artifacts_complete,
    _quantized_model_complete,
    _retire_condition_evaluation_outputs,
)


def test_modal_runner_is_locked_to_l40s_and_expected_mounts():
    assert GPU_TYPE == "L40S"
    assert PROJECT_REMOTE_ROOT == "/root/project"
    assert QRESAFE_REMOTE_ROOT == "/root/Qresafe/quant-without-ft"


def test_modal_image_includes_zstandard_for_wikitext_dataset_downloads():
    assert any(package.startswith("zstandard==") for package in DATASET_RUNTIME_PACKAGES)


def test_condition_resume_requires_all_locked_artifacts(tmp_path):
    assert not _condition_artifacts_complete(tmp_path, "fp16")

    (tmp_path / "fp16.summary.json").write_text(
        '{"condition":"fp16","n":510,"asr_heldout_refusal_heuristic":0.5}',
        encoding="utf-8",
    )
    (tmp_path / "fp16.ppl.json").write_text('{"ppl":7.0}', encoding="utf-8")
    assert _condition_artifacts_complete(tmp_path, "fp16")

    assert not _condition_artifacts_complete(tmp_path, "pile")
    model_dir = tmp_path / "models" / "awq_pile"
    model_dir.mkdir(parents=True)
    (tmp_path / "awq_pile.summary.json").write_text(
        '{"condition":"pile","n":510,"asr_heldout_refusal_heuristic":0.5}',
        encoding="utf-8",
    )
    (tmp_path / "awq_pile.ppl.json").write_text('{"ppl":7.0}', encoding="utf-8")
    assert not _condition_artifacts_complete(tmp_path, "pile")
    (model_dir / "quantization_metadata.json").write_text(
        '{"condition":"pile","bits":4,"max_calib_samples":128,"max_calib_seq_len":512,'
        '"calibration_block_length_policy":"qresafe_default_512"}',
        encoding="utf-8",
    )
    assert _condition_artifacts_complete(tmp_path, "pile")


def test_quantized_model_resume_requires_saved_model_and_metadata(tmp_path):
    assert not _quantized_model_complete(tmp_path, "pile")

    model_dir = tmp_path / "models" / "awq_pile"
    model_dir.mkdir(parents=True)
    (model_dir / "quantization_metadata.json").write_text(
        '{"condition":"pile","bits":4,"max_calib_samples":128,"max_calib_seq_len":512,'
        '"calibration_block_length_policy":"qresafe_default_512"}',
        encoding="utf-8",
    )
    assert not _quantized_model_complete(tmp_path, "pile")

    for filename in ("model.safetensors", "config.json", "tokenizer.json"):
        (model_dir / filename).write_text("saved", encoding="utf-8")
    assert _quantized_model_complete(tmp_path, "pile")


def test_risk_resume_rejects_a_legacy_512_token_calibration_model(tmp_path):
    model_dir = tmp_path / "models" / "awq_risk1"
    model_dir.mkdir(parents=True)
    for filename in ("model.safetensors", "config.json", "tokenizer.json"):
        (model_dir / filename).write_text("saved", encoding="utf-8")
    (model_dir / "quantization_metadata.json").write_text(
        '{"condition":"risk1","bits":4,"max_calib_samples":10,"max_calib_seq_len":512,'
        '"calibration_block_length_policy":"common_risk_128"}',
        encoding="utf-8",
    )

    assert not _quantized_model_complete(tmp_path, "risk1")


def test_requantization_retires_stale_evaluation_outputs_without_deleting_them(tmp_path):
    for suffix in ("jsonl", "summary.json", "ppl.json"):
        path = tmp_path / f"awq_risk1.{suffix}"
        path.write_text(f"old-{suffix}", encoding="utf-8")

    _retire_condition_evaluation_outputs(tmp_path, "risk1")

    assert not (tmp_path / "awq_risk1.jsonl").exists()
    stale = list(tmp_path.glob("awq_risk1.*.stale*"))
    assert len(stale) == 3
    assert sorted(path.read_text(encoding="utf-8") for path in stale) == [
        "old-jsonl",
        "old-ppl.json",
        "old-summary.json",
    ]


def test_token_matched_resume_requires_the_token_matched_metadata(tmp_path):
    model_dir = tmp_path / "models" / "awq_risk1"
    model_dir.mkdir(parents=True)
    for filename in ("model.safetensors", "config.json", "tokenizer.json"):
        (model_dir / filename).write_text("saved", encoding="utf-8")
    (model_dir / "quantization_metadata.json").write_text(
        '{"condition":"risk1","bits":4,"max_calib_samples":10,'
        '"max_calib_seq_len":128,"calibration_block_length_policy":'
        '"token_matched_common_384","token_match_target_tokens":384}',
        encoding="utf-8",
    )

    assert _quantized_model_complete(tmp_path, "risk1", protocol="token_matched")
    assert not _quantized_model_complete(tmp_path, "risk1", protocol="standard")
