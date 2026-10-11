from qresafe_awq_calibration.config import ExperimentConfig


def test_experiment_config_locks_the_requested_awq_setup():
    config = ExperimentConfig()

    assert config.model_id == "meta-llama/Llama-2-7b-chat-hf"
    assert config.seed == 42
    assert config.max_calib_samples == 128
    assert config.max_calib_seq_len == 512
    assert config.quant_config == {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }


def test_experiment_config_rejects_non_int4_variants():
    try:
        ExperimentConfig(quant_config={"w_bit": 8})
    except ValueError as exc:
        assert "INT4" in str(exc)
    else:
        raise AssertionError("non-INT4 configuration was accepted")
