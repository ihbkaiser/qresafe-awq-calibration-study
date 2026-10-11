from qresafe_awq_calibration.safety import compute_asr, is_refusal


def test_refusal_detector_is_deterministic_for_common_refusal_prefixes():
    assert is_refusal("I can't help with that request.") is True
    assert is_refusal("Here are the steps you requested.") is False


def test_asr_is_fraction_of_non_refusals():
    responses = [
        "I can't help with that.",
        "Here is an answer.",
        "I must refuse this request.",
        "A direct answer follows.",
    ]

    assert compute_asr(responses) == 0.5
