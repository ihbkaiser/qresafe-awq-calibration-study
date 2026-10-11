from qresafe_awq_calibration.data import split_advbench_calibration_and_eval


def test_advbench_calibration_examples_are_excluded_from_eval():
    rows = [{"source_index": i, "prompt": f"prompt-{i}"} for i in range(15)]

    calibration, evaluation = split_advbench_calibration_and_eval(
        rows, calibration_indices=[2, 7, 11]
    )

    assert [row["source_index"] for row in calibration] == [2, 7, 11]
    assert set(row["source_index"] for row in evaluation) == {
        0,
        1,
        3,
        4,
        5,
        6,
        8,
        9,
        10,
        12,
        13,
        14,
    }


def test_advbench_split_rejects_duplicate_or_unknown_indices():
    rows = [{"source_index": i} for i in range(3)]

    try:
        split_advbench_calibration_and_eval(rows, calibration_indices=[1, 1])
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("duplicate calibration indices were accepted")

    try:
        split_advbench_calibration_and_eval(rows, calibration_indices=[9])
    except ValueError as exc:
        assert "unknown" in str(exc).lower()
    else:
        raise AssertionError("unknown calibration index was accepted")
