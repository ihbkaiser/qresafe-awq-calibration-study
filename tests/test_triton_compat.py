import sys
import types


def test_legacy_triton_interleave_is_installed_with_existing_primitives(monkeypatch):
    from qresafe_awq_calibration.runtime import install_qresafe_triton_compat

    fake_triton = types.ModuleType("triton")
    fake_language = types.ModuleType("triton.language")
    fake_triton.language = fake_language
    fake_triton.__version__ = "2.1.0-test"
    fake_triton.jit = lambda fn: fn

    def expand_dims(value, axis):
        return ("expand_dims", value, axis)

    def broadcast_to(value, shape):
        return ("broadcast", value, shape)

    def view(value, shape, _builder=None):
        return ("view", value, shape)

    fake_language.expand_dims = expand_dims
    fake_language.broadcast_to = broadcast_to
    fake_language.reshape = lambda value, shape: ("reshape", value, shape)
    fake_language.view = view

    class FakeTensor:
        shape = (3, 4)

        def __getitem__(self, key):
            return ("slice", key)

    fake_language.tensor = FakeTensor
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.language", fake_language)

    result = install_qresafe_triton_compat()

    assert result == {"available": True, "patched": True, "triton": "2.1.0-test"}
    assert hasattr(fake_language, "interleave")

    tensor = FakeTensor()
    assert fake_language.interleave(tensor, tensor) == (
        "view",
        (
            "broadcast",
            ("expand_dims", tensor, 2),
            (3, 4, 2),
        ),
        (3, 8),
    )
    assert tensor.reshape(8, _builder="builder") == ("view", tensor, (8,))


def test_existing_triton_interleave_is_not_replaced(monkeypatch):
    from qresafe_awq_calibration.runtime import install_qresafe_triton_compat

    fake_triton = types.ModuleType("triton")
    fake_language = types.ModuleType("triton.language")
    fake_triton.language = fake_language
    fake_triton.__version__ = "new-test"
    sentinel = object()
    fake_language.interleave = sentinel
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.language", fake_language)

    result = install_qresafe_triton_compat()

    assert result == {"available": True, "patched": False, "triton": "new-test"}
    assert fake_language.interleave is sentinel


def test_legacy_triton_dot_gets_qresafe_accumulator_adapter(monkeypatch):
    from qresafe_awq_calibration.runtime import install_qresafe_triton_compat

    fake_triton = types.ModuleType("triton")
    fake_language = types.ModuleType("triton.language")
    fake_triton.language = fake_language
    fake_triton.__version__ = "2.1.0-test"
    fake_triton.jit = lambda fn: fn

    class LegacyDotResult:
        def __add__(self, accumulator):
            return ("dot_accumulated", accumulator)

    def legacy_dot(input_value, other_value, allow_tf32=True, out_dtype="fp32", _builder=None):
        return LegacyDotResult()

    fake_language.dot = legacy_dot
    fake_language.float32 = "fp32"
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.language", fake_language)

    install_qresafe_triton_compat()

    assert fake_language.dot is not legacy_dot
    assert fake_language.dot("a", "b", "acc", out_dtype="fp32") == (
        "dot_accumulated",
        "acc",
    )


def test_legacy_triton_tensor_reshape_is_replaced_when_present(monkeypatch):
    from qresafe_awq_calibration.runtime import install_qresafe_triton_compat

    fake_triton = types.ModuleType("triton")
    fake_language = types.ModuleType("triton.language")
    fake_triton.language = fake_language
    fake_triton.__version__ = "2.1.0-test"
    fake_triton.jit = lambda fn: fn

    def view(value, shape, _builder=None):
        return ("view", value, shape)

    fake_language.view = view
    fake_language.reshape = lambda value, shape: ("reshape", value, shape)

    class FakeTensor:
        # Triton 2.1 exposes a frontend reshape method, but the upstream
        # Q-resafe kernel still fails for this shape on Llama GEMM.
        def reshape(self, shape, _builder=None):
            return ("legacy-reshape", shape)

    fake_language.tensor = FakeTensor
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.language", fake_language)

    install_qresafe_triton_compat()

    tensor = FakeTensor()
    assert tensor.reshape(8, _builder="builder") == (
        "view",
        tensor,
        (8,),
    )
