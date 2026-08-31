import triton

import os
import pathlib
import hashlib
import pytest
from triton._internal_testing import is_cuda
from triton._instrumentation import instrument, is_enabled, register_instrumentation, unregister_instrumentation


def test_instrumentation_dispatch():
    calls = []
    options = {"instrumentation_mode": "gsan,consan"}

    def callback(pm, callback_options):
        calls.append((pm, callback_options))

    assert is_enabled(options, "consan")
    assert not is_enabled(options, "proton")
    register_instrumentation(name="consan", point="prepare-captures", backend="test", callback=callback)
    try:
        assert instrument("pm", name="consan", point="prepare-captures", backend="test", options=options)
        assert calls == [("pm", options)]
    finally:
        unregister_instrumentation(name="consan", point="prepare-captures", backend="test")


def test_dynamic_instrumentation_and_dialect_loading():
    calls = []
    options = {"instrumentation_mode": "proton"}

    assert not instrument("context", name="proton", point="load-dialects", backend="test")
    register_instrumentation(name="proton", point="load-dialects", backend="test",
                             callback=lambda context, _options: calls.append(("dialects", context)))
    register_instrumentation(name="proton", point="pre-lower", backend="test",
                             callback=lambda _pm, _options: calls.append("passes"))
    try:
        assert instrument("context", name="proton", point="load-dialects", backend="test")
        assert instrument("pm", name="proton", point="pre-lower", backend="test", options=options)
        assert calls == [("dialects", "context"), "passes"]
    finally:
        unregister_instrumentation(name="proton", point="load-dialects", backend="test")
        unregister_instrumentation(name="proton", point="pre-lower", backend="test")

    assert is_enabled(options, "proton")
    assert not instrument("context", name="proton", point="load-dialects", backend="test")


@pytest.mark.skipif(not is_cuda(), reason="only currently tested on CUDA")
def test_inspection(monkeypatch, fresh_knobs, tmp_path: pathlib.Path):
    stage_name = 'make_ttgir'
    curr_repro_path = tmp_path / ("repro_prefix." + stage_name + ".repro.mlir")
    repro_path = tmp_path / "repro_prefix"

    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_REPRODUCER_PATH", str(repro_path))

    inspect_stages_hook_called = False
    make_ttgir_wrapper_called = False

    def get_key():
        return pathlib.Path(__file__).read_text()

    def get_hash():
        return hashlib.sha256(get_key().encode('utf-8')).hexdigest()

    def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
        if all(arg is None for arg in (stages, options, language, capability)):
            return get_key(), get_hash()
        nonlocal inspect_stages_hook_called
        inspect_stages_hook_called = True

        def make_ttgir_wrapper(src, metadata, options, capability):
            nonlocal make_ttgir_wrapper_called
            make_ttgir_wrapper_called = True
            return self.make_ttgir(src, metadata, options, capability)

        stages["ttgir"] = lambda src, metadata: make_ttgir_wrapper(src, metadata, options, capability)

    @triton.jit
    def k1():
        return

    @triton.jit
    def k2():
        return

    # Run once to get the clean/golden repro dump
    k1[(1, )]()
    assert not inspect_stages_hook_called and not make_ttgir_wrapper_called
    assert os.path.exists(curr_repro_path)
    golden_repro = curr_repro_path.read_text()
    curr_repro_path.unlink()

    # Setup hook and call again, check if hooks got called
    fresh_knobs.runtime.add_stages_inspection_hook = inspect_stages_hook
    k2[(1, )]()
    assert inspect_stages_hook_called and make_ttgir_wrapper_called
    assert os.path.exists(curr_repro_path)
    hook_repro = curr_repro_path.read_text()

    # Check that repros match
    assert golden_repro.replace('k1', 'dummy') == hook_repro.replace('k2', 'dummy')
