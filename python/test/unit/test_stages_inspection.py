import triton

import os
import pathlib
import hashlib
import pytest
from triton._internal_testing import is_cuda


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


@pytest.mark.skipif(not is_cuda(), reason="only currently tested on CUDA")
def test_inspection_hash_cached_per_hook(fresh_knobs):
    # The hook's no-arg form is consulted on every kernel launch to key the
    # compilation cache; its result must be memoized per hook object so an
    # expensive key computation does not become per-launch overhead.
    key_calls = 0

    def make_hook(tag):

        def hook(self=None, stages=None, options=None, language=None, capability=None):
            if all(arg is None for arg in (stages, options, language, capability)):
                nonlocal key_calls
                key_calls += 1
                return tag, hashlib.sha256(tag.encode('utf-8')).hexdigest()

        return hook

    @triton.jit
    def k():
        return

    fresh_knobs.runtime.add_stages_inspection_hook = make_hook("first")
    for _ in range(4):
        k[(1, )]()
    assert key_calls == 1

    # Installing a different hook object must re-derive the hash.
    fresh_knobs.runtime.add_stages_inspection_hook = make_hook("second")
    k[(1, )]()
    assert key_calls == 2
