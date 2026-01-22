"""
Test module for proton's CPP API functionality.
No GPU kernel should be declared in this test.
Python API correctness tests involving GPU kernels should be placed in `test_api.py`.
Profile correctness tests involving GPU kernels should be placed in `test_profile.py`.
"""
import pathlib
import pytest

import triton._C.libproton.proton as libproton
from triton.profiler.profile import _select_backend


def test_record():
    id0 = libproton.record_scope()
    id1 = libproton.record_scope()
    assert id1 == id0 + 1


def test_state():
    libproton.enter_state("zero")
    libproton.exit_state()


def test_scope():
    id0 = libproton.record_scope()
    libproton.enter_scope(id0, "zero")
    id1 = libproton.record_scope()
    libproton.enter_scope(id1, "one")
    libproton.exit_scope(id1, "one")
    libproton.exit_scope(id0, "zero")


def test_op():
    id0 = libproton.record_scope()
    libproton.enter_op(id0, "zero")
    libproton.exit_op(id0, "zero")


@pytest.mark.parametrize("source", ["shadow", "python"])
def test_context(source: str, tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_context.hatchet"
    session_id = libproton.start(str(temp_file.with_suffix("")), source, "tree", _select_backend())
    depth = libproton.get_context_depth(session_id)
    libproton.finalize(session_id, "hatchet")
    assert depth >= 0
    assert temp_file.exists()


def test_session(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_session.hatchet"
    session_id = libproton.start(str(temp_file.with_suffix("")), "shadow", "tree", _select_backend())
    libproton.deactivate(session_id, False)
    libproton.activate(session_id)
    libproton.finalize(session_id, "hatchet")
    libproton.finalize_all("hatchet")
    assert temp_file.exists()


def test_add_metrics(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_add_metrics.hatchet"
    libproton.start(str(temp_file.with_suffix("")), "shadow", "tree", _select_backend())
    id1 = libproton.record_scope()
    libproton.enter_scope(id1, "one")
    libproton.add_metrics(id1, {"a": 1.0, "b": 2.0})
    libproton.exit_scope(id1, "one")
    libproton.finalize_all("hatchet")
    assert temp_file.exists()


def test_init_function_metadata(tmp_path: pathlib.Path):
    metadata_file = tmp_path / "meta.json"
    metadata_file.write_text("{}")
    libproton.init_function_metadata(
        0,
        "dummy_fn",
        [(0, "root")],
        [],
        str(metadata_file),
    )


def test_instrumented_op_entry_exit():
    libproton.enter_instrumented_op(0, 0, 0, 0)
    libproton.exit_instrumented_op(0, 0, 0, 0)


def test_set_metric_kernels():
    libproton.set_metric_kernels(0, 0, 0)


def test_tensor_metric_construction():
    metric = libproton.TensorMetric(123, libproton.metric_double_index)
    assert metric.ptr == 123
    assert metric.index == libproton.metric_double_index
