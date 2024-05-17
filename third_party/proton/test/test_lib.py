import triton._C.libproton.proton as libproton
import tempfile
import pathlib
from triton.profiler.profile import _select_backend


def test_record():
    id0 = libproton.record_scope()
    id1 = libproton.record_scope()
    assert id1 == id0 + 1


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


def test_session():
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        session_id = libproton.start(f.name.split(".")[0], "shadow", "tree", _select_backend())
        libproton.deactivate(session_id)
        libproton.activate(session_id)
        libproton.finalize(session_id, "hatchet")
        libproton.finalize_all("hatchet")
        assert pathlib.Path(f.name).exists()


def test_add_metrics():
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        libproton.start(f.name.split(".")[0], "shadow", "tree", _select_backend())
        id1 = libproton.record_scope()
        libproton.enter_scope(id1, "one")
        libproton.add_metrics(id1, {"a": 1.0, "b": 2.0})
        libproton.exit_scope(id1, "one")
        libproton.finalize_all("hatchet")
        assert pathlib.Path(f.name).exists()
