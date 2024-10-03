import json
import triton.profiler as proton
import tempfile
import pathlib


def test_profile():
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        session_id0 = proton.start(f.name.split(".")[0])
        proton.activate()
        proton.deactivate()
        proton.finalize()
        assert session_id0 == 0

    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        session_id1 = proton.start(f.name.split(".")[0])
        proton.activate(session_id1)
        proton.deactivate(session_id1)
        proton.finalize(session_id1)
        assert session_id1 == session_id0 + 1

    session_id2 = proton.start("test")
    proton.activate(session_id2)
    proton.deactivate(session_id2)
    proton.finalize()
    assert session_id2 == session_id1 + 1
    assert pathlib.Path("test.hatchet").exists()
    pathlib.Path("test.hatchet").unlink()


def test_profile_decorator():
    f = tempfile.NamedTemporaryFile(delete=True)
    name = f.name.split(".")[0]

    @proton.profile(name=name)
    def foo0(a, b):
        return a + b

    foo0(1, 2)
    proton.finalize()
    assert pathlib.Path(f.name).exists()

    f.close()

    @proton.profile
    def foo1(a, b):
        return a + b

    foo1(1, 2)
    proton.finalize()
    assert pathlib.Path(proton.DEFAULT_PROFILE_NAME + ".hatchet").exists()


def test_scope():
    # Scope can be annotated even when profiling is off
    with proton.scope("test"):
        pass

    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        proton.start(f.name.split(".")[0])
        with proton.scope("test"):
            pass

        @proton.scope("test")
        def foo():
            pass

        foo()

        proton.enter_scope("test")
        proton.exit_scope()
        proton.finalize()
        assert pathlib.Path(f.name).exists()


def test_hook():
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        session_id0 = proton.start(f.name.split(".")[0], hook="triton")
        proton.activate(session_id0)
        proton.deactivate(session_id0)
        proton.finalize(None)
        assert pathlib.Path(f.name).exists()


def test_scope_metrics():
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        session_id = proton.start(f.name.split(".")[0])
        # Test different scope creation methods
        with proton.scope("test0", {"a": 1.0}):
            pass

        @proton.scope("test1", {"a": 1.0})
        def foo():
            pass

        foo()

        # After deactivation, the metrics should be ignored
        proton.deactivate(session_id)
        proton.enter_scope("test2", metrics={"a": 1.0})
        proton.exit_scope()

        # Metrics should be recorded again after reactivation
        proton.activate(session_id)
        proton.enter_scope("test3", metrics={"a": 1.0})
        proton.exit_scope()

        proton.enter_scope("test3", metrics={"a": 1.0})
        proton.exit_scope()

        proton.finalize()
        assert pathlib.Path(f.name).exists()
        data = json.load(f)
        assert len(data[0]["children"]) == 3
        for child in data[0]["children"]:
            if child["frame"]["name"] == "test3":
                assert child["metrics"]["a"] == 2.0


def test_scope_properties():
    with open("test.hatchet", "w+") as f:
        proton.start(f.name.split(".")[0])
        # Test different scope creation methods
        # Different from metrics, properties could be str
        with proton.scope("test0", properties={"a": "1"}):
            pass

        @proton.scope("test1", properties={"a": "1"})
        def foo():
            pass

        foo()

        # Properties do not aggregate
        proton.enter_scope("test2", properties={"a": 1.0})
        proton.exit_scope()

        proton.enter_scope("test2", properties={"a": 1.0})
        proton.exit_scope()

        proton.finalize()
        assert pathlib.Path(f.name).exists()
        data = json.load(f)
        for child in data[0]["children"]:
            if child["frame"]["name"] == "test2":
                assert child["metrics"]["a"] == 1.0
            elif child["frame"]["name"] == "test0":
                assert child["metrics"]["a"] == "1"


def test_throw():
    # Catch an exception thrown by c++
    session_id = 100
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        activate_error = ""
        try:
            session_id = proton.start(f.name.split(".")[0])
            proton.activate(session_id + 1)
        except Exception as e:
            activate_error = str(e)
        finally:
            proton.finalize()
        assert "Session has not been initialized: " + str(session_id + 1) in activate_error

        deactivate_error = ""
        try:
            session_id = proton.start(f.name.split(".")[0])
            proton.deactivate(session_id + 1)
        except Exception as e:
            deactivate_error = str(e)
        finally:
            proton.finalize()
        assert "Session has not been initialized: " + str(session_id + 1) in deactivate_error
