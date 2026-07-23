import subprocess

from python import build_helpers


def _create_rocm_headers(root):
    include_dir = root / "include"
    (include_dir / "hip").mkdir(parents=True)
    (include_dir / "hip" / "hip_runtime.h").touch()
    (include_dir / "rocprofiler-sdk").mkdir()
    (include_dir / "rocprofiler-sdk" / "fwd.h").touch()
    return include_dir


def test_find_therock_rocm_include_dir(monkeypatch, tmp_path):
    include_dir = _create_rocm_headers(tmp_path)
    calls = []

    monkeypatch.setattr(build_helpers.sys, "executable", "/missing/bin/python")
    monkeypatch.setattr(build_helpers.shutil, "which", lambda command: f"/venv/bin/{command}")

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return f"{tmp_path}\n"

    monkeypatch.setattr(build_helpers.subprocess, "check_output", check_output)

    assert build_helpers.find_therock_rocm_include_dir() == str(include_dir)
    expected_env = build_helpers.os.environ.copy()
    expected_env.pop("PYTHONPATH", None)
    assert calls == [
        (
            ["/venv/bin/rocm-sdk", "path", "--root"],
            {
                "stderr": subprocess.DEVNULL,
                "text": True,
                "env": expected_env,
            },
        )
    ]


def test_find_therock_rocm_include_dir_missing_command(monkeypatch):
    monkeypatch.setattr(build_helpers.sys, "executable", "/missing/bin/python")
    monkeypatch.setattr(build_helpers.shutil, "which", lambda command: None)
    assert build_helpers.find_therock_rocm_include_dir() is None


def test_find_therock_rocm_include_dir_missing_headers(monkeypatch, tmp_path):
    monkeypatch.setattr(build_helpers.sys, "executable", "/missing/bin/python")
    monkeypatch.setattr(build_helpers.shutil, "which", lambda command: f"/venv/bin/{command}")
    monkeypatch.setattr(
        build_helpers.subprocess,
        "check_output",
        lambda *args, **kwargs: str(tmp_path),
    )
    assert build_helpers.find_therock_rocm_include_dir() is None


def test_find_therock_rocm_include_dir_failed_command(monkeypatch):
    monkeypatch.setattr(build_helpers.sys, "executable", "/missing/bin/python")
    monkeypatch.setattr(build_helpers.shutil, "which", lambda command: f"/venv/bin/{command}")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(build_helpers.subprocess, "check_output", fail)
    assert build_helpers.find_therock_rocm_include_dir() is None
