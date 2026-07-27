from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import pytest


_CALIBRATION_SCOPE = "__triton_ci_timing_calibration__"
_STATE: "_TimingState | None" = None


def _safe_component(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


class _TimingState:
    def __init__(self, config: Any, root: Path) -> None:
        worker_input = getattr(config, "workerinput", None)
        if worker_input is not None:
            self.invocation = worker_input.get("triton_ci_timing_invocation", f"orphan-{time.time_ns()}")
            self.worker_id = worker_input.get("workerid", "worker")
            self.role = "worker"
        else:
            self.invocation = f"{time.time_ns()}-{os.getpid()}"
            self.worker_id = "controller"
            num_processes = getattr(config.option, "numprocesses", None)
            self.role = "controller" if num_processes not in (None, 0, "0") else "main"

        self.label = os.environ.get("TRITON_CI_TIMING_LABEL", "unlabeled")
        process_name = f"{_safe_component(self.worker_id)}-{os.getpid()}"
        self.output_dir = root / _safe_component(self.label) / _safe_component(self.invocation) / process_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile_base = self.output_dir / "gpu"

        self.compile_events: list[list[int]] = []
        self.compile_lock = threading.Lock()
        self.listener_wrappers: list[Callable[..., None]] = []
        self.errors: list[str] = []
        self.calibration: dict[str, int] | None = None
        self.proton_session: int | None = None
        self.mono_start_ns: int | None = None
        self.mono_tests_end_ns: int | None = None
        self.mono_end_ns: int | None = None
        self.wall_start_ns: int | None = None
        self.wall_end_ns: int | None = None
        self.exit_status: int | None = None
        self.finished = False

    def install_compilation_listener(self) -> None:
        from triton import knobs

        previous = knobs.compilation.listener
        if getattr(previous, "_triton_ci_timing_owner", None) is self:
            return

        def listener(*, src, metadata, metadata_group, times, cache_hit) -> None:
            end_ns = time.monotonic_ns()
            total_ns = times.total * 1000
            event = [
                end_ns - total_ns,
                end_ns,
                int(cache_hit),
                times.ir_initialization * 1000,
                times.total_lowering * 1000,
                times.store_results * 1000,
            ]
            with self.compile_lock:
                self.compile_events.append(event)
            if previous is not None:
                previous(
                    src=src,
                    metadata=metadata,
                    metadata_group=metadata_group,
                    times=times,
                    cache_hit=cache_hit,
                )

        listener._triton_ci_timing_owner = self  # type: ignore[attr-defined]
        self.listener_wrappers.append(listener)
        knobs.compilation.listener = listener

    def start(self) -> None:
        self.wall_start_ns = time.time_ns()
        self.mono_start_ns = time.monotonic_ns()
        self.install_compilation_listener()
        if self.role == "controller":
            return

        try:
            import triton.profiler as proton

            self.proton_session = proton.start(
                str(self.profile_base),
                backend="cupti",
                context="shadow",
                data="trace",
            )
            enter_before_ns = time.monotonic_ns()
            proton.enter_scope(_CALIBRATION_SCOPE)
            enter_after_ns = time.monotonic_ns()
            time.sleep(0.002)
            exit_before_ns = time.monotonic_ns()
            proton.exit_scope(_CALIBRATION_SCOPE)
            exit_after_ns = time.monotonic_ns()
            self.calibration = {
                "enter_before_ns": enter_before_ns,
                "enter_after_ns": enter_after_ns,
                "exit_before_ns": exit_before_ns,
                "exit_after_ns": exit_after_ns,
            }
        except Exception:
            self.errors.append("proton start failed:\n" + traceback.format_exc())

    def finish(self, exit_status: int | None) -> None:
        if self.finished:
            return
        self.finished = True
        self.exit_status = exit_status
        self.mono_tests_end_ns = time.monotonic_ns()
        if self.proton_session is not None:
            try:
                import triton.profiler as proton

                proton.finalize(self.proton_session)
            except Exception:
                self.errors.append("proton finalize failed:\n" + traceback.format_exc())
        self.mono_end_ns = time.monotonic_ns()
        self.wall_end_ns = time.time_ns()
        self.write_summary()

    def write_summary(self) -> None:
        with self.compile_lock:
            compile_events = list(self.compile_events)
        payload = {
            "schema_version": 1,
            "label": self.label,
            "invocation": self.invocation,
            "role": self.role,
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_start_ns": self.wall_start_ns,
            "wall_end_ns": self.wall_end_ns,
            "mono_start_ns": self.mono_start_ns,
            "mono_tests_end_ns": self.mono_tests_end_ns,
            "mono_end_ns": self.mono_end_ns,
            "exit_status": self.exit_status,
            "calibration": self.calibration,
            "profile": str(self.profile_base.with_suffix(".chrome_trace")),
            "errors": self.errors,
            # Each event is [start, end, cache_hit, ir_init, lowering, store], all times in ns.
            "compile_events": compile_events,
        }
        destination = self.output_dir / "summary.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")))
        os.replace(temporary, destination)


def pytest_configure(config: Any) -> None:
    global _STATE
    root = os.environ.get("TRITON_CI_TIMING_DIR")
    if not root:
        return
    _STATE = _TimingState(config, Path(root))


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    if _STATE is not None:
        node.workerinput["triton_ci_timing_invocation"] = _STATE.invocation


def pytest_sessionstart(session: Any) -> None:
    if _STATE is not None:
        _STATE.start()


def pytest_runtest_setup(item: Any) -> None:
    if _STATE is not None:
        _STATE.install_compilation_listener()


def pytest_runtest_call(item: Any) -> None:
    if _STATE is not None:
        _STATE.install_compilation_listener()


def pytest_runtest_teardown(item: Any) -> None:
    if _STATE is not None:
        _STATE.install_compilation_listener()


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    if _STATE is not None:
        _STATE.finish(exitstatus)


def pytest_unconfigure(config: Any) -> None:
    if _STATE is not None and not _STATE.finished:
        _STATE.finish(None)
