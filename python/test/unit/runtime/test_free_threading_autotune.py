import importlib
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest


class _ObservedRLock:

    def __init__(self, on_contention=None):
        self._lock = threading.RLock()
        self._attempt_lock = threading.Lock()
        self._attempts = 0
        self.second_attempted = threading.Event()
        self.on_contention = on_contention

    def __enter__(self):
        with self._attempt_lock:
            self._attempts += 1
            if self._attempts == 2:
                self.second_attempted.set()
        if not self._lock.acquire(blocking=False):
            if self.on_contention is not None:
                self.on_contention()
            self._lock.acquire()
        return self

    def __exit__(self, *exc_info):
        self._lock.release()


class _FakeJITFunction:

    def __init__(self):
        self.fn = lambda: None

    def run(self, *args, **kwargs):
        return kwargs["BLOCK"]


def _make_autotuner(monkeypatch, blocks=(1, 2)):
    autotuner = importlib.import_module("triton.runtime.autotuner")
    local = threading.local()
    active = types.SimpleNamespace(
        set_device=lambda device: setattr(local, "device", device),
        get_current_device=lambda: local.device,
    )
    monkeypatch.setattr(autotuner, "driver", types.SimpleNamespace(active=active))
    monkeypatch.setattr(autotuner, "JITFunction", _FakeJITFunction)
    configs = [autotuner.Config({"BLOCK": block}) for block in blocks]
    tuner = autotuner.Autotuner(
        _FakeJITFunction(),
        arg_names=["N"],
        configs=configs,
        key=["N"],
        reset_to_zero=None,
        restore_value=None,
        do_bench=lambda kernel_call, quantiles: (0.0, 0.0, 0.0),
    )
    return autotuner, tuner, active


def test_autotuner_concurrent_failure_retries_cleanly(monkeypatch):
    _, tuner, active = _make_autotuner(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    observed_lock = _ObservedRLock()
    tuner._cache_lock = observed_lock

    def bench(*args, config, nargs, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(10)
            raise RuntimeError("first benchmark failed")
        return (float(config.kwargs["BLOCK"] != 1), ) * 3

    tuner._bench = bench

    def run():
        active.set_device(0)
        return tuner.run(1, grid=(1, ), warmup=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run)
        try:
            assert entered.wait(10)
            second = pool.submit(run)
            assert observed_lock.second_attempted.wait(10)
        finally:
            release.set()

        with pytest.raises(RuntimeError, match="first benchmark failed"):
            first.result(timeout=10)
        assert second.result(timeout=10) == 1
    assert len(tuner.cache) == 1


def test_autotuner_concurrent_keys_keep_listener_timings_together(monkeypatch):
    autotuner, tuner, active = _make_autotuner(monkeypatch)
    first_reset = threading.Event()
    release_first = threading.Event()
    captured = []

    # The old per-key implementation leaves this lock before benchmarking.
    # The new cold-miss lock is still held, so contention releases key 1.
    tuner._cache_lock = _ObservedRLock(on_contention=release_first.set)

    def bench(*args, config, nargs, **kwargs):
        return (float(config.kwargs["BLOCK"] != nargs["N"]), ) * 3

    def pre_hook(kwargs, reset_only=False):
        if not reset_only:
            return
        if kwargs["N"] == 1:
            first_reset.set()
            assert release_first.wait(10)
        else:
            release_first.set()

    def listener(*, fn, key, best_config, configs_timings, duration, cache_hit):
        captured.append((key[0], best_config.kwargs["BLOCK"],
                         {config.kwargs["BLOCK"]: timing[0]
                          for config, timing in configs_timings.items()}, duration, cache_hit))

    tuner._bench = bench
    tuner.pre_hook = pre_hook
    monkeypatch.setattr(autotuner.knobs.autotuning, "listener", listener)

    def run(key):
        active.set_device(0)
        return tuner.run(key, grid=(1, ), warmup=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, 1)
        try:
            assert first_reset.wait(10)
            second = pool.submit(run, 2)
            assert first.result(timeout=10) == 1
            assert second.result(timeout=10) == 2
        finally:
            release_first.set()

    records = sorted(captured)
    assert [record[:3] for record in records] == [(1, 1, {1: 0.0, 2: 1.0}), (2, 2, {1: 1.0, 2: 0.0})]
    assert all(duration is not None and not cache_hit for _, _, _, duration, cache_hit in records)


def test_autotuner_listener_can_wait_for_another_tuning_thread(monkeypatch):
    autotuner, tuner, active = _make_autotuner(monkeypatch)
    tuner._bench = lambda *args, config, nargs, **kwargs: (float(config.kwargs["BLOCK"] != nargs["N"]), ) * 3

    def run(key):
        active.set_device(0)
        return tuner.run(key, grid=(1, ), warmup=False)

    def listener(*, key, **kwargs):
        if key[0] == 1:
            assert pool.submit(run, 2).result(timeout=5) == 2

    monkeypatch.setattr(autotuner.knobs.autotuning, "listener", listener)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert pool.submit(run, 1).result(timeout=10) == 1


def test_autotuner_concurrent_devices_do_not_share_tuning_key(monkeypatch):
    _, tuner, active = _make_autotuner(monkeypatch)
    start = threading.Barrier(2)

    def bench(*args, config, nargs, **kwargs):
        best = active.get_current_device() + 1
        return (float(config.kwargs["BLOCK"] != best), ) * 3

    tuner._bench = bench

    def run(device):
        active.set_device(device)
        start.wait(10)
        return device, tuner.run(1, grid=(1, ), warmup=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(run, (0, 1))) == [(0, 1), (1, 2)]
    assert {key[-1] for key in tuner.cache} == {0, 1}


def test_autotuner_interpreter_does_not_initialize_driver(monkeypatch):
    autotuner, tuner, active = _make_autotuner(monkeypatch)
    monkeypatch.setattr(autotuner.knobs.runtime, "interpret", True)
    monkeypatch.setattr(active, "get_current_device", lambda: pytest.fail("interpreter initialized the GPU driver"))
    tuner._bench = lambda *args, config, nargs, **kwargs: (float(config.kwargs["BLOCK"] != 1), ) * 3

    assert tuner.run(1, grid=(1, ), warmup=False) == 1
    assert tuple(tuner.cache) == ((1, ), )
