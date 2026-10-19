import pytest

import triton._test_runner as test_runner


@pytest.mark.parametrize(
    ("return_codes", "expected"),
    [
        ([0, 0], 0),
        ([0, -15], -15),
        ([0, 2], 2),
        ([1, 2], 2),
    ],
)
def test_concurrent_returns_highest_failure_status(monkeypatch, return_codes, expected):
    statuses = iter(return_codes)

    class Process:
        def wait(self):
            return next(statuses)

    monkeypatch.setattr(test_runner.subprocess, "Popen", lambda *args, **kwargs: Process())
    commands = [([], ".", {}) for _ in return_codes]

    assert test_runner._concurrent(commands) == expected
