# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption("--warmup", action="store_true", default=False, help="Ignore test failures")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    # Make all the tests pass during warmup
    if item.config.getoption("--warmup"):
        report.outcome = "passed"
        report.longrepr = None
