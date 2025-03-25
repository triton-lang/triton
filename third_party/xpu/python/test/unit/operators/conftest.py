# content of conftest.py


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")


def pytest_sessionfinish(session, exitstatus):
    # If all tests are skipped (exit code 5), modify the exit code to 0
    if exitstatus == 5:
        session.exitstatus = 0
