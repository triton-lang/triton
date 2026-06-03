import pytest


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")
    parser.addoption(
        "--tdm-disable-hint",
        action="store_true",
        default=False,
        help="Run TDM merge tests without warp_used_hint for manual comparison.",
    )


@pytest.fixture
def device(request):
    return request.config.getoption("--device")
