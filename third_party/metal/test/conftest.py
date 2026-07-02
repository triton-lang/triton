import pytest
import sys


def pytest_collection_modifyitems(config, items):
    """Skip Metal tests if not on macOS."""
    if sys.platform != 'darwin':
        skip_metal = pytest.mark.skip(reason="Metal backend requires macOS")
        for item in items:
            item.add_marker(skip_metal)
