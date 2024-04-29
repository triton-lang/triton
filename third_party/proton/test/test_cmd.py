import subprocess
import tempfile
import json


def test_help():
    # Only check if the viewer can be invoked
    ret = subprocess.check_call(["proton", "-h"], stdout=subprocess.DEVNULL)
    assert ret == 0


def test_python():
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        ret = subprocess.check_call(["proton", "-n", "test", "helper.py"])
        assert ret == 0
        data = json.load(f)
        kernels = data[0]["children"]
        assert len(kernels) == 2
        assert kernels[1]["frame"]["name"] == "test"


def test_pytest():
    ret = subprocess.check_call(["proton", "-n", "test", "pytest", "helper.py"], stdout=subprocess.DEVNULL)
    assert ret == 0
