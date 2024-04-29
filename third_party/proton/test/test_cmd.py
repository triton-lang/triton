import pytest
import subprocess
import tempfile
import json


def test_help():
    # Only check if the viewer can be invoked
    ret = subprocess.check_call(["proton", "-h"], stdout=subprocess.DEVNULL)
    assert ret == 0


@pytest.mark.parametrize("mode", ["script", "python", "pytest"])
def test_exec(mode):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".hatchet") as f:
        name = f.name.split(".")[0]
        if mode == "script":
            ret = subprocess.check_call(["proton", "-n", name, "helper.py", "test"], stdout=subprocess.DEVNULL)
        elif mode == "python":
            ret = subprocess.check_call(["proton", "-n", name, "python", "helper.py", "test"],
                                        stdout=subprocess.DEVNULL)
        elif mode == "pytest":
            ret = subprocess.check_call(["proton", "-n", name, "pytest", "helper.py", "test"],
                                        stdout=subprocess.DEVNULL)
        assert ret == 0
        data = json.load(f, )
        kernels = data[0]["children"]
        assert len(kernels) == 2
        assert kernels[1]["frame"]["name"] == "test"
