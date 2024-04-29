import subprocess


def test_help():
    # Only check if the viewer can be invoked
    ret = subprocess.check_call(["proton", "-h"], stdout=subprocess.DEVNULL)
    assert ret == 0


def test_python():
    ret = subprocess.check_call(["proton", "-n", "test", "helper.py"])
    assert ret == 0


def test_pytest():
    ret = subprocess.check_call(["proton", "-n", "test", "pytest", "helper.py"], stdout=subprocess.DEVNULL)
    assert ret == 0
