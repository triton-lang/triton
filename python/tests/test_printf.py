import os
import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))
printf_path = os.path.join(dir_path, "printf_helper.py")


def test_printf():
    proc = subprocess.Popen(["python", printf_path], stdout=subprocess.PIPE, shell=False)
    (outs, err) = proc.communicate()
    outs = outs.split()
    new_lines = set()
    for line in outs:
        try:
            value = int(float(line))
            new_lines.add(value)
        except Exception as e:
            print(e)
    for i in range(128):
        assert i in new_lines
    assert len(new_lines) == 128
