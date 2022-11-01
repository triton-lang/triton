import os


def test_printf():
    os.system('./_printf.sh')
    lines = None
    with open("/tmp/___printf.log", "r") as f:
        lines = f.readlines()

    new_lines = set()
    for line in lines:
        try:
            value = int(float(line))
            new_lines.add(value)
        except Exception as e:
            print(e)

    for i in range(128):
        assert i in new_lines
