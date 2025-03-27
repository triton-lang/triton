import os


def front_end_debugging():
    return os.getenv("TRITON_FRONT_END_DEBUGGING", "0") == "1"
