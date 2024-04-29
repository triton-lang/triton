import argparse
import sys
import os
import torch
from .profile import start, finalize
from .flags import set_command_line


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="The proton command utility for profiling scripts and pytest tests.", usage="""
    proton [options] script.py [script_args] [script_options]
    python -m proton [options] script.py [script_args] [script_options]
    proton [options] pytest [pytest_args] [script_options]
""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--name", type=str, help="Name of the profiling session")
    parser.add_argument("-b", "--backend", type=str, help="Profiling backend", default="cupti", choices=["cupti"])
    parser.add_argument("-c", "--context", type=str, help="Profiling context", default="shadow",
                        choices=["shadow", "python"])
    parser.add_argument("-d", "--data", type=str, help="Profiling data", default="tree", choices=["tree"])
    parser.add_argument("-k", "--hook", type=str, help="Profiling hook", default=None, choices=[None, "triton"])
    args, target_args = parser.parse_known_args()
    return args, target_args


def is_pytest(script):
    return os.path.basename(script) == 'pytest'


def run_profiling(args, target_args):
    script = target_args[0]
    script_args = target_args[1:] if len(target_args) > 1 else []
    sys.argv = target_args

    # Init ROCM and CUDA
    torch.cuda.init()

    start(args.name, backend=args.backend, context=args.context, data=args.data, hook=args.hook)

    # Set the command line mode to avoid any `start` calls in the script.
    set_command_line()

    if is_pytest(script):
        import pytest
        pytest.main([script] + script_args)
    else:
        with open(script, 'rb') as file:
            code = compile(file.read(), script, 'exec')

        exec(code, globals())

    finalize()


def main():
    args, target_args = parse_arguments()
    run_profiling(args, target_args)


if __name__ == "__main__":
    main()
