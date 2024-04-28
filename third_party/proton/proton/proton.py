import argparse
import sys
import os
from .profile import start, finalize
from .flags import set_command_line


def parse_arguments():
    parser = argparse.ArgumentParser(description="""
    Profile Triton kernels with proton command line.
    We support three ways to run the script:

    proton [profiling_options] script.py [script_args] [script_options]
    python -m proton [profiling_options] script.py [script_args] [script_options]
    proton [profiling_options] pytest [pytest_args] [script_options]

    See `proton -h` for more information.
""")
    parser.add_argument("name", type=str, help="Name of the profiling session")
    parser.add_argument("backend", type=str, help="Profiling backend", default="cupti")
    parser.add_argument("context", type=str, help="Profiling context", default="shadow")
    parser.add_argument("data", type=str, help="Profiling data", default="tree")
    parser.add_argument("hook", type=str, help="Profiling hook", default=None)
    args, target_args = parser.parse_known_args()
    return args, target_args


def is_pytest(script):
    return os.path.basename(script) == 'pytest'


def run_pytest(script, script_args):
    import pytest
    pytest.main([script] + script_args)


def run_profiling(args, target_args):
    script = target_args[0]
    script_args = target_args[1:] if len(target_args) > 1 else []
    sys.argv = target_args

    start(args.name, backend=args.backend, context=args.context, data=args.data, hook=args.hook)
    set_command_line()

    if is_pytest(script):
        run_pytest(script, script_args)
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
