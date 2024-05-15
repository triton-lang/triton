import argparse
import sys
import os
from .profile import start, finalize, _select_backend
from .flags import set_command_line


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="The proton command utility for profiling scripts and pytest tests.", usage="""
    proton [options] script.py [script_args] [script_options]
    proton [options] pytest [pytest_args] [script_options]
    python -m triton.profiler.proton [options] script.py [script_args] [script_options]
""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--name", type=str, help="Name of the profiling session")
    parser.add_argument("-b", "--backend", type=str, help="Profiling backend", default=None, choices=["cupti"])
    parser.add_argument("-c", "--context", type=str, help="Profiling context", default="shadow",
                        choices=["shadow", "python"])
    parser.add_argument("-d", "--data", type=str, help="Profiling data", default="tree", choices=["tree"])
    parser.add_argument("-k", "--hook", type=str, help="Profiling hook", default=None, choices=[None, "triton"])
    args, target_args = parser.parse_known_args()
    return args, target_args


def is_pytest(script):
    return os.path.basename(script) == 'pytest'


def execute_as_main(script, args):
    script_path = os.path.abspath(script)
    # Prepare a clean global environment
    clean_globals = {
        "__name__": "__main__",
        "__file__": script_path,
        "__builtins__": __builtins__,
        sys.__name__: sys,
    }

    original_argv = sys.argv
    sys.argv = [script] + args

    # Execute in the isolated environment
    try:
        with open(script_path, 'rb') as file:
            code = compile(file.read(), script_path, 'exec')
        exec(code, clean_globals)
    except Exception as e:
        print(f"An error occurred while executing the script: {e}")
    finally:
        sys.argv = original_argv


def run_profiling(args, target_args):
    backend = args.backend if args.backend else _select_backend()

    start(args.name, context=args.context, data=args.data, backend=backend, hook=args.hook)

    # Set the command line mode to avoid any `start` calls in the script.
    set_command_line()

    script = target_args[0]
    script_args = target_args[1:] if len(target_args) > 1 else []
    if is_pytest(script):
        import pytest
        pytest.main(script_args)
    else:
        execute_as_main(script, script_args)

    finalize()


def main():
    args, target_args = parse_arguments()
    run_profiling(args, target_args)


if __name__ == "__main__":
    main()
