import argparse
import sys
import os
import runpy
import traceback
from .profile import start, finalize, _select_backend
from .flags import flags


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="The proton command utility for profiling scripts and pytest tests.", usage="""
    proton [options] script.py [script_args] [script_options]
    proton [options] pytest [pytest_args] [script_options]
    python -m triton.profiler.proton [options] script.py [script_args] [script_options]
""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--name", type=str, help="Name of the profiling session")
    parser.add_argument("-b", "--backend", type=str, help="Profiling backend", default=None,
                        choices=["cupti", "roctracer", "instrumentation"])
    parser.add_argument("-c", "--context", type=str, help="Profiling context", default="shadow",
                        choices=["shadow", "python"])
    parser.add_argument("-m", "--mode", type=str, help="Profiling mode", default=None)
    parser.add_argument("-d", "--data", type=str, help="Profiling data", default="tree", choices=["tree", "trace"])
    parser.add_argument("-k", "--hook", type=str, help="Profiling hook", default=None, choices=[None, "triton"])
    parser.add_argument('target_args', nargs=argparse.REMAINDER, help='Subcommand and its arguments')
    args = parser.parse_args()
    return args, args.target_args


def is_pytest(script):
    return os.path.basename(script) == 'pytest'


def execute_as_main(script, args):
    script_path = os.path.abspath(script)

    original_argv = sys.argv
    sys.argv = [script] + args
    # Append the script's directory in case the script uses relative imports
    sys.path.append(os.path.dirname(script_path))

    # Execute in the isolated environment
    try:
        runpy.run_path(script, run_name="__main__")
    except Exception as e:
        print("An error occurred while executing the script:")
        traceback.print_exception(e)
        return 1
    except SystemExit as e:
        return e.code
    except KeyboardInterrupt:
        return 1
    finally:
        sys.argv = original_argv
    return 0


def do_setup_and_execute(target_args):
    # Set the command line mode to avoid any `start` calls in the script.
    flags.command_line = True

    script = target_args[0]
    script_args = target_args[1:] if len(target_args) > 1 else []
    if is_pytest(script):
        import pytest
        return pytest.main(script_args)
    else:
        return execute_as_main(script, script_args)


def run_profiling(args, target_args):
    backend = args.backend if args.backend else _select_backend()

    start(args.name, context=args.context, data=args.data, backend=backend, hook=args.hook)

    exitcode = do_setup_and_execute(target_args)

    finalize()
    sys.exit(exitcode)


def main():
    args, target_args = parse_arguments()
    run_profiling(args, target_args)


if __name__ == "__main__":
    main()
