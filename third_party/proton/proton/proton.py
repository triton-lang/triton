import argparse
import sys
import os

_PROTON_REEXEC_MARKER = "_PROTON_ROCM_REEXEC"


def _is_rocm_system():
    return (os.path.exists('/opt/rocm') or os.environ.get('ROCM_PATH') or os.environ.get('HIP_PATH'))


def _get_proton_lib_path():
    try:
        import triton._C.libproton as libproton
        lib_dir = os.path.dirname(libproton.__file__)
        for name in ['libproton.so', 'proton.so', 'libproton_backend.so']:
            candidate = os.path.join(lib_dir, name)
            if os.path.exists(candidate):
                return candidate
    except ImportError:
        pass
    return None


def _get_rocm_roctx_lib():
    candidates = [
        '/opt/rocm/lib/librocprofiler-sdk-roctx.so',
        os.path.join(os.environ.get('ROCM_PATH', ''), 'lib/librocprofiler-sdk-roctx.so'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _maybe_reexec_for_rocm():
    """
    ROCProfiler-SDK requires:
    1. ROCP_TOOL_LIBRARIES - tool library loaded before HIP initializes
    2. LD_PRELOAD with librocprofiler-sdk-roctx.so - for roctx/nvtx marker interception
    """
    if os.environ.get(_PROTON_REEXEC_MARKER) or os.environ.get('ROCP_TOOL_LIBRARIES'):
        return

    if not _is_rocm_system():
        return

    lib_path = _get_proton_lib_path()
    if not lib_path:
        return

    os.environ['ROCP_TOOL_LIBRARIES'] = lib_path
    os.environ[_PROTON_REEXEC_MARKER] = '1'

    # Set LD_PRELOAD for roctx marker interception so it doesn't use the old library
    # probably not best way
    roctx_lib = _get_rocm_roctx_lib()
    if roctx_lib:
        existing_preload = os.environ.get('LD_PRELOAD', '')
        if existing_preload:
            os.environ['LD_PRELOAD'] = f"{roctx_lib}:{existing_preload}"
        else:
            os.environ['LD_PRELOAD'] = roctx_lib

    os.execv(sys.executable, [sys.executable] + sys.argv)


_maybe_reexec_for_rocm()

# these imports MUST be after _maybe_reexec_for_rocm() to ensure rocprofiler-sdk
# is configured before HIP initializes (triggered by importing triton).
from .profile import start, finalize, _select_backend, _normalize_backend  # noqa: E402
from .flags import flags  # noqa: E402


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="The proton command utility for profiling scripts and pytest tests.", usage="""
    proton [options] script.py [script_args] [script_options]
    proton [options] pytest [pytest_args] [script_options]
    python -m triton.profiler.proton [options] script.py [script_args] [script_options]
""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--name", type=str, help="Name of the profiling session")
    parser.add_argument("-b", "--backend", type=str, help="Profiling backend", default=None,
                        choices=["cupti", "rocprofiler", "roctracer", "instrumentation"])
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
    # Prepare a clean global environment
    clean_globals = {
        "__name__": "__main__",
        "__file__": script_path,
        "__builtins__": __builtins__,
        sys.__name__: sys,
    }

    original_argv = sys.argv
    sys.argv = [script] + args
    # Append the script's directory in case the script uses relative imports
    sys.path.append(os.path.dirname(script_path))

    # Execute in the isolated environment
    try:
        with open(script_path, 'rb') as file:
            code = compile(file.read(), script_path, 'exec')
        exec(code, clean_globals)
    except Exception as e:
        print(f"An error occurred while executing the script: {e}")
        sys.exit(1)
    finally:
        sys.argv = original_argv


def do_setup_and_execute(target_args):
    # Set the command line mode to avoid any `start` calls in the script.
    flags.command_line = True

    script = target_args[0]
    script_args = target_args[1:] if len(target_args) > 1 else []
    if is_pytest(script):
        import pytest
        pytest.main(script_args)
    else:
        execute_as_main(script, script_args)


def run_profiling(args, target_args):
    backend = args.backend if args.backend else _select_backend()
    backend = _normalize_backend(backend)

    start(args.name, context=args.context, data=args.data, backend=backend, hook=args.hook)

    do_setup_and_execute(target_args)

    finalize()


def main():
    args, target_args = parse_arguments()
    run_profiling(args, target_args)


if __name__ == "__main__":
    main()
