"""Run Triton test suites and globally coordinated compile warmup."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from triton._compile_warmup import _require_complete_warmup, summarize_compile_trace
from triton._compile_warmup_pool import SharedWarmupCoordinator

ROOT = Path(__file__).resolve().parents[2]
PHASES = (
    ("python/test/unit", "warmup-unit"),
    ("python/triton_kernels/tests", "warmup-triton-kernels"),
    ("python/test/gluon", "warmup-gluon"),
    ("python/tutorials/gluon", "warmup-gluon"),
    ("python/examples/gluon", "warmup-gluon-examples"),
    ("python/tutorials/06-fused-attention.py", "warmup-attention"),
    ("python/test/regression", "warmup-regression"),
)
IMPORT_PATHS = (ROOT / "python" / "triton_kernels", ROOT / "python" / "test" / "unit" / "language",
                ROOT / "python" / "tutorials" / "gluon")


def _pythonpath(environment):
    entries = [str(path) for path in IMPORT_PATHS]
    if environment.get("PYTHONPATH"):
        entries.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(entries)


def _pytest(*arguments, workers=None, import_mode=None, distribution="worksteal"):
    command = [sys.executable, "-m", "pytest", "-p", "triton._compile_warmup", "-s", "--tb=short"]
    if import_mode is not None:
        command.append(f"--import-mode={import_mode}")
    if workers is not None:
        command.extend(("-n", str(workers), f"--dist={distribution}"))
    command.extend(arguments)
    return command


def _environment(phase=None, num_gpus=None):
    environment = os.environ.copy()
    _pythonpath(environment)
    if phase is not None:
        environment["TRITON_CI_CACHE_PHASE"] = phase
    if num_gpus is not None:
        environment["TRITON_TEST_NUM_GPUS"] = str(num_gpus)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            environment["TRITON_TEST_VISIBLE_GPUS"] = visible
    return environment


def _run(command, *, phase=None, num_gpus=None, cwd=ROOT, environment=None):
    return subprocess.run(command, cwd=cwd, env=environment or _environment(phase, num_gpus), check=False).returncode


def _concurrent(commands):
    processes = [subprocess.Popen(command, cwd=cwd, env=environment) for command, cwd, environment in commands]
    return max(process.wait() for process in processes)


def _capability():
    import torch

    if not torch.cuda.is_available() or torch.version.hip is not None:
        return 0
    return torch.cuda.get_device_capability()[0]


def _validate_gpus(count):
    import torch

    available = torch.cuda.device_count()
    if count < 1 or count > available:
        raise SystemExit(f"requested {count} GPU shards, but {available} CUDA devices are visible")


def _warmup(args):
    _validate_gpus(1)
    _pythonpath(os.environ)
    for entry in IMPORT_PATHS:
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    directory = os.environ.get("TRITON_CI_COMPILE_TRACE_DIR")
    coordinator = SharedWarmupCoordinator(max_workers=args.warmup_procs, trace_directory=directory)
    environment = _environment("warmup-unit", 1)
    environment["TRITON_WARMUP_COORDINATOR"] = coordinator.address
    capture_procs = args.capture_procs or min(8, max(2, args.warmup_procs // 12))
    command = _pytest("--warmup-only", "--warmup-workers", str(args.warmup_procs), workers=capture_procs,
                      import_mode="importlib")
    for path, phase in PHASES:
        command.extend(("--warmup-phase", f"{path}={phase}"))
    command.extend(
        args.targets
        or ("python/test/unit/language", "python/triton_kernels/tests", "python/test/gluon", "python/tutorials/gluon",
            "python/examples/gluon", "python/tutorials/06-fused-attention.py", "python/test/regression"))
    try:
        status = _run(command, environment=environment)
    finally:
        coordinator.close()
    return status


def _unit(args):
    capability = _capability()
    unit_directory = ROOT / "python" / "test" / "unit"
    if capability >= 9:
        main = _pytest("--ignore-glob=plugins/*", "--ignore=test_debug.py", "--ignore=language/test_subprocess.py",
                       workers=args.num_procs)
        debug = _pytest("test_debug.py", "language/test_subprocess.py", workers=args.debug_procs)
        status = _concurrent(
            ((main, unit_directory, _environment("unit")), (debug, unit_directory, _environment("unit"))))
    else:
        status = _run(_pytest("--ignore-glob=plugins/*", workers=args.num_procs), phase="unit", cwd=unit_directory)
    if status:
        return status

    kernel_procs = args.kernel_procs or (3 if capability >= 9 else 6)
    status = _run(_pytest("python/triton_kernels/tests/", workers=kernel_procs), phase="triton-kernels")
    if status:
        return status
    status = _run(_pytest("python/tutorials/06-fused-attention.py", workers=1), phase="attention")
    if status:
        return status

    environment = _environment("instrumentation")
    environment.update({
        "TRITON_ALWAYS_COMPILE": "1", "TRITON_DISABLE_LINE_INFO": "0", "LLVM_PASS_PLUGIN_PATH":
        "python/triton/instrumentation/libGPUInstrumentationTestLib.so"
    })
    return _run(
        _pytest("--capture=tee-sys", "-rfs", "-vvv", "python/test/unit/instrumentation/test_gpuhello.py", workers=1),
        environment=environment)


def _gluon(args):
    _validate_gpus(args.num_gpus)
    capability = _capability()
    general_workers = min(args.num_procs, args.gluon_procs * args.num_gpus)
    if capability >= 9:
        general = _pytest("python/test/gluon/", "python/tutorials/gluon/", "--ignore=python/test/gluon/test_consan.py",
                          workers=general_workers)
        consan = _pytest("python/test/gluon/test_consan.py", workers=args.consan_procs * args.num_gpus)
        status = _concurrent(
            ((general, ROOT, _environment("gluon", args.num_gpus)), (consan, ROOT, _environment("gluon",
                                                                                                args.num_gpus))))
    else:
        status = _run(_pytest("python/test/gluon/", "python/tutorials/gluon/", workers=general_workers), phase="gluon",
                      num_gpus=args.num_gpus)
    if status:
        return status

    example_workers = min(args.num_procs, args.example_procs * args.num_gpus)
    return _run(_pytest("python/examples/gluon/", workers=example_workers, import_mode="importlib"),
                phase="gluon-examples", num_gpus=args.num_gpus)


def _gsan(args):
    environment = _environment("gsan")
    environment["TRITON_DISABLE_LINE_INFO"] = "0"
    symmetric_memory = "python/test/gsan/test_symmetric_memory.py"

    if args.num_gpus == 1:
        return _run(_pytest("python/test/gsan", workers=args.num_procs, distribution="loadgroup"),
                    environment=environment)

    status = _run(
        _pytest(f"--ignore={symmetric_memory}", "python/test/gsan", workers=args.num_procs, distribution="loadgroup"),
        environment=environment)
    if status:
        return status
    return _run(_pytest(symmetric_memory, workers=1, distribution="loadgroup"), environment=environment)


def _suite(args):
    if args.name == "unit":
        return _unit(args)
    if args.name == "gsan":
        return _gsan(args)
    return _gluon(args)


def _report(args):
    if args.directory is None:
        raise SystemExit("set TRITON_CI_COMPILE_TRACE_DIR or pass --directory")
    report = summarize_compile_trace(args.directory)
    print(f"TRITON_CI_COMPILE_TRACE {json.dumps(report, sort_keys=True)}")
    if args.require_complete:
        _require_complete_warmup(report)
    return 0


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    warmup = commands.add_parser("warmup", help="capture marked tests using one shared compiler pool")
    warmup.add_argument("--warmup-procs", type=int, default=int(os.environ.get("WARMUP_PROCS", "8")))
    warmup.add_argument("--capture-procs", type=int)
    warmup.add_argument("targets", nargs="*", help="optional marked pytest targets for focused warmup")
    warmup.set_defaults(run=_warmup)

    suite = commands.add_parser("suite", help="execute a dynamically scheduled test suite")
    suite.add_argument("name", choices=("unit", "gluon", "gsan"))
    suite.add_argument("--num-gpus", type=int, default=int(os.environ.get("NUM_GPUS", "1")))
    suite.add_argument("--num-procs", type=int, default=int(os.environ.get("NUM_PROCS", "8")))
    suite.add_argument("--gluon-procs", type=int, default=8)
    suite.add_argument("--consan-procs", type=int, default=4)
    suite.add_argument("--example-procs", type=int, default=4)
    suite.add_argument("--debug-procs", type=int, default=4)
    suite.add_argument("--kernel-procs", type=int)
    suite.set_defaults(run=_suite)

    report = commands.add_parser("report", help="summarize cache coverage and validate completeness")
    report.add_argument("--directory", default=os.environ.get("TRITON_CI_COMPILE_TRACE_DIR"))
    report.add_argument("--require-complete", action="store_true")
    report.set_defaults(run=_report)

    args = parser.parse_args(argv)
    raise SystemExit(args.run(args))


if __name__ == "__main__":
    _main()
