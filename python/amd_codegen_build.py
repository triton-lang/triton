"""Minimal entry path for an already configured standalone AMD build."""
import hashlib
import json
import os
import sys


def file_state(path):
    stat = os.stat(path)
    return [os.path.realpath(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def record_path(argv):
    cache = argv[argv.index("--triton-cache-path") + 1]
    key = hashlib.sha256(json.dumps(argv).encode()).hexdigest()
    return os.path.join(cache, "amd", "fast-build", key + ".json")


def environment():
    return {key: os.environ.get(key) for key in ("PATH", "SDKROOT", "CC", "CXX")}


def try_build(argv):
    # CMake translates this variable, including its empty-value semantics.
    # Retain that interface through the full helper when it is supplied.
    if not argv or argv[0] != "prepare_amd_codegen" or "CMAKE_BUILD_PARALLEL_LEVEL" in os.environ:
        return False
    try:
        if os.path.exists(os.path.join(argv[argv.index("--triton-cache-path") + 1], "archives")):
            return False
        with open(record_path(argv)) as file:
            record = json.load(file)
        if record["environment"] != environment():
            return False
        if any(file_state(path) != state for path, state in record["inputs"]):
            return False
        if not os.path.isfile(os.path.join(record["build"], "build.ninja")):
            return False
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False

    import subprocess
    subprocess.check_call([record["ninja"], "-C", record["build"], "triton_amd_codegen"])
    try:
        return all(file_state(path) == state for path, state in record["delivery"])
    except (OSError, KeyError, TypeError, ValueError):
        return False


def save(argv, build, ninja, source, destination, inputs):
    path = record_path(argv)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = dict(build=build, ninja=ninja, environment=environment(), inputs=[[p, file_state(p)] for p in inputs],
                  delivery=[[p, file_state(p)] for p in (source, destination)])
    temporary = path + "." + str(os.getpid())
    try:
        with open(temporary, "w") as file:
            json.dump(record, file)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if try_build(argv):
        return

    from build_helpers import main as build_helpers_main
    build_helpers_main(argv)


if __name__ == "__main__":
    main()
