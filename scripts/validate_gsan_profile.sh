#!/usr/bin/env bash
# Run on a CUDA devbox. Install the profiler before pytest initializes CUDA.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export PYTHONPATH="$root/python"
export TRITON_INSTRUMENTATION_MODE=
python=${PYTHON:-python3}
case "${1:-test}" in
  test)
    shift
    "$python" -c 'from triton.experimental.gsan.profile import install_allocator; install_allocator(); import pytest, sys; sys.exit(pytest.main(sys.argv[1:]))' \
      -s --tb=short --maxfail=1 -q python/test/unit/backend/test_gsan_profile.py "$@"
    ;;
  distributed)
    shift
    "$python" -m torch.distributed.run "$@" --no-python "$python" -c \
      'from triton.experimental.gsan.profile import install_allocator; install_allocator(); import pytest, sys; sys.exit(pytest.main(sys.argv[1:]))' \
      -s --tb=short --maxfail=1 -q python/test/unit/backend/test_gsan_profile.py::test_gsan_profile_distributed
    ;;
  *) echo "usage: $0 {test [pytest options]|distributed [torchrun options]}" >&2; exit 2 ;;
esac
