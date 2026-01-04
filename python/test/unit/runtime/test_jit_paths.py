import pytest
import torch

from triton._C.libtriton import native_specialize_impl
from triton._utils import find_paths_if
from triton.runtime.jit import _find_paths_if_leaf_is_constexpr, _find_paths_if_leaf_is_str


def _available_backends():
    backends = []
    try:
        from triton.backends.nvidia.compiler import CUDABackend

        backends.append(CUDABackend)
    except Exception:
        pass
    try:
        from triton.backends.amd.compiler import HIPBackend

        backends.append(HIPBackend)
    except Exception:
        pass
    return backends


@pytest.mark.parametrize("backend", _available_backends() or [pytest.param(None, marks=pytest.mark.skip)])
@pytest.mark.parametrize(
    "arg",
    [
        # tuple constexpr markers at nested paths
        (1, 2, 1),
        # nested tuple constexpr markers at deeper paths
        ((1, 1), (2, 1)),
        # tuple attrs (e.g. alignment divisibility "D") at nested paths
        (torch.empty((16, ), device="cpu"), torch.empty((16, ), device="cpu")),
    ],
)
def test_jit_precomputed_paths_match_find_paths_if(backend, arg):
    # Specialize a *single* Python argument that itself may be a tuple/nested tuple.
    # native_specialize_impl returns a nested (tys, keys) structure for tuple inputs.
    tys, keys = native_specialize_impl(backend, arg, False, True, True)

    # Mirror the old _pack_args behavior:
    #   sigvals = [x[0] for x in specialization]
    #   attrvals = [x[1] for x in specialization]
    sigvals = [tys]
    attrvals = [keys]

    expected_constexpr = tuple(find_paths_if(sigvals, lambda _, v: v == "constexpr"))
    expected_attrs = tuple(find_paths_if(attrvals, lambda _, v: isinstance(v, str)))

    got_constexpr = _find_paths_if_leaf_is_constexpr(sigvals)
    got_attrs = _find_paths_if_leaf_is_str(attrvals)

    assert got_constexpr == expected_constexpr
    assert got_attrs == expected_attrs
