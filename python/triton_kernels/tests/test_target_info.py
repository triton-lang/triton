from types import SimpleNamespace

import pytest
import triton.language as tl
from triton_kernels import target_info


@pytest.mark.parametrize("arch, expected", [(90, False), (100, True), (103, True), (120, False), (121, False)])
def test_has_tma_scatter(arch, expected, monkeypatch):
    # No CI runner is sm_12x: fake the target. `current_target` is imported by name into
    # `triton_kernels.target_info`, so it is patched there and where `cuda_capability_geq` reads it.
    def fake():
        return SimpleNamespace(backend="cuda", arch=arch)

    monkeypatch.setattr(tl.target_info, "current_target", fake)
    monkeypatch.setattr(target_info, "current_target", fake)
    assert target_info.has_tma_scatter() == expected
    assert target_info.has_tma_gather() == (arch >= 100)
