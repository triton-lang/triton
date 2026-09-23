import pytest

from triton.backends.amd import compiler


@pytest.mark.parametrize(("arch", "expected_architected_sgprs"), [
    ("gfx90a", False),
    ("gfx942", False),
    ("gfx950", False),
    ("gfx1250", True),
])
def test_amd_codegen_reports_target_properties(arch, expected_architected_sgprs):
    data_layout = compiler.get_amdgpu_data_layout(arch, "")
    assert "p1:64:64" in data_layout
    assert "p3:32:32" in data_layout
    assert compiler.has_architected_sgprs(arch) is expected_architected_sgprs
