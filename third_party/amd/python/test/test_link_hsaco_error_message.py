import tempfile
import os

import pytest
import triton


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@pytest.mark.skipif(not is_hip(), reason="link_hsaco only available on HIP backend")
def test_nonexistent_input_reports_lld_error_details():
    """Verify that lld linker errors are captured and surfaced in exceptions.

    Regression test for the fix that redirects lld::lldMain's error output
    to the captured errStream rather than llvm::errs(), ensuring meaningful
    diagnostics appear in the RuntimeError raised by link_hsaco.
    """
    from triton._C.libtriton import amd

    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as tmp_out:
        fake_input = "/tmp/nonexistent_triton_test_input.o"
        try:
            with pytest.raises(RuntimeError) as exc_info:
                amd.link_hsaco(fake_input, tmp_out.name)

            error_msg = str(exc_info.value)
            assert "LLD failed to link" in error_msg
            lld_detail = error_msg.split("because ")[-1]
            assert len(lld_detail.strip()) > 0, ("LLD error details are empty — lld stderr is not being captured")
        finally:
            os.unlink(tmp_out.name)
