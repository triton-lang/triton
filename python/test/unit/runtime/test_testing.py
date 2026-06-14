from unittest import mock

import pytest
import torch

import triton


@pytest.mark.parametrize(
    "bench_name",
    ["do_bench_cudagraph", "do_bench_cudagraph_proton"],
)
def test_cudagraph_bench_rejects_active_capture(bench_name):
    bench = getattr(triton.testing, bench_name)

    with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            mock.patch.object(torch.cuda, "synchronize") as synchronize,
    ):
        with pytest.raises(RuntimeError, match="Warm up autotuned kernels"):
            bench(lambda: None)

    synchronize.assert_not_called()
