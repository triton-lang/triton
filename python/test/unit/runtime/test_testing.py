from unittest import mock

import pytest
import torch

import triton


def test_do_bench_cudagraph_rejects_active_capture():
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
        mock.patch.object(torch.cuda, "synchronize") as synchronize,
    ):
        with pytest.raises(RuntimeError, match="Warm up autotuned kernels"):
            triton.testing.do_bench_cudagraph(lambda: None)

    synchronize.assert_not_called()
