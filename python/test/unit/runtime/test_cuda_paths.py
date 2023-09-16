from unittest.mock import call, patch, Mock, ANY
from triton.common.backend import _path_to_binary
import pytest
import os


def test_cuda_paths():
    env_vars = {"TRITON_PTXAS_PATH": "/nonexistent/ptax",
                "CUDA_ROOT": "/nonexistent/cuda_root",
                "CUDA_HOME": "/nonexistent/cuda_home",
                "CUDA_PATH": "/nonexistent/cuda_path",
                }

    with patch.dict(os.environ, env_vars), \
            patch('os.path.exists', return_value=False) as mocked_path_exists:
        with pytest.raises(RuntimeError) as expected_exception:
            _path_to_binary("x")

        assert "Cannot find x" == str(expected_exception.value)

    assert mocked_path_exists.call_count == len(env_vars) + 1  # third_party/bin/binary
    mocked_path_exists.assert_has_calls([call("/nonexistent/ptax"),
                                         ANY,
                                         call("/nonexistent/cuda_root/bin/x"),
                                         call("/nonexistent/cuda_home/bin/x"),
                                         call("/nonexistent/cuda_path/bin/x"),])

    assert "/third_party/cuda/bin/x" in mocked_path_exists.call_args_list[1].args[0]
