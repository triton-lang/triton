import torch
import pytest
import pathlib
import triton

from triton._internal_testing import is_cuda


@pytest.mark.skipif(not is_cuda(), reason="warp specialization is only supported on NVIDIA")
def test_warp_specialize_basic_ir(tmp_path: pathlib.Path):
    ir = """
    tt.func @kernel(%arg0: !tt.ptr<i32>) {
      %c42_i32 = arith.constant 42 : i32
      gpu.barrier
      ttg.warp_specialize(%arg0)
      default {
        tt.store %arg0, %c42_i32 : !tt.ptr<i32>
        gpu.barrier
        ttg.warp_yield
      }
      partition0(%arg1: !tt.ptr<i32>) num_warps(1) {
        %c5555_i32 = arith.constant 5555 : i32
        %c1_i32 = arith.constant 1 : i32
        gpu.barrier
        %ptr = tt.addptr %arg1, %c1_i32 : !tt.ptr<i32>, i32
        tt.store %ptr, %c5555_i32 : !tt.ptr<i32>
        ttg.warp_return
      } : (!tt.ptr<i32>) -> ()
      tt.return
    }
    """

    temp_file = tmp_path / "test_warp_specialize_basic_ir.ttir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    input = torch.empty(2, dtype=torch.int32, device='cuda')
    kernel[(1, 1, 1)](input)
    assert input[0] == 42
    assert input[1] == 5555


@pytest.mark.skipif(not is_cuda(), reason="warp specialization is only supported on NVIDIA")
def test_warpgroup_reduction(tmp_path: pathlib.Path):

    def template(i, num_warps, in_ptr, out_ptr):
        return f"""
          %range = tt.make_range {{end = {(i+1)*256} : i32, start = {i*256} : i32}} : tensor<256xi32, #blocked{num_warps}>
          %splatted = tt.splat {in_ptr} : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #blocked{num_warps}>
          %ptrs = tt.addptr %splatted, %range : tensor<256x!tt.ptr<i32>, #blocked{num_warps}>, tensor<256xi32, #blocked{num_warps}>
          %input = tt.load %ptrs : tensor<256x!tt.ptr<i32>, #blocked{num_warps}>
          %result = "tt.reduce"(%input) ({{
          ^bb0(%lhs: i32, %rhs: i32):
            %result = arith.addi %lhs, %rhs : i32
            tt.reduce.return %result : i32
          }}) {{axis = 0 : i32}} : (tensor<256xi32, #blocked{num_warps}>) -> i32
          %offset = arith.constant {i} : i32
          %output = tt.addptr {out_ptr}, %offset : !tt.ptr<i32>, i32
          tt.store %output, %result : !tt.ptr<i32>
        """

    ir = """
    #blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    #blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
    #blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

    module attributes {"ttg.num-warps" = 4 : i32} {

    tt.func @kernel(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
      ttg.warp_specialize(%arg0, %arg1)
      default {
      """ + template(0, 4, "%arg0", "%arg1") + """
        ttg.warp_yield
      }
      partition0(%arg2: !tt.ptr<i32>, %arg3: !tt.ptr<i32>) num_warps(4) {
      """ + template(1, 4, "%arg2", "%arg3") + """
        ttg.warp_return
      }
      partition1(%arg4: !tt.ptr<i32>, %arg5: !tt.ptr<i32>) num_warps(2) {
      """ + template(2, 2, "%arg4", "%arg5") + """
        ttg.warp_return
      }
      partition2(%arg6: !tt.ptr<i32>, %arg7: !tt.ptr<i32>) num_warps(1) {
      """ + template(3, 1, "%arg6", "%arg7") + """
        ttg.warp_return
      } : (!tt.ptr<i32>, !tt.ptr<i32>) -> ()
      tt.return
    }

    }
    """

    temp_file = tmp_path / "test_warpgroup_reduction.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    input = torch.arange(1024, dtype=torch.int32, device='cuda')
    output = torch.empty(4, dtype=torch.int32, device='cuda')
    kernel[(1, 1, 1)](input, output)
    assert output[0] == torch.arange(0, 256).sum()
    assert output[1] == torch.arange(256, 512).sum()
    assert output[2] == torch.arange(512, 768).sum()
    assert output[3] == torch.arange(768, 1024).sum()
