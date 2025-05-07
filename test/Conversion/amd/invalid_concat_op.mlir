// RUN: triton-opt -split-input-file %s --convert-triton-amdgpu-to-llvm='arch=gfx942' -verify-diagnostics

// Invalid shapes
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @invalid_concat(
    %arg0: tensor<32x64xf32, #blocked>,
    %arg1: tensor<32x64xf32, #blocked>,
    %arg2: tensor<32x64xf32, #blocked>,
    %arg3: tensor<32x64xf32, #blocked>,
    %arg4: tensor<32x64xf32, #blocked>,
    %arg5: tensor<32x64xf32, #blocked>,
    %arg6: tensor<32x64xf32, #blocked>,
    %arg7: tensor<32x64xf32, #blocked>) {

    // expected-error @+1 {{Number of source tiles (8) doesn't match required count (16).}}
    %1 = amdgpu.concat %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7:
    tensor<32x64xf32, #blocked>,tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked> -> tensor<256x128xf32, #blocked>
    tt.return
  }
}

// -----

// Different layouts 1
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @invalid_concat(
    %arg0: tensor<32x64xf32, #blocked1>,
    %arg1: tensor<32x64xf32, #blocked>,
    %arg2: tensor<32x64xf32, #blocked>,
    %arg3: tensor<32x64xf32, #blocked>,
    %arg4: tensor<32x64xf32, #blocked>,
    %arg5: tensor<32x64xf32, #blocked>,
    %arg6: tensor<32x64xf32, #blocked>,
    %arg7: tensor<32x64xf32, #blocked>) {

    // expected-error @+1 {{All sources must have identical tensor types.}}
    %1 = amdgpu.concat %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7:
    tensor<32x64xf32, #blocked1>,tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked> -> tensor<128x128xf32, #blocked>
    tt.return
  }
}

// -----

// Different layouts 2
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @invalid_concat(
    %arg0: tensor<32x64xf32, #blocked>,
    %arg1: tensor<32x64xf32, #blocked>,
    %arg2: tensor<32x64xf32, #blocked>,
    %arg3: tensor<32x64xf32, #blocked>,
    %arg4: tensor<32x64xf32, #blocked>,
    %arg5: tensor<32x64xf32, #blocked>,
    %arg6: tensor<32x64xf32, #blocked>,
    %arg7: tensor<32x64xf32, #blocked>) {

    // expected-error @+1 {{CTA tile shapes must match between source and destination tensors.}}
    %1 = amdgpu.concat %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7:
    tensor<32x64xf32, #blocked>,tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked> -> tensor<128x128xf32, #blocked1>
    tt.return
  }
}
