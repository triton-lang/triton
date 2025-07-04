// RUN: triton-opt -split-input-file %s --convert-triton-amdgpu-to-llvm='arch=gfx942' -verify-diagnostics


// Invalid ranks
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

    // expected-error @+1 {{Source and destination tensors must have the same rank.}}
    %1 = amdgpu.concat %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7:
    tensor<32x64xf32, #blocked>,tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked> -> tensor<256xf32, #blocked>
    tt.return
  }
}

// -----

// Invalid shapes 1
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

    // expected-error @+1 {{Source and destination tensor shapes don't match.}}
    %1 = amdgpu.concat %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7:
    tensor<32x64xf32, #blocked>,tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked> -> tensor<257x128xf32, #blocked>
    tt.return
  }
}

// -----

// Invalid shapes 2
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

// Invalid shapes 3
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

// -----

// Different types
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

// Invalid element types
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

    // expected-error @+1 {{Element types of sources and destination must match.}}
    %1 = amdgpu.concat %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7:
    tensor<32x64xf32, #blocked>,tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked>, tensor<32x64xf32, #blocked> -> tensor<256x64xf16, #blocked>
    tt.return
  }
}


// -----

// Different layouts 1
#src_layout = #ttg.linear<{register=[[0, 1], [0, 2], [0, 8], [0, 16], [0, 64], [64, 0]], lane=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp=[[0, 32], [32, 0]], block=[]}>
#dst_layout = #ttg.linear<{register=[[0, 1], [0, 2], [0, 8], [0, 16], [0, 64], [0, 128], [64, 0], [128, 0]], lane=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4], [0, 0]], warp=[[0, 32], [32, 0]], block=[]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @invalid_concat(
    %arg0: tensor<128x128xf32, #src_layout>,
    %arg1: tensor<128x128xf32, #src_layout>,
    %arg2: tensor<128x128xf32, #src_layout>,
    %arg3: tensor<128x128xf32, #src_layout>) {

    // expected-error @+1 {{Lane and warp dim basis must match between source and destination layout.}}
    %1 = amdgpu.concat %arg0, %arg1, %arg2, %arg3:
    tensor<128x128xf32, #src_layout>, tensor<128x128xf32, #src_layout>, tensor<128x128xf32, #src_layout>, tensor<128x128xf32, #src_layout> -> tensor<256x256xf32, #dst_layout>
    tt.return
  }
}

// -----

// Different layouts 2
// Case when src and dst layouts have same CTA tile shape, but different number of registers
#src_layout = #ttg.linear<{register=[[1, 0], [2, 0]], lane=[[4, 0], [8, 0], [16, 0], [0, 1], [0, 2], [0, 4]], warp=[[0, 0], [0, 8]], block=[]}>
#dst_layout = #ttg.linear<{register=[[1, 0]], lane=[[4, 0], [8, 0], [16, 0], [0, 1], [0, 2], [0, 4]], warp=[[2, 0], [0, 8]], block=[]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @invalid_concat(
    %arg0: tensor<32x16xf32, #src_layout>,
    %arg1: tensor<32x16xf32, #src_layout>,
    %arg2: tensor<32x16xf32, #src_layout>,
    %arg3: tensor<32x16xf32, #src_layout>) {

    // expected-error @+1 {{Register basis must match on a CTA tile between source and destination.}}
    %1 = amdgpu.concat %arg0, %arg1, %arg2, %arg3:
    tensor<32x16xf32, #src_layout>, tensor<32x16xf32, #src_layout>, tensor<32x16xf32, #src_layout>, tensor<32x16xf32, #src_layout> -> tensor<64x32xf32, #dst_layout>
    tt.return
  }
}
