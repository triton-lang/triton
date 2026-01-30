// RUN: triton-opt --split-input-file %s --verify-diagnostics

// expected-error @+1 {{WMMA version must be in the [1, 3] range}}
#wmma = #ttg.amd_wmma<{version = 0, isTranspose = false, ctaLayout = {warp = [[0, 1], [1, 0]]}}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<32x32x!tt.ptr<i32,1>, #wmma>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [1, 0], [2, 0]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_output_encoding(%arg0: tensor<32x32xf16, #blocked>) {
// expected-error-re @+15 {{Expect output layout to be transposed per thread:{{.*}}- register=1 -> (1, 0){{.*}}register=2 -> (2, 0){{.*}}register=4 -> (0, 1){{.*}}register=8 -> (0, 2)}}
// Full expected layout is following:
// - register=1 -> (1, 0)
//   register=2 -> (2, 0)
//   register=4 -> (0, 1)
//   register=8 -> (0, 2)}}
// - lane=1 -> (0, 4)
//   lane=2 -> (0, 8)
//   lane=4 -> (0, 16)
//   lane=8 -> (4, 0)
//   lane=16 -> (8, 0)
//   lane=32 -> (16, 0)
// - warp is a size 1 dimension
// - block is a size 1 dimension
// where out dims are: [dim0 (size 32), dim1 (size 32)]
    %0 = amdg.in_thread_transpose %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_input_encoding(%arg0: tensor<32x32xf16, #mfma>) {
// expected-error @+1 {{Expect input tensor in Blocked encoding}}
    %0 = amdg.in_thread_transpose %arg0 : tensor<32x32xf16, #mfma> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_shape(%arg0: tensor<64x64xf16, #blocked>) {
// expected-error @+1 {{Expect equal input and output shapes}}
    %0 = amdg.in_thread_transpose %arg0 : tensor<64x64xf16, #blocked> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_dtype(%arg0: tensor<32x32xf16, #blocked>) {
// expected-error @+1 {{Expect input and output tensor to have same dtype}}
    %0 = amdg.in_thread_transpose %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf32, #linear>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4, 4], threadsPerWarp = [1, 8, 8], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 0, 1], [0, 0, 2]], lane = [[0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_3d_shape(%arg0: tensor<2x32x32xf16, #blocked>) {
// expected-error @+1 {{Expect 2d tensor}}
    %0 = amdg.in_thread_transpose %arg0 : tensor<2x32x32xf16, #blocked> -> tensor<2x32x32xf16, #linear>
    tt.return
  }
}

// -----

#mma32 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @local_load_packed_tranposed_wrong_op_idx(%arg0: !ttg.memdesc<16x64xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x16xi8, #shared1, #smem, mutable>) {
// expected-error @+1 {{Order of dimensions don't match expected}}
    %1 = amdg.local_load_packed_tranposed %arg0 : !ttg.memdesc<16x64xi8, #shared, #smem, mutable> -> tensor<32x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }

  tt.func @local_load_packed_tranposed_wrong_op_idx2(%arg0: !ttg.memdesc<64x16xi8, #shared, #smem, mutable>) {
// expected-error @+1 {{Input and output dimensions don't match after packing changes}}
    %1 = amdg.local_load_packed_tranposed %arg0 : !ttg.memdesc<64x16xi8, #shared, #smem, mutable> -> tensor<32x32xi8, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.return
  }
  //  CHECK-LABEL: ds_transpose_t_fp4_mfma16
  tt.func @local_load_packed_tranposed_wrong_shape(%arg0: !ttg.memdesc<8x128xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x8xi8, #shared1, #smem, mutable>) {
// expected-error @+1 {{only works with DotOperandEncodingAttr dst encoding}}
    %1 = amdg.local_load_packed_tranposed %arg0 : !ttg.memdesc<8x128xi8, #shared, #smem, mutable> -> tensor<256x128xi32, #blocked>
    tt.return
  }

}

// -----

#wmma_v3 = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#wmma_v2 = #ttg.amd_wmma<{version = 2, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#wmma_diff_warp = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [0, 0]]}, instrShape = [16, 16, 32]}>
#wmma_diff_shape = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 64]}>
#wmma_diff_transpose = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 64], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wmma_dot_incompatible_versions(
              %arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>>,
              %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_v2, kWidth = 8}>>,
              %dst: tensor<16x16xf32, #wmma_v3>
  ) {
    // expected-error @+2 {{'tt.dot' op failed to infer returned types}}
    // expected-error @+1 {{Incompatible parent encoding}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_v2, kWidth = 8}>> -> tensor<16x16xf32, #wmma_v3>
    tt.return
  }

  tt.func @wmma_dot_incompatible_warp_layouts(
              %arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>>,
              %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_warp, kWidth = 8}>>,
              %dst: tensor<16x16xf32, #wmma_v3>
  ) {
    // expected-error @+2 {{'tt.dot' op failed to infer returned types}}
    // expected-error @+1 {{Incompatible parent encoding}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_warp, kWidth = 8}>> -> tensor<16x16xf32, #wmma_v3>
    tt.return
  }

  tt.func @wmma_dot_incomptible_shapes(
              %arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>>,
              %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_shape, kWidth = 8}>>,
              %dst: tensor<16x16xf32, #wmma_v3>
  ) {
    // expected-error @+2 {{'tt.dot' op failed to infer returned types}}
    // expected-error @+1 {{Incompatible parent encoding}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_shape, kWidth = 8}>> -> tensor<16x16xf32, #wmma_v3>
    tt.return
  }

  tt.func @wmma_dot_incomptible_transpose(
              %arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>>,
              %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_transpose, kWidth = 8}>>,
              %dst: tensor<16x16xf32, #wmma_v3>
  ) {
    // expected-error @+2 {{'tt.dot' op failed to infer returned types}}
    // expected-error @+1 {{Incompatible parent encoding}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_transpose, kWidth = 8}>> -> tensor<16x16xf32, #wmma_v3>
    tt.return
  }
}

// -----

#wmma_acc = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, CGALayout = [[1, 0], [0, 1]], instrShape = [16, 16, 32]}>
#wmma_dim1 = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, CGALayout = [[1, 0], [0, 0]], instrShape = [16, 16, 32]}>
#wmma_dim2 = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, CGALayout = [[0, 0], [0, 1]], instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wmma_invalid_cga_split_operand_0(
              %arg0: tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim2, kWidth = 8}>>,
              %arg1: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim2, kWidth = 8}>>,
              %dst: tensor<32x32xf32, #wmma_acc>
  ) {
    // expected-error @+1 {{Incompatible CGA layout for operand 0}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim2, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim2, kWidth = 8}>> -> tensor<32x32xf32, #wmma_acc>
    tt.return
  }

  tt.func @wmma_invalid_cga_split_operand_1(
              %arg0: tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim1, kWidth = 8}>>,
              %arg1: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim1, kWidth = 8}>>,
              %dst: tensor<32x32xf32, #wmma_acc>
  ) {
    // expected-error @+1 {{Incompatible CGA layout for operand 1}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim1, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim1, kWidth = 8}>> -> tensor<32x32xf32, #wmma_acc>
    tt.return
  }

  tt.func @wmma_invalid_cga_split_accumulator(
              %arg0: tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim2, kWidth = 8}>>,
              %arg1: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim2, kWidth = 8}>>,
              %dst: tensor<32x32xf32, #wmma_dim1>
  ) {
    // expected-error @+1 {{Accumulator CGA layout should not broadcast or have repeated rows}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim2, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim2, kWidth = 8}>> -> tensor<32x32xf32, #wmma_dim1>
    tt.return
  }
}
