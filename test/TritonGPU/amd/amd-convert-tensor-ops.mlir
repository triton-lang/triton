// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-tensor-ops | FileCheck %s

// Gather i32: indices arrive with the NVIDIA-oriented slice encoding (#nv_slice)
// from the shared TritonToTritonGPU pass. The pass re-layouts them to AMD's TDM
// encoding and lowers to amdg.async_tdm_gather.

#nv_slice = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked_res = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#padded_desc = #ttg.padded_shared<[32:+16] {order = [1, 0], shape = [1, 32]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-DAG: #[[$AMD_IDX_I32:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK-LABEL: @test_gather_i32
// CHECK: ttg.convert_layout %{{.*}} -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #[[$AMD_IDX_I32]]}>>
// CHECK: ttg.local_alloc
// CHECK: amdg.async_tdm_gather {{.*}} pred = %{{.*}} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #[[$AMD_IDX_I32]]
// CHECK: amdg.async_tdm_wait {num = 0 : i32}
// CHECK: ttg.local_load
tt.func public @test_gather_i32(%desc: !tt.tensordesc<1x32xi8, #padded_desc>,
                                %indices: tensor<32xi32, #ttg.slice<{dim = 0, parent = #nv_slice}>>,
                                %y_off: i32) -> tensor<32x32xi8, #blocked_res> {
  %0 = tt.descriptor_gather %desc[%indices, %y_off] : (!tt.tensordesc<1x32xi8, #padded_desc>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #nv_slice}>>, i32) -> tensor<32x32xi8, #blocked_res>
  tt.return %0 : tensor<32x32xi8, #blocked_res>
}
}

// -----

// Gather i16: same NVIDIA-oriented slice encoding on the indices (the shared
// pass doesn't special-case index element type). i16 indices are AMD-only.

#nv_slice16 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked_res16 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#padded_desc16 = #ttg.padded_shared<[32:+16] {order = [1, 0], shape = [1, 32]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-DAG: #[[$AMD_IDX_I16:.*]] = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK-LABEL: @test_gather_i16
// CHECK: ttg.convert_layout %{{.*}} -> tensor<32xi16, #ttg.slice<{dim = 0, parent = #[[$AMD_IDX_I16]]}>>
// CHECK: ttg.local_alloc
// CHECK: amdg.async_tdm_gather {{.*}} pred = %{{.*}} : tensor<32xi16, #ttg.slice<{dim = 0, parent = #[[$AMD_IDX_I16]]
// CHECK: amdg.async_tdm_wait {num = 0 : i32}
// CHECK: ttg.local_load
tt.func public @test_gather_i16(%desc: !tt.tensordesc<1x32xi8, #padded_desc16>,
                                %indices: tensor<32xi16, #ttg.slice<{dim = 0, parent = #nv_slice16}>>,
                                %y_off: i32) -> tensor<32x32xi8, #blocked_res16> {
  %0 = tt.descriptor_gather %desc[%indices, %y_off] : (!tt.tensordesc<1x32xi8, #padded_desc16>, tensor<32xi16, #ttg.slice<{dim = 0, parent = #nv_slice16}>>, i32) -> tensor<32x32xi8, #blocked_res16>
  tt.return %0 : tensor<32x32xi8, #blocked_res16>
}
}

// -----

// Scatter i32: indices arrive with the NVIDIA-oriented slice encoding (#nv_slice)
// from the shared TritonToTritonGPU pass. The pass re-layouts them to AMD's TDM
// encoding and lowers to amdg.async_tdm_scatter.

#nv_slice_s = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked_src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#padded_desc_s = #ttg.padded_shared<[32:+16] {order = [1, 0], shape = [1, 32]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-DAG: #[[$AMD_IDX_S_I32:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK-LABEL: @test_scatter_i32
// CHECK: ttg.convert_layout %{{.*}} -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #[[$AMD_IDX_S_I32]]}>>
// CHECK: ttg.local_alloc {{.*}} : (tensor<32x32xi8
// CHECK: amdg.async_tdm_scatter {{.*}} from {{.*}} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #[[$AMD_IDX_S_I32]]
// CHECK: amdg.async_tdm_wait {num = 0 : i32}
// CHECK-NOT: ttg.local_load
tt.func public @test_scatter_i32(%desc: !tt.tensordesc<1x32xi8, #padded_desc_s>,
                                 %indices: tensor<32xi32, #ttg.slice<{dim = 0, parent = #nv_slice_s}>>,
                                 %y_off: i32,
                                 %src: tensor<32x32xi8, #blocked_src>) {
  tt.descriptor_scatter %desc[%indices, %y_off], %src : !tt.tensordesc<1x32xi8, #padded_desc_s>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #nv_slice_s}>>, i32, tensor<32x32xi8, #blocked_src>
  tt.return
}
}

// -----

// Scatter i16: same NVIDIA-oriented slice encoding on the indices (the shared
// pass doesn't special-case index element type). i16 indices are AMD-only.

#nv_slice_s16 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked_src16 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#padded_desc_s16 = #ttg.padded_shared<[32:+16] {order = [1, 0], shape = [1, 32]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-DAG: #[[$AMD_IDX_S_I16:.*]] = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK-LABEL: @test_scatter_i16
// CHECK: ttg.convert_layout %{{.*}} -> tensor<32xi16, #ttg.slice<{dim = 0, parent = #[[$AMD_IDX_S_I16]]}>>
// CHECK: ttg.local_alloc {{.*}} : (tensor<32x32xi8
// CHECK: amdg.async_tdm_scatter {{.*}} from {{.*}} : tensor<32xi16, #ttg.slice<{dim = 0, parent = #[[$AMD_IDX_S_I16]]
// CHECK: amdg.async_tdm_wait {num = 0 : i32}
// CHECK-NOT: ttg.local_load
tt.func public @test_scatter_i16(%desc: !tt.tensordesc<1x32xi8, #padded_desc_s16>,
                                 %indices: tensor<32xi16, #ttg.slice<{dim = 0, parent = #nv_slice_s16}>>,
                                 %y_off: i32,
                                 %src: tensor<32x32xi8, #blocked_src16>) {
  tt.descriptor_scatter %desc[%indices, %y_off], %src : !tt.tensordesc<1x32xi8, #padded_desc_s16>, tensor<32xi16, #ttg.slice<{dim = 0, parent = #nv_slice_s16}>>, i32, tensor<32x32xi8, #blocked_src16>
  tt.return
}
}

// -----

// CHECK-LABEL: test_cvt1
// CHECK: amdg.async_tdm_copy_global_to_local {{.*}}: !tt.tensordesc<128x16xf16, #shared> -> !ttg.memdesc<128x16xf16, #shared, #smem, mutable>
// CHECK: amdg.async_tdm_wait  {num = 0 : i32}
// CHECK: amdg.async_tdm_copy_local_to_global {{.*}} : !ttg.memdesc<128x128xf16, #shared2, #smem, mutable> -> !tt.tensordesc<128x128xf16, #shared2>
// CHECK: amdg.async_tdm_wait  {num = 0 : i32}

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = [[0, 0], [1, 0]]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0], CGALayout = [[0, 1], [0, 0]]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = [[0, 1], [1, 0]]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [0, 2], [1, 0]]}, CGALayout = [[0, 1], [1, 0]], instrShape = [16, 16, 32]}>
#shared = #ttg.padded_shared<[128:+8] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], block = [[0, 0], [64, 0]]}>
#shared1 = #ttg.padded_shared<[128:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0], [8, 0]], block = [[0, 64], [0, 0]]}>
#shared2 = #ttg.padded_shared<[64:+8] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], block = [[0, 64], [64, 0]]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_cvt1(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                            %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                            %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c512_i64 = arith.constant 512 : i64
    %c512_i32 = arith.constant 512 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c128_i32 : i32
    %3 = arith.muli %1, %c128_i32 : i32
    %4 = tt.make_tensor_descriptor %a_ptr, [%c1024_i32, %c256_i32], [%c256_i64, %c1_i64] : <f16>, <128x16xf16, #shared>
    %5 = tt.make_tensor_descriptor %b_ptr, [%c256_i32, %c512_i32], [%c512_i64, %c1_i64] : <f16>, <16x128xf16, #shared1>
    %6 = tt.make_tensor_descriptor %c_ptr, [%c1024_i32, %c512_i32], [%c512_i64, %c1_i64] : <f16>, <128x128xf16, #shared2>
    %7 = tt.descriptor_load %4[%2, %c0_i32] : !tt.tensordesc<128x16xf16, #shared> -> tensor<128x16xf16, #blocked>
    %8 = tt.descriptor_load %5[%c0_i32, %3] : !tt.tensordesc<16x128xf16, #shared1> -> tensor<16x128xf16, #blocked1>
    %9 = ttg.convert_layout %7 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %10 = ttg.convert_layout %8 : tensor<16x128xf16, #blocked1> -> tensor<16x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %11 = tt.dot %9, %10, %cst : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
    %12 = arith.truncf %11 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %13 = ttg.convert_layout %12 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked2>
    tt.descriptor_store %6[%2, %3], %13 : !tt.tensordesc<128x128xf16, #shared2>, tensor<128x128xf16, #blocked2>
    tt.return
  }
}
