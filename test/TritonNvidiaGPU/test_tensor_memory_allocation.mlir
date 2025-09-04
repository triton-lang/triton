// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -triton-tensor-memory-allocation | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, unpacked = true>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: ttg.tensor_memory_size = 512
  // CHECK: alloc_tensor_memory
  tt.func public @alloc_tensor_memory(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blocked1>
    %cst2 = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked2>
    %cst3 = arith.constant dense<0> : tensor<64x4xi8, #linear>
    %cst4 = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked2>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %0 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32}
    %1 = ttng.tmem_alloc %cst0 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 256 : i32, tensor_memory_row_offset = 0 : i32}
    %2 = ttng.tmem_alloc %cst1 : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 320 : i32, tensor_memory_row_offset = 0 : i32}
    %3 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    ttng.tmem_store %cst, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst0, %1, %true : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst1, %2, %true : tensor<64x64xf16, #blocked1> -> !ttg.memdesc<64x64xf16, #tmem1, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %3, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %4 = ttng.tmem_alloc %cst4 : (tensor<64x128xf16, #blocked2>) -> !ttg.memdesc<64x128xf16, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 16 : i32}
    %5 = ttng.tmem_alloc %cst4 : (tensor<64x128xf16, #blocked2>) -> !ttg.memdesc<64x128xf16, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32}
    %6 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    ttng.tmem_store %cst2, %4, %true : tensor<64x128xf16, #blocked2> -> !ttg.memdesc<64x128xf16, #tmem2, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst2, %5, %true : tensor<64x128xf16, #blocked2> -> !ttg.memdesc<64x128xf16, #tmem2, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %6, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    %7 = ttng.tmem_alloc : () -> !ttg.memdesc<64x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc  {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %8 = ttng.tmem_alloc : () -> !ttg.memdesc<64x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc  {tensor_memory_col_offset = 4 : i32, tensor_memory_row_offset = 0 : i32}

    ttng.tmem_store %cst3, %7, %true : tensor<64x4xi8, #linear> -> !ttg.memdesc<64x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst3, %8, %true : tensor<64x4xi8, #linear> -> !ttg.memdesc<64x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>


    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: ttg.tensor_memory_size = 512
  // CHECK: alloc_tensor_memory_re_use
  tt.func public @alloc_tensor_memory_re_use(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %true = arith.constant true
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blocked>
    %cst2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked1>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %a = ttng.tmem_alloc %cst0 : (tensor<128x256xf32, #blocked>) -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %0 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %1 = ttng.tmem_alloc %cst2 : (tensor<128x64xf32, #blocked1>) -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32}
    %2 = ttng.tmem_alloc %cst2 : (tensor<128x64xf32, #blocked1>) -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst2, %1, %true : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst2, %2, %true : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>

    // Test that the 2 allocations above are re-used.
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %3 = ttng.tmem_alloc %cst0 : (tensor<128x256xf32, #blocked>) -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %4 = ttng.tmem_alloc %cst2 : (tensor<128x64xf32, #blocked1>) -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32}
    %5 = ttng.tmem_alloc %cst2 : (tensor<128x64xf32, #blocked1>) -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst2, %4, %true : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>

    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32}
    %6 = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %s = ttg.memdesc_index %6[%c1] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %7 = ttng.tmem_alloc %cst2 : (tensor<128x64xf32, #blocked1>) -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 384 : i32, tensor_memory_row_offset = 0 : i32}
    %8 = ttng.tmem_alloc %cst2 : (tensor<128x64xf32, #blocked1>) -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>

    ttng.tmem_store %cst, %s, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst2, %7, %true : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst2, %5, %true : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: ttg.tensor_memory_size = 128
  // CHECK: alloc_tensor_memory_re_use_liverange_end_collision
  tt.func public @alloc_tensor_memory_re_use_liverange_end_collision(
                                             %arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>,
                                             %lb: index, %ub: index, %step: index) {
    %true = arith.constant true
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %cst2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %a = ttng.tmem_alloc %cst0 : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32}
    %b = ttng.tmem_alloc %cst : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>

    scf.for %i = %lb to %ub step %step {
      ttng.tmem_store %cst2, %a, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tmem_store %cst2, %b, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield
    }
    // Liveranges of both allocations end at the same time, at the boundary of the loop. Make sure we can handle this case.

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %c = ttng.tmem_alloc %cst0 : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32}
    %d = ttng.tmem_alloc %cst : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>

    ttng.tmem_store %cst2, %c, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst2, %d, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>

    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [2, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [2, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true, CTASplitM = 2>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true, CTASplitN = 2>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, ttg.shared = 65536 : i32} {
  // CHECK-LABEL: multi_ctas
  tt.func public @multi_ctas() {
    %true = arith.constant true
    %cst0 = arith.constant dense<0.000000e+00> : tensor<256x128xf16, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<256x128xf16, #blocked1>

    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<256x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32}
    %1 = ttng.tmem_alloc : () -> !ttg.memdesc<256x128xf16, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 256 : i32, tensor_memory_row_offset = 0 : i32}
    %2 = ttng.tmem_alloc : () -> !ttg.memdesc<256x128xf16, #tmem, #ttng.tensor_memory, mutable>

    ttng.tmem_store %cst1, %0, %true : tensor<256x128xf16, #blocked1> -> !ttg.memdesc<256x128xf16, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst0, %1, %true : tensor<256x128xf16, #blocked> -> !ttg.memdesc<256x128xf16, #tmem1, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst1, %2, %true : tensor<256x128xf16, #blocked1> -> !ttg.memdesc<256x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#layout = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem = #ttng.tensor_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @alloc_warp_specialize
tt.func @alloc_warp_specialize() {
  // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
  %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
  ttg.warp_specialize()
  default {
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32}
    %1 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32}
    %2 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 256 : i32, tensor_memory_row_offset = 0 : i32}
    %1 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 384 : i32, tensor_memory_row_offset = 0 : i32}
    %2 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
    "use"(%1) : (!ttg.memdesc<128x128xf32, #layout, #tmem, mutable>) -> ()
    ttg.warp_return
  } : () -> ()
  "use"(%0) : (!ttg.memdesc<128x128xf32, #layout, #tmem, mutable>) -> ()
  tt.return
}

// CHECK-LABEL: @alloc_warp_specialize_explicit_capture
tt.func @alloc_warp_specialize_explicit_capture() {
  // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
  %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
  ttg.warp_specialize(%0)
  default {
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32}
    %1 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>) num_warps(1) {
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 256 : i32, tensor_memory_row_offset = 0 : i32}
    %1 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x128xf32, #layout, #tmem, mutable>) -> ()
  tt.return
}

}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32} {

// CHECK-LABEL: @mma_lhs_tmem
tt.func @mma_lhs_tmem(
  %b: !ttg.memdesc<64x64xf16, #shared1, #ttg.shared_memory>,
  %useAcc: i1,
  %pred: i1,
  %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
  %barrierPred: i1
) {
  // CHECK-COUNT-2: ttng.tmem_alloc {{.*}} tensor_memory_row_offset = 0 : i32
  // CHECK-NOT: tensor_memory_row_offset
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>
  %c = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
  ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier[%barrierPred] {is_async} :
    !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<64x64xf16, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
  tt.return
}

// CHECK-LABEL: @mma_scaled_lhs_tmem
tt.func @mma_scaled_lhs_tmem(
  %b: !ttg.memdesc<64x64xf16, #shared1, #ttg.shared_memory>,
  %scale_a: !ttg.memdesc<128x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>,
  %scale_b: !ttg.memdesc<256x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>,
  %useAcc: i1,
  %pred: i1,
  %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
  %barrierPred: i1
) {
  // CHECK-COUNT-2: ttng.tmem_alloc {{.*}} tensor_memory_row_offset = 0 : i32
  // CHECK-NOT: tensor_memory_row_offset
  %a = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>
  %c = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
  ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e2m1 rhs = e2m1, %barrier[%barrierPred] {is_async} :
    !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<64x64xf16, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<128x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<256x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
  tt.return
}

}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = false>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @alloc_warp_specialize_explicit_capture_subview
tt.func @alloc_warp_specialize_explicit_capture_subview() {
  // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
  %0 = ttg.local_alloc {allocation.offset = 196880 : i32} : () -> !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
  %1 = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x128xbf16, #tmem1, #ttng.tensor_memory, mutable>
  %2 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xbf16, #shared1, #smem, mutable>
  // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32}
  %3 = ttng.tmem_alloc : () -> !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>
  ttg.warp_specialize(%2, %1, %3, %0)
  default {
    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<2x128x128xbf16, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<1x64x128xbf16, #tmem1, #ttng.tensor_memory, mutable>, %arg2: !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg3: !ttg.memdesc<2x1xi64, #shared, #smem, mutable>) num_warps(1) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32

    %b = ttg.memdesc_index %arg0[%c0_i32] : !ttg.memdesc<2x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem>
    %a = ttg.memdesc_index %arg1[%c0_i32] : !ttg.memdesc<1x64x128xbf16, #tmem1, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xbf16, #tmem1, #ttng.tensor_memory, mutable, 1x64x128>
    %d = ttg.memdesc_index %arg2[%c0_i32] : !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x64x128>
    %barrier = ttg.memdesc_index %arg3[%c0_i32] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>

    ttng.tc_gen5_mma %a, %b, %d, %true, %true, %barrier[%true] {is_async} : !ttg.memdesc<64x128xbf16, #tmem1, #ttng.tensor_memory, mutable, 1x64x128>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x64x128>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<2x128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<1x64x128xbf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x1xi64, #shared, #smem, mutable>) -> ()
  tt.return
}

// CHECK-LABEL: @alloc_warp_specialize_explicit_capture
tt.func @alloc_warp_specialize_explicit_capture() {
  // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
  %0 = ttg.local_alloc {allocation.offset = 196880 : i32} : () -> !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
  %1 = ttng.tmem_alloc : () -> !ttg.memdesc<64x128xbf16, #tmem1, #ttng.tensor_memory, mutable>
  %2 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xbf16, #shared1, #smem, mutable>
  // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32}
  %3 = ttng.tmem_alloc : () -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
  ttg.warp_specialize(%2, %1, %3, %0)
  default {
    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<2x128x128xbf16, #shared1, #smem, mutable>, %arg1: !ttg.memdesc<64x128xbf16, #tmem1, #ttng.tensor_memory, mutable>, %arg2: !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg3: !ttg.memdesc<2x1xi64, #shared, #smem, mutable>) num_warps(1) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32

    %b = ttg.memdesc_index %arg0[%c0_i32] : !ttg.memdesc<2x128x128xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xbf16, #shared1, #smem>
    %barrier = ttg.memdesc_index %arg3[%c0_i32] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>

    ttng.tc_gen5_mma %arg1, %b, %arg2, %true, %true, %barrier[%true] {is_async} : !ttg.memdesc<64x128xbf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<2x128x128xbf16, #shared1, #smem, mutable>, !ttg.memdesc<64x128xbf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x1xi64, #shared, #smem, mutable>) -> ()
  tt.return
}

}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32} {

// CHECK-LABEL: @mma_lhs_tmem
tt.func @mma_lhs_tmem(
  %b: !ttg.memdesc<64x64xf16, #shared1, #ttg.shared_memory>,
  %useAcc: i1,
  %pred: i1,
  %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
  %barrierPred: i1
) {
  // CHECK-COUNT-4: ttng.tmem_alloc {{.*}} tensor_memory_row_offset = 0 : i32
  // CHECK-NOT: tensor_memory_row_offset
  %a0 = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>
  %a1 = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>
  %a2 = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>
  %c = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>

  %a = arith.select %barrierPred, %a0, %a1 : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>

  cf.cond_br %barrierPred, ^switch, ^bb1(%a : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>)

^switch:
  cf.br ^bb1(%a2 : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>)

^bb1(%lhs: !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>):
  ttng.tc_gen5_mma %lhs, %b, %c, %useAcc, %pred, %barrier[%barrierPred] {is_async} :
    !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<64x64xf16, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
  tt.return
}

}
