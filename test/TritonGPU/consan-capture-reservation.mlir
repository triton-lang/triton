// RUN: split-file %s %t
// RUN: not triton-opt %t/missing.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/missing.mlir --check-prefix=MISSING
// RUN: not triton-opt %t/too-small.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/too-small.mlir --check-prefix=SMALL
// RUN: not triton-opt %t/cross-cta-affine.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/cross-cta-affine.mlir --check-prefix=AFFINE

//--- missing.mlir

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @missing_reservation() {
    // MISSING: WarpSpecialize op is missing 'consan.extra_capture_bytes'
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

//--- cross-cta-affine.mlir

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 0]]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @cross_cta_affine_subslice() {
    %src = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x32xi32, #shared, #smem, mutable>
    // AFFINE: error: memdesc subslices with cross-CTA affine offsets are unsupported by buffer region analysis
    %tile = ttg.memdesc_subslice %src [2, 0] : !ttg.memdesc<4x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32>
    %indices = arith.constant dense<0> : tensor<2x32xi32, #blocked>
    %gathered = ttg.local_gather %tile[%indices] {axis = 1 : i32} : !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32>, tensor<2x32xi32, #blocked> -> tensor<2x32xi32, #blocked>
    tt.return
  }
}

//--- too-small.mlir

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @small_reservation() {
    // SMALL: ConSan WarpSpecialize capture reservation is too small: reserved 0 bytes, but 1 captures require 8 bytes
    ttg.warp_specialize() attributes {consan.extra_capture_bytes = 0 : i32}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
