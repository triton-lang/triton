// RUN: split-file %s %t
// RUN: not triton-opt %t/missing.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/missing.mlir --check-prefix=MISSING
// RUN: not triton-opt %t/too-small.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/too-small.mlir --check-prefix=SMALL
// RUN: not triton-opt %t/cross-cta-affine.mlir -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/cross-cta-affine.mlir --check-prefix=AFFINE
// RUN: triton-opt %t/convert-only.mlir -tritoninstrument-prepare-consan-captures="target=nvidia" | FileCheck %t/convert-only.mlir --check-prefix=CONVERT
// RUN: triton-opt %t/reduce-only.mlir -tritoninstrument-prepare-consan-captures="target=nvidia" | FileCheck %t/reduce-only.mlir --check-prefix=REDUCE
// RUN: not triton-opt %t/convert-missing-metadata.mlir -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/convert-missing-metadata.mlir --check-prefix=CVT-MISSING
// RUN: not triton-opt %t/convert-partial-metadata.mlir -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/convert-partial-metadata.mlir --check-prefix=CVT-PARTIAL
// RUN: not triton-opt %t/convert-invalid-metadata.mlir -tritoninstrument-concurrency-sanitizer 2>&1 | FileCheck %t/convert-invalid-metadata.mlir --check-prefix=CVT-INVALID

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

//--- reduce-only.mlir

#reduce = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 16 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @reduce_only_reservation(%arg0: tensor<1x256xf32, #reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<1x256xf32, #reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #reduce}>>
    // REDUCE: ttg.warp_specialize() attributes {consan.extra_capture_bytes = 24 : i32}
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

//--- convert-only.mlir

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#dst = #ttg.slice<{dim = 1, parent = #dst_parent}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @convert_only_reservation(%arg0: tensor<128xi32, #src>) {
    %0 = ttg.convert_layout %arg0 : tensor<128xi32, #src> -> tensor<128xi32, #dst>
    // CONVERT: ttg.warp_specialize() attributes {consan.extra_capture_bytes = 24 : i32}
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

//--- convert-missing-metadata.mlir

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#dst = #ttg.slice<{dim = 1, parent = #dst_parent}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @convert_missing_metadata(%arg0: tensor<128xi32, #src>) {
    // CVT-MISSING: error: shared-memory convert_layout is missing scratch allocation metadata
    %0 = ttg.convert_layout %arg0 : tensor<128xi32, #src> -> tensor<128xi32, #dst>
    tt.return
  }
}

//--- convert-partial-metadata.mlir

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#dst = #ttg.slice<{dim = 1, parent = #dst_parent}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @convert_partial_metadata(%arg0: tensor<128xi32, #src>) {
    // CVT-PARTIAL: error: convert_layout scratch allocation metadata must include both allocation.offset and allocation.size
    %0 = ttg.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<128xi32, #src> -> tensor<128xi32, #dst>
    tt.return
  }
}

//--- convert-invalid-metadata.mlir

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#dst = #ttg.slice<{dim = 1, parent = #dst_parent}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  tt.func public @convert_invalid_metadata(%arg0: tensor<128xi32, #src>) {
    // CVT-INVALID: error: invalid convert_layout scratch allocation metadata: offset 16777215, size 2
    %0 = ttg.convert_layout %arg0 {allocation.offset = 16777215 : i32, allocation.size = 2 : i32} : tensor<128xi32, #src> -> tensor<128xi32, #dst>
    tt.return
  }
}
