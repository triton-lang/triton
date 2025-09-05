// RUN: triton-opt %s | FileCheck %s


#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#padded = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [256, 128]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: memdesc_subslice_spliting
  tt.func public @memdesc_subslice_spliting() {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    %2 = ttg.memdesc_subslice %1 [0, 0]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %3 = ttg.memdesc_subslice %1 [0, 32]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %4 = ttg.memdesc_subslice %1 [0, 64]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %5 = ttg.memdesc_subslice %1 [0, 96]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %6 = ttg.memdesc_subslice %1 [128, 0]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %7 = ttg.memdesc_subslice %1 [128, 32]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %8 = ttg.memdesc_subslice %1 [128, 64]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %9 = ttg.memdesc_subslice %1 [128, 96]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>

    %padded = ttg.local_alloc : () -> !ttg.memdesc<1x256x128xf16, #padded, #smem, mutable>
    %padded_indexed_explicit_alloc_shape = ttg.memdesc_index %padded[%c0_i32] : !ttg.memdesc<1x256x128xf16, #padded, #smem, mutable> -> !ttg.memdesc<256x128xf16, #padded, #smem, mutable, 1x256x128>
    %10 = ttg.memdesc_subslice %padded_indexed_explicit_alloc_shape [128, 96]  : !ttg.memdesc<256x128xf16, #padded, #smem, mutable, 1x256x128> -> !ttg.memdesc<128x32xf16, #padded, #smem, mutable, 1x256x128>
    %padded_indexed_implicit_alloc_shape = ttg.memdesc_index %padded[%c0_i32] : !ttg.memdesc<1x256x128xf16, #padded, #smem, mutable> -> !ttg.memdesc<256x128xf16, #padded, #smem, mutable>
    %11 = ttg.memdesc_subslice %padded_indexed_implicit_alloc_shape [128, 96]  : !ttg.memdesc<256x128xf16, #padded, #smem, mutable> -> !ttg.memdesc<128x32xf16, #padded, #smem, mutable, 256x128>
    tt.return
  }
}
