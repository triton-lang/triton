// RUN: triton-opt %s | FileCheck %s


#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: memdesc_subview_spliting
  tt.func public @memdesc_subview_spliting() attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>
    %1 = ttg.memdesc_subview %0[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    %c0_i32_0 = arith.constant 0 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %2 = ttg.memdesc_subview %1[%c0_i32_0, %c0_i32_1] : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %c0_i32_2 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %3 = ttg.memdesc_subview %1[%c0_i32_2, %c32_i32] : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %c0_i32_3 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %4 = ttg.memdesc_subview %1[%c0_i32_3, %c64_i32] : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %c0_i32_4 = arith.constant 0 : i32
    %c96_i32 = arith.constant 96 : i32
    %5 = ttg.memdesc_subview %1[%c0_i32_4, %c96_i32] : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32_5 = arith.constant 0 : i32
    %6 = ttg.memdesc_subview %1[%c128_i32, %c0_i32_5] : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %c128_i32_6 = arith.constant 128 : i32
    %c32_i32_7 = arith.constant 32 : i32
    %7 = ttg.memdesc_subview %1[%c128_i32_6, %c32_i32_7] : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %c128_i32_8 = arith.constant 128 : i32
    %c64_i32_9 = arith.constant 64 : i32
    %8 = ttg.memdesc_subview %1[%c128_i32_8, %c64_i32_9] : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %c128_i32_10 = arith.constant 128 : i32
    %c96_i32_11 = arith.constant 96 : i32
    %9 = ttg.memdesc_subview %1[%c128_i32_10, %c96_i32_11] : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    tt.return
  }
}
