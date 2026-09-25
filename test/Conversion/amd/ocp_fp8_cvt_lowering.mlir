// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1170 | FileCheck --check-prefixes=COMMON,HWCVT %s
// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 | FileCheck --check-prefixes=COMMON,NOHW %s
// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 | FileCheck --check-prefixes=COMMON,NOHW %s

// Whether OCP fp8 (f8E4M3FN / f8E5M2) casts lower to the hardware cvt
// instructions or to the software bit-manipulation fallback, per target. COMMON
// covers the fp16/bf16 <-> fp32 casts, which take the same scalar LLVM path
// everywhere.
//
// HWCVT (gfx1170) has the *plain* v_cvt_pk_{f32_fp8,f32_bf8,fp8_f32,bf8_f32}
// ops, which encode OCP on that target, but not the scaled ops CDNA4 and
// gfx1250 use: expect rocdl.cvt.pk.*, not rocdl.cvt.scalef32.pk.*. There is no
// plain 16-bit-source op, so f16/bf16 casts hop through f32; both hops are
// lossless here, leaving the hardware op as the only rounding site.
//
// NOHW keeps the software fallback, so the hardware path cannot leak onto a
// target that fails to run it. gfx1100/RDNA3 lacks the instructions outright;
// gfx1200/RDNA4 has them, but its packed upcast prints without the .l/.h suffix
// the assembler needs in gfx12's default real-true16 mode (LCOMPILER-2609).

// COMMON-LABEL: f16_to_f32
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f16_to_f32(%arg0: tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // COMMON-COUNT-8: llvm.fpext %{{.+}} : f16 to f32
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// COMMON-LABEL: f32_to_f16
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f32_to_f16(%arg0: tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // rtne: scalar fptrunc.
    // COMMON-COUNT-8: llvm.fptrunc %{{.+}} : f32 to f16
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    // rtz: packed round-to-zero conversion.
    // COMMON-COUNT-4: rocdl.cvt.pkrtz
    %1 = tt.fp_to_fp %arg0, rounding = rtz : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// f32 -> OCP fp8/bf8, RTNE. Each group of 4 elements becomes two packed
// converts, chained through the `old` operand so the second call fills the
// other half of the same dword. 8 elements per thread => 4 converts.

// COMMON-LABEL: downcast_f32_to_ocp_f8
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @downcast_f32_to_ocp_f8(%arg0: tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // HWCVT: %[[P0:.*]] = rocdl.cvt.pk.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // HWCVT: rocdl.cvt.pk.fp8.f32 %{{.*}}, %{{.*}} -> %[[P0]][true]
    // HWCVT: %[[P1:.*]] = rocdl.cvt.pk.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // HWCVT: rocdl.cvt.pk.fp8.f32 %{{.*}}, %{{.*}} -> %[[P1]][true]
    // NOHW-NOT: rocdl.cvt.pk.fp8.f32
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// COMMON-LABEL: downcast_f32_to_ocp_bf8
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @downcast_f32_to_ocp_bf8(%arg0: tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // HWCVT: %[[P0:.*]] = rocdl.cvt.pk.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // HWCVT: rocdl.cvt.pk.bf8.f32 %{{.*}}, %{{.*}} -> %[[P0]][true]
    // HWCVT: %[[P1:.*]] = rocdl.cvt.pk.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // HWCVT: rocdl.cvt.pk.bf8.f32 %{{.*}}, %{{.*}} -> %[[P1]][true]
    // NOHW-NOT: rocdl.cvt.pk.bf8.f32
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// f16/bf16 -> OCP fp8/bf8, RTNE. No plain 16-bit-source op exists, so these
// widen to f32 first and then use the same packed downcast.

// COMMON-LABEL: downcast_16bit_to_ocp_f8
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @downcast_16bit_to_ocp_f8(%arg0: tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                                  %arg1: tensor<8x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // Converters run one group of 4 elements at a time, so the widening and the
    // packed converts interleave per group rather than being hoisted.
    // HWCVT-COUNT-4: llvm.fpext %{{.+}} : f16 to f32
    // HWCVT-COUNT-2: rocdl.cvt.pk.fp8.f32
    // HWCVT-COUNT-4: llvm.fpext %{{.+}} : f16 to f32
    // HWCVT-COUNT-2: rocdl.cvt.pk.fp8.f32
    // NOHW-NOT: rocdl.cvt.pk.fp8.f32
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>

    // HWCVT-COUNT-4: rocdl.cvt.pk.bf8.f32
    // NOHW-NOT: rocdl.cvt.pk.bf8.f32
    %1 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>

    // HWCVT-COUNT-4: rocdl.cvt.pk.fp8.f32
    // NOHW-NOT: rocdl.cvt.pk.fp8.f32
    %2 = tt.fp_to_fp %arg1, rounding = rtne : tensor<8x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>

    // HWCVT-COUNT-4: rocdl.cvt.pk.bf8.f32
    // NOHW-NOT: rocdl.cvt.pk.bf8.f32
    %3 = tt.fp_to_fp %arg1, rounding = rtne : tensor<8x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// OCP fp8/bf8 -> f32. Two packed upcasts per group of 4, both reading the same
// source dword and selecting opposite halves via wordSel.

// COMMON-LABEL: upcast_ocp_f8_to_f32
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @upcast_ocp_f8_to_f32(%arg0: tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // HWCVT: rocdl.cvt.pk.f32.fp8 %[[V0:.*]][false]
    // HWCVT: rocdl.cvt.pk.f32.fp8 %[[V0]][true]
    // HWCVT: rocdl.cvt.pk.f32.fp8 %[[V1:.*]][false]
    // HWCVT: rocdl.cvt.pk.f32.fp8 %[[V1]][true]
    // NOHW-NOT: rocdl.cvt.pk.f32.fp8
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// COMMON-LABEL: upcast_ocp_bf8_to_f32
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @upcast_ocp_bf8_to_f32(%arg0: tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // HWCVT: rocdl.cvt.pk.f32.bf8 %[[V0:.*]][false]
    // HWCVT: rocdl.cvt.pk.f32.bf8 %[[V0]][true]
    // HWCVT: rocdl.cvt.pk.f32.bf8 %[[V1:.*]][false]
    // HWCVT: rocdl.cvt.pk.f32.bf8 %[[V1]][true]
    // NOHW-NOT: rocdl.cvt.pk.f32.bf8
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// OCP fp8/bf8 -> f16/bf16. Packed upcast to f32, then narrow. Every OCP fp8
// value is exactly representable in both f16 and bf16, so the narrowing step
// cannot introduce error.

// COMMON-LABEL: upcast_ocp_f8_to_16bit
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @upcast_ocp_f8_to_16bit(%arg0: tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                                  %arg1: tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // Per group of 4: two packed upcasts, then four narrowing truncs.
    // HWCVT-COUNT-2: rocdl.cvt.pk.f32.fp8
    // HWCVT-COUNT-4: llvm.fptrunc %{{.+}} : f32 to f16
    // HWCVT-COUNT-2: rocdl.cvt.pk.f32.fp8
    // HWCVT-COUNT-4: llvm.fptrunc %{{.+}} : f32 to f16
    // NOHW-NOT: rocdl.cvt.pk.f32.fp8
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>

    // HWCVT-COUNT-4: rocdl.cvt.pk.f32.bf8
    // NOHW-NOT: rocdl.cvt.pk.f32.bf8
    %1 = tt.fp_to_fp %arg1 : tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>

    // HWCVT-COUNT-4: rocdl.cvt.pk.f32.fp8
    // NOHW-NOT: rocdl.cvt.pk.f32.fp8
    %2 = tt.fp_to_fp %arg0 : tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>

    // HWCVT-COUNT-4: rocdl.cvt.pk.f32.bf8
    // NOHW-NOT: rocdl.cvt.pk.f32.bf8
    %3 = tt.fp_to_fp %arg1 : tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// The plain hardware downcast is round-to-nearest-even only, so RTZ requests
// must stay on the software path even on HWCVT targets. Guards against a future
// change wiring RTZ to the RTNE instruction.

// COMMON-LABEL: downcast_rtz_stays_software
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @downcast_rtz_stays_software(%arg0: tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                                       %arg1: tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // COMMON-NOT: rocdl.cvt.pk.bf8.f32
    // COMMON-COUNT-4: rocdl.cvt.pkrtz
    %0 = tt.fp_to_fp %arg0, rounding = rtz : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    // COMMON-NOT: rocdl.cvt.pk.bf8.f32
    %1 = tt.fp_to_fp %arg1, rounding = rtz : tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// A layout that leaves a single fp8 value per thread upcasts with the scalar
// converter rather than the packed one. The packed op would need the value
// padded out with undef and would discard three quarters of its result.

// COMMON-LABEL: single_element_upcast_is_scalar
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @single_element_upcast_is_scalar(%arg0: tensor<128xf8E4M3FN, #blocked>,
                                           %arg1: tensor<128xf8E5M2, #blocked>) {
    // HWCVT-COUNT-1: rocdl.cvt.f32.fp8
    // HWCVT-NOT: rocdl.cvt.pk.f32.fp8
    // NOHW-NOT: rocdl.cvt.f32.fp8
    %0 = tt.fp_to_fp %arg0 : tensor<128xf8E4M3FN, #blocked> -> tensor<128xf32, #blocked>

    // HWCVT-COUNT-1: rocdl.cvt.f32.bf8
    // HWCVT-NOT: rocdl.cvt.pk.f32.bf8
    // NOHW-NOT: rocdl.cvt.f32.bf8
    %1 = tt.fp_to_fp %arg1 : tensor<128xf8E5M2, #blocked> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

// Two values per thread still use the packed converter: only the single-value
// case is special-cased, so this pins the boundary.

// COMMON-LABEL: two_element_upcast_stays_packed
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @two_element_upcast_stays_packed(%arg0: tensor<256xf8E4M3FN, #blocked>) {
    // HWCVT-NOT: rocdl.cvt.f32.fp8
    // HWCVT-COUNT-2: rocdl.cvt.pk.f32.fp8
    %0 = tt.fp_to_fp %arg0 : tensor<256xf8E4M3FN, #blocked> -> tensor<256xf32, #blocked>
    tt.return
  }
}

// -----

// The scalar converter is used only for an f32 destination. A single fp8 value
// headed for f16/bf16 keeps the packed upcast, which here still discards three
// quarters of its result: there is no plain op landing directly in 16 bits, so
// the f32 intermediate is needed either way.

// COMMON-LABEL: single_element_upcast_to_16bit_stays_packed
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @single_element_upcast_to_16bit_stays_packed(%arg0: tensor<128xf8E4M3FN, #blocked>,
                                                      %arg1: tensor<128xf8E5M2, #blocked>) {
    // HWCVT-NOT: rocdl.cvt.f32.fp8
    // HWCVT-COUNT-2: rocdl.cvt.pk.f32.fp8
    %0 = tt.fp_to_fp %arg0 : tensor<128xf8E4M3FN, #blocked> -> tensor<128xf16, #blocked>

    // HWCVT-NOT: rocdl.cvt.f32.bf8
    // HWCVT-COUNT-2: rocdl.cvt.pk.f32.bf8
    %1 = tt.fp_to_fp %arg1 : tensor<128xf8E5M2, #blocked> -> tensor<128xbf16, #blocked>
    tt.return
  }
}

// -----

// FNUZ fp8 is the gfx942 encoding. These targets have no FNUZ hardware, so the
// plain ops must not be selected for it even though the mnemonics match.

// COMMON-LABEL: fnuz_stays_software
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @fnuz_stays_software(%arg0: tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                               %arg1: tensor<8x8xf8E4M3FNUZ, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // COMMON-NOT: rocdl.cvt.pk.fp8.f32
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf8E4M3FNUZ, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    // COMMON-NOT: rocdl.cvt.pk.f32.fp8
    %1 = tt.fp_to_fp %arg1 : tensor<8x8xf8E4M3FNUZ, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}
