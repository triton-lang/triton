// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1170 | FileCheck --check-prefixes=COMMON,HWCVT %s
// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 | FileCheck --check-prefixes=COMMON,HWCVT %s
// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 | FileCheck --check-prefixes=COMMON,NOHW %s

// OCP fp8 (f8E4M3FN / f8E5M2) casts: hardware cvt vs. software fallback.
// HWCVT (gfx1170, gfx1200) uses the unscaled cvt ops for downcasts and for
// upcasts, except bf8 -> f16 which is a shift in software. NOHW targets use
// software only.

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

// f16/bf16 -> OCP fp8/bf8, RTNE. No unscaled 16-bit-source op exists, so these
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

// The unscaled hardware downcast is round-to-nearest-even only, so RTZ requests
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

// OCP fp8/bf8 -> f32. The packed upcast does not assemble on gfx12
// (LCOMPILER-2609), so each group of 4 values is packed into one dword and
// converted one byte at a time.

// COMMON-LABEL: upcast_ocp_to_f32
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @upcast_ocp_to_f32(%arg0: tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                             %arg1: tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // HWCVT: rocdl.cvt.f32.fp8 %[[W0:.*]][0] : f32
    // HWCVT: rocdl.cvt.f32.fp8 %[[W0]][1] : f32
    // HWCVT: rocdl.cvt.f32.fp8 %[[W0]][2] : f32
    // HWCVT: rocdl.cvt.f32.fp8 %[[W0]][3] : f32
    // HWCVT-NOT: rocdl.cvt.pk.f32.fp8
    // NOHW-NOT: rocdl.cvt.f32.fp8
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>

    // HWCVT: rocdl.cvt.f32.bf8 %[[W1:.*]][0] : f32
    // HWCVT: rocdl.cvt.f32.bf8 %[[W1]][1] : f32
    // HWCVT: rocdl.cvt.f32.bf8 %[[W1]][2] : f32
    // HWCVT: rocdl.cvt.f32.bf8 %[[W1]][3] : f32
    // HWCVT-NOT: rocdl.cvt.pk.f32.bf8
    // NOHW-NOT: rocdl.cvt.f32.bf8
    %1 = tt.fp_to_fp %arg1 : tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// A single fp8 value per thread still goes through the byte-wise converter;
// the padding lanes are dead and get cleaned up later.

// COMMON-LABEL: single_element_upcast_to_f32
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @single_element_upcast_to_f32(%arg0: tensor<128xf8E4M3FN, #blocked>,
                                        %arg1: tensor<128xf8E5M2, #blocked>) {
    // HWCVT: rocdl.cvt.f32.fp8 %{{.*}}[0] : f32
    // NOHW-NOT: rocdl.cvt.f32.fp8
    %0 = tt.fp_to_fp %arg0 : tensor<128xf8E4M3FN, #blocked> -> tensor<128xf32, #blocked>
    // HWCVT: rocdl.cvt.f32.bf8 %{{.*}}[0] : f32
    // NOHW-NOT: rocdl.cvt.f32.bf8
    %1 = tt.fp_to_fp %arg1 : tensor<128xf8E5M2, #blocked> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

// OCP fp8/bf8 -> f16/bf16 converts to f32 byte-wise, then narrows exactly.
// bf8 -> f16 only widens the mantissa, so it stays in software.

// COMMON-LABEL: upcast_ocp_to_16bit
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @upcast_ocp_to_16bit(%arg0: tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
                               %arg1: tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // COMMON-NOT: rocdl.cvt.pk.f32.fp8
    // HWCVT-COUNT-4: rocdl.cvt.f32.fp8
    // HWCVT-COUNT-4: llvm.fptrunc %{{.+}} : f32 to f16
    // HWCVT-COUNT-4: rocdl.cvt.f32.fp8
    // HWCVT-COUNT-4: llvm.fptrunc %{{.+}} : f32 to f16
    // NOHW-NOT: rocdl.cvt.f32.fp8
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    // COMMON-NOT: rocdl.cvt.f32.bf8
    %1 = tt.fp_to_fp %arg1 : tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    // HWCVT-COUNT-8: rocdl.cvt.f32.fp8
    // NOHW-NOT: rocdl.cvt.f32.fp8
    %2 = tt.fp_to_fp %arg0 : tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    // HWCVT-COUNT-8: rocdl.cvt.f32.bf8
    // COMMON-NOT: rocdl.cvt.pk.f32.bf8
    // NOHW-NOT: rocdl.cvt.f32.bf8
    %3 = tt.fp_to_fp %arg1 : tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// FNUZ fp8 is the gfx942 encoding. These targets have no FNUZ hardware, so the
// unscaled ops must not be selected for it even though the mnemonics match.

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
