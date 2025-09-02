// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck --check-prefixes=COMMON,GFX942 %s
// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx950 | FileCheck --check-prefixes=COMMON,GFX950 %s

//  CHECK-LABEL: f16_to_f32
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f16_to_f32(%arg0: tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // GFX942-COUNT-8: llvm.fpext %{{.+}} : f16 to f32
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

//  CHECK-LABEL: bf16_to_f32
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @bf16_to_f32(%arg0: tensor<8x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>) {
    // GFX942-COUNT-8: llvm.bitcast
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    tt.return
  }
}

// -----

//  CHECK-LABEL: f32_to_f16
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f32_to_f16(%arg0: tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>) {
    // GFX942-COUNT-8: llvm.fptrunc %{{.+}} : f32 to f16
    // GFX950-COUNT-4: llvm.fptrunc %{{.+}} : vector<2xf32> to vector<2xf16>
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    // COMMON-COUNT-4: rocdl.cvt.pkrtz
    %1 = tt.fp_to_fp %arg0, rounding = rtz : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    tt.return
  }
}

// -----

//  CHECK-LABEL: f32_to_f16_single_value
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @f32_to_f16_single_value(%arg0: tensor<1x128xf32, #blocked>) {
    // COMMON: llvm.fptrunc %{{.+}} : f32 to f16
    // COMMON-NOT: llvm.fptrunc
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<1x128xf32, #blocked> -> tensor<1x128xf16, #blocked>
    // COMMON: rocdl.cvt.pkrtz
    // COMMON-NOT: rocdl.cvt.pkrtz
    %1 = tt.fp_to_fp %arg0, rounding = rtz : tensor<1x128xf32, #blocked> -> tensor<1x128xf16, #blocked>
    tt.return
  }
}

// -----

//  CHECK-LABEL: downcast_to_f8
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @downcast_to_f8(%arg0: tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>,
                     %arg1: tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>,
                     %arg2: tensor<8x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>) {
    // GFX950: rocdl.cvt.scalef32.pk.bf8.f32  %{{.*}}, %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX950: rocdl.cvt.scalef32.pk.bf8.f32  %{{.*}}, %{{.*}}, %{{.*}} -> %{{.*}}[true]
    // GFX950: rocdl.cvt.scalef32.pk.bf8.f32  %{{.*}}, %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX950: rocdl.cvt.scalef32.pk.bf8.f32  %{{.*}}, %{{.*}}, %{{.*}} -> %{{.*}}[true]
    %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>

    // GFX950: rocdl.cvt.scalef32.pk.bf8.f16 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX950: rocdl.cvt.scalef32.pk.bf8.f16 %{{.*}}, %{{.*}} -> %{{.*}}[true]
    // GFX950: rocdl.cvt.scalef32.pk.bf8.f16 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX950: rocdl.cvt.scalef32.pk.bf8.f16 %{{.*}}, %{{.*}} -> %{{.*}}[true]
    %1 = tt.fp_to_fp %arg1, rounding = rtne : tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>

    // GFX950: rocdl.cvt.scalef32.pk.bf8.bf16 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX950: rocdl.cvt.scalef32.pk.bf8.bf16 %{{.*}}, %{{.*}} -> %{{.*}}[true]
    // GFX950: rocdl.cvt.scalef32.pk.bf8.bf16 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX950: rocdl.cvt.scalef32.pk.bf8.bf16 %{{.*}}, %{{.*}} -> %{{.*}}[true]
    %2 = tt.fp_to_fp %arg2, rounding = rtne : tensor<8x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>

    // GFX950: rocdl.cvt.scalef32.pk.fp8.f32 %{{.*}}, %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX950: rocdl.cvt.scalef32.pk.fp8.f32 %{{.*}}, %{{.*}}, %{{.*}} -> %{{.*}}[true]
    // GFX950: rocdl.cvt.scalef32.pk.fp8.f32 %{{.*}}, %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX950: rocdl.cvt.scalef32.pk.fp8.f32 %{{.*}}, %{{.*}}, %{{.*}} -> %{{.*}}[true]
    %3 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>

    // GFX950: rocdl.cvt.scalef32.pk.fp8.f16 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX950: rocdl.cvt.scalef32.pk.fp8.f16 %{{.*}}, %{{.*}} -> %{{.*}}[true]
    // GFX950: rocdl.cvt.scalef32.pk.fp8.f16 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX950: rocdl.cvt.scalef32.pk.fp8.f16 %{{.*}}, %{{.*}} -> %{{.*}}[true]
    %4 = tt.fp_to_fp %arg1, rounding = rtne : tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>

    // GFX950: rocdl.cvt.scalef32.pk.fp8.bf16 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX950: rocdl.cvt.scalef32.pk.fp8.bf16 %{{.*}}, %{{.*}} -> %{{.*}}[true]
    // GFX950: rocdl.cvt.scalef32.pk.fp8.bf16 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX950: rocdl.cvt.scalef32.pk.fp8.bf16 %{{.*}}, %{{.*}} -> %{{.*}}[true]
    %5 = tt.fp_to_fp %arg2, rounding = rtne : tensor<8x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: f32_to_bf8
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @downcast_to_bf8(%arg0: tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>) {
    // GFX942: rocdl.cvt.pk.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX942: rocdl.cvt.pk.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[true]
    // GFX942: rocdl.cvt.pk.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX942: rocdl.cvt.pk.bf8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[true]
    // GFX950-COUNT-16: llvm.trunc %{{.+}} : i32 to i8
    %6 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf8E5M2FNUZ, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: f32_to_f8
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f32_to_f8(%arg0: tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>) {
    // GFX942: rocdl.cvt.pk.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX942: rocdl.cvt.pk.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[true]
    // GFX942: rocdl.cvt.pk.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[false]
    // GFX942: rocdl.cvt.pk.fp8.f32 %{{.*}}, %{{.*}} -> %{{.*}}[true]
    // GFX950-COUNT-16: llvm.trunc %{{.+}} : i32 to i8
    %7 = tt.fp_to_fp %arg0, rounding = rtne : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf8E4M3FNUZ, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    tt.return
  }
}

// -----

//  CHECK-LABEL: upcast_from_f8
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @upcast_from_f8(%arg0: tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>,
                     %arg1: tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>,
                     %arg2: tensor<8x8xf8E5M2FNUZ, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>,
                     %arg3: tensor<8x8xf8E4M3FNUZ, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>) {
    // GFX950: rocdl.cvt.scalef32.pk.f32.bf8 %[[VR1:.*]][false]
    // GFX950: rocdl.cvt.scalef32.pk.f32.bf8 %[[VR1]][true]
    // GFX950: rocdl.cvt.scalef32.pk.f32.bf8 %[[VR2:.*]][false]
    // GFX950: rocdl.cvt.scalef32.pk.f32.bf8 %[[VR2]][true]
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>

    // GFX950: rocdl.cvt.scalef32.pk.f16.bf8 %[[VR3:.*]][false]
    // GFX950: rocdl.cvt.scalef32.pk.f16.bf8 %[[VR3]][true]
    // GFX950: rocdl.cvt.scalef32.pk.f16.bf8 %[[VR4:.*]][false]
    // GFX950: rocdl.cvt.scalef32.pk.f16.bf8 %[[VR4]][true]
    %1 = tt.fp_to_fp %arg0 : tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>

    // GFX950: rocdl.cvt.scalef32.pk.bf16.bf8 %[[VR5:.*]][false]
    // GFX950: rocdl.cvt.scalef32.pk.bf16.bf8 %[[VR5]][true]
    // GFX950: rocdl.cvt.scalef32.pk.bf16.bf8 %[[VR6:.*]][false]
    // GFX950: rocdl.cvt.scalef32.pk.bf16.bf8 %[[VR6]][true]
    %2 = tt.fp_to_fp %arg0 : tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>

    // GFX950: rocdl.cvt.scalef32.pk.f32.fp8 %[[VR7:.*]][false]
    // GFX950: rocdl.cvt.scalef32.pk.f32.fp8 %[[VR7]][true]
    // GFX950: rocdl.cvt.scalef32.pk.f32.fp8 %[[VR8:.*]][false]
    // GFX950: rocdl.cvt.scalef32.pk.f32.fp8 %[[VR8]][true]
    %3 = tt.fp_to_fp %arg1 : tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>

    // GFX950: rocdl.cvt.scalef32.pk.f16.fp8 %[[VR9:.*]][false]
    // GFX950: rocdl.cvt.scalef32.pk.f16.fp8 %[[VR9]][true]
    // GFX950: rocdl.cvt.scalef32.pk.f16.fp8 %[[VR10:.*]][false]
    // GFX950: rocdl.cvt.scalef32.pk.f16.fp8 %[[VR10]][true]
    %4 = tt.fp_to_fp %arg1 : tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>

    // GFX950: rocdl.cvt.scalef32.pk.bf16.fp8 %[[VR11:.*]][false]
    // GFX950: rocdl.cvt.scalef32.pk.bf16.fp8 %[[VR11]][true]
    // GFX950: rocdl.cvt.scalef32.pk.bf16.fp8 %[[VR12:.*]][false]
    // GFX950: rocdl.cvt.scalef32.pk.bf16.fp8 %[[VR12]][true]
    %5 = tt.fp_to_fp %arg1 : tensor<8x8xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>

    // GFX942: rocdl.cvt.pk.f32.bf8 %[[VR13:.*]][false]
    // GFX942: rocdl.cvt.pk.f32.bf8 %[[VR13]][true]
    // GFX942: rocdl.cvt.pk.f32.bf8 %[[VR14:.*]][false]
    // GFX942: rocdl.cvt.pk.f32.bf8 %[[VR14]][true]
    %6 = tt.fp_to_fp %arg2 : tensor<8x8xf8E5M2FNUZ, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>

    // GFX942: rocdl.cvt.pk.f32.fp8 %[[VR15:.*]][false]
    // GFX942: rocdl.cvt.pk.f32.fp8 %[[VR15]][true]
    // GFX942: rocdl.cvt.pk.f32.fp8 %[[VR16:.*]][false]
    // GFX942: rocdl.cvt.pk.f32.fp8 %[[VR16]][true]
    %7 = tt.fp_to_fp %arg3 : tensor<8x8xf8E4M3FNUZ, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    tt.return
  }
}

// -----

//  CHECK-LABEL: f8_rtz
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @f8_rtz(%arg0: tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>,
                     %arg1: tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>) {
    // GFX950-NOT: rocdl.cvt.scalef32.pk.f32.bf8
    // GFX950-COUNT-4: rocdl.cvt.pkrtz
    %1 = tt.fp_to_fp %arg0, rounding = rtz : tensor<8x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    // GFX950-NOT: rocdl.cvt.scalef32.pk.f16.bf8
    %2 = tt.fp_to_fp %arg1, rounding = rtz : tensor<8x8xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    tt.return
  }
}
