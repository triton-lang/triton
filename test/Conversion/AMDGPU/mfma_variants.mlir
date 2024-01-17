// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm="target=rocdl" 2>/dev/null | FileCheck --check-prefixes=CHECK,GCN %s

!a_ty = f8E4M3FNUZ
!b_ty = f8E4M3FNUZ
!c_ty = f32
#k_width = 8
#non_k_dim = 32
#mfmaVersion = 3
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_32x32x16_fp8_fp8
  tt.func @convert_dot_mfma_f32_32x32x16_fp8_fp8(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-64: rocdl.mfma.f32.32x32x16.fp8.fp8
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = f8E4M3FNUZ
!b_ty = f8E5M2FNUZ
!c_ty = f32
#k_width = 8
#non_k_dim = 32
#mfmaVersion = 3
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_32x32x16_fp8_bf8
  tt.func @convert_dot_mfma_f32_32x32x16_fp8_bf8(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-64: rocdl.mfma.f32.32x32x16.fp8.bf8
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = f8E5M2FNUZ
!b_ty = f8E4M3FNUZ
!c_ty = f32
#k_width = 8
#non_k_dim = 32
#mfmaVersion = 3
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_32x32x16_bf8_fp8
  tt.func @convert_dot_mfma_f32_32x32x16_bf8_fp8(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-64: rocdl.mfma.f32.32x32x16.bf8.fp8
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = f8E5M2FNUZ
!b_ty = f8E5M2FNUZ
!c_ty = f32
#k_width = 8
#non_k_dim = 32
#mfmaVersion = 3
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_32x32x16_bf8_bf8
  tt.func @convert_dot_mfma_f32_32x32x16_bf8_bf8(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-64: rocdl.mfma.f32.32x32x16.bf8.bf8
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = f16
!b_ty = f16
!c_ty = f32
#k_width = 4
#non_k_dim = 32
#mfmaVersion = 2
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion , warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_32x32x8f16
  tt.func @convert_dot_mfma_f32_32x32x8f16(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-128: rocdl.mfma.f32.32x32x8f16
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = bf16
!b_ty = bf16
!c_ty = f32
#k_width = 2
#non_k_dim = 32
#mfmaVersion = 1
#mfma = #triton_gpu.mfma<{versionMajor = 1, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_32x32x4bf16
  tt.func @convert_dot_mfma_f32_32x32x4bf16(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-256: rocdl.mfma.f32.32x32x4bf16
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = bf16
!b_ty = bf16
!c_ty = f32
#k_width = 4
#non_k_dim = 32
#mfmaVersion = 2
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion , warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_32x32x8bf16_1k
  tt.func @convert_dot_mfma_f32_32x32x8bf16_1k(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-128: rocdl.mfma.f32.32x32x8bf16.1k
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = f32
!b_ty = f32
!c_ty = f32
#k_width = 1
#non_k_dim = 32
#mfmaVersion = 2
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion , warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_32x32x2f32
  tt.func @convert_dot_mfma_f32_32x32x2f32(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-512: rocdl.mfma.f32.32x32x2f32
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = i8
!b_ty = i8
!c_ty = i32
#k_width = 4
#non_k_dim = 32
#mfmaVersion = 2
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion , warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_i32_32x32x8i8
  tt.func @convert_dot_mfma_i32_32x32x8i8(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-128: rocdl.mfma.i32.32x32x8i8
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = i8
!b_ty = i8
!c_ty = i32
#k_width = 8
#non_k_dim = 32
#mfmaVersion = 3
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_i32_32x32x16_i8
  tt.func @convert_dot_mfma_i32_32x32x16_i8(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-64: rocdl.mfma.i32.32x32x16.i8
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = f8E4M3FNUZ
!b_ty = f8E4M3FNUZ
!c_ty = f32
#k_width = 8
#non_k_dim = 16
#mfmaVersion = 3
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_16x16x32_fp8_fp8
  tt.func @convert_dot_mfma_f32_16x16x32_fp8_fp8(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-128: rocdl.mfma.f32.16x16x32.fp8.fp8
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = f8E4M3FNUZ
!b_ty = f8E5M2FNUZ
!c_ty = f32
#k_width = 8
#non_k_dim = 16
#mfmaVersion = 3
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_16x16x32_fp8_bf8
  tt.func @convert_dot_mfma_f32_16x16x32_fp8_bf8(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-128: rocdl.mfma.f32.16x16x32.fp8.bf8
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = f8E5M2FNUZ
!b_ty = f8E4M3FNUZ
!c_ty = f32
#k_width = 8
#non_k_dim = 16
#mfmaVersion = 3
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_16x16x32_bf8_fp8
  tt.func @convert_dot_mfma_f32_16x16x32_bf8_fp8(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-128: rocdl.mfma.f32.16x16x32.bf8.fp8
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = f8E5M2FNUZ
!b_ty = f8E5M2FNUZ
!c_ty = f32
#k_width = 8
#non_k_dim = 16
#mfmaVersion = 3
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_16x16x32_bf8_bf8
  tt.func @convert_dot_mfma_f32_16x16x32_bf8_bf8(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-128: rocdl.mfma.f32.16x16x32.bf8.bf8
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = f16
!b_ty = f16
!c_ty = f32
#k_width = 4
#non_k_dim = 16
#mfmaVersion = 2
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_16x16x16f16
  tt.func @convert_dot_mfma_f32_16x16x16f16(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-256: rocdl.mfma.f32.16x16x16f16
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = bf16
!b_ty = bf16
!c_ty = f32
#k_width = 2
#non_k_dim = 16
#mfmaVersion = 1
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_16x16x8bf16
  tt.func @convert_dot_mfma_f32_16x16x8bf16(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-512: rocdl.mfma.f32.16x16x8bf16
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = bf16
!b_ty = bf16
!c_ty = f32
#k_width = 4
#non_k_dim = 16
#mfmaVersion = 2
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_16x16x16bf16_1k
  tt.func @convert_dot_mfma_f32_16x16x16bf16_1k(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-256: rocdl.mfma.f32.16x16x16bf16.1k
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = f32
!b_ty = f32
!c_ty = f32
#k_width = 1
#non_k_dim = 16
#mfmaVersion = 2
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_16x16x4f32
  tt.func @convert_dot_mfma_f32_16x16x4f32(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-1024: rocdl.mfma.f32.16x16x4f32
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = i8
!b_ty = i8
!c_ty = i32
#k_width = 4
#non_k_dim = 16
#mfmaVersion = 2
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_i32_16x16x16i8
  tt.func @convert_dot_mfma_i32_16x16x16i8(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-256: rocdl.mfma.i32.16x16x16i8
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = i8
!b_ty = i8
!c_ty = i32
#k_width = 8
#non_k_dim = 16
#mfmaVersion = 3
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_i32_16x16x32_i8
  tt.func @convert_dot_mfma_i32_16x16x32_i8(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-128: rocdl.mfma.i32.16x16x32.i8
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = f16
!b_ty = f16
!c_ty = f32
#k_width = 4
#non_k_dim = 4
#mfmaVersion = 2
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_4x4x4f16
  tt.func @convert_dot_mfma_f32_4x4x4f16(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-1024: rocdl.mfma.f32.4x4x4f16
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = bf16
!b_ty = bf16
!c_ty = f32
#k_width = 2
#non_k_dim = 4
#mfmaVersion = 1
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_4x4x2bf16
  tt.func @convert_dot_mfma_f32_4x4x2bf16(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-2048: rocdl.mfma.f32.4x4x2bf16
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = bf16
!b_ty = bf16
!c_ty = f32
#k_width = 4
#non_k_dim = 4
#mfmaVersion = 2
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_4x4x4bf16_1k
  tt.func @convert_dot_mfma_f32_4x4x4bf16_1k(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-1024: rocdl.mfma.f32.4x4x4bf16.1k
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = f32
!b_ty = f32
!c_ty = f32
#k_width = 1
#non_k_dim = 4
#mfmaVersion = 2
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_f32_4x4x1f32
  tt.func @convert_dot_mfma_f32_4x4x1f32(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0.000000e+00> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-4096: rocdl.mfma.f32.4x4x1f32
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

// -----

!a_ty = i8
!b_ty = i8
!c_ty = i32
#k_width = 4
#non_k_dim = 4
#mfmaVersion = 2
#mfma = #triton_gpu.mfma<{versionMajor = #mfmaVersion, warpsPerCTA=[1,1], instrShape = [#non_k_dim, #non_k_dim], isTranspose=false}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mfma, kWidth = #k_width}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mfma, kWidth = #k_width}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot_mfma_i32_4x4x4i8
  tt.func @convert_dot_mfma_i32_4x4x4i8(%a: tensor<128x256x!a_ty, #dot_operand_a>, %b: tensor<256x32x!b_ty, #dot_operand_b>) {
    %cst_c = arith.constant dense<0> : tensor<128x32x!c_ty, #mfma>
    // GCN-COUNT-1024: rocdl.mfma.i32.4x4x4i8
    %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x256x!a_ty, #dot_operand_a> * tensor<256x32x!b_ty, #dot_operand_b> -> tensor<128x32x!c_ty, #mfma>
    tt.return
  }
}

