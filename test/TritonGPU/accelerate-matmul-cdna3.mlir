// RUN: (! triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul=arch-generation-name=gfx940 --mlir-pass-pipeline-crash-reproducer=%t 2>/dev/null) | FileCheck --check-prefixes=CHECK %s

!a_ty = f16
!b_ty = f16
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
// CHECK: convert_dot_32_32_32_f16_f16_f32
    tt.func @convert_dot_32_32_32_f16_f16_f32(%a: tensor<32x32x!a_ty, #dot_operand_a>, %b: tensor<32x32x!b_ty, #dot_operand_b>) -> tensor<32x32x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<32x32x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<32x32x!a_ty, #dot_operand_a> * tensor<32x32x!b_ty, #dot_operand_b> -> tensor<32x32x!c_ty, #blocked>
        tt.return %D: tensor<32x32x!c_ty, #blocked>
    }
}

// -----

!a_ty = bf16
!b_ty = bf16
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
// CHECK: convert_dot_32_32_32_bf16_bf16_f32
    tt.func @convert_dot_32_32_32_bf16_bf16_f32(%a: tensor<32x32x!a_ty, #dot_operand_a>, %b: tensor<32x32x!b_ty, #dot_operand_b>) -> tensor<32x32x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<32x32x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<32x32x!a_ty, #dot_operand_a> * tensor<32x32x!b_ty, #dot_operand_b> -> tensor<32x32x!c_ty, #blocked>
        tt.return %D: tensor<32x32x!c_ty, #blocked>
    }
}

// -----

!a_ty = f32
!b_ty = f32
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
// CHECK: convert_dot_32_32_32_f32_f32_f32
    tt.func @convert_dot_32_32_32_f32_f32_f32(%a: tensor<32x32x!a_ty, #dot_operand_a>, %b: tensor<32x32x!b_ty, #dot_operand_b>) -> tensor<32x32x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<32x32x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 1}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 1}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<32x32x!a_ty, #dot_operand_a> * tensor<32x32x!b_ty, #dot_operand_b> -> tensor<32x32x!c_ty, #blocked>
        tt.return %D: tensor<32x32x!c_ty, #blocked>
    }
}

// -----

!a_ty = i8
!b_ty = i8
!c_ty = i32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
// CHECK: convert_dot_32_32_32_i8_i8_i32
    tt.func @convert_dot_32_32_32_i8_i8_i32(%a: tensor<32x32x!a_ty, #dot_operand_a>, %b: tensor<32x32x!b_ty, #dot_operand_b>) -> tensor<32x32x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0> : tensor<32x32x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<32x32x!a_ty, #dot_operand_a> * tensor<32x32x!b_ty, #dot_operand_b> -> tensor<32x32x!c_ty, #blocked>
        tt.return %D: tensor<32x32x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E4M3FNUZ
!b_ty = f8E4M3FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
// CHECK: convert_dot_32_32_32_f8E4M3FNUZ_f8E4M3FNUZ_f32
    tt.func @convert_dot_32_32_32_f8E4M3FNUZ_f8E4M3FNUZ_f32(%a: tensor<32x32x!a_ty, #dot_operand_a>, %b: tensor<32x32x!b_ty, #dot_operand_b>) -> tensor<32x32x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<32x32x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<32x32x!a_ty, #dot_operand_a> * tensor<32x32x!b_ty, #dot_operand_b> -> tensor<32x32x!c_ty, #blocked>
        tt.return %D: tensor<32x32x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E4M3FNUZ
!b_ty = f8E5M2FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
// CHECK: convert_dot_32_32_32_f8E4M3FNUZ_f8E5M2FNUZ_f32
    tt.func @convert_dot_32_32_32_f8E4M3FNUZ_f8E5M2FNUZ_f32(%a: tensor<32x32x!a_ty, #dot_operand_a>, %b: tensor<32x32x!b_ty, #dot_operand_b>) -> tensor<32x32x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<32x32x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<32x32x!a_ty, #dot_operand_a> * tensor<32x32x!b_ty, #dot_operand_b> -> tensor<32x32x!c_ty, #blocked>
        tt.return %D: tensor<32x32x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E5M2FNUZ
!b_ty = f8E4M3FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
// CHECK: convert_dot_32_32_32_f8E5M2FNUZ_f8E4M3FNUZ_f32
    tt.func @convert_dot_32_32_32_f8E5M2FNUZ_f8E4M3FNUZ_f32(%a: tensor<32x32x!a_ty, #dot_operand_a>, %b: tensor<32x32x!b_ty, #dot_operand_b>) -> tensor<32x32x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<32x32x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<32x32x!a_ty, #dot_operand_a> * tensor<32x32x!b_ty, #dot_operand_b> -> tensor<32x32x!c_ty, #blocked>
        tt.return %D: tensor<32x32x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E5M2FNUZ
!b_ty = f8E5M2FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
// CHECK: convert_dot_32_32_32_f8E5M2FNUZ_f8E5M2FNUZ_f32
    tt.func @convert_dot_32_32_32_f8E5M2FNUZ_f8E5M2FNUZ_f32(%a: tensor<32x32x!a_ty, #dot_operand_a>, %b: tensor<32x32x!b_ty, #dot_operand_b>) -> tensor<32x32x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<32x32x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<32x32x!a_ty, #dot_operand_a> * tensor<32x32x!b_ty, #dot_operand_b> -> tensor<32x32x!c_ty, #blocked>
        tt.return %D: tensor<32x32x!c_ty, #blocked>
    }
}

// -----

!a_ty = f16
!b_ty = f16
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = false}>
// CHECK: convert_dot_16_16_32_f16_f16_f32
    tt.func @convert_dot_16_16_32_f16_f16_f32(%a: tensor<16x32x!a_ty, #dot_operand_a>, %b: tensor<32x16x!b_ty, #dot_operand_b>) -> tensor<16x16x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<16x16x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<16x32x!a_ty, #dot_operand_a> * tensor<32x16x!b_ty, #dot_operand_b> -> tensor<16x16x!c_ty, #blocked>
        tt.return %D: tensor<16x16x!c_ty, #blocked>
    }
}

// -----

!a_ty = bf16
!b_ty = bf16
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = false}>
// CHECK: convert_dot_16_16_32_bf16_bf16_f32
    tt.func @convert_dot_16_16_32_bf16_bf16_f32(%a: tensor<16x32x!a_ty, #dot_operand_a>, %b: tensor<32x16x!b_ty, #dot_operand_b>) -> tensor<16x16x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<16x16x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<16x32x!a_ty, #dot_operand_a> * tensor<32x16x!b_ty, #dot_operand_b> -> tensor<16x16x!c_ty, #blocked>
        tt.return %D: tensor<16x16x!c_ty, #blocked>
    }
}

// -----

!a_ty = f32
!b_ty = f32
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = false}>
// CHECK: convert_dot_16_16_32_f32_f32_f32
    tt.func @convert_dot_16_16_32_f32_f32_f32(%a: tensor<16x32x!a_ty, #dot_operand_a>, %b: tensor<32x16x!b_ty, #dot_operand_b>) -> tensor<16x16x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<16x16x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 1}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 1}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<16x32x!a_ty, #dot_operand_a> * tensor<32x16x!b_ty, #dot_operand_b> -> tensor<16x16x!c_ty, #blocked>
        tt.return %D: tensor<16x16x!c_ty, #blocked>
    }
}

// -----

!a_ty = i8
!b_ty = i8
!c_ty = i32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = false}>
// CHECK: convert_dot_16_16_32_i8_i8_i32
    tt.func @convert_dot_16_16_32_i8_i8_i32(%a: tensor<16x32x!a_ty, #dot_operand_a>, %b: tensor<32x16x!b_ty, #dot_operand_b>) -> tensor<16x16x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0> : tensor<16x16x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<16x32x!a_ty, #dot_operand_a> * tensor<32x16x!b_ty, #dot_operand_b> -> tensor<16x16x!c_ty, #blocked>
        tt.return %D: tensor<16x16x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E4M3FNUZ
!b_ty = f8E4M3FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = false}>
// CHECK: convert_dot_16_16_32_f8E4M3FNUZ_f8E4M3FNUZ_f32
    tt.func @convert_dot_16_16_32_f8E4M3FNUZ_f8E4M3FNUZ_f32(%a: tensor<16x32x!a_ty, #dot_operand_a>, %b: tensor<32x16x!b_ty, #dot_operand_b>) -> tensor<16x16x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<16x16x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<16x32x!a_ty, #dot_operand_a> * tensor<32x16x!b_ty, #dot_operand_b> -> tensor<16x16x!c_ty, #blocked>
        tt.return %D: tensor<16x16x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E4M3FNUZ
!b_ty = f8E5M2FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = false}>
// CHECK: convert_dot_16_16_32_f8E4M3FNUZ_f8E5M2FNUZ_f32
    tt.func @convert_dot_16_16_32_f8E4M3FNUZ_f8E5M2FNUZ_f32(%a: tensor<16x32x!a_ty, #dot_operand_a>, %b: tensor<32x16x!b_ty, #dot_operand_b>) -> tensor<16x16x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<16x16x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<16x32x!a_ty, #dot_operand_a> * tensor<32x16x!b_ty, #dot_operand_b> -> tensor<16x16x!c_ty, #blocked>
        tt.return %D: tensor<16x16x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E5M2FNUZ
!b_ty = f8E4M3FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = false}>
// CHECK: convert_dot_16_16_32_f8E5M2FNUZ_f8E4M3FNUZ_f32
    tt.func @convert_dot_16_16_32_f8E5M2FNUZ_f8E4M3FNUZ_f32(%a: tensor<16x32x!a_ty, #dot_operand_a>, %b: tensor<32x16x!b_ty, #dot_operand_b>) -> tensor<16x16x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<16x16x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<16x32x!a_ty, #dot_operand_a> * tensor<32x16x!b_ty, #dot_operand_b> -> tensor<16x16x!c_ty, #blocked>
        tt.return %D: tensor<16x16x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E5M2FNUZ
!b_ty = f8E5M2FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = false}>
// CHECK: convert_dot_16_16_32_f8E5M2FNUZ_f8E5M2FNUZ_f32
    tt.func @convert_dot_16_16_32_f8E5M2FNUZ_f8E5M2FNUZ_f32(%a: tensor<16x32x!a_ty, #dot_operand_a>, %b: tensor<32x16x!b_ty, #dot_operand_b>) -> tensor<16x16x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<16x16x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<16x32x!a_ty, #dot_operand_a> * tensor<32x16x!b_ty, #dot_operand_b> -> tensor<16x16x!c_ty, #blocked>
        tt.return %D: tensor<16x16x!c_ty, #blocked>
    }
}

// -----

!a_ty = f16
!b_ty = f16
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [4, 4], isTransposed = false}>
// CHECK: convert_dot_4_4_64_f16_f16_f32
    tt.func @convert_dot_4_4_64_f16_f16_f32(%a: tensor<4x64x!a_ty, #dot_operand_a>, %b: tensor<64x4x!b_ty, #dot_operand_b>) -> tensor<4x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<4x4x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x64x!a_ty, #dot_operand_a> * tensor<64x4x!b_ty, #dot_operand_b> -> tensor<4x4x!c_ty, #blocked>
        tt.return %D: tensor<4x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = bf16
!b_ty = bf16
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [4, 4], isTransposed = false}>
// CHECK: convert_dot_4_4_64_bf16_bf16_f32
    tt.func @convert_dot_4_4_64_bf16_bf16_f32(%a: tensor<4x64x!a_ty, #dot_operand_a>, %b: tensor<64x4x!b_ty, #dot_operand_b>) -> tensor<4x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<4x4x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x64x!a_ty, #dot_operand_a> * tensor<64x4x!b_ty, #dot_operand_b> -> tensor<4x4x!c_ty, #blocked>
        tt.return %D: tensor<4x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = f32
!b_ty = f32
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [4, 4], isTransposed = false}>
// CHECK: convert_dot_4_4_64_f32_f32_f32
    tt.func @convert_dot_4_4_64_f32_f32_f32(%a: tensor<4x64x!a_ty, #dot_operand_a>, %b: tensor<64x4x!b_ty, #dot_operand_b>) -> tensor<4x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<4x4x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 1}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 1}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x64x!a_ty, #dot_operand_a> * tensor<64x4x!b_ty, #dot_operand_b> -> tensor<4x4x!c_ty, #blocked>
        tt.return %D: tensor<4x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = i8
!b_ty = i8
!c_ty = i32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [4, 4], isTransposed = false}>
// CHECK: convert_dot_4_4_64_i8_i8_i32
    tt.func @convert_dot_4_4_64_i8_i8_i32(%a: tensor<4x64x!a_ty, #dot_operand_a>, %b: tensor<64x4x!b_ty, #dot_operand_b>) -> tensor<4x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0> : tensor<4x4x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x64x!a_ty, #dot_operand_a> * tensor<64x4x!b_ty, #dot_operand_b> -> tensor<4x4x!c_ty, #blocked>
        tt.return %D: tensor<4x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E4M3FNUZ
!b_ty = f8E4M3FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {

// CHECK-NOT: convert_dot_4_4_64_f8E4M3FNUZ_f8E4M3FNUZ_f32
    tt.func @convert_dot_4_4_64_f8E4M3FNUZ_f8E4M3FNUZ_f32(%a: tensor<4x64x!a_ty, #dot_operand_a>, %b: tensor<64x4x!b_ty, #dot_operand_b>) -> tensor<4x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<4x4x!c_ty, #blocked>

        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x64x!a_ty, #dot_operand_a> * tensor<64x4x!b_ty, #dot_operand_b> -> tensor<4x4x!c_ty, #blocked>
        tt.return %D: tensor<4x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E4M3FNUZ
!b_ty = f8E5M2FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {

// CHECK-NOT: convert_dot_4_4_64_f8E4M3FNUZ_f8E5M2FNUZ_f32
    tt.func @convert_dot_4_4_64_f8E4M3FNUZ_f8E5M2FNUZ_f32(%a: tensor<4x64x!a_ty, #dot_operand_a>, %b: tensor<64x4x!b_ty, #dot_operand_b>) -> tensor<4x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<4x4x!c_ty, #blocked>

        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x64x!a_ty, #dot_operand_a> * tensor<64x4x!b_ty, #dot_operand_b> -> tensor<4x4x!c_ty, #blocked>
        tt.return %D: tensor<4x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E5M2FNUZ
!b_ty = f8E4M3FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {

// CHECK-NOT: convert_dot_4_4_64_f8E5M2FNUZ_f8E4M3FNUZ_f32
    tt.func @convert_dot_4_4_64_f8E5M2FNUZ_f8E4M3FNUZ_f32(%a: tensor<4x64x!a_ty, #dot_operand_a>, %b: tensor<64x4x!b_ty, #dot_operand_b>) -> tensor<4x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<4x4x!c_ty, #blocked>

        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x64x!a_ty, #dot_operand_a> * tensor<64x4x!b_ty, #dot_operand_b> -> tensor<4x4x!c_ty, #blocked>
        tt.return %D: tensor<4x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E5M2FNUZ
!b_ty = f8E5M2FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {

// CHECK-NOT: convert_dot_4_4_64_f8E5M2FNUZ_f8E5M2FNUZ_f32
    tt.func @convert_dot_4_4_64_f8E5M2FNUZ_f8E5M2FNUZ_f32(%a: tensor<4x64x!a_ty, #dot_operand_a>, %b: tensor<64x4x!b_ty, #dot_operand_b>) -> tensor<4x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<4x4x!c_ty, #blocked>

        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x64x!a_ty, #dot_operand_a> * tensor<64x4x!b_ty, #dot_operand_b> -> tensor<4x4x!c_ty, #blocked>
        tt.return %D: tensor<4x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = f16
!b_ty = f16
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [64, 4], isTransposed = false}>
// CHECK: convert_dot_64_4_4_f16_f16_f32
    tt.func @convert_dot_64_4_4_f16_f16_f32(%a: tensor<64x4x!a_ty, #dot_operand_a>, %b: tensor<4x4x!b_ty, #dot_operand_b>) -> tensor<64x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<64x4x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<64x4x!a_ty, #dot_operand_a> * tensor<4x4x!b_ty, #dot_operand_b> -> tensor<64x4x!c_ty, #blocked>
        tt.return %D: tensor<64x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = bf16
!b_ty = bf16
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [64, 4], isTransposed = false}>
// CHECK: convert_dot_64_4_4_bf16_bf16_f32
    tt.func @convert_dot_64_4_4_bf16_bf16_f32(%a: tensor<64x4x!a_ty, #dot_operand_a>, %b: tensor<4x4x!b_ty, #dot_operand_b>) -> tensor<64x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<64x4x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<64x4x!a_ty, #dot_operand_a> * tensor<4x4x!b_ty, #dot_operand_b> -> tensor<64x4x!c_ty, #blocked>
        tt.return %D: tensor<64x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = f32
!b_ty = f32
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [64, 4], isTransposed = false}>
// CHECK: convert_dot_64_4_4_f32_f32_f32
    tt.func @convert_dot_64_4_4_f32_f32_f32(%a: tensor<64x4x!a_ty, #dot_operand_a>, %b: tensor<4x4x!b_ty, #dot_operand_b>) -> tensor<64x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<64x4x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 1}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 1}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<64x4x!a_ty, #dot_operand_a> * tensor<4x4x!b_ty, #dot_operand_b> -> tensor<64x4x!c_ty, #blocked>
        tt.return %D: tensor<64x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = i8
!b_ty = i8
!c_ty = i32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [64, 4], isTransposed = false}>
// CHECK: convert_dot_64_4_4_i8_i8_i32
    tt.func @convert_dot_64_4_4_i8_i8_i32(%a: tensor<64x4x!a_ty, #dot_operand_a>, %b: tensor<4x4x!b_ty, #dot_operand_b>) -> tensor<64x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0> : tensor<64x4x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<64x4x!a_ty, #dot_operand_a> * tensor<4x4x!b_ty, #dot_operand_b> -> tensor<64x4x!c_ty, #blocked>
        tt.return %D: tensor<64x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E4M3FNUZ
!b_ty = f8E4M3FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {

// CHECK-NOT: convert_dot_64_4_4_f8E4M3FNUZ_f8E4M3FNUZ_f32
    tt.func @convert_dot_64_4_4_f8E4M3FNUZ_f8E4M3FNUZ_f32(%a: tensor<64x4x!a_ty, #dot_operand_a>, %b: tensor<4x4x!b_ty, #dot_operand_b>) -> tensor<64x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<64x4x!c_ty, #blocked>

        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<64x4x!a_ty, #dot_operand_a> * tensor<4x4x!b_ty, #dot_operand_b> -> tensor<64x4x!c_ty, #blocked>
        tt.return %D: tensor<64x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E4M3FNUZ
!b_ty = f8E5M2FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {

// CHECK-NOT: convert_dot_64_4_4_f8E4M3FNUZ_f8E5M2FNUZ_f32
    tt.func @convert_dot_64_4_4_f8E4M3FNUZ_f8E5M2FNUZ_f32(%a: tensor<64x4x!a_ty, #dot_operand_a>, %b: tensor<4x4x!b_ty, #dot_operand_b>) -> tensor<64x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<64x4x!c_ty, #blocked>

        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<64x4x!a_ty, #dot_operand_a> * tensor<4x4x!b_ty, #dot_operand_b> -> tensor<64x4x!c_ty, #blocked>
        tt.return %D: tensor<64x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E5M2FNUZ
!b_ty = f8E4M3FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {

// CHECK-NOT: convert_dot_64_4_4_f8E5M2FNUZ_f8E4M3FNUZ_f32
    tt.func @convert_dot_64_4_4_f8E5M2FNUZ_f8E4M3FNUZ_f32(%a: tensor<64x4x!a_ty, #dot_operand_a>, %b: tensor<4x4x!b_ty, #dot_operand_b>) -> tensor<64x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<64x4x!c_ty, #blocked>

        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<64x4x!a_ty, #dot_operand_a> * tensor<4x4x!b_ty, #dot_operand_b> -> tensor<64x4x!c_ty, #blocked>
        tt.return %D: tensor<64x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E5M2FNUZ
!b_ty = f8E5M2FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {

// CHECK-NOT: convert_dot_64_4_4_f8E5M2FNUZ_f8E5M2FNUZ_f32
    tt.func @convert_dot_64_4_4_f8E5M2FNUZ_f8E5M2FNUZ_f32(%a: tensor<64x4x!a_ty, #dot_operand_a>, %b: tensor<4x4x!b_ty, #dot_operand_b>) -> tensor<64x4x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<64x4x!c_ty, #blocked>

        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<64x4x!a_ty, #dot_operand_a> * tensor<4x4x!b_ty, #dot_operand_b> -> tensor<64x4x!c_ty, #blocked>
        tt.return %D: tensor<64x4x!c_ty, #blocked>
    }
}

// -----

!a_ty = f16
!b_ty = f16
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [4, 64], isTransposed = false}>
// CHECK: convert_dot_4_64_4_f16_f16_f32
    tt.func @convert_dot_4_64_4_f16_f16_f32(%a: tensor<4x4x!a_ty, #dot_operand_a>, %b: tensor<4x64x!b_ty, #dot_operand_b>) -> tensor<4x64x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<4x64x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x4x!a_ty, #dot_operand_a> * tensor<4x64x!b_ty, #dot_operand_b> -> tensor<4x64x!c_ty, #blocked>
        tt.return %D: tensor<4x64x!c_ty, #blocked>
    }
}

// -----

!a_ty = bf16
!b_ty = bf16
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [4, 64], isTransposed = false}>
// CHECK: convert_dot_4_64_4_bf16_bf16_f32
    tt.func @convert_dot_4_64_4_bf16_bf16_f32(%a: tensor<4x4x!a_ty, #dot_operand_a>, %b: tensor<4x64x!b_ty, #dot_operand_b>) -> tensor<4x64x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<4x64x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x4x!a_ty, #dot_operand_a> * tensor<4x64x!b_ty, #dot_operand_b> -> tensor<4x64x!c_ty, #blocked>
        tt.return %D: tensor<4x64x!c_ty, #blocked>
    }
}

// -----

!a_ty = f32
!b_ty = f32
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [4, 64], isTransposed = false}>
// CHECK: convert_dot_4_64_4_f32_f32_f32
    tt.func @convert_dot_4_64_4_f32_f32_f32(%a: tensor<4x4x!a_ty, #dot_operand_a>, %b: tensor<4x64x!b_ty, #dot_operand_b>) -> tensor<4x64x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<4x64x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 1}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 1}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x4x!a_ty, #dot_operand_a> * tensor<4x64x!b_ty, #dot_operand_b> -> tensor<4x64x!c_ty, #blocked>
        tt.return %D: tensor<4x64x!c_ty, #blocked>
    }
}

// -----

!a_ty = i8
!b_ty = i8
!c_ty = i32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
// CHECK: #mfma = #triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [4, 64], isTransposed = false}>
// CHECK: convert_dot_4_64_4_i8_i8_i32
    tt.func @convert_dot_4_64_4_i8_i8_i32(%a: tensor<4x4x!a_ty, #dot_operand_a>, %b: tensor<4x64x!b_ty, #dot_operand_b>) -> tensor<4x64x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0> : tensor<4x64x!c_ty, #blocked>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mfma>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
// CHECK: triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<{{.*}}, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x4x!a_ty, #dot_operand_a> * tensor<4x64x!b_ty, #dot_operand_b> -> tensor<4x64x!c_ty, #blocked>
        tt.return %D: tensor<4x64x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E4M3FNUZ
!b_ty = f8E4M3FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {

// CHECK-NOT: convert_dot_4_64_4_f8E4M3FNUZ_f8E4M3FNUZ_f32
    tt.func @convert_dot_4_64_4_f8E4M3FNUZ_f8E4M3FNUZ_f32(%a: tensor<4x4x!a_ty, #dot_operand_a>, %b: tensor<4x64x!b_ty, #dot_operand_b>) -> tensor<4x64x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<4x64x!c_ty, #blocked>

        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x4x!a_ty, #dot_operand_a> * tensor<4x64x!b_ty, #dot_operand_b> -> tensor<4x64x!c_ty, #blocked>
        tt.return %D: tensor<4x64x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E4M3FNUZ
!b_ty = f8E5M2FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {

// CHECK-NOT: convert_dot_4_64_4_f8E4M3FNUZ_f8E5M2FNUZ_f32
    tt.func @convert_dot_4_64_4_f8E4M3FNUZ_f8E5M2FNUZ_f32(%a: tensor<4x4x!a_ty, #dot_operand_a>, %b: tensor<4x64x!b_ty, #dot_operand_b>) -> tensor<4x64x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<4x64x!c_ty, #blocked>

        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x4x!a_ty, #dot_operand_a> * tensor<4x64x!b_ty, #dot_operand_b> -> tensor<4x64x!c_ty, #blocked>
        tt.return %D: tensor<4x64x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E5M2FNUZ
!b_ty = f8E4M3FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {

// CHECK-NOT: convert_dot_4_64_4_f8E5M2FNUZ_f8E4M3FNUZ_f32
    tt.func @convert_dot_4_64_4_f8E5M2FNUZ_f8E4M3FNUZ_f32(%a: tensor<4x4x!a_ty, #dot_operand_a>, %b: tensor<4x64x!b_ty, #dot_operand_b>) -> tensor<4x64x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<4x64x!c_ty, #blocked>

        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x4x!a_ty, #dot_operand_a> * tensor<4x64x!b_ty, #dot_operand_b> -> tensor<4x64x!c_ty, #blocked>
        tt.return %D: tensor<4x64x!c_ty, #blocked>
    }
}

// -----

!a_ty = f8E5M2FNUZ
!b_ty = f8E5M2FNUZ
!c_ty = f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {

// CHECK-NOT: convert_dot_4_64_4_f8E5M2FNUZ_f8E5M2FNUZ_f32
    tt.func @convert_dot_4_64_4_f8E5M2FNUZ_f8E5M2FNUZ_f32(%a: tensor<4x4x!a_ty, #dot_operand_a>, %b: tensor<4x64x!b_ty, #dot_operand_b>) -> tensor<4x64x!c_ty, #blocked> {
        %cst_c = arith.constant dense<0.000000e+00> : tensor<4x64x!c_ty, #blocked>

        %D = tt.dot %a, %b, %cst_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<4x4x!a_ty, #dot_operand_a> * tensor<4x64x!b_ty, #dot_operand_b> -> tensor<4x64x!c_ty, #blocked>
        tt.return %D: tensor<4x64x!c_ty, #blocked>
    }
}

