// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm | FileCheck --check-prefixes=CHECK,GCN %s

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 1, perPhase=1, maxPhase=1, order = [1, 0]}>
#mma0 = #triton_gpu.mma<{versionMajor=3, warpsPerCTA=[1,1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot
  func.func @convert_dot(%A: tensor<32x32xf16, #blocked0>, %B: tensor<32x32xf16, #blocked0>) {
    %AA = triton_gpu.convert_layout %A : (tensor<32x32xf16, #blocked0>) -> tensor<32x32xf16, #shared0>
    %BB = triton_gpu.convert_layout %B : (tensor<32x32xf16, #blocked0>) -> tensor<32x32xf16, #shared0>
    // GCN-COUNT-32:  llvm.load {{.*}} : !llvm.ptr<f16, 3>
    %AA_DOT = triton_gpu.convert_layout %AA : (tensor<32x32xf16, #shared0>) -> tensor<32x32xf16, #dot_operand_a>
    %BB_DOT = triton_gpu.convert_layout %BB : (tensor<32x32xf16, #shared0>) -> tensor<32x32xf16, #dot_operand_b>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma0>

    // GCN-COUNT-4: rocdl.mfma.f32.32x32x8f16
    %D = tt.dot %AA_DOT, %BB_DOT, %cst0 {allowTF32 = true, transA = false, transB = false} : tensor<32x32xf16, #dot_operand_a> * tensor<32x32xf16, #dot_operand_b> -> tensor<32x32xf32, #mma0>

    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [64, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #triton_gpu.mma<{versionMajor = 3, warpsPerCTA = [2, 2]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global external @global_smem() {addr_space = 3 : i32} : !llvm.array<0 x i8>
  // CHECK-LABEL: convert_layout_mmav3_block
  func.func @convert_layout_mmav3_blocked(%arg0: tensor<32x32xf32, #mma>) {
    // GCN-COUNT-16: llvm.store {{.*}} : !llvm.ptr<vector<1xf32>, 3>
    // GCN-NEXT: rocdl.barrier
    %0 = triton_gpu.convert_layout %arg0 : (tensor<32x32xf32, #mma>) -> tensor<32x32xf32, #blocked0>
    return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #triton_gpu.mma<{versionMajor = 3, warpsPerCTA = [2, 2]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: matmul_kernel_dot_operand_layout
  func.func @matmul_kernel_dot_operand_layout(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
    %a:tensor<128x32xf16, #shared>, %b:tensor<32x256xf16, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    // GCN-COUNT-96: llvm.load {{.*}} : !llvm.ptr<f16, 3>
    %a_mat = triton_gpu.convert_layout %a : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #dot_operand_a>
    %b_mat = triton_gpu.convert_layout %b : (tensor<32x256xf16, #shared>) -> tensor<32x256xf16, #dot_operand_b>
    
    %28 = tt.dot %a_mat, %b_mat, %cst {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #dot_operand_a> * tensor<32x256xf16, #dot_operand_b> -> tensor<128x256xf32, #mma>
    // GCN-COUNT-32: rocdl.mfma.f32.32x32x8f16
    %38 = triton_gpu.convert_layout %28 : (tensor<128x256xf32, #mma>) -> tensor<128x256xf32, #blocked>

    %30 = tt.splat %ptr : (!tt.ptr<f32>) -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : (tensor<128x1x!tt.ptr<f32>, #blocked>) -> tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<128x256xf32, #blocked>
    return
  }
}