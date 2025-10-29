// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s

#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape=[16, 16, 128]}>
#mma1 = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape=[16, 16, 64]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  //  CHECK-LABEL: wmma_scaled_dot_fp4
  tt.func @wmma_scaled_dot_fp4(%arg0: tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<32x4xi8, #linear>, %arg2: tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg3: tensor<32x4xi8, #linear1>, %out0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // Matrix A
    // CHECK-COUNT-32: llvm.extractvalue {{.*}} :  !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-32: llvm.insertelement {{.*}} : vector<32xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<32xi8> to vector<8xi32>
    // Matrix B
    // CHECK-COUNT-32: llvm.extractvalue {{.*}} :  !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-32: llvm.insertelement {{.*}} : vector<32xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<32xi8> to vector<8xi32>
    // Scale A
    // CHECK-COUNT-4: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8)>
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // Scale B
    // CHECK-COUNT-4: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8)>
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // Matrix C
    // CHECK-COUNT-8:  llvm.insertelement {{.*}} : vector<8xf32>
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4"{{.*}} : (i32, vector<8xi32>, i32, vector<8xi32>, i16, vector<8xf32>, i32, i32, i32, i32, i32, i32, i1, i1) -> vector<8xf32>
    %c = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, tensor<32x4xi8, #linear> * tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, tensor<32x4xi8, #linear1> -> tensor<32x32xf32, #mma>
    // CHECK-COUNT-8: llvm.extractelement {{.*}} : vector<8xf32>
    // CHECK-COUNT-8: llvm.insertelement {{.*}} : vector<1xf32>
    %ptr0 = tt.splat %out0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #mma>
    tt.store %ptr0, %c : tensor<32x32x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape=[16, 16, 128]}>
#mma1 = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape=[16, 16, 64]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_scaled_dot_fp4_fp8
  tt.func @wmma_scaled_dot_fp4_fp8(%arg0: tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<32x4xi8, #linear>, %arg2: tensor<128x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<32x4xi8, #linear1>, %out0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // Matrix A
    // CHECK-COUNT-32: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-32: llvm.insertelement {{.*}} : vector<32xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<32xi8> to vector<8xi32>
    // Matrix B
    // CHECK-COUNT-64: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,  i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-64: llvm.insertelement {{.*}} : vector<64xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<64xi8> to vector<16xi32>
    // Scale A
    // CHECK-COUNT-4: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8)>
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // Scale B
    // CHECK-COUNT-4: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8)>
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // Matrix C
    // CHECK-COUNT-8:  llvm.insertelement {{.*}} : vector<8xf32>
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4"{{.*}} : (i32, vector<16xi32>, i32, vector<8xi32>, i16, vector<8xf32>, i32, i32, i32, i32, i32, i32, i1, i1) -> vector<8xf32>
    %c = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e2m1 rhs = e4m3 {fastMath = false} : tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, tensor<32x4xi8, #linear> * tensor<128x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<32x4xi8, #linear1> -> tensor<32x32xf32, #mma>
    // CHECK-COUNT-8: llvm.extractelement {{.*}} : vector<8xf32>
    // CHECK-COUNT-8: llvm.insertelement {{.*}} : vector<1xf32>
    %ptr0 = tt.splat %out0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #mma>
    tt.store %ptr0, %c : tensor<32x32x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape=[16, 16, 128]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_scaled_dot_fp8
  tt.func @wmma_scaled_dot_fp8(%arg0: tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<32x4xi8, #linear>, %arg2: tensor<128x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<32x4xi8, #linear1>, %out0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // Matrix A
    // CHECK-COUNT-64: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,  i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-64: llvm.insertelement {{.*}} : vector<64xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<64xi8> to vector<16xi32>
    // Matrix B
    // CHECK-COUNT-64: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,  i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-64: llvm.insertelement {{.*}} : vector<64xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<64xi8> to vector<16xi32>
    // Scale A
    // CHECK-COUNT-4: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8)>
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // Scale B
    // CHECK-COUNT-4: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8)>
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // Matrix C
    // CHECK-COUNT-8:  llvm.insertelement {{.*}} : vector<8xf32>
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4"{{.*}} : (i32, vector<16xi32>, i32, vector<16xi32>, i16, vector<8xf32>, i32, i32, i32, i32, i32, i32, i1, i1) -> vector<8xf32>
    %c = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<32x4xi8, #linear> * tensor<128x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<32x4xi8, #linear1> -> tensor<32x32xf32, #mma>
    // CHECK-COUNT-8: llvm.extractelement {{.*}} : vector<8xf32>
    // CHECK-COUNT-8: llvm.insertelement {{.*}} : vector<1xf32>
    %ptr0 = tt.splat %out0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #mma>
    tt.store %ptr0, %c : tensor<32x32x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape=[16, 16, 128]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_scaled_dot_fp8_k64
  tt.func @wmma_scaled_dot_fp8_k64(%arg0: tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<32x2xi8, #linear>, %arg2: tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<32x2xi8, #linear1>, %out0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // Adjust for acc
    // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i8) : i8
    // Matrix A
    // CHECK-COUNT-32: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,  i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-32: llvm.insertelement {{.*}} : vector<64xi8>
    // CHECK-COUNT-32: llvm.insertelement %[[ZERO]], {{.*}} : vector<64xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<64xi8> to vector<16xi32>
    // Matrix B
    // CHECK-COUNT-32: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8,  i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-32: llvm.insertelement {{.*}} : vector<64xi8>
    // CHECK-COUNT-32: llvm.insertelement %[[ZERO]], {{.*}} : vector<64xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<64xi8> to vector<16xi32>
    // Scale A
    // CHECK-COUNT-4: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8)>
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // Scale B
    // CHECK-COUNT-4: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8)>
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // Matrix C
    // CHECK-COUNT-8:  llvm.insertelement {{.*}} : vector<8xf32>
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4"{{.*}} : (i32, vector<16xi32>, i32, vector<16xi32>, i16, vector<8xf32>, i32, i32, i32, i32, i32, i32, i1, i1) -> vector<8xf32>
    %c = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<32x2xi8, #linear> * tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<32x2xi8, #linear1> -> tensor<32x32xf32, #mma>
    // CHECK-COUNT-8: llvm.extractelement {{.*}} : vector<8xf32>
    // CHECK-COUNT-8: llvm.insertelement {{.*}} : vector<1xf32>
    %ptr0 = tt.splat %out0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #mma>
    tt.store %ptr0, %c : tensor<32x32x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape=[16, 16, 128]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_scaled_dot_fp8_repeat_k
  tt.func @wmma_scaled_dot_fp8_repeat_k(%arg0: tensor<32x256xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<32x8xi8, #linear>, %arg2: tensor<256x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<32x8xi8, #linear1>, %out0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // Matrix A
    // CHECK-COUNT-128: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK: llvm.bitcast {{.*}} : vector<64xi8> to vector<16xi32>
    // CHECK-COUNT-64: llvm.insertelement {{.*}} : vector<64xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<64xi8> to vector<16xi32>
    // Matrix B
    // CHECK-COUNT-128: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-64: llvm.insertelement {{.*}} : vector<64xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<64xi8> to vector<16xi32>
    // CHECK-COUNT-64: llvm.insertelement {{.*}} : vector<64xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<64xi8> to vector<16xi32>
    // Scale A
    // CHECK-COUNT-8: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // Scale B
    // CHECK-COUNT-8: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // Matrix C
    // CHECK-COUNT-8:  llvm.insertelement {{.*}} : vector<8xf32>
    // CHECK-COUNT-2: llvm.call_intrinsic "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4"{{.*}} : (i32, vector<16xi32>, i32, vector<16xi32>, i16, vector<8xf32>, i32, i32, i32, i32, i32, i32, i1, i1) -> vector<8xf32>
    %c = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x256xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<32x8xi8, #linear> * tensor<256x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<32x8xi8, #linear1> -> tensor<32x32xf32, #mma>
    // CHECK-COUNT-8: llvm.extractelement {{.*}} : vector<8xf32>
    // CHECK-COUNT-8: llvm.insertelement {{.*}} : vector<1xf32>
    %ptr0 = tt.splat %out0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #mma>
    tt.store %ptr0, %c : tensor<32x32x!tt.ptr<f32>, #mma>
    tt.return
  }
}
