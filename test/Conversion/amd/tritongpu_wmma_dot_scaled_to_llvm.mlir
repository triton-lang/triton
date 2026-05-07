// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s

#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = true, instrShape=[16, 16, 128]}>
#mma1 = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = true, instrShape=[16, 16, 64]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  //  CHECK-LABEL: wmma_scaled_dot_fp4
  tt.func @wmma_scaled_dot_fp4(%arg0: tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<32x4xi8, #linear>, %arg2: tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg3: tensor<32x4xi8, #linear1>, %out0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // Matrix C
    // CHECK-COUNT-8:  llvm.insertelement {{.*}} : vector<8xf32>
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
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = true, instrShape=[16, 16, 128]}>
#mma1 = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = true, instrShape=[16, 16, 64]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_scaled_dot_fp4_fp8
  tt.func @wmma_scaled_dot_fp4_fp8(%arg0: tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<32x4xi8, #linear>, %arg2: tensor<128x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<32x4xi8, #linear1>, %out0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // Matrix C
    // CHECK-COUNT-8:  llvm.insertelement {{.*}} : vector<8xf32>
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
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = true, instrShape=[16, 16, 128]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_scaled_dot_fp8
  tt.func @wmma_scaled_dot_fp8(%arg0: tensor<32x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<32x4xi8, #linear>, %arg2: tensor<128x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<32x4xi8, #linear1>, %out0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // Matrix C
    // CHECK-COUNT-8:  llvm.insertelement {{.*}} : vector<8xf32>
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
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = true, instrShape=[16, 16, 128]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_scaled_dot_fp8_k64
  tt.func @wmma_scaled_dot_fp8_k64(%arg0: tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<32x2xi8, #linear>, %arg2: tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<32x2xi8, #linear1>, %out0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // Adjust for acc
    // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i8) : i8
    // Matrix C
    // CHECK-COUNT-8:  llvm.insertelement {{.*}} : vector<8xf32>
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
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = true, instrShape=[16, 16, 128]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_scaled_dot_fp8_repeat_k
  tt.func @wmma_scaled_dot_fp8_repeat_k(%arg0: tensor<32x256xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<32x8xi8, #linear>, %arg2: tensor<256x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<32x8xi8, #linear1>, %out0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // Matrix C
    // CHECK-COUNT-8:  llvm.insertelement {{.*}} : vector<8xf32>
    // Matrix A
    // CHECK-COUNT-64: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-64: llvm.insertelement {{.*}} : vector<64xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<64xi8> to vector<16xi32>
    // Matrix B
    // CHECK-COUNT-64: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-64: llvm.insertelement {{.*}} : vector<64xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<64xi8> to vector<16xi32>
    // Scale A
    // CHECK-COUNT-4: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // Scale B
    // CHECK-COUNT-4: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4"{{.*}} : (i32, vector<16xi32>, i32, vector<16xi32>, i16, vector<8xf32>, i32, i32, i32, i32, i32, i32, i1, i1) -> vector<8xf32>
    // Matrix A
    // CHECK-COUNT-64: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-64: llvm.insertelement {{.*}} : vector<64xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<64xi8> to vector<16xi32>
    // Matrix B
    // CHECK-COUNT-64: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-64: llvm.insertelement {{.*}} : vector<64xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<64xi8> to vector<16xi32>
    // Scale A
    // CHECK-COUNT-4: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // Scale B
    // CHECK-COUNT-4: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-4: llvm.insertelement {{.*}} : vector<4xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to i32
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4"{{.*}} : (i32, vector<16xi32>, i32, vector<16xi32>, i16, vector<8xf32>, i32, i32, i32, i32, i32, i32, i1, i1) -> vector<8xf32>
    %c = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x256xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<32x8xi8, #linear> * tensor<256x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<32x8xi8, #linear1> -> tensor<32x32xf32, #mma>
    // CHECK-COUNT-8: llvm.extractelement {{.*}} : vector<8xf32>
    // CHECK-COUNT-8: llvm.insertelement {{.*}} : vector<1xf32>
    %ptr0 = tt.splat %out0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #mma>
    tt.store %ptr0, %c : tensor<32x32x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [16, 0], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[1, 0], [2, 0]]}, isTranspose = true, instrShape=[16, 16, 128]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_scaled_dot_fp8_chained
  tt.func @wmma_scaled_dot_fp8_chained(%arg0: tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg2: tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg3: tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, %out0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %scale0 = arith.constant dense<127> :  tensor<128x4xi8, #linear>
    %scale1 = arith.constant dense<127> :  tensor<128x4xi8, #linear1>
    // CHECK-COUNT-16: llvm.call_intrinsic "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4"{{.*}} : (i32, vector<16xi32>, i32, vector<16xi32>, i16, vector<8xf32>, i32, i32, i32, i32, i32, i32, i1, i1) -> vector<8xf32>
    %mm0 = tt.dot_scaled %arg0 scale %scale0, %arg2 scale %scale1, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, tensor<128x4xi8, #linear> * tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, tensor<128x4xi8, #linear1> -> tensor<128x128xf32, #mma>
    // CHECK-NOT: rocdl.ds_swizzle
    // CHECK-NOT: llvm.call_intrinsic "llvm.amdgcn.permlane16.swap"
    %op0 = ttg.convert_layout %mm0 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %op1 = tt.fp_to_fp %op0, rounding = rtne : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> -> tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    // CHECK-COUNT-16: llvm.call_intrinsic "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4"{{.*}} : (i32, vector<16xi32>, i32, vector<16xi32>, i16, vector<8xf32>, i32, i32, i32, i32, i32, i32, i1, i1) -> vector<8xf32>
    %mm1 = tt.dot_scaled %op1 scale %scale0, %arg3 scale %scale1, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>, tensor<128x4xi8, #linear> * tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, tensor<128x4xi8, #linear1> -> tensor<128x128xf32, #mma>
    %ptr0 = tt.splat %out0 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #mma>
    tt.store %ptr0, %mm1 : tensor<128x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// Scale A: 32x16 WMMA — all 32 lanes cover M (depth=1), no broadcast
#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [32, 0]], block = []}>
// Scale B: 16 lanes cover N (depth=2), 1 broadcast bit
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = false, instrShape=[32, 16, 128]}>
#mma1 = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = false, instrShape=[32, 16, 64]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_scaled_dot_fp4_32x16
  tt.func @wmma_scaled_dot_fp4_32x16(%arg0: tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<64x4xi8, #linear>, %arg2: tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg3: tensor<32x4xi8, #linear1>, %out0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
    // Matrix C
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xf32>
    // Matrix A
    // CHECK-COUNT-64: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-64: llvm.insertelement {{.*}} : vector<64xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<64xi8> to vector<16xi32>
    // Matrix B
    // CHECK-COUNT-32: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
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
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.scale.f32.32x16x128.f4"{{.*}} : (vector<16xi32>, vector<8xi32>, i16, vector<16xf32>, i32, i32, i32, i32, i32, i32, i1, i1) -> vector<16xf32>
    %c = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, tensor<64x4xi8, #linear> * tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, tensor<32x4xi8, #linear1> -> tensor<64x32xf32, #mma>
    // CHECK-COUNT-16: llvm.extractelement {{.*}} : vector<16xf32>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<1xf32>
    %ptr0 = tt.splat %out0 : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #mma>
    tt.store %ptr0, %c : tensor<64x32x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// 32x16 WMMA with isTranspose=true
// Scale A 16 lanes cover M, 1 broadcast bit.
#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
// Scale B 32 lanes cover N, no broadcast.
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = true, instrShape=[32, 16, 128]}>
#mma1 = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = true, instrShape=[32, 16, 64]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_scaled_dot_fp4_32x16_transposed
  tt.func @wmma_scaled_dot_fp4_32x16_transposed(%arg0: tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<32x4xi8, #linear>, %arg2: tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg3: tensor<64x4xi8, #linear1>, %out0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
    // Matrix C
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xf32>
    // Matrix A (frontend A -> slot-2, kBase=32 i8 after swap; vector<32xi8>)
    // CHECK-COUNT-32: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-32: llvm.insertelement {{.*}} : vector<32xi8>
    // CHECK: llvm.bitcast {{.*}} : vector<32xi8> to vector<8xi32>
    // Matrix B (frontend B -> slot-1, kBase=64 i8 after swap; vector<64xi8>)
    // CHECK-COUNT-64: llvm.extractvalue {{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
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
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.scale.f32.32x16x128.f4"{{.*}} : (vector<16xi32>, vector<8xi32>, i16, vector<16xf32>, i32, i32, i32, i32, i32, i32, i1, i1) -> vector<16xf32>
    %c = tt.dot_scaled %arg0 scale %arg1, %arg2 scale %arg3, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, tensor<32x4xi8, #linear> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, tensor<64x4xi8, #linear1> -> tensor<32x64xf32, #mma>
    // CHECK-COUNT-16: llvm.extractelement {{.*}} : vector<16xf32>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<1xf32>
    %ptr0 = tt.splat %out0 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #mma>
    tt.store %ptr0, %c : tensor<32x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}
