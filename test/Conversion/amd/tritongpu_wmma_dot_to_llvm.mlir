// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck %s

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
#mma = #triton_gpu.amd_wmma<{warpsPerCTA = [2, 2]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  //  CHECK-LABEL: wmma_dot_operand
  tt.func @wmma_dot_operand(%arg0: !tt.memdesc<64x64xf16, #shared>) {
    // 2 CTA * 4 rep * load_per_thread_per_instr
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xf16>
    %0 = triton_gpu.local_load %arg0 : !tt.memdesc<64x64xf16, #shared> -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    // CHECK-COUNT-128: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<1xf16>
    %1 = triton_gpu.local_load %arg0 : !tt.memdesc<64x64xf16, #shared> -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: wmma_dot
  tt.func @wmma_dot(%arg0: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg2: tensor<16x16xf16, #mma>) {
    // CHECK-COUNT-32: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK: llvm.mlir.undef : vector<16xf16>
    // CHECK-COUNT-8: llvm.insertelement {{.*}} : vector<16xf16>
    // CHECK: rocdl.wmma.f16.16x16x16.f16 {{.*}} : (vector<16xf16>, vector<16xf16>, vector<16xf16>, i1) -> vector<16xf16>
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<16x16xf16, #mma>
    // CHECK-COUNT-8: llvm.extractelement {{.*}} : vector<16xf16>
    // CHECK: llvm.mlir.undef : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-COUNT-8: llvm.insertvalue {{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    tt.return
  }

  //  CHECK-LABEL: wmma_dot_int8_32
  tt.func @wmma_dot_int8_32(%arg0: tensor<16x16xui8, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<16x16xui8, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg2: tensor<16x16xi32, #mma>) {
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xi8>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xi8> to vector<4xi32>
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xi8>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xi8> to vector<4xi32>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK: rocdl.wmma.i32.16x16x16.iu8 {{.*}} : (i1, vector<4xi32>, i1, vector<4xi32>, vector<8xi32>, i1) -> vector<8xi32>
    %0 = tt.dot %arg0, %arg1, %arg2 {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xui8, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<16x16xui8, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<16x16xi32, #mma>
    // CHECK-COUNT-8: llvm.insertvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    tt.return
  }

  //  CHECK-LABEL: wmma_dot_int4_32
  tt.func @wmma_dot_int4_32(%arg0: tensor<16x16xui4, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<16x16xui4, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg2: tensor<16x16xi32, #mma>) {
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4)>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xi4>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xi4> to vector<2xi32>
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4)>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xi4>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xi4> to vector<2xi32>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK: rocdl.wmma.i32.16x16x16.iu4 {{.*}} : (i1, vector<2xi32>, i1, vector<2xi32>, vector<8xi32>, i1) -> vector<8xi32>
    %0 = tt.dot %arg0, %arg1, %arg2 {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xui4, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<16x16xui4, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<16x16xi32, #mma>
    // CHECK-COUNT-8: llvm.insertvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    tt.return
  }
}

// -----

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0], hasLeadingOffset = false}>
#mma = #triton_gpu.amd_wmma<{warpsPerCTA = [2, 1, 4]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_dot_operand3d
  tt.func @wmma_dot_operand3d(%arg0: !tt.memdesc<4x16x32xf16, #shared>) {
    // CHECK-COUNT-4: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xf16>
    %0 = triton_gpu.local_load %arg0 : !tt.memdesc<4x16x32xf16, #shared> -> tensor<4x16x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    // CHECK-COUNT-32: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<1xf16>
    %1 = triton_gpu.local_load %arg0 : !tt.memdesc<4x16x32xf16, #shared> -> tensor<4x16x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    tt.return
  }

  // CHECK-LABEL: wmma_dot3d
  tt.func @wmma_dot3d(%arg0: tensor<2x16x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>, %arg1: tensor<2x32x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>, %arg2: tensor<2x16x16xf16, #mma>) {
    // CHECK-COUNT-32: llvm.extractvalue %arg0
    // CHECK-COUNT-32: llvm.insertelement
    // CHECK-COUNT-32: llvm.extractvalue %arg1
    // CHECK-COUNT-32: llvm.insertelement
    // CHECK-COUNT-8: llvm.extractvalue %arg2
    // CHECK-COUNT-8: llvm.insertelement
    // CHECK-COUNT-2: rocdl.wmma.f16.16x16x16.f16 {{.*}} : (vector<16xf16>, vector<16xf16>, vector<16xf16>, i1) -> vector<16xf16>
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<2x16x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<2x32x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<2x16x16xf16, #mma>
    // CHECK-COUNT-8: llvm.extractelement
    // CHECK-COUNT-8: llvm.insertvalue
    tt.return
  }
}
