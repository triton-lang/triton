// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx1100 | FileCheck %s

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
#mma1 = #triton_gpu.amd_wmma<{version = 1, warpsPerCTA = [2, 2]}>
#mma2 = #triton_gpu.amd_wmma<{version = 2, warpsPerCTA = [2, 2]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  //  CHECK-LABEL: wmma1_dot_operand
  tt.func @wmma1_dot_operand(%arg0: !triton_gpu.memdesc<64x64xf16, #shared>) {
    // 2 CTA * 4 rep * load_per_thread_per_instr
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xf16>
    %0 = triton_gpu.local_load %arg0 : !triton_gpu.memdesc<64x64xf16, #shared> -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>
    // CHECK-COUNT-128: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<1xf16>
    %1 = triton_gpu.local_load %arg0 : !triton_gpu.memdesc<64x64xf16, #shared> -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>
    tt.return
  }

  //  CHECK-LABEL: wmma2_dot_operand
  tt.func @wmma2_dot_operand(%arg0: !triton_gpu.memdesc<64x64xf16, #shared>) {
    // 2 CTA * 4 rep * load_per_thread_per_instr
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xf16>
    %0 = triton_gpu.local_load %arg0 : !triton_gpu.memdesc<64x64xf16, #shared> -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma2, kWidth = 8}>>
    // CHECK-COUNT-64: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<1xf16>
    %1 = triton_gpu.local_load %arg0 : !triton_gpu.memdesc<64x64xf16, #shared> -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma2, kWidth = 8}>>
    tt.return
  }

  //  CHECK-LABEL: wmma1_dot
  tt.func @wmma1_dot(%arg0: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg2: tensor<16x16xf16, #mma1>) {
    // CHECK-COUNT-32: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK: llvm.mlir.undef : vector<16xf16>
    // CHECK-COUNT-8: llvm.insertelement {{.*}} : vector<16xf16>
    // CHECK: rocdl.wmma.f16.16x16x16.f16 {{.*}} : (vector<16xf16>, vector<16xf16>, vector<16xf16>, i1) -> vector<16xf16>
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>> -> tensor<16x16xf16, #mma1>
    // CHECK-COUNT-8: llvm.extractelement {{.*}} : vector<16xf16>
    // CHECK: llvm.mlir.undef : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-COUNT-8: llvm.insertvalue {{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    tt.return
  }

  //  CHECK-LABEL: wmma1_dot_bf16
  tt.func @wmma1_dot_bf16(%arg0: tensor<16x16xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<16x16xbf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg2: tensor<16x16xbf16, #mma1>) {
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16)>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xbf16> to vector<16xi16>
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16)>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xbf16> to vector<16xi16>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16)>
    // CHECK: llvm.mlir.undef : vector<16xbf16>
    // CHECK-COUNT-8: llvm.insertelement {{.*}} : vector<16xbf16>
    // CHECK: rocdl.wmma.bf16.16x16x16.bf16 {{.*}} : (vector<16xi16>, vector<16xi16>, vector<16xbf16>, i1) -> vector<16xbf16>
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>> * tensor<16x16xbf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>> -> tensor<16x16xbf16, #mma1>
    tt.return
  }

  //  CHECK-LABEL: wmma1_dot_int8_32
  tt.func @wmma1_dot_int8_32(%arg0: tensor<16x16xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<16x16xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg2: tensor<16x16xi32, #mma1>) {
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xi8>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xi8> to vector<4xi32>
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xi8>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xi8> to vector<4xi32>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK: rocdl.wmma.i32.16x16x16.iu8 {{.*}} : (i1, vector<4xi32>, i1, vector<4xi32>, vector<8xi32>, i1) -> vector<8xi32>
    %0 = tt.dot %arg0, %arg1, %arg2 {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>> * tensor<16x16xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>> -> tensor<16x16xi32, #mma1>
    // CHECK-COUNT-8: llvm.insertvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    tt.return
  }

  //  CHECK-LABEL: wmma1_dot_int4_32
  tt.func @wmma1_dot_int4_32(%arg0: tensor<16x16xi4, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<16x16xi4, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg2: tensor<16x16xi32, #mma1>) {
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4)>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xi4>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xi4> to vector<2xi32>
    // CHECK-COUNT-16: llvm.extractvalue %{{.*}} : !llvm.struct<(i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4, i4)>
    // CHECK-COUNT-16: llvm.insertelement {{.*}} : vector<16xi4>
    // CHECK: llvm.bitcast %{{.*}} : vector<16xi4> to vector<2xi32>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK: rocdl.wmma.i32.16x16x16.iu4 {{.*}} : (i1, vector<2xi32>, i1, vector<2xi32>, vector<8xi32>, i1) -> vector<8xi32>
    %0 = tt.dot %arg0, %arg1, %arg2 {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xi4, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>> * tensor<16x16xi4, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>> -> tensor<16x16xi32, #mma1>
    // CHECK-COUNT-8: llvm.insertvalue {{.*}} : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    tt.return
  }

  //  CHECK-LABEL: wmma2_dot
  tt.func @wmma2_dot(%arg0: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma2, kWidth = 8}>>, %arg1: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma2, kWidth = 8}>>, %arg2: tensor<16x16xf16, #mma2>) {
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK: llvm.mlir.undef : vector<8xf16>
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.wmma.f16.16x16x16.f16.v8f16.v8f16"{{.*}} : (vector<8xf16>, vector<8xf16>, vector<8xf16>, i1) -> vector<8xf16>
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma2, kWidth = 8}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma2, kWidth = 8}>> -> tensor<16x16xf16, #mma2>
    // CHECK-COUNT-8: llvm.extractelement {{.*}} : vector<8xf16>
    // CHECK: llvm.mlir.undef : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-COUNT-8: llvm.insertvalue {{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    tt.return
  }
}

// -----

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0], hasLeadingOffset = false}>
#mma1 = #triton_gpu.amd_wmma<{version = 1, warpsPerCTA = [2, 1, 4]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_dot_operand3d
  tt.func @wmma_dot_operand3d(%arg0: !triton_gpu.memdesc<4x16x32xf16, #shared>) {
    // CHECK-COUNT-4: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xf16>
    %0 = triton_gpu.local_load %arg0 : !triton_gpu.memdesc<4x16x32xf16, #shared> -> tensor<4x16x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>
    // CHECK-COUNT-32: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<1xf16>
    %1 = triton_gpu.local_load %arg0 : !triton_gpu.memdesc<4x16x32xf16, #shared> -> tensor<4x16x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>
    tt.return
  }

  // CHECK-LABEL: wmma_dot3d
  tt.func @wmma_dot3d(%arg0: tensor<2x16x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>, %arg1: tensor<2x32x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>, %arg2: tensor<2x16x16xf16, #mma1>) {
    // CHECK-COUNT-32: llvm.extractvalue %arg0
    // CHECK-COUNT-32: llvm.insertelement
    // CHECK-COUNT-32: llvm.extractvalue %arg1
    // CHECK-COUNT-32: llvm.insertelement
    // CHECK-COUNT-8: llvm.extractvalue %arg2
    // CHECK-COUNT-8: llvm.insertelement
    // CHECK-COUNT-2: rocdl.wmma.f16.16x16x16.f16 {{.*}} : (vector<16xf16>, vector<16xf16>, vector<16xf16>, i1) -> vector<16xf16>
    %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<2x16x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>> * tensor<2x32x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>> -> tensor<2x16x16xf16, #mma1>
    // CHECK-COUNT-8: llvm.extractelement
    // CHECK-COUNT-8: llvm.insertvalue
    tt.return
  }
}
