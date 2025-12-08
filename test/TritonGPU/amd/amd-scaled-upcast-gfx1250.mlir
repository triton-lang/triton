// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm="arch=gfx1250" --canonicalize --cse | FileCheck %s

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_scaled_mxfp8_bf16(%arg0: tensor<32x128xf8E4M3FN, #blocked>, %arg1: tensor<32x128xi8, #blocked>, %arg2: tensor<32x128x!tt.ptr<bf16>, #blocked>) {
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK: %[[SCALE_1:.*]] = llvm.extractvalue %arg1[8] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK: %[[SCALE_2:.*]] = llvm.extractvalue %arg1[16] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK: %[[SCALE_3:.*]] = llvm.extractvalue %arg1[24] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>

    // CHECK: llvm.insertelement %[[SCALE]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE]], {{.*}} : vector<4xi8>
    // CHECK: %[[V0:.*]] = llvm.insertelement %[[SCALE]], {{.*}} : vector<4xi8>
    // CHECK: %[[SCALE_INT32:.*]] = llvm.bitcast %[[V0]] : vector<4xi8> to i32
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp8 {{.*}}, %[[SCALE_INT32]][0] : vector<8xbf16>

    // CHECK: llvm.insertelement %[[SCALE_1]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE_1]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE_1]], {{.*}} : vector<4xi8>
    // CHECK: %[[V1:.*]] = llvm.insertelement %[[SCALE_1]], {{.*}} : vector<4xi8>
    // CHECK: %[[SCALE_INT32_1:.*]] = llvm.bitcast %[[V1]] : vector<4xi8> to i32
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp8 {{.*}}, %[[SCALE_INT32_1]][0] : vector<8xbf16>

    // CHECK: llvm.insertelement %[[SCALE_2]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE_2]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE_2]], {{.*}} : vector<4xi8>
    // CHECK: %[[V2:.*]] = llvm.insertelement %[[SCALE_2]], {{.*}} : vector<4xi8>
    // CHECK: %[[SCALE_INT32_2:.*]] = llvm.bitcast %[[V2]] : vector<4xi8> to i32
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp8 {{.*}}, %[[SCALE_INT32_2]][0] : vector<8xbf16>

    // CHECK: llvm.insertelement %[[SCALE_3]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE_3]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE_3]], {{.*}} : vector<4xi8>
    // CHECK: %[[V3:.*]] = llvm.insertelement %[[SCALE_3]], {{.*}} : vector<4xi8>
    // CHECK: %[[SCALE_INT32_3:.*]] = llvm.bitcast %[[V3]] : vector<4xi8> to i32
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp8 {{.*}}, %[[SCALE_INT32_3]][0] : vector<8xbf16>
    %7 = amdg.scaled_upcast_fp8 %arg0 scale %arg1 : tensor<32x128xf8E4M3FN, #blocked>, tensor<32x128xi8, #blocked> -> tensor<32x128xbf16, #blocked>
    tt.store %arg2, %7 : tensor<32x128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [4, 1], instrShape = [16, 16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 2048 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk8_bf16_fp4(%output: tensor<16x64x!tt.ptr<bf16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>, %15: tensor<16x32xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, %27: tensor<16x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>) attributes {noinline = false} {
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg2[0] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK: %[[SCALE_1:.*]] = llvm.extractvalue %arg2[8] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK: %[[SCALE_2:.*]] = llvm.extractvalue %arg2[16] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>
    // CHECK: %[[SCALE_3:.*]] = llvm.extractvalue %arg2[24] : !llvm.struct<(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8)>

    // CHECK: llvm.insertelement %[[SCALE]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE]], {{.*}} : vector<4xi8>
    // CHECK: %[[V0:.*]] = llvm.insertelement %[[SCALE]], {{.*}} : vector<4xi8>
    // CHECK: %[[SCALE_INT32:.*]] = llvm.bitcast %[[V0]] : vector<4xi8> to i32
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp4 {{.*}}, %[[SCALE_INT32]][0] : vector<8xbf16>

    // CHECK: llvm.insertelement %[[SCALE_1]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE_1]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE_1]], {{.*}} : vector<4xi8>
    // CHECK: %[[V1:.*]] = llvm.insertelement %[[SCALE_1]], {{.*}} : vector<4xi8>
    // CHECK: %[[SCALE_INT32_1:.*]] = llvm.bitcast %[[V1]] : vector<4xi8> to i32
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp4 {{.*}}, %[[SCALE_INT32_1]][0] : vector<8xbf16>

    // CHECK: llvm.insertelement %[[SCALE_2]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE_2]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE_2]], {{.*}} : vector<4xi8>
    // CHECK: %[[V2:.*]] = llvm.insertelement %[[SCALE_2]], {{.*}} : vector<4xi8>
    // CHECK: %[[SCALE_INT32_2:.*]] = llvm.bitcast %[[V2]] : vector<4xi8> to i32
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp4 {{.*}}, %[[SCALE_INT32_2]][0] : vector<8xbf16>

    // CHECK: llvm.insertelement %[[SCALE_3]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE_3]], {{.*}} : vector<4xi8>
    // CHECK: llvm.insertelement %[[SCALE_3]], {{.*}} : vector<4xi8>
    // CHECK: %[[V3:.*]] = llvm.insertelement %[[SCALE_3]], {{.*}} : vector<4xi8>
    // CHECK: %[[SCALE_INT32_3:.*]] = llvm.bitcast %[[V3]] : vector<4xi8> to i32
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp4 {{.*}}, %[[SCALE_INT32_3]][0] : vector<8xbf16>

    %28 = amdg.scaled_upcast_fp4 %15 scale %27 {axis = 1 : i32} : tensor<16x32xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<16x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> -> tensor<16x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    tt.store %output, %28 : tensor<16x64x!tt.ptr<bf16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    tt.return
  }
}
