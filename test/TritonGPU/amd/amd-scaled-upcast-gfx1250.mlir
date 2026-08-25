// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1250" --canonicalize --cse | FileCheck %s

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, isTranspose = true, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_scaled_mxfp8_bf16(%arg0: tensor<32x128xf8E4M3FN, #blocked>, %arg1: tensor<32x128xi8, #blocked>, %arg2: tensor<32x128x!tt.ptr<bf16>, #blocked>) {
    // Non-broadcast scale layouts use FP8 Block16 mode (opSel=8): byte 0 feeds
    // lanes 0..15 and byte 1 feeds lanes 16..31. Byte 1 is lane (j^16)'s scale,
    // gathered via permlanex16.
    // CHECK: %[[S:.+]] = llvm.extractvalue %arg1[0] : !llvm.struct
    // CHECK: %[[Z:.+]] = llvm.zext %[[S]] : i8 to i32
    // CHECK: %[[P:.+]] = rocdl.permlanex16 %[[Z]], %[[Z]], {{.*}}, true, false : i32, i32
    // CHECK: %[[X:.+]] = llvm.trunc %[[P]] : i32 to i8
    // CHECK: %[[V0:.+]] = llvm.insertelement %[[S]], %{{.+}}[%{{.+}} : i32] : vector<4xi8>
    // CHECK: %[[V1:.+]] = llvm.insertelement %[[X]], %[[V0]][%{{.+}} : i32] : vector<4xi8>
    // CHECK: %[[SI:.+]] = llvm.bitcast %[[V1]] : vector<4xi8> to i32
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp8 %{{.+}}, %[[SI]][8] : vector<8xbf16>
    %7 = amdg.scaled_upcast_fp8 %arg0 scale %arg1 : tensor<32x128xf8E4M3FN, #blocked>, tensor<32x128xi8, #blocked> -> tensor<32x128xbf16, #blocked>
    tt.store %arg2, %7 : tensor<32x128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[1, 0], [2, 0]]}, isTranspose = true, instrShape = [16, 16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 2048 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk8_bf16_fp4(%output: tensor<16x64x!tt.ptr<bf16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>, %15: tensor<16x32xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, %27: tensor<16x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>) attributes {noinline = false} {
    // This scale layout is not broadcast across the lane^16 split, so the 4
    // scale registers used by the emitted pk8 groups are exchanged cross-lane
    // (permlanex16) to co-locate lane (j^16)'s scale into byte 1 for the upper
    // output lanes.
    // CHECK-COUNT-4: rocdl.permlanex16 {{.*}}, true, false : i32, i32
    // CHECK-NOT: rocdl.permlanex16
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp4 {{.*}} : vector<8xbf16>

    %28 = amdg.scaled_upcast_fp4 %15 scale %27 {axis = 1 : i32} : tensor<16x32xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<16x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> -> tensor<16x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    tt.store %output, %28 : tensor<16x64x!tt.ptr<bf16>, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    tt.return
  }
}

// -----

// Compact scale: one E8M0 scale per 32 output elements along the K axis.
#packed = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#compact = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk8_bf16_fp4_compact(%output: tensor<1x1024x!tt.ptr<bf16>, #unpacked>, %x: tensor<1x512xi8, #packed>, %scale: tensor<1x32xi8, #compact>) {
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp4
    %up = amdg.scaled_upcast_fp4 %x scale %scale {axis = 1 : i32} : tensor<1x512xi8, #packed>, tensor<1x32xi8, #compact> -> tensor<1x1024xbf16, #unpacked>
    tt.store %output, %up : tensor<1x1024x!tt.ptr<bf16>, #unpacked>
    tt.return
  }
}

// -----

#packed = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @cvt_scale_pk8_bf16_fp4_contig8
  tt.func public @cvt_scale_pk8_bf16_fp4_contig8(%output: tensor<8x64x!tt.ptr<bf16>, #unpacked>, %x: tensor<8x32xi8, #packed>, %scale: tensor<8x2xi8, #scale>) {
    // Two K blocks -> block 0 uses scale[0], block 1 uses scale[1].
    // CHECK: %[[S0:.+]] = llvm.extractvalue %{{.+}}[0] : !llvm.struct<(i8, i8)>
    // CHECK: %[[S1:.+]] = llvm.extractvalue %{{.+}}[1] : !llvm.struct<(i8, i8)>
    // CHECK: llvm.insertelement %[[S0]]
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp4
    // CHECK: llvm.insertelement %[[S1]]
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp4
    %up = amdg.scaled_upcast_fp4 %x scale %scale {axis = 1 : i32} : tensor<8x32xi8, #packed>, tensor<8x2xi8, #scale> -> tensor<8x64xbf16, #unpacked>
    tt.store %output, %up : tensor<8x64x!tt.ptr<bf16>, #unpacked>
    tt.return
  }
}

// -----

#packed = #ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
#unpacked = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
#scale = #ttg.linear<{register = [[0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @cvt_scale_pk8_bf16_fp4_interleaved
  tt.func public @cvt_scale_pk8_bf16_fp4_interleaved(%output: tensor<32x32x!tt.ptr<bf16>, #unpacked>, %x: tensor<32x16xi8, #packed>, %scale: tensor<32x2xi8, #scale>) {
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp4 {{.*}}, %[[SI0:.+]][0] : vector<8xbf16>
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp4 {{.*}}, %[[SI1:.+]][0] : vector<8xbf16>
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp4 {{.*}}, %[[SI0]][0] : vector<8xbf16>
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp4 {{.*}}, %[[SI1]][0] : vector<8xbf16>
    %up = amdg.scaled_upcast_fp4 %x scale %scale {axis = 1 : i32} : tensor<32x16xi8, #packed>, tensor<32x2xi8, #scale> -> tensor<32x32xbf16, #unpacked>
    tt.store %output, %up : tensor<32x32x!tt.ptr<bf16>, #unpacked>
    tt.return
  }
}
