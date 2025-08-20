// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx950 | FileCheck --check-prefixes=GFX950 %s

// -----

// GFX950-LABEL: upcast_mxfp4
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 4], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 4096 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @upcast_mxfp4(%arg0 : tensor<32x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, %arg1 : tensor<32x2xi8, #blocked>) {
    // GFX950-DAG: %[[CST:.*]] = llvm.mlir.constant(23 : i32) : i32
    // GFX950-DAG: %[[ISCALE:.*]] = llvm.zext %{{.*}} : i8 to i32
    // GFX950: %[[INTS:.*]] = llvm.shl %[[ISCALE]], %[[CST]] : i32
    // GFX950: %[[SCALE:.*]] = llvm.bitcast %[[INTS]] : i32 to f32
    // GFX950: rocdl.cvt.scalef32.pk.bf16.fp4 %[[REG:.*]][0], %[[SCALE]] : vector<2xbf16>
    // GFX950: rocdl.cvt.scalef32.pk.bf16.fp4 %[[REG]][1], %[[SCALE]] : vector<2xbf16>
    // GFX950: rocdl.cvt.scalef32.pk.bf16.fp4 %[[REG]][2], %[[SCALE]] : vector<2xbf16>
    // GFX950: rocdl.cvt.scalef32.pk.bf16.fp4 %[[REG]][3], %[[SCALE]] : vector<2xbf16>
    %1 = amdgpu.upcast_mxfp %arg0, %arg1 fp_type = e2m1 {fastMath = false} : tensor<32x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, tensor<32x2xi8, #blocked> -> tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    tt.return
  }
}


// -----

// GFX950-LABEL: upcast_mxfp8
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 4], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 4096 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @upcast_mxfp8(%arg0 : tensor<64x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, %arg1 : tensor<32x2xi8, #blocked>) {
    // GFX950-DAG: %[[CST:.*]] = llvm.mlir.constant(23 : i32) : i32
    // GFX950-DAG: %[[ISCALE:.*]] = llvm.zext %{{.*}} : i8 to i32
    // GFX950: %[[INTS:.*]] = llvm.shl %[[ISCALE]], %[[CST]] : i32
    // GFX950: %[[SCALE:.*]] = llvm.bitcast %[[INTS]] : i32 to f32
    // GFX950: rocdl.cvt.scalef32.pk.bf16.fp8 %[[REG:.*]][false], %[[SCALE]] : vector<2xbf16>
    // GFX950: rocdl.cvt.scalef32.pk.bf16.fp8 %[[REG]][true], %[[SCALE]] : vector<2xbf16>
    %1 = amdgpu.upcast_mxfp %arg0, %arg1 fp_type = e4m3 {fastMath = false} : tensor<64x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, tensor<32x2xi8, #blocked> -> tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    tt.return
  }
}

// -----

// GFX950-LABEL: upcast_mxbf8
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 4], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 4096 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @upcast_mxbf8(%arg0 : tensor<64x32xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, %arg1 : tensor<32x2xi8, #blocked>) {
    // GFX950-DAG: %[[CST:.*]] = llvm.mlir.constant(23 : i32) : i32
    // GFX950-DAG: %[[ISCALE:.*]] = llvm.zext %{{.*}} : i8 to i32
    // GFX950: %[[INTS:.*]] = llvm.shl %[[ISCALE]], %[[CST]] : i32
    // GFX950: %[[SCALE:.*]] = llvm.bitcast %[[INTS]] : i32 to f32
    // GFX950: rocdl.cvt.scalef32.pk.f16.bf8 %[[REG:.*]][false], %[[SCALE]] : vector<2xf16>
    // GFX950: rocdl.cvt.scalef32.pk.f16.bf8 %[[REG]][true], %[[SCALE]] : vector<2xf16>
    %1 = amdgpu.upcast_mxfp %arg0, %arg1 fp_type = e5m2 {fastMath = false} : tensor<64x32xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, tensor<32x2xi8, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    tt.return
  }
}
