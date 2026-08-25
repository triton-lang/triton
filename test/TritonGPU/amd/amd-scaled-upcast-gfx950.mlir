// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --canonicalize --cse | FileCheck %s

#packed = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @cvt_scalef32_bf16_fp4_compact_e8m0
  tt.func public @cvt_scalef32_bf16_fp4_compact_e8m0(%output: tensor<8x512x!tt.ptr<bf16>, #unpacked>, %x: tensor<8x256xi8, #packed>, %scale: tensor<8x16xi8, #scale>) {
    // A raw E8M0 payload is shifted into the f32 exponent by 23.
    // CHECK-DAG: %[[C23:.+]] = llvm.mlir.constant(23 : i32) : i32
    // The 8 pk groups a thread holds span 2 scale blocks: groups 0-3 (16
    // intrinsics) take scale register 0, groups 4-7 take scale register 1.
    // CHECK: %[[R0:.+]] = llvm.extractvalue %{{.+}}[0] : !llvm.struct<(i8, i8)>
    // CHECK: %[[R1:.+]] = llvm.extractvalue %{{.+}}[1] : !llvm.struct<(i8, i8)>
    // CHECK: %[[Z0:.+]] = llvm.zext %[[R0]] : i8 to i32
    // CHECK: %[[H0:.+]] = llvm.shl %[[Z0]], %[[C23]] : i32
    // CHECK: %[[S0:.+]] = llvm.bitcast %[[H0]] : i32 to f32
    // CHECK-COUNT-16: rocdl.cvt.scalef32.pk.bf16.fp4 %{{.+}}, %[[S0]] : vector<2xbf16>
    // CHECK: %[[Z1:.+]] = llvm.zext %[[R1]] : i8 to i32
    // CHECK: %[[H1:.+]] = llvm.shl %[[Z1]], %[[C23]] : i32
    // CHECK: %[[S1:.+]] = llvm.bitcast %[[H1]] : i32 to f32
    // CHECK-COUNT-16: rocdl.cvt.scalef32.pk.bf16.fp4 %{{.+}}, %[[S1]] : vector<2xbf16>
    // CHECK-NOT: rocdl.cvt.scalef32.pk.bf16.fp4
    %up = amdg.scaled_upcast_fp4 %x scale %scale {axis = 1 : i32} : tensor<8x256xi8, #packed>, tensor<8x16xi8, #scale> -> tensor<8x512xbf16, #unpacked>
    tt.store %output, %up : tensor<8x512x!tt.ptr<bf16>, #unpacked>
    tt.return
  }
}

// -----

#packed = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @cvt_scalef32_bf16_fp4_compact_bf16
  tt.func public @cvt_scalef32_bf16_fp4_compact_bf16(%output: tensor<8x512x!tt.ptr<bf16>, #unpacked>, %x: tensor<8x256xi8, #packed>, %scale: tensor<8x16xbf16, #scale>) {
    // A bf16 scale is pre-shifted by 7, so it only needs 16 more bits.
    // CHECK-DAG: %[[C16:.+]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: %[[R0:.+]] = llvm.extractvalue %{{.+}}[0] : !llvm.struct<(bf16, bf16)>
    // CHECK: %[[R1:.+]] = llvm.extractvalue %{{.+}}[1] : !llvm.struct<(bf16, bf16)>
    // CHECK: %[[B0:.+]] = llvm.bitcast %[[R0]] : bf16 to i16
    // CHECK: %[[Z0:.+]] = llvm.zext %[[B0]] : i16 to i32
    // CHECK: %[[H0:.+]] = llvm.shl %[[Z0]], %[[C16]] : i32
    // CHECK: %[[S0:.+]] = llvm.bitcast %[[H0]] : i32 to f32
    // CHECK-COUNT-16: rocdl.cvt.scalef32.pk.bf16.fp4 %{{.+}}, %[[S0]] : vector<2xbf16>
    // CHECK: %[[B1:.+]] = llvm.bitcast %[[R1]] : bf16 to i16
    // CHECK: %[[Z1:.+]] = llvm.zext %[[B1]] : i16 to i32
    // CHECK: %[[H1:.+]] = llvm.shl %[[Z1]], %[[C16]] : i32
    // CHECK: %[[S1:.+]] = llvm.bitcast %[[H1]] : i32 to f32
    // CHECK-COUNT-16: rocdl.cvt.scalef32.pk.bf16.fp4 %{{.+}}, %[[S1]] : vector<2xbf16>
    // CHECK-NOT: rocdl.cvt.scalef32.pk.bf16.fp4
    %up = amdg.scaled_upcast_fp4 %x scale %scale {axis = 1 : i32} : tensor<8x256xi8, #packed>, tensor<8x16xbf16, #scale> -> tensor<8x512xbf16, #unpacked>
    tt.store %output, %up : tensor<8x512x!tt.ptr<bf16>, #unpacked>
    tt.return
  }
}

// -----

#packed = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @cvt_scalef32_f16_fp4_compact
  tt.func public @cvt_scalef32_f16_fp4_compact(%output: tensor<8x512x!tt.ptr<f16>, #unpacked>, %x: tensor<8x256xi8, #packed>, %scale: tensor<8x16xi8, #scale>) {
    // CHECK: %[[S0:.+]] = llvm.bitcast %{{.+}} : i32 to f32
    // CHECK-COUNT-16: rocdl.cvt.scalef32.pk.f16.fp4 %{{.+}}, %[[S0]] : vector<2xf16>
    // CHECK: %[[S1:.+]] = llvm.bitcast %{{.+}} : i32 to f32
    // CHECK-COUNT-16: rocdl.cvt.scalef32.pk.f16.fp4 %{{.+}}, %[[S1]] : vector<2xf16>
    // CHECK-NOT: rocdl.cvt.scalef32.pk.f16.fp4
    %up = amdg.scaled_upcast_fp4 %x scale %scale {axis = 1 : i32} : tensor<8x256xi8, #packed>, tensor<8x16xi8, #scale> -> tensor<8x512xf16, #unpacked>
    tt.store %output, %up : tensor<8x512x!tt.ptr<f16>, #unpacked>
    tt.return
  }
}

// -----

#packed = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @cvt_scalef32_bf16_fp4_compact_single_reg
  tt.func public @cvt_scalef32_bf16_fp4_compact_single_reg(%output: tensor<8x256x!tt.ptr<bf16>, #unpacked>, %x: tensor<8x128xi8, #packed>, %scale: tensor<8x8xi8, #scale>) {
    // CHECK: %[[R0:.+]] = llvm.extractvalue %{{.+}}[0] : !llvm.struct<(i8)>
    // CHECK: %[[Z0:.+]] = llvm.zext %[[R0]] : i8 to i32
    // CHECK: %[[S0:.+]] = llvm.bitcast %{{.+}} : i32 to f32
    // CHECK-COUNT-16: rocdl.cvt.scalef32.pk.bf16.fp4 %{{.+}}, %[[S0]] : vector<2xbf16>
    // CHECK-NOT: rocdl.cvt.scalef32.pk.bf16.fp4
    %up = amdg.scaled_upcast_fp4 %x scale %scale {axis = 1 : i32} : tensor<8x128xi8, #packed>, tensor<8x8xi8, #scale> -> tensor<8x256xbf16, #unpacked>
    tt.store %output, %up : tensor<8x256x!tt.ptr<bf16>, #unpacked>
    tt.return
  }
}
