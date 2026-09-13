// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --canonicalize --cse | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942" --canonicalize --cse | FileCheck %s --check-prefix=SW

#packed = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @cvt_scalef32_bf16_fp4_compact_e8m0
  // SW-LABEL: llvm.func @cvt_scalef32_bf16_fp4_compact_e8m0
  tt.func public @cvt_scalef32_bf16_fp4_compact_e8m0(%output: tensor<8x512x!tt.ptr<bf16>, #unpacked>, %x: tensor<8x256xi8, #packed>, %scale: tensor<8x16xi8, #scale>) {
    // Software multiplication needs numeric scales, including byte 0 and NaN.
    // SW-DAG: %[[ZERO:.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // SW-DAG: %[[MIN:.+]] = llvm.mlir.constant(4194304 : i32) : i32
    // SW: %[[BITS:.+]] = llvm.intr.umax(%{{.+}}, %[[MIN]]) : (i32, i32) -> i32
    // SW: %[[VALUE:.+]] = llvm.bitcast %[[BITS]] : i32 to f32
    // SW: %[[SCALE:.+]] = llvm.intr.fma(%[[VALUE]], %[[ZERO]], %[[VALUE]]) : (f32, f32, f32) -> f32
    // SW: llvm.fmul %{{.+}}, %[[SCALE]] : f32
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
  // SW-LABEL: llvm.func @cvt_scalef32_bf16_fp4_compact_bf16
  tt.func public @cvt_scalef32_bf16_fp4_compact_bf16(%output: tensor<8x512x!tt.ptr<bf16>, #unpacked>, %x: tensor<8x256xi8, #packed>, %scale: tensor<8x16xbf16, #scale>) {
    // Pre-shifted scales still require NaN restoration for software arithmetic.
    // SW: %[[ZERO:.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // SW: %[[VALUE:.+]] = llvm.bitcast %{{.+}} : i32 to f32
    // SW: %[[SCALE:.+]] = llvm.intr.fma(%[[VALUE]], %[[ZERO]], %[[VALUE]]) : (f32, f32, f32) -> f32
    // SW: llvm.fmul %{{.+}}, %[[SCALE]] : f32
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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @cvt_scalef32_bf16_fp8_1_element
  // SW-LABEL: llvm.func @cvt_scalef32_bf16_fp8_1_element
  tt.func public @cvt_scalef32_bf16_fp8_1_element(%x: tensor<32x16xf8E5M2, #blocked>, %scale: tensor<32x16xbf16, #blocked>) -> tensor<32x16xbf16, #blocked> {
    // CHECK-COUNT-1: rocdl.cvt.scalef32.pk.bf16.bf8
    // CHECK-NOT: rocdl.cvt.scalef32.pk.bf16.bf8
    // CHECK: llvm.return
    // SW-COUNT-1: llvm.fmul
    // SW-NOT: llvm.fmul
    // SW: llvm.return
    %up = amdg.scaled_upcast_fp8 %x scale %scale : tensor<32x16xf8E5M2, #blocked>, tensor<32x16xbf16, #blocked> -> tensor<32x16xbf16, #blocked>
    tt.return %up : tensor<32x16xbf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @cvt_scalef32_bf16_fp8_2_element
  // SW-LABEL: llvm.func @cvt_scalef32_bf16_fp8_2_element
  tt.func public @cvt_scalef32_bf16_fp8_2_element(%x: tensor<64x16xf8E5M2, #blocked>, %scale: tensor<64x16xbf16, #blocked>) -> tensor<64x16xbf16, #blocked> {
    // Each register belongs to a different scale block.
    // CHECK: %[[S0:.+]] = llvm.extractvalue %arg1[0]
    // CHECK: %[[S1:.+]] = llvm.extractvalue %arg1[1]
    // CHECK: %[[B0:.+]] = llvm.bitcast %[[S0]] : bf16 to i16
    // CHECK: %[[E0:.+]] = llvm.zext %[[B0]] : i16 to i32
    // CHECK: %[[H0:.+]] = llvm.shl %[[E0]], %{{.+}} : i32
    // CHECK: %[[F0:.+]] = llvm.bitcast %[[H0]] : i32 to f32
    // CHECK: %[[C0:.+]] = rocdl.cvt.scalef32.pk.bf16.bf8 %{{.+}}[false], %[[F0]]
    // CHECK: %[[O0:.+]] = llvm.extractelement %[[C0]]
    // CHECK: %[[B1:.+]] = llvm.bitcast %[[S1]] : bf16 to i16
    // CHECK: %[[E1:.+]] = llvm.zext %[[B1]] : i16 to i32
    // CHECK: %[[H1:.+]] = llvm.shl %[[E1]], %{{.+}} : i32
    // CHECK: %[[F1:.+]] = llvm.bitcast %[[H1]] : i32 to f32
    // CHECK: %[[C1:.+]] = rocdl.cvt.scalef32.pk.bf16.bf8 %{{.+}}[false], %[[F1]]
    // CHECK: %[[O1:.+]] = llvm.extractelement %[[C1]]
    // CHECK: %[[R0:.+]] = llvm.insertvalue %[[O0]], %{{.+}}[0]
    // CHECK: %[[R1:.+]] = llvm.insertvalue %[[O1]], %[[R0]][1]
    // CHECK: llvm.return %[[R1]]
    // SW: %[[S0:.+]] = llvm.extractvalue %arg1[0]
    // SW: %[[S1:.+]] = llvm.extractvalue %arg1[1]
    // SW: %[[B0:.+]] = llvm.bitcast %[[S0]] : bf16 to i16
    // SW: %[[E0:.+]] = llvm.zext %[[B0]] : i16 to i32
    // SW: %[[H0:.+]] = llvm.shl %[[E0]], %{{.+}} : i32
    // SW: %[[F0:.+]] = llvm.bitcast %[[H0]] : i32 to f32
    // SW: %[[N0:.+]] = llvm.intr.fma(%[[F0]], %{{.+}}, %[[F0]])
    // SW: llvm.fmul %{{.+}}, %[[N0]]
    // SW: %[[B1:.+]] = llvm.bitcast %[[S1]] : bf16 to i16
    // SW: %[[E1:.+]] = llvm.zext %[[B1]] : i16 to i32
    // SW: %[[H1:.+]] = llvm.shl %[[E1]], %{{.+}} : i32
    // SW: %[[F1:.+]] = llvm.bitcast %[[H1]] : i32 to f32
    // SW: %[[N1:.+]] = llvm.intr.fma(%[[F1]], %{{.+}}, %[[F1]])
    // SW: llvm.fmul %{{.+}}, %[[N1]]
    %up = amdg.scaled_upcast_fp8 %x scale %scale : tensor<64x16xf8E5M2, #blocked>, tensor<64x16xbf16, #blocked> -> tensor<64x16xbf16, #blocked>
    tt.return %up : tensor<64x16xbf16, #blocked>
  }
}

// -----

#packed = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#unpacked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // Each register has its own scale, including the two fp4 values in a byte.
  // CHECK-LABEL: llvm.func @cvt_scalef32_bf16_fp4_distinct_scales
  // SW-LABEL: llvm.func @cvt_scalef32_bf16_fp4_distinct_scales
  tt.func public @cvt_scalef32_bf16_fp4_distinct_scales(%x: tensor<256xi8, #packed>, %scale: tensor<512xbf16, #unpacked>) -> tensor<512xbf16, #unpacked> {
    // CHECK: %[[S0:.+]] = llvm.extractvalue %{{.+}}[0] : !llvm.struct<(bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16)>
    // CHECK: %[[S1:.+]] = llvm.extractvalue %{{.+}}[1] : !llvm.struct<(bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16)>
    // CHECK: %[[B0:.+]] = llvm.bitcast %[[S0]] : bf16 to i16
    // CHECK: %[[Z0:.+]] = llvm.zext %[[B0]] : i16 to i32
    // CHECK: %[[H0:.+]] = llvm.shl %[[Z0]], %{{.+}} : i32
    // CHECK: %[[F0:.+]] = llvm.bitcast %[[H0]] : i32 to f32
    // CHECK: rocdl.cvt.scalef32.pk.bf16.fp4 %{{.+}}, %[[F0]] : vector<2xbf16>
    // CHECK: %[[B1:.+]] = llvm.bitcast %[[S1]] : bf16 to i16
    // CHECK: %[[Z1:.+]] = llvm.zext %[[B1]] : i16 to i32
    // CHECK: %[[H1:.+]] = llvm.shl %[[Z1]], %{{.+}} : i32
    // CHECK: %[[F1:.+]] = llvm.bitcast %[[H1]] : i32 to f32
    // CHECK: rocdl.cvt.scalef32.pk.bf16.fp4 %{{.+}}, %[[F1]] : vector<2xbf16>
    // SW: %[[S0:.+]] = llvm.extractvalue %{{.+}}[0] : !llvm.struct<(bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16)>
    // SW: %[[S1:.+]] = llvm.extractvalue %{{.+}}[1] : !llvm.struct<(bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16)>
    // SW: %[[B0:.+]] = llvm.bitcast %[[S0]] : bf16 to i16
    // SW: %[[Z0:.+]] = llvm.zext %[[B0]] : i16 to i32
    // SW: %[[H0:.+]] = llvm.shl %[[Z0]], %{{.+}} : i32
    // SW: %[[F0:.+]] = llvm.bitcast %[[H0]] : i32 to f32
    // SW: %[[N0:.+]] = llvm.intr.fma(%[[F0]], %{{.+}}, %[[F0]]) : (f32, f32, f32) -> f32
    // SW: llvm.fmul %{{.+}}, %[[N0]] : f32
    // SW: %[[B1:.+]] = llvm.bitcast %[[S1]] : bf16 to i16
    // SW: %[[Z1:.+]] = llvm.zext %[[B1]] : i16 to i32
    // SW: %[[H1:.+]] = llvm.shl %[[Z1]], %{{.+}} : i32
    // SW: %[[F1:.+]] = llvm.bitcast %[[H1]] : i32 to f32
    // SW: %[[N1:.+]] = llvm.intr.fma(%[[F1]], %{{.+}}, %[[F1]]) : (f32, f32, f32) -> f32
    // SW: llvm.fmul %{{.+}}, %[[N1]] : f32
    %up = amdg.scaled_upcast_fp4 %x scale %scale {axis = 0 : i32} : tensor<256xi8, #packed>, tensor<512xbf16, #unpacked> -> tensor<512xbf16, #unpacked>
    tt.return %up : tensor<512xbf16, #unpacked>
  }
}
