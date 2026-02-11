// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck --check-prefixes=GFX942 %s
// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx1250 | FileCheck --check-prefixes=GFX1250 %s


#layout0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#layout1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [64, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: fp4_upcast_to_f32
  tt.func @fp4_upcast_to_f32() -> tensor<128x64xf32, #layout1> {
    // GFX942: [[i64_0:%.*]] = rocdl.cvt.pk.f32.fp8 %{{.*}}[false] : i64
    // GFX942: [[i64_1:%.*]] = rocdl.cvt.pk.f32.fp8 %{{.*}}[true] : i64
    // GFX942: [[i64_2:%.*]] = rocdl.cvt.pk.f32.fp8 %{{.*}}[false] : i64
    // GFX942: [[i64_3:%.*]] = rocdl.cvt.pk.f32.fp8 %{{.*}}[true] : i64
    // GFX942: [[i32x2_0:%.*]] = llvm.bitcast [[i64_0]] : i64 to vector<2xi32>
    // GFX942: [[i32x2_1:%.*]] = llvm.bitcast [[i64_1]] : i64 to vector<2xi32>
    // GFX942: [[i32x2_2:%.*]] = llvm.bitcast [[i64_2]] : i64 to vector<2xi32>
    // GFX942: [[i32x2_3:%.*]] = llvm.bitcast [[i64_3]] : i64 to vector<2xi32>
    // GFX942: [[i32_0:%.*]] = llvm.extractelement [[i32x2_0]][%{{.*}} : i32] : vector<2xi32>
    // GFX942: [[i32_1:%.*]] = llvm.extractelement [[i32x2_2]][%{{.*}} : i32] : vector<2xi32>
    // GFX942: [[i32_2:%.*]] = llvm.extractelement [[i32x2_0]][%{{.*}} : i32] : vector<2xi32>
    // GFX942: [[i32_3:%.*]] = llvm.extractelement [[i32x2_2]][%{{.*}} : i32] : vector<2xi32>
    // GFX942: [[i32_4:%.*]] = llvm.extractelement [[i32x2_1]][%{{.*}} : i32] : vector<2xi32>
    // GFX942: [[i32_5:%.*]] = llvm.extractelement [[i32x2_3]][%{{.*}} : i32] : vector<2xi32>
    // GFX942: [[i32_6:%.*]] = llvm.extractelement [[i32x2_1]][%{{.*}} : i32] : vector<2xi32>
    // GFX942: [[i32_7:%.*]] = llvm.extractelement [[i32x2_3]][%{{.*}} : i32] : vector<2xi32>
    // GFX942: [[f32_0:%.*]] = llvm.bitcast [[i32_0]] : i32 to f32
    // GFX942: [[f32_1:%.*]] = llvm.bitcast [[i32_1]] : i32 to f32
    // GFX942: [[f32_2:%.*]] = llvm.bitcast [[i32_2]] : i32 to f32
    // GFX942: [[f32_3:%.*]] = llvm.bitcast [[i32_3]] : i32 to f32
    // GFX942: [[f32_4:%.*]] = llvm.bitcast [[i32_4]] : i32 to f32
    // GFX942: [[f32_5:%.*]] = llvm.bitcast [[i32_5]] : i32 to f32
    // GFX942: [[f32_6:%.*]] = llvm.bitcast [[i32_6]] : i32 to f32
    // GFX942: [[f32_7:%.*]] = llvm.bitcast [[i32_7]] : i32 to f32

    // GFX1250: [[c16:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // GFX1250: [[i32_0:%.*]] = llvm.shl %{{.*}}, [[c16]] : i32
    // GFX1250: [[f32_0:%.*]] = llvm.bitcast [[i32_0]] : i32 to f32
    // GFX1250: [[c_0xFFFF0000:%.*]] = llvm.mlir.constant(-65536 : i32) : i32
    // GFX1250: [[i32_1:%.*]] = llvm.and %{{.*}}, [[c_0xFFFF0000]] : i32
    // GFX1250: [[f32_1:%.*]] = llvm.bitcast [[i32_1]] : i32 to f32
    // GFX1250: [[c16:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // GFX1250: [[i32_2:%.*]] = llvm.shl %{{.*}}, [[c16]] : i32
    // GFX1250: [[f32_2:%.*]] = llvm.bitcast [[i32_2]] : i32 to f32
    // GFX1250: [[c_0xFFFF0000:%.*]] = llvm.mlir.constant(-65536 : i32) : i32
    // GFX1250: [[i32_3:%.*]] = llvm.and %{{.*}}, [[c_0xFFFF0000]] : i32
    // GFX1250: [[f32_3:%.*]] = llvm.bitcast [[i32_3]] : i32 to f32
    // GFX1250: [[c16:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // GFX1250: [[i32_4:%.*]] = llvm.shl %{{.*}}, [[c16]] : i32
    // GFX1250: [[f32_4:%.*]] = llvm.bitcast [[i32_4]] : i32 to f32
    // GFX1250: [[c_0xFFFF0000:%.*]] = llvm.mlir.constant(-65536 : i32) : i32
    // GFX1250: [[i32_5:%.*]] = llvm.and %{{.*}}, [[c_0xFFFF0000]] : i32
    // GFX1250: [[f32_5:%.*]] = llvm.bitcast [[i32_5]] : i32 to f32
    // GFX1250: [[c16:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // GFX1250: [[i32_6:%.*]] = llvm.shl %{{.*}}, [[c16]] : i32
    // GFX1250: [[f32_6:%.*]] = llvm.bitcast [[i32_6]] : i32 to f32
    // GFX1250: [[c_0xFFFF0000:%.*]] = llvm.mlir.constant(-65536 : i32) : i32
    // GFX1250: [[i32_7:%.*]] = llvm.and %{{.*}}, [[c_0xFFFF0000]] : i32
    // GFX1250: [[f32_7:%.*]] = llvm.bitcast [[i32_7]] : i32 to f32

    %0 = arith.constant dense<0> : tensor<128x32xi8, #layout0>
    %fp32 = ttg.fp4_to_fp %0 {axis = 1 : i32} : tensor<128x32xi8, #layout0> -> tensor<128x64xf32, #layout1>
    tt.return %fp32: tensor<128x64xf32, #layout1>
  }
}
