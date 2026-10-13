// RUN: triton-opt %s -triton-cpu-math-to-vec-lib="cpu_features=neon" | FileCheck %s --check-prefix=NEON --implicit-check-not=vector.vscale --implicit-check-not=scf.for --implicit-check-not=sve --implicit-check-not=math.sin --implicit-check-not="func.call @Sleef_"
// RUN: triton-opt %s -triton-cpu-math-to-vec-lib="cpu_features=avx" | FileCheck %s --check-prefix=AVX --implicit-check-not=vector.vscale --implicit-check-not=scf.for --implicit-check-not=sve --implicit-check-not=math.sin --implicit-check-not="func.call @Sleef_"

// Fixed-width math calls retain their existing signatures and decomposition.
// NEON-DAG: func.func private @Sleef_sinf4_u10(vector<4xf32>) -> vector<4xf32>
// NEON-DAG: func.func private @Sleef_sind2_u10(vector<2xf64>) -> vector<2xf64>
// AVX-DAG: func.func private @Sleef_sinf8_u10(vector<8xf32>) -> vector<8xf32>
// AVX-DAG: func.func private @Sleef_sind4_u10(vector<4xf64>) -> vector<4xf64>

module {
  // NEON-LABEL: llvm.func @sin_fixed_f32(
  // NEON-COUNT-4: func.call @Sleef_sinf4_u10
  // NEON-NOT: func.call @Sleef_sinf4_u10
  // NEON: llvm.return
  // AVX-LABEL: llvm.func @sin_fixed_f32(
  // AVX-COUNT-2: func.call @Sleef_sinf8_u10
  // AVX-NOT: func.call @Sleef_sinf8_u10
  // AVX: llvm.return
  llvm.func @sin_fixed_f32(%arg: vector<16xf32>) -> vector<16xf32> {
    %result = math.sin %arg : vector<16xf32>
    llvm.return %result : vector<16xf32>
  }

  // NEON-LABEL: llvm.func @sin_fixed_f64(
  // NEON-COUNT-8: func.call @Sleef_sind2_u10
  // NEON-NOT: func.call @Sleef_sind2_u10
  // NEON: llvm.return
  // AVX-LABEL: llvm.func @sin_fixed_f64(
  // AVX-COUNT-4: func.call @Sleef_sind4_u10
  // AVX-NOT: func.call @Sleef_sind4_u10
  // AVX: llvm.return
  llvm.func @sin_fixed_f64(%arg: vector<16xf64>) -> vector<16xf64> {
    %result = math.sin %arg : vector<16xf64>
    llvm.return %result : vector<16xf64>
  }
}
