// RUN: triton-opt %s --allocate-shared-memory-nv --convert-triton-gpu-to-llvm --cse | FileCheck %s

// CHECK-LABEL: dont_divide_0
// CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NOT: llvm.urem %{{.*}}, %[[C0]]
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dont_divide_0() {
    %zero = arith.constant dense<0.000000e+00> : tensor<16x1xf32, #mma>
    %cvt = ttg.convert_layout %zero : tensor<16x1xf32, #mma> -> tensor<16x1xf32, #blocked>
    tt.return
  }

  // NVPTX expands llvm.frem inline as a - trunc(a/b) * b, which loses the
  // dividend's sign on an exact-zero remainder (#11986). frem lowers to
  // libdevice fmod instead; NaN/inf/denormal edge cases come with it.
  // CHECK-LABEL: remf_lowers_to_libdevice_fmod
  // CHECK: llvm.call @__nv_fmodf
  tt.func public @remf_lowers_to_libdevice_fmod(%arg0: tensor<32x2xf32, #blocked>, %arg1: tensor<32x2xf32, #blocked>) {
    %0 = arith.remf %arg0, %arg1 : tensor<32x2xf32, #blocked>
    tt.return
  }
}
