// RUN: triton-opt %s --split-input-file --decompose-unsupported-amd-conversions=arch=gfx942 | FileCheck %s

// CHECK-DAG: #[[DST_ENC:.+]] = #triton_gpu.blocked<{{.*}}>
// CHECK-DAG: #[[SRC_ENC:.+]] = #triton_gpu.amd_mfma<{{.*}}>
// CHECK-DAG: #[[TMP_ENC:.+]] = #triton_gpu.amd_mfma<{{.*}}>
// CHECK: large_tensor_conversion
#src = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [32, 32], isTransposed = false}>
#dst = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @large_tensor_conversion(%arg0: tensor<128x128xf32, #src>) {
    // CHECK: %[[TMP:.*]] = triton_gpu.convert_layout {{.*}} : tensor<128x128xf32, #[[SRC_ENC]]> -> tensor<128x128xf32, #[[TMP_ENC]]>
    // CHECK: %{{.*}} = triton_gpu.convert_layout [[TMP]] : tensor<128x128xf32, #[[TMP_ENC]]> -> tensor<128x128xf32, #[[DST_ENC]]>
    %0 = triton_gpu.convert_layout %arg0 : tensor<128x128xf32, #src> -> tensor<128x128xf32, #dst>
    tt.return
  }
}

// -----

// CHECK-DAG: #[[DST_ENC:.+]] = #triton_gpu.blocked<{{.*}}>
// CHECK-DAG: #[[SRC_ENC:.+]] = #triton_gpu.amd_mfma<{{.*}}>
// CHECK-DAG: #[[TMP_ENC:.+]] = #triton_gpu.amd_mfma<{{.*}}>
// CHECK: large_tensor_3d_conversion
#src = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 1, 2], instrShape = [32, 32], isTransposed = false}>
#dst = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 64, 1], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @large_tensor_3d_conversion(%arg0: tensor<2x128x64xf32, #src>) {
    // CHECK: %[[TMP:.*]] = triton_gpu.convert_layout {{.*}} : tensor<2x128x64xf32, #[[SRC_ENC]]> -> tensor<2x128x64xf32, #[[TMP_ENC]]>
    // CHECK: {{.*}} = triton_gpu.convert_layout %[[TMP]] : tensor<2x128x64xf32, #[[TMP_ENC]]> -> tensor<2x128x64xf32, #[[DST_ENC]]>
    %0 = triton_gpu.convert_layout %arg0 : tensor<2x128x64xf32, #src> -> tensor<2x128x64xf32, #dst>
    tt.return
  }
}
