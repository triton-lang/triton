// RUN: triton-opt %s -split-input-file --tritonamdgpu-optimize-atomic-layouts='arch-generation-name=gfx942' | FileCheck %s

// CHECK: #[[OLD_LAYOUT:.+]] = #triton_gpu.blocked<{sizePerThread = [1]{{.*}}}>
// CHECK: #[[NEW_LAYOUT:.+]] = #triton_gpu.blocked<{sizePerThread = [2]{{.*}}}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @atomic_add_f16x2(%arg0 : tensor<256x!tt.ptr<f16>, #blocked>, %arg1 : tensor<256xi1, #blocked>, %arg2 : tensor<256xf16, #blocked>) {
    // CHECK-3: triton_gpu.convert_layout {{.*}}tensor<{{.*}}#[[OLD_LAYOUT]]> -> tensor<{{.*}}#[[NEW_LAYOUT]]>
    // CHECK: atomic_rmw fadd{{.*}} -> tensor<{{.*}}#[[NEW_LAYOUT]]>
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
    // CHECK: triton_gpu.convert_layout {{.*}}tensor<{{.*}} #[[NEW_LAYOUT]]> -> tensor<{{.*}} #[[OLD_LAYOUT]]>
    tt.store %arg0, %0 : tensor<256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK: #[[OLD_LAYOUT:.+]] = #triton_gpu.blocked<{sizePerThread = [3]{{.*}}}>
// CHECK: #[[NEW_LAYOUT:.+]] = #triton_gpu.blocked<{sizePerThread = [6]{{.*}}}>
#blocked = #triton_gpu.blocked<{sizePerThread = [3], threadsPerWarp = [64], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @atomic_add_bf16x6(%arg0 : tensor<256x!tt.ptr<bf16>, #blocked>, %arg1 : tensor<256xi1, #blocked>, %arg2 : tensor<256xbf16, #blocked>) {
    // CHECK-3: triton_gpu.convert_layout {{.*}}tensor<{{.*}}#[[OLD_LAYOUT]]> -> tensor<{{.*}}#[[NEW_LAYOUT]]>
    // CHECK: atomic_rmw fadd{{.*}} -> tensor<{{.*}}#[[NEW_LAYOUT]]>
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xbf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xbf16, #blocked>
    // CHECK: triton_gpu.convert_layout {{.*}}tensor<{{.*}} #[[NEW_LAYOUT]]> -> tensor<{{.*}} #[[OLD_LAYOUT]]>
    tt.store %arg0, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}
