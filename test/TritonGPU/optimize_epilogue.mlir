// RUN: triton-opt %s -split-input-file --tritongpu-optimize-epilogue | FileCheck --check-prefixes=CHECK %s
// RUN: triton-opt %s -split-input-file --tritonamdgpu-optimize-epilogue | FileCheck --check-prefixes=GCN %s

#mma = #triton_gpu.mfma<{warpsPerCTA=[1,1], nonKDim = 32, isTranspose=false}>
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // GCN-LABEL: mma_epilogue_simple
  // CHECK-LABEL: mma_epilogue_simple
  tt.func public @mma_epilogue_simple(%data: tensor<64x64xf16, #mma>, %ptr: tensor<64x64x!tt.ptr<f16>, #blocked>) {
    // GCN: [[PTR:%[a-z0-9]+]] = triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mma>
    // GCN: tt.store [[PTR]], {{.*}} : tensor<{{.*}}, #mma>
    // CHECK: [[DATA:%[a-z0-9]+]] = triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #mma>) -> tensor<{{.*}}, #blocked>
    // CHECK: tt.store {{.*}}, [[DATA]] {{.*}} : tensor<{{.*}}, #blocked>
    %converted_data = triton_gpu.convert_layout %data : (tensor<64x64xf16, #mma>) -> tensor<64x64xf16, #blocked>
    tt.store %ptr, %converted_data {cache = 1 : i32, evict = 1 : i32} : tensor<64x64xf16, #blocked>
    tt.return
  }
}

// -----

#mma = #triton_gpu.mfma<{warpsPerCTA=[1,1], nonKDim = 32, isTranspose=false}>
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // GCN-LABEL: mma_epilogue_chained_elementwise
  // CHECK-LABEL: mma_epilogue_chained_elementwise
  tt.func public @mma_epilogue_chained_elementwise(%data: tensor<64x64xf32, #mma>, %ptr: tensor<64x64x!tt.ptr<f16>, #blocked>) {
    // GCN: [[PTR:%[a-z0-9]+]] = triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #blocked>) -> tensor<{{.*}}, #mma>
    // GCN: tt.store [[PTR]], {{.*}} : tensor<{{.*}}, #mma>
    // CHECK: [[DATA:%[a-z0-9]+]] = triton_gpu.convert_layout {{.*}} : (tensor<{{.*}}, #mma>) -> tensor<{{.*}}, #blocked>
    // CHECK: [[TDATA:%[a-z0-9]+]] = arith.truncf [[DATA]] {{.*}}
    // CHECK: tt.store {{.*}}, [[TDATA]] {{.*}} : tensor<{{.*}}, #blocked>
    %converted_data = triton_gpu.convert_layout %data : (tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #blocked>
    %trunked = arith.truncf %converted_data : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    tt.store %ptr, %trunked {cache = 1 : i32, evict = 1 : i32} : tensor<64x64xf16, #blocked>
    tt.return
  }
}
