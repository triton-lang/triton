// RUN: triton-opt %s -split-input-file -tritongpu-coalesce-async-copy | FileCheck %s

// CHECK: #[[NEW_BLOCKED:.*]] = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: %{{.*}} = triton_gpu.convert_layout %{{.*}} : {{.*}} -> tensor<64x16x!tt.ptr<i8>, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = triton_gpu.convert_layout %{{.*}} : {{.*}} -> tensor<64x16xi1, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = triton_gpu.convert_layout %{{.*}} : {{.*}} -> tensor<64x16xi8, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = triton_gpu.async_copy_global_to_local %{{.*}}: tensor<64x16x!tt.ptr<i8>, #[[NEW_BLOCKED]]>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
tt.func @async_copy_i8(%input: tensor<64x16x!tt.ptr<i8>, #blocked>,
    %view: !triton_gpu.memdesc<64x16xi8, #shared, #triton_gpu.shared_memory, mutable>,
    %mask: tensor<64x16xi1, #blocked>,
    %other: tensor<64x16xi8, #blocked>) {
  %token = triton_gpu.async_copy_global_to_local %input, %view mask %mask other %other: tensor<64x16x!tt.ptr<i8>, #blocked> -> <64x16xi8, #shared, #triton_gpu.shared_memory, mutable>
  tt.return
}
}

// -----

// CHECK: #[[NEW_BLOCKED:.*]] = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: %{{.*}} = triton_gpu.convert_layout %{{.*}} : {{.*}} -> tensor<64x16x!tt.ptr<i8>, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = triton_gpu.async_copy_global_to_local %{{.*}}: tensor<64x16x!tt.ptr<i8>, #[[NEW_BLOCKED]]>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
tt.func @async_copy_i8_no_mask_or_other(%input: tensor<64x16x!tt.ptr<i8>, #blocked>,
    %view: !triton_gpu.memdesc<64x16xi8, #shared, #triton_gpu.shared_memory, mutable>) {
  %token = triton_gpu.async_copy_global_to_local %input, %view : tensor<64x16x!tt.ptr<i8>, #blocked> -> <64x16xi8, #shared, #triton_gpu.shared_memory, mutable>
  tt.return
}
}
