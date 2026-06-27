// RUN: triton-opt %s -split-input-file -tritongpu-coalesce-async-copy | FileCheck %s

// CHECK: #[[NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64x16x!tt.ptr<i8>, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64x16xi1, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64x16xi8, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<64x16x!tt.ptr<i8>, #[[NEW_BLOCKED]]>
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_copy_i8(%input: tensor<64x16x!tt.ptr<i8>, #blocked>,
    %view: !ttg.memdesc<64x16xi8, #shared, #smem, mutable>,
    %mask: tensor<64x16xi1, #blocked>,
    %other: tensor<64x16xi8, #blocked>) {
  %token = ttg.async_copy_global_to_local %input, %view mask %mask other %other: tensor<64x16x!tt.ptr<i8>, #blocked> -> <64x16xi8, #shared, #smem, mutable>
  tt.return
}
}

// -----

// CHECK: #[[NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64x16x!tt.ptr<i8>, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<64x16x!tt.ptr<i8>, #[[NEW_BLOCKED]]>
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_copy_i8_no_mask_or_other(%input: tensor<64x16x!tt.ptr<i8>, #blocked>,
    %view: !ttg.memdesc<64x16xi8, #shared, #smem, mutable>) {
  %token = ttg.async_copy_global_to_local %input, %view : tensor<64x16x!tt.ptr<i8>, #blocked> -> <64x16xi8, #shared, #smem, mutable>
  tt.return
}
}

// -----

// CHECK: #[[NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64x!tt.ptr<i32>, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64xi1, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64xi32, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<64x!tt.ptr<i32>, #[[NEW_BLOCKED]]>
#blocked_small = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared_large_vec = #ttg.swizzled_shared<{vec = 64, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_copy_i32_small(%input: tensor<64x!tt.ptr<i32>, #blocked_small>,
    %view: !ttg.memdesc<64xi32, #shared_large_vec, #smem, mutable>,
    %mask: tensor<64xi1, #blocked_small>,
    %other: tensor<64xi32, #blocked_small>) {
  %token = ttg.async_copy_global_to_local %input, %view mask %mask other %other
      : tensor<64x!tt.ptr<i32>, #blocked_small> -> <64xi32, #shared_large_vec, #smem, mutable>
  tt.return
}
}

// -----

// CHECK: #[[F32_CP_ASYNC_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<1024x!tt.ptr<f32>, #[[F32_CP_ASYNC_BLOCKED]]>
// CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<1024x!tt.ptr<f32>, #[[F32_CP_ASYNC_BLOCKED]]>
#blocked_f32_256b = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared_f32_256b = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_copy_f32_clips_to_cp_async_limit(%input: tensor<1024x!tt.ptr<f32>, #blocked_f32_256b>,
    %view: !ttg.memdesc<1024xf32, #shared_f32_256b, #smem, mutable>) {
  %token = ttg.async_copy_global_to_local %input, %view
      : tensor<1024x!tt.ptr<f32>, #blocked_f32_256b> -> <1024xf32, #shared_f32_256b, #smem, mutable>
  tt.return
}
}

// -----

// CHECK: #[[F64_CP_ASYNC_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<1024x!tt.ptr<f64>, #[[F64_CP_ASYNC_BLOCKED]]>
// CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<1024x!tt.ptr<f64>, #[[F64_CP_ASYNC_BLOCKED]]>
#blocked_f64_256b = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared_f64_256b = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_copy_f64_clips_to_cp_async_limit(%input: tensor<1024x!tt.ptr<f64>, #blocked_f64_256b>,
    %view: !ttg.memdesc<1024xf64, #shared_f64_256b, #smem, mutable>) {
  %token = ttg.async_copy_global_to_local %input, %view
      : tensor<1024x!tt.ptr<f64>, #blocked_f64_256b> -> <1024xf64, #shared_f64_256b, #smem, mutable>
  tt.return
}
}
