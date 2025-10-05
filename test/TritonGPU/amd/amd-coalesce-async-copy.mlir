// RUN: triton-opt %s -split-input-file --tritonamdgpu-coalesce-async-copy=arch-generation-name=gfx950 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
// sizePerThread = [1] because we have no information about contiguity of src pointers
// CHECK: #[[$NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: async_copy_1d
tt.func @async_copy_1d(%input: tensor<1024x!tt.ptr<f32>, #blocked>,
    %view: !ttg.memdesc<1024xf32, #shared, #smem, mutable>) {
  // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<1024x!tt.ptr<f32>, #[[$NEW_BLOCKED]]>
  // CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<1024x!tt.ptr<f32>, #[[$NEW_BLOCKED]]>
  %token = ttg.async_copy_global_to_local %input, %view: tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.padded_shared<[4:+4] {order = [0], shape = [1024]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
// Padded encoding with an identity mapping does produce coalesced writes so we should not change the blocked encoding
// CHECK: #[[$NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: async_copy_with_padding
tt.func @async_copy_with_padding(%input: tensor<1024x!tt.ptr<f32>, #blocked>,
    %view: !ttg.memdesc<1024xf32, #shared, #smem, mutable>) {
  // CHECK-NOT: ttg.convert_layout
  // CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<1024x!tt.ptr<f32>, #[[$NEW_BLOCKED]]>
  %token = ttg.async_copy_global_to_local %input, %view: tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
// sizePerThread = [1, 1] because we have no information about contiguity of src pointers
// CHECK: #[[$NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK-LABEL: async_copy_2d
tt.func @async_copy_2d(%input: tensor<64x64x!tt.ptr<f32>, #blocked>,
    %view: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>) {
  // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64x64x!tt.ptr<f32>, #[[$NEW_BLOCKED]]>
  // CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<64x64x!tt.ptr<f32>, #[[$NEW_BLOCKED]]>
  %token = ttg.async_copy_global_to_local %input, %view: tensor<64x64x!tt.ptr<f32>, #blocked> -> <64x64xf32, #shared, #smem, mutable>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [1,2,2], order = [0,1,2]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0,1,2]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
// sizePerThread = [1, 1, 1] because we have no information about contiguity of src pointers
// CHECK: #[[$NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
// CHECK-LABEL: async_copy_3d
tt.func @async_copy_3d(%input: tensor<1024x1024x1024x!tt.ptr<f32>, #blocked>,
    %view: !ttg.memdesc<1024x1024x1024xf32, #shared, #smem, mutable>) {
  // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<1024x1024x1024x!tt.ptr<f32>, #[[$NEW_BLOCKED]]>
  // CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<1024x1024x1024x!tt.ptr<f32>, #[[$NEW_BLOCKED]]>
  %token = ttg.async_copy_global_to_local %input, %view: tensor<1024x1024x1024x!tt.ptr<f32>, #blocked> -> <1024x1024x1024xf32, #shared, #smem, mutable>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
// CHECK: #[[$NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK-LABEL: async_copy_with_mask_and_other
tt.func @async_copy_with_mask_and_other(%input: tensor<64x64x!tt.ptr<f32>, #blocked>,
    %view: !ttg.memdesc<64x64xf32, #shared, #smem, mutable>,
    %mask: tensor<64x64xi1, #blocked>,
    %other: tensor<64x64xf32, #blocked>) {
  // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64x64x!tt.ptr<f32>, #[[$NEW_BLOCKED]]>
  // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64x64xi1, #[[$NEW_BLOCKED]]>
  // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64x64xf32, #[[$NEW_BLOCKED]]>
  // CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<64x64x!tt.ptr<f32>, #[[$NEW_BLOCKED]]>
  %token = ttg.async_copy_global_to_local %input, %view mask %mask other %other: tensor<64x64x!tt.ptr<f32>, #blocked> -> <64x64xf32, #shared, #smem, mutable>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // Clip to vector size 2 (32bit) because we do not support 64 bit loads to lds
  // CHECK: #[[$NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: async_copy_vector_size_2
  tt.func public @async_copy_vector_size_2(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) {
    // We need the index calculation so AxisAnalysis sees that we can vectorize the load
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>

    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<32x64x!tt.ptr<f16>, #[[$NEW_BLOCKED]]>
    // CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<32x64x!tt.ptr<f16>, #[[$NEW_BLOCKED]]>
    %6 = ttg.async_copy_global_to_local %5, %arg2 : tensor<32x64x!tt.ptr<f16>, #blocked> -> <32x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // Clip to vector size 4 (128bit) which is the largest supported load width
  // CHECK: #[[$NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: async_copy_vector_size_8
  tt.func public @async_copy_vector_size_8(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<32x64xf16, #shared, #smem, mutable>) {
    // We need the index calculation so AxisAnalysis sees that we can vectorize the load based on the src contiguity
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>

    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<32x64x!tt.ptr<f16>, #[[$NEW_BLOCKED]]>
    // CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<32x64x!tt.ptr<f16>, #[[$NEW_BLOCKED]]>
    %6 = ttg.async_copy_global_to_local %5, %arg2 : tensor<32x64x!tt.ptr<f16>, #blocked> -> <32x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // The order of #blocked and #shared are different so we need to clip to 1 element
  // CHECK: #[[$NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: async_copy_different_order
  tt.func public @async_copy_different_order(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: i32 {tt.divisibility = 16 : i32},
                                %arg2: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>) {
    // We need the index calculation so AxisAnalysis sees that we can vectorize the load based on the src contiguity
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<32x64x!tt.ptr<f32>, #blocked>, tensor<32x64xi32, #blocked>

    // CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<32x64x!tt.ptr<f32>, #[[$NEW_BLOCKED]]>
    // CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<32x64x!tt.ptr<f32>, #[[$NEW_BLOCKED]]>
    %6 = ttg.async_copy_global_to_local %5, %arg2 : tensor<32x64x!tt.ptr<f32>, #blocked> -> <32x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
// The shared layout should not be changed
// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 2, maxPhase = 4, order = [1, 0]}>
// CHECK-NOT: #shared1
// CHECK-LABEL: async_copy_2d_swizzled
tt.func @async_copy_2d_swizzled(%input: tensor<64x64x!tt.ptr<f16>, #blocked>,
    %view: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>) {
  // CHECK: %{{.*}} = ttg.async_copy_global_to_local {{.*}} -> <64x64xf16, #shared, #smem, mutable>
  %token = ttg.async_copy_global_to_local %input, %view: tensor<64x64x!tt.ptr<f16>, #blocked> -> <64x64xf16, #shared, #smem, mutable>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.padded_shared<[4:+4] {order = [0], shape = [256]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
// Padded encoding with an identity mapping has vec=1 whereas the blocked has vec=4 so we need to rewrite it
// CHECK: #[[$NEW_SRC_ENCODING:.*]] = #ttg.linear
// CHECK-SAME{LITERAL}: register = [[64], [128]], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []
// CHECK-LABEL: async_copy_with_padding_different_vec
tt.func @async_copy_with_padding_different_vec(%input: tensor<256x!tt.ptr<f32>, #blocked>,
    %view: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
  // CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<256x!tt.ptr<f32>, #[[$NEW_SRC_ENCODING]]>
  %token = ttg.async_copy_global_to_local %input, %view: tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.padded_shared<[64:+4] {offset = [[1], [2], [4], [8], [64], [128], [16], [32]], block = []}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
// We rearrange in 4 blocks of 16 elements, check that we transfer it to the src encoding to write coalesced to lds
// CHECK: #[[$NEW_SRC_ENCODING:.*]] = #ttg.linear
// CHECK-SAME{LITERAL}: register = [], lane = [[1], [2], [4], [8], [64], [128]], warp = [[16], [32]], block = []
// CHECK-LABEL: async_copy_padded_layout_with_simple_rearanging
tt.func @async_copy_padded_layout_with_simple_rearanging(%input: tensor<256x!tt.ptr<f32>, #blocked>,
    %view: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
  // CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<256x!tt.ptr<f32>, #[[$NEW_SRC_ENCODING]]>
  %token = ttg.async_copy_global_to_local %input, %view: tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.padded_shared<[64:+4] {offset = [[1], [2], [4], [8], [16], [32], [256], [512], [64], [128]], block = []}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
// We rearrange in 4 blocks of 16 elements, check that we transfer it to the src encoding to write coalesced to lds
// CHECK: #[[$NEW_SRC_ENCODING:.*]] = #ttg.linear
// CHECK-SAME{LITERAL}: register = [[1], [2]], lane = [[4], [8], [16], [32], [256], [512]], warp = [[64], [128]], block = []
// CHECK-LABEL: async_copy_padded_layout_with_vectorization_and_rearanging
tt.func @async_copy_padded_layout_with_vectorization_and_rearanging(%input: tensor<1024x!tt.ptr<f32>, #blocked> {tt.contiguity = 4 : i32, tt.divisibility = 16 : i32},
    %view: !ttg.memdesc<1024xf32, #shared, #smem, mutable>) {
  // CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<1024x!tt.ptr<f32>, #[[$NEW_SRC_ENCODING]]>
  %token = ttg.async_copy_global_to_local %input, %view: tensor<1024x!tt.ptr<f32>, #blocked> -> <1024xf32, #shared, #smem, mutable>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.padded_shared<[64:+4] {offset = [[1], [2], [4], [8], [64], [16], [32]], block = []}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
// Check that we add a broadcast in case not each lane in the WG can read unique data
// CHECK: #[[$NEW_SRC_ENCODING:.*]] = #ttg.linear
// CHECK-SAME{LITERAL}: register = [], lane = [[1], [2], [4], [8], [64], [16]], warp = [[32], [0]], block = []
// CHECK-LABEL: async_copy_padded_layout_requiring_broadcasting
tt.func @async_copy_padded_layout_requiring_broadcasting(%input: tensor<128x!tt.ptr<f32>, #blocked>,
    %view: !ttg.memdesc<128xf32, #shared, #smem, mutable>) {
  // CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<128x!tt.ptr<f32>, #[[$NEW_SRC_ENCODING]]>
  %token = ttg.async_copy_global_to_local %input, %view: tensor<128x!tt.ptr<f32>, #blocked> -> <128xf32, #shared, #smem, mutable>
  tt.return
}
}
