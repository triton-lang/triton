// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory | FileCheck %s

#blocked1 = #ttg.blocked<{sizePerThread = [8, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

// This test checks padding based converter.
//
// Converter allocates temporary buffer, stores and reads parts or tensor in few transactions, which are named repeats.
// Size of temporary buffer is computed using the following algorithm:
// - get CTA tile shape of blocked1 layout: [8*8*4, 4*8*1] = [256, 32]
// - get CTA tile shape of blocked2 layout: [1*8*4, 1*8*1] = [32, 8]
// - compute common tile shape is [max(256, 32), max(32, 8)] = [256, 32].
// - pad fastest dimension(same as output layout, 1 in this case) with size of memory access to reduce bank conflicts. 16 bytes in this case.
//
// Therefore total memory consuption for scratch buffer is 256*(32 * 4(size of one element) + 16(padding)) = 36864 bytes
//
// For implementation see mlir::triton::getNumScratchElemsPaddedCvt function.

// CHECK: ttg.shared = 36864 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: @convert_layout_padded
tt.func @convert_layout_padded(%arg0: tensor<256x256xi32, #blocked1>) {
  // CHECK-NEXT: allocation.offset = 0 : i32
  %0 = ttg.convert_layout %arg0 {amdgpu.use_padded_scratch_shmem} : tensor<256x256xi32, #blocked1> -> tensor<256x256xi32, #blocked2>
  tt.return
}

}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [8, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

// This test checks swizzling based converter.
//
// Swizzling converter tries to find swizzling pattern, which provides widest load and store instructions and avoids as much back conflicts as possible.
// Current converter implementation decides that best swizzling patter requires allocation of tile with shape [256, 128], which takes 256*128*4(size of one element) = 131072 bytes
//
// For implementation see mlir::triton::getNumScratchElemsSwizzledCvt function,
// in particular mlir::triton::gpu::optimalSwizzling to get shape of repeat tile.

// CHECK: ttg.shared = 131072 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: @convert_layout_swizzled
tt.func @convert_layout_swizzled(%arg0: tensor<256x256xi32, #blocked1>) {
  // CHECK-NEXT: allocation.offset = 0 : i32
  %0 = ttg.convert_layout %arg0 : tensor<256x256xi32, #blocked1> -> tensor<256x256xi32, #blocked2>
  tt.return
}

}
