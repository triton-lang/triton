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

// The source owns four contiguous FP32 elements, but the transposed NVMMA
// destination permits only scalar copies. Redistribute the copy source rather
// than issuing strided scalar copies with the original per-thread ownership.
// CHECK: #[[COPY:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK: tt.func @async_copy_nvmma_f32_transposed
// CHECK: ttg.convert_layout {{.*}} -> tensor<32x128x!tt.ptr<f32>, #[[COPY]]>
// CHECK: ttg.convert_layout {{.*}} -> tensor<32x128xi1, #[[COPY]]>
// CHECK: ttg.convert_layout {{.*}} -> tensor<32x128xf32, #[[COPY]]>
// CHECK: ttg.async_copy_global_to_local {{.*}} tensor<32x128x!tt.ptr<f32>, #[[COPY]]>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @async_copy_nvmma_f32_transposed(%input: tensor<32x128x!tt.ptr<f32>, #blocked>,
      %view: !ttg.memdesc<32x128xf32, #shared, #smem, mutable>,
      %mask: tensor<32x128xi1, #blocked>, %other: tensor<32x128xf32, #blocked>) {
    %token = ttg.async_copy_global_to_local %input, %view mask %mask other %other
        : tensor<32x128x!tt.ptr<f32>, #blocked> -> <32x128xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A contiguous, non-transposed NVMMA destination already permits vector copies.
// CHECK-LABEL: tt.func @async_copy_nvmma_f32_contiguous
// CHECK-NOT: ttg.convert_layout
// CHECK: ttg.async_copy_global_to_local
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @async_copy_nvmma_f32_contiguous(%input: tensor<128x32x!tt.ptr<f32>, #blocked>,
      %view: !ttg.memdesc<128x32xf32, #shared, #smem, mutable>) {
    %token = ttg.async_copy_global_to_local %input, %view
        : tensor<128x32x!tt.ptr<f32>, #blocked> -> <128x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A transposed FP32 NVMMA layout needs scalar copies at every swizzle width.
// CHECK: #[[CLIP0:.*]] = #ttg.blocked<{{.*}}sizePerThread = [1, 1]{{.*}}>
// CHECK: tt.func @nvmma_swizzle_0
// CHECK: ttg.convert_layout {{.*}} -> tensor<32x128x!tt.ptr<f32>, #[[CLIP0]]>
// CHECK: ttg.async_copy_global_to_local {{.*}} tensor<32x128x!tt.ptr<f32>, #[[CLIP0]]>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
 tt.func @nvmma_swizzle_0(%p: tensor<32x128x!tt.ptr<f32>, #blocked>, %s: !ttg.memdesc<32x128xf32, #shared, #smem, mutable>) {
  %t = ttg.async_copy_global_to_local %p, %s : tensor<32x128x!tt.ptr<f32>, #blocked> -> <32x128xf32, #shared, #smem, mutable>
  tt.return
 }
}

// -----

// A transposed FP32 NVMMA layout needs scalar copies at every swizzle width.
// CHECK: #[[CLIP32:.*]] = #ttg.blocked<{{.*}}sizePerThread = [1, 1]{{.*}}>
// CHECK: tt.func @nvmma_swizzle_32
// CHECK: ttg.convert_layout {{.*}} -> tensor<32x128x!tt.ptr<f32>, #[[CLIP32]]>
// CHECK: ttg.async_copy_global_to_local {{.*}} tensor<32x128x!tt.ptr<f32>, #[[CLIP32]]>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
 tt.func @nvmma_swizzle_32(%p: tensor<32x128x!tt.ptr<f32>, #blocked>, %s: !ttg.memdesc<32x128xf32, #shared, #smem, mutable>) {
  %t = ttg.async_copy_global_to_local %p, %s : tensor<32x128x!tt.ptr<f32>, #blocked> -> <32x128xf32, #shared, #smem, mutable>
  tt.return
 }
}

// -----

// A transposed FP32 NVMMA layout needs scalar copies at every swizzle width.
// CHECK: #[[CLIP64:.*]] = #ttg.blocked<{{.*}}sizePerThread = [1, 1]{{.*}}>
// CHECK: tt.func @nvmma_swizzle_64
// CHECK: ttg.convert_layout {{.*}} -> tensor<32x128x!tt.ptr<f32>, #[[CLIP64]]>
// CHECK: ttg.async_copy_global_to_local {{.*}} tensor<32x128x!tt.ptr<f32>, #[[CLIP64]]>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
 tt.func @nvmma_swizzle_64(%p: tensor<32x128x!tt.ptr<f32>, #blocked>, %s: !ttg.memdesc<32x128xf32, #shared, #smem, mutable>) {
  %t = ttg.async_copy_global_to_local %p, %s : tensor<32x128x!tt.ptr<f32>, #blocked> -> <32x128xf32, #shared, #smem, mutable>
  tt.return
 }
}

// -----

// Packed/padded NVMMA is deliberately not included in per-element clipping.
// CHECK-LABEL: tt.func @nvmma_fp4_unchanged
// CHECK-NOT: ttg.convert_layout
// CHECK: ttg.async_copy_global_to_local %{{.*}}, %{{.*}} : tensor<128x128x!tt.ptr<i8>, #{{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, fp4Padded = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
 tt.func @nvmma_fp4_unchanged(%p: tensor<128x128x!tt.ptr<i8>, #blocked>, %s: !ttg.memdesc<128x128xi8, #shared, #smem, mutable>) {
  %t = ttg.async_copy_global_to_local %p, %s : tensor<128x128x!tt.ptr<i8>, #blocked> -> <128x128xi8, #shared, #smem, mutable>
  tt.return
 }
}

// -----

// Packed/padded NVMMA is deliberately not included in per-element clipping.
// CHECK-LABEL: tt.func @nvmma_fp6_packed_unchanged
// CHECK-NOT: ttg.convert_layout
// CHECK: ttg.async_copy_global_to_local %{{.*}}, %{{.*}} : tensor<128x128x!tt.ptr<i8>, #{{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, padding = "b6x16_p32"}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
 tt.func @nvmma_fp6_packed_unchanged(%p: tensor<128x128x!tt.ptr<i8>, #blocked>, %s: !ttg.memdesc<128x128xi8, #shared, #smem, mutable>) {
  %t = ttg.async_copy_global_to_local %p, %s : tensor<128x128x!tt.ptr<i8>, #blocked> -> <128x128xi8, #shared, #smem, mutable>
  tt.return
 }
}

// -----

// Packed/padded NVMMA is deliberately not included in per-element clipping.
// CHECK-LABEL: tt.func @nvmma_fp6_byte_unchanged
// CHECK-NOT: ttg.convert_layout
// CHECK: ttg.async_copy_global_to_local %{{.*}}, %{{.*}} : tensor<128x128x!tt.ptr<i8>, #{{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, padding = "b6p2x16"}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
 tt.func @nvmma_fp6_byte_unchanged(%p: tensor<128x128x!tt.ptr<i8>, #blocked>, %s: !ttg.memdesc<128x128xi8, #shared, #smem, mutable>) {
  %t = ttg.async_copy_global_to_local %p, %s : tensor<128x128x!tt.ptr<i8>, #blocked> -> <128x128xi8, #shared, #smem, mutable>
  tt.return
 }
}

// -----

// Preserve the CTA partition/replication bases while redistributing registers.
// CHECK: #[[CGA_split_m:.*]] = #ttg.blocked<{{.*}}sizePerThread = [1, 1]{{.*}}CGALayout = {{\[\[1, 0\]\]}}{{.*}}>
// CHECK: tt.func @nvmma_split_m
// CHECK: ttg.async_copy_global_to_local {{.*}} tensor<64x128x!tt.ptr<f32>, #[[CGA_split_m]]>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32, CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
 tt.func @nvmma_split_m(%p: tensor<64x128x!tt.ptr<f32>, #blocked>, %s: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>) {
  %t = ttg.async_copy_global_to_local %p, %s : tensor<64x128x!tt.ptr<f32>, #blocked> -> <64x128xf32, #shared, #smem, mutable>
  tt.return
 }
}

// -----

// Preserve the CTA partition/replication bases while redistributing registers.
// CHECK: #[[CGA_split_n:.*]] = #ttg.blocked<{{.*}}sizePerThread = [1, 1]{{.*}}CGALayout = {{\[\[0, 1\]\]}}{{.*}}>
// CHECK: tt.func @nvmma_split_n
// CHECK: ttg.async_copy_global_to_local {{.*}} tensor<64x128x!tt.ptr<f32>, #[[CGA_split_n]]>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32, CGALayout = [[0, 1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
 tt.func @nvmma_split_n(%p: tensor<64x128x!tt.ptr<f32>, #blocked>, %s: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>) {
  %t = ttg.async_copy_global_to_local %p, %s : tensor<64x128x!tt.ptr<f32>, #blocked> -> <64x128xf32, #shared, #smem, mutable>
  tt.return
 }
}

// -----

// Preserve the CTA partition/replication bases while redistributing registers.
// CHECK: #[[CGA_replicated:.*]] = #ttg.blocked<{{.*}}sizePerThread = [1, 1]{{.*}}CGALayout = {{\[\[0, 0\]\]}}{{.*}}>
// CHECK: tt.func @nvmma_replicated
// CHECK: ttg.async_copy_global_to_local {{.*}} tensor<64x128x!tt.ptr<f32>, #[[CGA_replicated]]>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32, CGALayout = [[0, 0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
 tt.func @nvmma_replicated(%p: tensor<64x128x!tt.ptr<f32>, #blocked>, %s: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>) {
  %t = ttg.async_copy_global_to_local %p, %s : tensor<64x128x!tt.ptr<f32>, #blocked> -> <64x128xf32, #shared, #smem, mutable>
  tt.return
 }
}

// -----

// Pointer/mask contiguity may lower vector width, but a compatible shared
// layout alone must not trigger this shared-layout clipping rule.
// CHECK-LABEL: tt.func @nvmma_mask_contiguity_unchanged
// CHECK-NOT: ttg.convert_layout
// CHECK: ttg.async_copy_global_to_local {{.*}} mask
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
 tt.func @nvmma_mask_contiguity_unchanged(%p: tensor<128x32x!tt.ptr<f32>, #blocked>, %s: !ttg.memdesc<128x32xf32, #shared, #smem, mutable>, %mask: tensor<128x32xi1, #blocked>) {
  %t = ttg.async_copy_global_to_local %p, %s mask %mask {contiguity = 1 : i32} : tensor<128x32x!tt.ptr<f32>, #blocked> -> <128x32xf32, #shared, #smem, mutable>
  tt.return
 }
}
