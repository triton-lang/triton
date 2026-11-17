// RUN: split-file %s %t
// RUN: triton-opt %t/nvmma-copy-contiguity.mlir -split-input-file --tritongpu-coalesce-async-copy | FileCheck %t/nvmma-copy-contiguity.mlir --check-prefix=HINT
// RUN: triton-opt %t/nvmma-copy-contiguity.mlir -split-input-file --tritongpu-coalesce-async-copy --allocate-shared-memory-nv='compute-capability=90 ptx-version=80' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=80' | FileCheck %t/nvmma-copy-contiguity.mlir --check-prefix=LOWER
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=80' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=80' 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_f32_nomask(%dest_ptrs: tensor<256x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %data: tensor<256xf32, #blocked>) {
    // CHECK-LABEL: atomic_add_f32_nomask
    // CHECK: llvm.atomicrmw fadd {{.*}} syncscope("device") acq_rel
    // CHECK: llvm.atomicrmw fadd {{.*}} syncscope("device") acq_rel
    // CHECK: llvm.atomicrmw fadd {{.*}} syncscope("device") acq_rel
    // CHECK: llvm.atomicrmw fadd {{.*}} syncscope("device") acq_rel
    %0 = tt.atomic_rmw fadd, acq_rel, gpu, %dest_ptrs, %data : (tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xf32, #blocked>) -> tensor<256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_f32_withmask(%dest_ptrs: tensor<256x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %data: tensor<256xf32, #blocked>, %mask: tensor<256xi1, #blocked> {tt.constancy = 2 : i32}) {
    // CHECK-LABEL: atomic_add_f32_withmask
    // CHECK: atom.global.gpu.acq_rel.add.f32
    // CHECK: atom.global.gpu.acq_rel.add.f32
    // CHECK: atom.global.gpu.acq_rel.add.f32
    // CHECK: atom.global.gpu.acq_rel.add.f32
    %0 = tt.atomic_rmw fadd, acq_rel, gpu, %dest_ptrs, %data, %mask : (tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xf32, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_f16_withmask(%dest_ptrs: tensor<256x!tt.ptr<f16>, #blocked> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %data: tensor<256xf16, #blocked>, %mask: tensor<256xi1, #blocked> {tt.constancy = 4 : i32}) {
    // CHECK-LABEL: atomic_add_f16_withmask
    // CHECK: atom.global.gpu.acq_rel.add.noftz.f16x2
    // CHECK: atom.global.gpu.acq_rel.add.noftz.f16x2
    // CHECK: atom.global.gpu.acq_rel.add.noftz.f16x2
    // CHECK: atom.global.gpu.acq_rel.add.noftz.f16x2
    %0 = tt.atomic_rmw fadd, acq_rel, gpu, %dest_ptrs, %data, %mask : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
    tt.return
  }
}

// -----

//--- nvmma-copy-contiguity.mlir

// HINT: #[[CLIPPED:.*]] = #ttg.blocked<{{.*}}sizePerThread = [1, 8]{{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
 // The clipped layout supports eight contiguous FP16 elements. A hint of 1
 // would cause lowering to attempt an illegal 2-byte cp.async transfer.
 // HINT: tt.func @nvmma_f16_hint_clamped
 // HINT: ttg.async_copy_global_to_local {{.*}} {contiguity = 8 : i32} : tensor<32x64x!tt.ptr<f16>, #[[CLIPPED]]>
 // LOWER-LABEL: llvm.func @nvmma_f16_hint_clamped
 // LOWER: nvvm.cp.async.shared.global {{.*}}, 16, cache = cg
 tt.func @nvmma_f16_hint_clamped(%input: tensor<32x64x!tt.ptr<f16>, #blocked>) {
  %view = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
  %token = ttg.async_copy_global_to_local %input, %view {contiguity = 16 : i32} : tensor<32x64x!tt.ptr<f16>, #blocked> -> <32x64xf16, #shared, #smem, mutable>
  ttg.async_commit_group tokens %token
  ttg.async_wait {num = 0 : i32}
  tt.return
 }
}

// -----

// HINT: #[[F32COPY:.*]] = #ttg.blocked<{{.*}}sizePerThread = [1, 1]{{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#k = #ttg.slice<{dim = 1, parent = #blocked}>
#n = #ttg.slice<{dim = 0, parent = #blocked}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
 // The old register axis spans adjacent columns; the clipped axis spans rows
 // with a stride of 128. Neither the old hint nor the old axis analysis applies.
 // HINT: tt.func @nvmma_tf32_row_major_b
 // HINT: ttg.async_copy_global_to_local {{%[^, ]+, %[^, ]+}} : tensor<32x128x!tt.ptr<f32>, #[[F32COPY]]>
 // HINT: ttg.async_copy_global_to_local {{%[^, ]+, %[^, ]+}} : tensor<32x128x!tt.ptr<f32>, #[[F32COPY]]>
 // LOWER-LABEL: llvm.func @nvmma_tf32_row_major_b
 // LOWER-NOT: nvvm.cp.async.shared.global {{.*}}, {{(8|16)}},
 // LOWER: nvvm.cp.async.shared.global {{.*}}, 4, cache = ca
 // LOWER-NOT: nvvm.cp.async.shared.global {{.*}}, {{(8|16)}},
 // LOWER: llvm.return
 tt.func @nvmma_tf32_row_major_b(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %rk = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #k>
  %rn = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #n>
  %stride = arith.constant dense<128> : tensor<32xi32, #k>
  %rows = arith.muli %rk, %stride : tensor<32xi32, #k>
  %rows2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32, #k> -> tensor<32x1xi32, #blocked>
  %cols2d = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32, #n> -> tensor<1x128xi32, #blocked>
  %rows_b = tt.broadcast %rows2d : tensor<32x1xi32, #blocked> -> tensor<32x128xi32, #blocked>
  %cols_b = tt.broadcast %cols2d : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
  %offsets = arith.addi %rows_b, %cols_b : tensor<32x128xi32, #blocked>
  %bases = tt.splat %base : !tt.ptr<f32> -> tensor<32x128x!tt.ptr<f32>, #blocked>
  %ptrs = tt.addptr %bases, %offsets : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
  %view0 = ttg.local_alloc : () -> !ttg.memdesc<32x128xf32, #shared, #smem, mutable>
  %view1 = ttg.local_alloc : () -> !ttg.memdesc<32x128xf32, #shared, #smem, mutable>
  %token0 = ttg.async_copy_global_to_local %ptrs, %view0 {contiguity = 4 : i32} : tensor<32x128x!tt.ptr<f32>, #blocked> -> <32x128xf32, #shared, #smem, mutable>
  %token1 = ttg.async_copy_global_to_local %ptrs, %view1 : tensor<32x128x!tt.ptr<f32>, #blocked> -> <32x128xf32, #shared, #smem, mutable>
  ttg.async_commit_group tokens %token0, %token1
  ttg.async_wait {num = 0 : i32}
  tt.return
 }
}
