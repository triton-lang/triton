// RUN: split-file %s %t
// RUN: triton-opt %t/nvmma-copy-contiguity.mlir --tritongpu-coalesce-async-copy | FileCheck %t/nvmma-copy-contiguity.mlir --check-prefix=HINT
// RUN: triton-opt %t/nvmma-copy-contiguity.mlir --tritongpu-coalesce-async-copy --allocate-shared-memory-nv='compute-capability=90 ptx-version=80' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=80' | FileCheck %t/nvmma-copy-contiguity.mlir --check-prefix=LOWER
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
 // HINT: tt.func @nvmma_f16_hint
 // HINT: ttg.async_copy_global_to_local {{.*}} {contiguity = 8 : i32} : tensor<32x64x!tt.ptr<f16>, #[[CLIPPED]]>
 // LOWER-LABEL: llvm.func @nvmma_f16_hint
 // LOWER: nvvm.cp.async.shared.global {{.*}}, 16, cache = cg
 tt.func @nvmma_f16_hint(%input: tensor<32x64x!tt.ptr<f16>, #blocked>) {
  %view = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, #shared, #smem, mutable>
  %token = ttg.async_copy_global_to_local %input, %view {contiguity = 8 : i32} : tensor<32x64x!tt.ptr<f16>, #blocked> -> <32x64xf16, #shared, #smem, mutable>
  ttg.async_commit_group tokens %token
  ttg.async_wait {num = 0 : i32}
  tt.return
 }
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
