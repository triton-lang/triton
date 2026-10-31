// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=83' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83' | FileCheck %s --check-prefix=NV
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s --check-prefix=AMD
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s --check-prefix=AMD
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s --check-prefix=AMD

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // NV-LABEL: @fma_rn
  // AMD-LABEL: @fma_rn
  tt.func public @fma_rn(%ptr: !tt.ptr<f32>, %arg0: f32, %arg1: f32, %arg2: f32) {
    // NV: llvm.intr.fma
    // AMD: llvm.intr.fma
    %0 = math.fma %arg0, %arg1, %arg2 : f32
    tt.store %ptr, %0 : !tt.ptr<f32>
    tt.return
  }

  // NV-LABEL: @fma_rz
  // AMD-LABEL: @fma_rz
  tt.func public @fma_rz(%ptr: !tt.ptr<f32>, %arg0: f32, %arg1: f32, %arg2: f32) {
    // NV: nvvm.fma {{.*}} rnd = <rz> : f32
    // AMD: llvm.call_intrinsic "llvm.amdgcn.s.setreg"
    // AMD: llvm.intr.experimental.constrained.fma {{.*}} towardzero ignore : f32
    // AMD: llvm.call_intrinsic "llvm.amdgcn.s.setreg"
    %0 = math.fma %arg0, %arg1, %arg2 toward_zero : f32
    tt.store %ptr, %0 : !tt.ptr<f32>
    tt.return
  }

  // NV-LABEL: @fma_rd
  // AMD-LABEL: @fma_rd
  tt.func public @fma_rd(%ptr: !tt.ptr<f64>, %arg0: f64, %arg1: f64, %arg2: f64) {
    // NV: nvvm.fma {{.*}} rnd = <rm> : f64
    // AMD: llvm.call_intrinsic "llvm.amdgcn.s.setreg"
    // AMD: llvm.intr.experimental.constrained.fma {{.*}} downward ignore : f64
    // AMD: llvm.call_intrinsic "llvm.amdgcn.s.setreg"
    %0 = math.fma %arg0, %arg1, %arg2 downward : f64
    tt.store %ptr, %0 : !tt.ptr<f64>
    tt.return
  }

  // NV-LABEL: @fma_ru
  // AMD-LABEL: @fma_ru
  tt.func public @fma_ru(%ptr: !tt.ptr<f32>, %arg0: f32, %arg1: f32, %arg2: f32) {
    // NV: nvvm.fma {{.*}} rnd = <rp> : f32
    // AMD: llvm.call_intrinsic "llvm.amdgcn.s.setreg"
    // AMD: llvm.intr.experimental.constrained.fma {{.*}} upward ignore : f32
    // AMD: llvm.call_intrinsic "llvm.amdgcn.s.setreg"
    %0 = math.fma %arg0, %arg1, %arg2 upward : f32
    tt.store %ptr, %0 : !tt.ptr<f32>
    tt.return
  }
}
