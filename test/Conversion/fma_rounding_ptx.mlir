// RUN: split-file %s %t
// RUN: triton-opt %t/common.mlir --allocate-shared-memory-nv='compute-capability=90 ptx-version=83' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_90 -mattr=+ptx83 | FileCheck %s --check-prefix=CHECK
// RUN: triton-opt %t/common.mlir --allocate-shared-memory-nv='compute-capability=100 ptx-version=87' --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_100 -mattr=+ptx87 | FileCheck %s --check-prefix=CHECK

//--- common.mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_rz(%ptr: !tt.ptr<f32>, %arg0: f32, %arg1: f32, %arg2: f32) {
    // CHECK: fma.rz.f32
    %0 = math.fma %arg0, %arg1, %arg2 toward_zero : f32
    tt.store %ptr, %0 : !tt.ptr<f32>
    tt.return
  }
  tt.func public @fma_rd(%ptr: !tt.ptr<f32>, %arg0: f32, %arg1: f32, %arg2: f32) {
    // CHECK: fma.rm.f32
    %0 = math.fma %arg0, %arg1, %arg2 downward : f32
    tt.store %ptr, %0 : !tt.ptr<f32>
    tt.return
  }
  tt.func public @fma_ru(%ptr: !tt.ptr<f64>, %arg0: f64, %arg1: f64, %arg2: f64) {
    // CHECK: fma.rp.f64
    %0 = math.fma %arg0, %arg1, %arg2 upward : f64
    tt.store %ptr, %0 : !tt.ptr<f64>
    tt.return
  }
}
