// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx90a  | FileCheck %s --check-prefixes=CHECK,ARITH
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942  | FileCheck %s --check-prefixes=CHECK,ARITH
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950  | FileCheck %s --check-prefixes=CHECK,ARITH
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 | FileCheck %s --check-prefixes=CHECK,ARITH
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 | FileCheck %s --check-prefixes=CHECK,WAVEID
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s --check-prefixes=CHECK,WAVEID

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: @wave_id
tt.func public @wave_id() {
  //       ARITH: %[[C64:.+]] = llvm.mlir.constant(64 : i32) : i32
  //  ARITH-NEXT: %[[IDX:.+]] = rocdl.workitem.id.x : i32
  //  ARITH-NEXT: %[[C63:.+]] = llvm.mlir.constant(63 : i32) : i32
  //  ARITH-NEXT: %[[AND:.+]] = llvm.and %[[IDX]], %[[C63]] : i32
  //  ARITH-NEXT: %[[DIV:.+]] = llvm.udiv %[[AND]], %[[C64]] : i32
  //  ARITH-NEXT: %{{.+}} = rocdl.readfirstlane %[[DIV]] : i32

  // WAVEID-NEXT: rocdl.wave.id
  //       CHECK: scf.for

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %1 = "ttg.warp_id"() : () -> i32
    scf.yield
  }
  tt.return
}

}
