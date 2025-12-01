// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942  | FileCheck %s --check-prefixes=CHECK,GFX9
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx950  | FileCheck %s --check-prefixes=CHECK,GFX9
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx1200 | FileCheck %s --check-prefixes=CHECK,GFX12
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx1250 | FileCheck %s --check-prefixes=CHECK,GFX12

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: @wave_id
tt.func public @wave_id() {
  //       GFX9: %[[C64:.+]] = llvm.mlir.constant(64 : i32) : i32
  //  GFX9-NEXT: %[[IDX:.+]] = rocdl.workitem.id.x : i32
  //  GFX9-NEXT: %[[C63:.+]] = llvm.mlir.constant(63 : i32) : i32
  //  GFX9-NEXT: %[[AND:.+]] = llvm.and %[[IDX]], %[[C63]] : i32
  //  GFX9-NEXT: %[[DIV:.+]] = llvm.udiv %[[AND]], %[[C64]] : i32
  //  GFX9-NEXT: %{{.+}} = rocdl.readfirstlane %[[DIV]] : i32

  // GFX12-NEXT: llvm.call_intrinsic "llvm.amdgcn.wave.id"
  //      CHECK: scf.for

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %1 = "ttg.warp_id"() : () -> i32
    scf.yield
  }
  tt.return
}

}
