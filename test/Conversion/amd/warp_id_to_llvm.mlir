// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx90a  | FileCheck %s --check-prefix=GFX9
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942  | FileCheck %s --check-prefix=GFX9
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950  | FileCheck %s --check-prefix=GFX9
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 | FileCheck %s --check-prefix=GFX11
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1170 | FileCheck %s --check-prefix=GFX11
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 | FileCheck %s --check-prefix=GFX12
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s --check-prefix=GFX12

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, "ttg.threads-per-warp" = 64 : i32} {

// GFX9-LABEL: @wave_id_wave64
tt.func public @wave_id_wave64() {
  //      GFX9: %[[C64:.+]] = llvm.mlir.constant(64 : i32) : i32
  // GFX9-NEXT: %[[IDX:.+]] = rocdl.workitem.id.x : i32
  // GFX9-NEXT: %[[C63:.+]] = llvm.mlir.constant(63 : i32) : i32
  // GFX9-NEXT: %[[AND:.+]] = llvm.and %[[IDX]], %[[C63]] : i32
  // GFX9-NEXT: %[[DIV:.+]] = llvm.udiv %[[AND]], %[[C64]] : i32
  // GFX9-NEXT: %{{.+}} = rocdl.readfirstlane %[[DIV]] : i32
  //      GFX9: llvm.br ^[[LOOP:bb[0-9]+]](
  // GFX9-NEXT: ^[[LOOP]](

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %1 = "ttg.warp_id"() : () -> i32
    scf.yield
  }
  tt.return
}

}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, "ttg.threads-per-warp" = 32 : i32} {

//  GFX11-LABEL: @wave_id_wave32
//  GFX12-LABEL: @wave_id_wave32
tt.func public @wave_id_wave32() {
  //       GFX11: %[[C32:.+]] = llvm.mlir.constant(32 : i32) : i32
  //  GFX11-NEXT: %[[IDX:.+]] = rocdl.workitem.id.x : i32
  //  GFX11-NEXT: %[[C31:.+]] = llvm.mlir.constant(31 : i32) : i32
  //  GFX11-NEXT: %[[AND:.+]] = llvm.and %[[IDX]], %[[C31]] : i32
  //  GFX11-NEXT: %[[DIV:.+]] = llvm.udiv %[[AND]], %[[C32]] : i32
  //  GFX11-NEXT: %{{.+}} = rocdl.readfirstlane %[[DIV]] : i32
  //  GFX11-NOT: rocdl.wave.id
  // GFX12-NEXT: rocdl.wave.id
  //       GFX11: llvm.br ^[[LOOP:bb[0-9]+]](
  //  GFX11-NEXT: ^[[LOOP]](
  //       GFX12: llvm.br ^[[LOOP:bb[0-9]+]](
  //  GFX12-NEXT: ^[[LOOP]](

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %1 = "ttg.warp_id"() : () -> i32
    scf.yield
  }
  tt.return
}

}
