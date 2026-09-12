// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1030 --convert-builtin-func-to-llvm 2>&1 | FileCheck %s --check-prefix=WAVE32
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 --convert-builtin-func-to-llvm 2>&1 | FileCheck %s --check-prefix=WAVE64

#l32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1030", "ttg.threads-per-warp" = 32 : i32} {

// WAVE32-LABEL: @bool_extui_w32
// WAVE32: rocdl.ballot
// WAVE32: rocdl.mbcnt.lo
// WAVE32-NOT: rocdl.mbcnt.hi
tt.func private @bool_extui_w32(%arg0: tensor<32xi1, #l32>) -> tensor<32xi32, #l32> {
  %b = arith.extui %arg0 : tensor<32xi1, #l32> to tensor<32xi32, #l32>
  %0 = "tt.scan"(%b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %c: i32):
    %1 = arith.addi %a, %c : i32
    tt.scan.return %1 : i32
  }) : (tensor<32xi32, #l32>) -> tensor<32xi32, #l32>
  tt.return %0 : tensor<32xi32, #l32>
}

}

// -----

#l64 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// WAVE32-LABEL: @bool_extui_w64
// WAVE64-LABEL: @bool_extui_w64
// WAVE64: rocdl.ballot
// WAVE64: rocdl.mbcnt.lo
// WAVE64: rocdl.mbcnt.hi
tt.func private @bool_extui_w64(%arg0: tensor<64xi1, #l64>) -> tensor<64xi32, #l64> {
  %b = arith.extui %arg0 : tensor<64xi1, #l64> to tensor<64xi32, #l64>
  %0 = "tt.scan"(%b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %c: i32):
    %1 = arith.addi %a, %c : i32
    tt.scan.return %1 : i32
  }) : (tensor<64xi32, #l64>) -> tensor<64xi32, #l64>
  tt.return %0 : tensor<64xi32, #l64>
}

}
