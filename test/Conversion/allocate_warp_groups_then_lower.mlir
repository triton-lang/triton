// RUN: triton-opt %s --tritongpu-allocate-warp-groups | FileCheck %s --check-prefix=ALLOC
// RUN: triton-opt %s --tritongpu-allocate-warp-groups --convert-warp-specialize-to-llvm | FileCheck %s --check-prefix=LOWER \
// RUN:   --implicit-check-not='nvvm.setmaxregister {{.*}} 16' --implicit-check-not='nvvm.setmaxregister increase 24'

module attributes {
  ttg.maxnreg = 80 : i32,
  "ttg.num-warps" = 4 : i32,
  ttg.target = "cuda:100"
} {
  llvm.mlir.global external @global_smem()
      {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // ALLOC-LABEL: llvm.func @padding_uses_legal_register_minimum
  // LOWER-LABEL: llvm.func @padding_uses_legal_register_minimum
  llvm.func @padding_uses_legal_register_minimum()
      attributes {allocation.offset = 0 : i32} {
    // ALLOC: actualRegisters = array<i32: 80, 80>
    ttg.warp_specialize() attributes {
      allocation.offset = 0 : i32,
      requestedRegisters = array<i32: 80>
    }
    default {
      ttg.warp_yield
    }
    partition0() num_warps(8) {
      ttg.warp_return
    } : () -> ()

    // ALLOC: actualRegisters = array<i32: 136, 80, 24, 80, 80>
    // ALLOC-SAME: requestedRegisters = array<i32: 80, 16, 16, 16>
    ttg.warp_specialize() attributes {
      allocation.offset = 0 : i32,
      requestedRegisters = array<i32: 80>
    }
    default {
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()

    // The one-warp worker and its two padding partitions share one cap change.
    // The other quartet stays at the idle cap of 24.
    // LOWER: llvm.switch {{.*}} : i8
    // LOWER: 2: [[SPLIT:\^bb[0-9]+]]
    // LOWER: [[SPLIT]]:
    // LOWER-NEXT: nvvm.setmaxregister increase 80
    // LOWER-NEXT: llvm.switch {{.*}} : i32
    // LOWER: [[RESTORE:\^bb[0-9]+]]:
    // LOWER: nvvm.barrier
    // LOWER-NEXT: nvvm.setmaxregister decrease 24
    // LOWER-COUNT-3: llvm.br [[RESTORE]]
    // LOWER: llvm.return
    llvm.return
  }
}
