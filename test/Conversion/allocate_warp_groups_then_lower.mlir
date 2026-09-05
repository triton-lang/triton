// RUN: triton-opt %s --tritongpu-allocate-warp-groups | FileCheck %s --check-prefix=ALLOC
// RUN: triton-opt %s --tritongpu-allocate-warp-groups --convert-warp-specialize-to-llvm | FileCheck %s --check-prefix=LOWER

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

    // ALLOC: actualRegisters = array<i32: 136, 80, 24>
    // ALLOC-SAME: requestedRegisters = array<i32: 80, 16>
    ttg.warp_specialize() attributes {
      allocation.offset = 0 : i32,
      requestedRegisters = array<i32: 80>
    }
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()

    // LOWER-NOT: nvvm.setmaxregister {{.*}} 16
    // LOWER: nvvm.setmaxregister increase 24
    // LOWER: llvm.return
    llvm.return
  }
}
