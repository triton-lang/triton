// RUN: triton-opt %s -split-input-file --tritongpu-allocate-warp-groups | FileCheck %s --check-prefix=ALLOC
// RUN: triton-opt %s -split-input-file --tritongpu-allocate-warp-groups --convert-warp-specialize-to-llvm | FileCheck %s --check-prefix=LOWER

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

// -----

module attributes {
  "ttg.num-warps" = 4 : i32,
  ttg.target = "cuda:100"
} {
  llvm.mlir.global external @global_smem()
      {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // ALLOC-LABEL: llvm.func @cap_and_elide_noop_setmaxnreg
  // LOWER-LABEL: llvm.func @cap_and_elide_noop_setmaxnreg
  llvm.func @cap_and_elide_noop_setmaxnreg(%run: i1)
      attributes {allocation.offset = 0 : i32} {
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.return
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.return
    llvm.cond_br %run, ^launch, ^exit
  ^launch:
    // ALLOC: actualRegisters = array<i32: 256, 256>
    ttg.warp_specialize() attributes {
      allocation.offset = 0 : i32,
      requestedRegisters = array<i32: 24>
    }
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %0 = llvm.mlir.constant(0 : i32) : i32
      ttg.warp_return
    } : () -> ()
    llvm.br ^exit
  ^exit:
    llvm.return
  }
}

// -----

module attributes {
  ttg.maxnreg = 168 : i32,
  "ttg.num-warps" = 4 : i32,
  ttg.target = "cuda:100"
} {
  llvm.mlir.global external @global_smem()
      {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // ALLOC-LABEL: llvm.func @cap_and_redistribute_to_workers
  // LOWER-LABEL: llvm.func @cap_and_redistribute_to_workers
  llvm.func @cap_and_redistribute_to_workers()
      attributes {allocation.offset = 0 : i32} {
    // ALLOC: actualRegisters = array<i32: 256, 200, 48>
    // LOWER: nvvm.setmaxregister decrease 120
    // LOWER-NOT: nvvm.setmaxregister {{.*}} 88
    // LOWER: nvvm.setmaxregister increase 200
    // LOWER: nvvm.setmaxregister decrease 120
    // LOWER: nvvm.setmaxregister decrease 48
    // LOWER: nvvm.setmaxregister increase 120
    // LOWER: nvvm.setmaxregister increase 256
    // LOWER: llvm.return
    ttg.warp_specialize() attributes {
      allocation.offset = 0 : i32,
      requestedRegisters = array<i32: 88, 24>
    }
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %0 = llvm.mlir.constant(0 : i32) : i32
      ttg.warp_return
    }
    partition1() num_warps(4) {
      %0 = llvm.mlir.constant(0 : i32) : i32
      ttg.warp_return
    } : () -> ()
    llvm.return
  }
}
