// RUN: triton-opt %s -allow-unregistered-dialect --tritongpu-allocate-warp-groups | FileCheck %s --check-prefix=ALLOC
// RUN: triton-opt %s -allow-unregistered-dialect --tritongpu-allocate-warp-groups --convert-warp-specialize-to-llvm -mlir-print-local-scope | FileCheck %s --check-prefix=LOWER

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

  // ALLOC-LABEL: llvm.func @warp_specialize_register_trampoline
  // LOWER-LABEL: llvm.func @warp_specialize_register_trampoline
  llvm.func @warp_specialize_register_trampoline()
      attributes {allocation.offset = 0 : i32} {
    // Each one-warp partition shares a warpgroup with another partition and
    // allocator-added padding. States with the same op and target share a
    // setter; equal targets in different ops remain separate.
    // ALLOC: actualRegisters = array<i32: 168, 48, 48, 24, 48>
    // ALLOC-SAME: requestedRegisters = array<i32: 32, 48, 16, 16>
    ttg.warp_specialize() attributes {
      allocation.offset = 0 : i32,
      requestedRegisters = array<i32: 32, 48>
    }
    default {
      "default0"() : () -> ()
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      "worker0"() : () -> ()
      ttg.warp_return
    }
    partition1() num_warps(1) {
      "worker1"() : () -> ()
      ttg.warp_return
    } : () -> ()

    // The second op's 24-register padding target must use a distinct setter
    // from the first op's 24-register padding target.
    // ALLOC: actualRegisters = array<i32: 152, 64, 64, 24, 64>
    // ALLOC-SAME: requestedRegisters = array<i32: 64, 32, 16, 16>
    ttg.warp_specialize() attributes {
      allocation.offset = 0 : i32,
      requestedRegisters = array<i32: 64, 32>
    }
    default {
      "default1"() : () -> ()
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      "worker2"() : () -> ()
      ttg.warp_return
    }
    partition1() num_warps(1) {
      "worker3"() : () -> ()
      ttg.warp_return
    } : () -> ()

    // LOWER: llvm.switch {{.*}} : i8, {{\^.*}} [
    // LOWER-NEXT: 0: [[REGS48:\^.*]],
    // LOWER-NEXT: 1: [[REGS48]],
    // LOWER-NEXT: 2: [[REGS24_FIRST:\^.*]],
    // LOWER-NEXT: 3: [[REGS48]],
    // LOWER-NEXT: 4: [[REGS64:\^.*]],
    // LOWER-NEXT: 5: [[REGS64]],
    // LOWER-NEXT: 6: [[REGS24_SECOND:\^.*]],
    // LOWER-NEXT: 7: [[REGS64]],
    // LOWER-NEXT: 8: [[EXIT:\^.*]]
    // LOWER: [[REGS48]]:
    // LOWER-NEXT: nvvm.setmaxregister increase 48
    // LOWER-NEXT: llvm.br [[PARTITION_DISPATCH:\^.*]]
    // LOWER: [[REGS24_FIRST]]:
    // LOWER-NEXT: nvvm.setmaxregister increase 24
    // LOWER-NEXT: llvm.br [[PARTITION_DISPATCH]]
    // LOWER: [[REGS64]]:
    // LOWER-NEXT: nvvm.setmaxregister increase 64
    // LOWER-NEXT: llvm.br [[PARTITION_DISPATCH]]
    // LOWER: [[REGS24_SECOND]]:
    // LOWER-NEXT: nvvm.setmaxregister increase 24
    // LOWER-NEXT: llvm.br [[PARTITION_DISPATCH]]
    // LOWER: [[PARTITION_DISPATCH]]:
    // LOWER-NEXT: llvm.switch {{.*}} : i8, [[EXIT]] [
    // LOWER-NEXT: 0: [[PARTITION0:\^.*]],
    // LOWER-NEXT: 1: [[PARTITION1:\^.*]],
    // LOWER-NEXT: 2: [[PARTITION2:\^.*]],
    // LOWER-NEXT: 3: [[PARTITION3:\^.*]],
    // LOWER-NEXT: 4: [[PARTITION4:\^.*]],
    // LOWER-NEXT: 5: [[PARTITION5:\^.*]],
    // LOWER-NEXT: 6: [[PARTITION6:\^.*]],
    // LOWER-NEXT: 7: [[PARTITION7:\^.*]]
    // LOWER: [[PARTITION0]]:
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: "worker0"()
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.br {{\^.*}}
    // LOWER: [[PARTITION1]]:
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: "worker1"()
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.br {{\^.*}}
    // LOWER: [[PARTITION2]]:
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.br {{\^.*}}
    // LOWER: [[PARTITION3]]:
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.br {{\^.*}}
    // LOWER: [[PARTITION4]]:
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: "worker2"()
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.br {{\^.*}}
    // LOWER: [[PARTITION5]]:
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: "worker3"()
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.br {{\^.*}}
    // LOWER: [[PARTITION6]]:
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.br {{\^.*}}
    // LOWER: [[PARTITION7]]:
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"({{.*}})
    // LOWER-NOT: nvvm.setmaxregister
    // LOWER: llvm.br {{\^.*}}
    llvm.return
  }
}
