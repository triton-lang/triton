// RUN: triton-opt %s -split-input-file --test-print-membar --convert-triton-gpu-to-llvm --convert-warp-specialize-to-llvm --convert-nv-gpu-to-llvm -allow-unregistered-dialect | FileCheck %s

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1, CGALayout = [[0, 0]]>

module attributes {"ttg.target" = "cuda:103", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, ttg.tensor_memory_size = 128 : i32, "ttng.two-ctas" = true} {
  // CHECK-LABEL: @automatic_tmem_lifecycle
  // CHECK: nvvm.cluster.arrive
  // CHECK-NEXT: nvvm.cluster.wait
  // CHECK-NEXT: {{.*}}tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32
  // CHECK: tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned
  // CHECK: nvvm.cluster.arrive
  // CHECK-NEXT: nvvm.cluster.wait
  // CHECK: tcgen05.dealloc.cta_group::2.sync.aligned.b32
  // CHECK-NOT: nvg.tensor_memory_base
  llvm.func @automatic_tmem_lifecycle() attributes {allocation.offset = 0 : i32, nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 256>} {
    ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %0 = nvg.tensor_memory_base
      %1 = llvm.ptrtoint %0 : !llvm.ptr<6> to i32
      "use"(%1) : (i32) -> ()
      ttg.warp_return
    } : () -> ()
    llvm.return
  }

  // CHECK-LABEL: @tmem_pipeline_stage_subslice_index
  // CHECK-DAG: llvm.mlir.constant(128 : i32)
  // CHECK-DAG: llvm.mlir.constant(64 : i32)
  // CHECK: llvm.add
  // CHECK: llvm.mul
  // CHECK: llvm.add
  tt.func private @tmem_pipeline_stage_subslice_index(%parent: !ttg.memdesc<5x128x64xf32, #tmem, #ttng.tensor_memory, mutable>, %index: i32) -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> attributes {"ws_num_warps" = 4 : i32} {
    %stages = ttng.tmem_subslice %parent {offset = 2 : i32, dim = 0 : i32} : !ttg.memdesc<5x128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x64xf32, #tmem, #ttng.tensor_memory, mutable, 5x128x64>
    %view = ttg.memdesc_index %stages[%index] : !ttg.memdesc<2x128x64xf32, #tmem, #ttng.tensor_memory, mutable, 5x128x64> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return %view : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
  }

}

// -----

module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 4 : i32, ttg.tensor_memory_size = 128 : i32} {
  // CHECK-LABEL: llvm.func internal @tmem_leaf(
  // CHECK-SAME: %[[LEAF_VALUE:.*]]: i32 {
  // CHECK-SAME: some.attr = "preserved"
  // CHECK-SAME: }, %[[LEAF_BASE:.*]]: !llvm.ptr<6>)
  // CHECK-NOT: tcgen05.alloc
  // CHECK-NOT: nvvm.barrier0
  // CHECK-NOT: nvg.tensor_memory_base
  // CHECK: llvm.ptrtoint %[[LEAF_BASE]] : !llvm.ptr<6> to i32
  llvm.func internal @tmem_leaf(%value: i32 {some.attr = "preserved"}) attributes {passthrough = ["noinline"], sym_visibility = "private"} {
    %base = nvg.tensor_memory_base
    %base_i32 = llvm.ptrtoint %base : !llvm.ptr<6> to i32
    "use"(%value, %base_i32) : (i32, i32) -> ()
    llvm.return
  }

  // CHECK-LABEL: llvm.func internal @tmem_middle(
  // CHECK-SAME: %[[MIDDLE_VALUE:.*]]: i32
  // CHECK-SAME: %[[MIDDLE_BASE:.*]]: !llvm.ptr<6>)
  // CHECK-NOT: tcgen05.alloc
  // CHECK-NOT: nvvm.barrier0
  // CHECK: llvm.call @tmem_leaf(%[[MIDDLE_VALUE]], %[[MIDDLE_BASE]]) : (i32, !llvm.ptr<6>) -> ()
  llvm.func internal @tmem_middle(%value: i32) attributes {passthrough = ["noinline"], sym_visibility = "private"} {
    llvm.call @tmem_leaf(%value) : (i32) -> ()
    llvm.return
  }

  // CHECK-LABEL: llvm.func @tmem_noinline_kernel()
  // CHECK: tcgen05.alloc
  // CHECK-NOT: tcgen05.alloc
  // CHECK: %[[BASE_I32:.*]] = llvm.load
  // CHECK: %[[KERNEL_BASE:.*]] = llvm.inttoptr %[[BASE_I32]] : i32 to !llvm.ptr<6>
  // CHECK: %[[SEVEN:.*]] = llvm.mlir.constant(7 : i32) : i32
  // CHECK: llvm.call @tmem_middle(%[[SEVEN]], %[[KERNEL_BASE]]) : (i32, !llvm.ptr<6>) -> ()
  // CHECK: llvm.call @tmem_middle(%[[SEVEN]], %[[KERNEL_BASE]]) : (i32, !llvm.ptr<6>) -> ()
  // CHECK: tcgen05.dealloc
  // CHECK-NOT: tcgen05.alloc
  // CHECK-NOT: nvg.tensor_memory_base
  llvm.func @tmem_noinline_kernel() attributes {allocation.offset = 0 : i32, nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 128>} {
    %seven = llvm.mlir.constant(7 : i32) : i32
    llvm.call @tmem_middle(%seven) : (i32) -> ()
    llvm.call @tmem_middle(%seven) : (i32) -> ()
    llvm.return
  }
}

// -----

module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 4 : i32, ttg.tensor_memory_size = 128 : i32} {
  // CHECK-LABEL: llvm.func internal @tmem_mixed_helper(
  // CHECK-SAME: %[[HELPER_BASE:.*]]: !llvm.ptr<6>)
  // CHECK-NOT: tcgen05.alloc
  // CHECK-NOT: nvvm.barrier0
  // CHECK: llvm.ptrtoint %[[HELPER_BASE]] : !llvm.ptr<6> to i32
  llvm.func internal @tmem_mixed_helper() attributes {passthrough = ["noinline"], sym_visibility = "private"} {
    %base = nvg.tensor_memory_base
    %base_i32 = llvm.ptrtoint %base : !llvm.ptr<6> to i32
    "use"(%base_i32) : (i32) -> ()
    llvm.return
  }

  // CHECK-LABEL: llvm.func @tmem_mixed_kernel()
  // CHECK: tcgen05.alloc
  // CHECK: %[[MIXED_LOAD:.*]] = llvm.load
  // CHECK: %[[MIXED_BASE:.*]] = llvm.inttoptr %[[MIXED_LOAD]] : i32 to !llvm.ptr<6>
  // CHECK: %[[MIXED_USE:.*]] = llvm.ptrtoint %[[MIXED_BASE]] : !llvm.ptr<6> to i32
  // CHECK: "use"(%[[MIXED_USE]])
  // CHECK: llvm.call @tmem_mixed_helper(%[[MIXED_BASE]]) : (!llvm.ptr<6>) -> ()
  // CHECK: tcgen05.dealloc
  // CHECK-NOT: tcgen05.alloc
  // CHECK-NOT: nvg.tensor_memory_base
  llvm.func @tmem_mixed_kernel() attributes {allocation.offset = 0 : i32, nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 128>} {
    %base = nvg.tensor_memory_base
    %base_i32 = llvm.ptrtoint %base : !llvm.ptr<6> to i32
    "use"(%base_i32) : (i32) -> ()
    llvm.call @tmem_mixed_helper() : () -> ()
    llvm.return
  }
}

// -----

module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 4 : i32, ttg.tensor_memory_size = 128 : i32} {
  // CHECK-LABEL: llvm.func internal @tmem_shared_helper(
  // CHECK-SAME: %[[SHARED_BASE:.*]]: !llvm.ptr<6>)
  // CHECK-NOT: tcgen05.alloc
  // CHECK-NOT: nvvm.barrier0
  // CHECK: llvm.ptrtoint %[[SHARED_BASE]] : !llvm.ptr<6> to i32
  llvm.func internal @tmem_shared_helper() attributes {passthrough = ["noinline"], sym_visibility = "private"} {
    %base = nvg.tensor_memory_base
    %base_i32 = llvm.ptrtoint %base : !llvm.ptr<6> to i32
    "use"(%base_i32) : (i32) -> ()
    llvm.return
  }

  // CHECK-LABEL: llvm.func @tmem_first_kernel()
  // CHECK: tcgen05.alloc
  // CHECK: %[[FIRST_LOAD:.*]] = llvm.load
  // CHECK: %[[FIRST_BASE:.*]] = llvm.inttoptr %[[FIRST_LOAD]] : i32 to !llvm.ptr<6>
  // CHECK: llvm.call @tmem_shared_helper(%[[FIRST_BASE]]) : (!llvm.ptr<6>) -> ()
  // CHECK: tcgen05.dealloc
  llvm.func @tmem_first_kernel() attributes {allocation.offset = 0 : i32, nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 128>} {
    llvm.call @tmem_shared_helper() : () -> ()
    llvm.return
  }

  // CHECK-LABEL: llvm.func @tmem_second_kernel()
  // CHECK: tcgen05.alloc
  // CHECK: %[[SECOND_LOAD:.*]] = llvm.load
  // CHECK: %[[SECOND_BASE:.*]] = llvm.inttoptr %[[SECOND_LOAD]] : i32 to !llvm.ptr<6>
  // CHECK: llvm.call @tmem_shared_helper(%[[SECOND_BASE]]) : (!llvm.ptr<6>) -> ()
  // CHECK: tcgen05.dealloc
  // CHECK-NOT: nvg.tensor_memory_base
  llvm.func @tmem_second_kernel() attributes {allocation.offset = 0 : i32, nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 128>} {
    llvm.call @tmem_shared_helper() : () -> ()
    llvm.return
  }
}
