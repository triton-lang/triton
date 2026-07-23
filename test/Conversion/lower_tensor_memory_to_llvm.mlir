// RUN: triton-opt %s --test-print-membar --convert-triton-gpu-to-llvm --convert-warp-specialize-to-llvm --convert-nv-gpu-to-llvm -allow-unregistered-dialect | FileCheck %s

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
