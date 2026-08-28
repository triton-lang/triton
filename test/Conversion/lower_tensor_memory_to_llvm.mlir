// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --allocate-shared-memory-nv=compute-capability=103 --triton-nvidia-gpu-tmem-barrier-insertion --test-print-membar --triton-nvidia-gpu-tmem-wait-insertion --convert-triton-gpu-to-llvm=compute-capability=103 --convert-warp-specialize-to-llvm --convert-nv-gpu-to-llvm -allow-unregistered-dialect | FileCheck %s

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

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

module attributes {"ttg.target" = "cuda:103", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.tensor_memory_size = 64 : i32} {
  // Disjoint stores to %a, %b, and %c share a wait before overwriting %a,
  // even when %c's predicate is false.
  // CHECK-LABEL: @tmem_store_wait_chain
  // CHECK: tcgen05.st.sync.aligned
  // CHECK-NOT: nvvm.tcgen05.wait <store>
  // CHECK: tcgen05.st.sync.aligned
  // CHECK-NOT: nvvm.tcgen05.wait <store>
  // CHECK: tcgen05.st.sync.aligned
  // CHECK-NEXT: nvvm.tcgen05.wait <store>
  // CHECK: tcgen05.st.sync.aligned
  // CHECK-NEXT: nvvm.tcgen05.wait <store>
  tt.func @tmem_store_wait_chain() {
    %true = arith.constant true
    %false = arith.constant false
    %data = arith.constant dense<0.0> : tensor<128x16xf32, #blocked>
    %a = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    %b = ttng.tmem_alloc {tensor_memory_col_offset = 16 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    %c = ttng.tmem_alloc {tensor_memory_col_offset = 32 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %a, %true : tensor<128x16xf32, #blocked> -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %b, %true : tensor<128x16xf32, #blocked> -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %c, %false : tensor<128x16xf32, #blocked> -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %a, %true : tensor<128x16xf32, #blocked> -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

module attributes {"ttg.target" = "cuda:103", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // %a and %b may alias, so wait before the second store.
  // CHECK-LABEL: @tmem_store_wait_unknown
  // CHECK: tcgen05.st.sync.aligned
  // CHECK-NEXT: nvvm.tcgen05.wait <store>
  // CHECK: tcgen05.st.sync.aligned
  // CHECK-NEXT: nvvm.tcgen05.wait <store>
  tt.func private @tmem_store_wait_unknown(%a: !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>, %b: !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>, %value: f32) {
    %true = arith.constant true
    %data = tt.splat %value : f32 -> tensor<128x16xf32, #blocked>
    ttng.tmem_store %data, %a, %true : tensor<128x16xf32, #blocked> -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %b, %true : tensor<128x16xf32, #blocked> -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mixed = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

module attributes {"ttg.target" = "cuda:103", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.tensor_memory_size = 64 : i32} {
  // Layout conversion and reduction lower to CTA barriers. The stores to
  // %a and %b share one wait before the last barrier.
  // CHECK-LABEL: @tmem_store_wait_scratch_boundaries
  // CHECK: tcgen05.st.sync.aligned
  // CHECK-NOT: nvvm.tcgen05.wait
  // CHECK: nvvm.barrier
  // CHECK-NOT: nvvm.tcgen05.wait
  // CHECK: tcgen05.st.sync.aligned
  // CHECK: nvvm.tcgen05.wait <store>
  // CHECK-NOT: nvvm.tcgen05.wait
  // CHECK: nvvm.barrier
  // CHECK-NOT: nvvm.tcgen05.wait
  // CHECK: tcgen05.st.sync.aligned
  // CHECK-NEXT: nvvm.tcgen05.wait <store>
  tt.func @tmem_store_wait_scratch_boundaries(%input: !tt.ptr<f32>) {
    %true = arith.constant true
    %rows = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %base = tt.splat %input : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %ptrs = tt.addptr %base, %rows : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %values = tt.load %ptrs : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %column = tt.expand_dims %values {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
    %data = tt.broadcast %column : tensor<128x1xf32, #blocked> -> tensor<128x16xf32, #blocked>
    %a = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    %b = ttng.tmem_alloc {tensor_memory_col_offset = 16 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    %c = ttng.tmem_alloc {tensor_memory_col_offset = 32 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %a, %true : tensor<128x16xf32, #blocked> -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    %converted = ttg.convert_layout %data : tensor<128x16xf32, #blocked> -> tensor<128x16xf32, #mixed>
    ttng.tmem_store %data, %b, %true : tensor<128x16xf32, #blocked> -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    %sum = "tt.reduce"(%converted) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) : (tensor<128x16xf32, #mixed>) -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #mixed}>>
    %sum_layout = ttg.convert_layout %sum : tensor<16xf32, #ttg.slice<{dim = 0, parent = #mixed}>> -> tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %sum_row = tt.expand_dims %sum_layout {axis = 0 : i32} : tensor<16xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xf32, #blocked>
    %sum_data = tt.broadcast %sum_row : tensor<1x16xf32, #blocked> -> tensor<128x16xf32, #blocked>
    ttng.tmem_store %sum_data, %c, %true : tensor<128x16xf32, #blocked> -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>

module attributes {"ttg.target" = "cuda:103", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.tensor_memory_size = 64 : i32} {
  // Stores to %a and %b share one wait before arrive_barrier. The load from %b
  // needs a store wait, but the disjoint store to %c needs no load wait.
  // Both outstanding accesses must complete before wait_barrier.
  // CHECK-LABEL: @tmem_store_wait_publication
  // CHECK: tcgen05.st.sync.aligned
  // CHECK-NEXT: nvvm.barrier
  // CHECK-NOT: nvvm.tcgen05.wait
  // CHECK: tcgen05.st.sync.aligned
  // CHECK-NEXT: nvvm.tcgen05.wait <store>
  // CHECK: nvvm.barrier
  // CHECK: mbarrier.arrive.shared::cta
  // CHECK: tcgen05.st.sync.aligned
  // CHECK-NEXT: nvvm.tcgen05.wait <store>
  // CHECK: tcgen05.ld.sync.aligned
  // CHECK-NOT: nvvm.tcgen05.wait
  // CHECK: tcgen05.st.sync.aligned
  // CHECK-NEXT: nvvm.tcgen05.wait <load>
  // CHECK-NEXT: nvvm.tcgen05.wait <store>
  // CHECK: mbarrier.try_wait.parity.shared::cta
  tt.func @tmem_store_wait_publication() {
    %true = arith.constant true
    %phase = arith.constant 0 : i32
    %data = arith.constant dense<0.0> : tensor<128x16xf32, #blocked>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
    %a = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    %b = ttng.tmem_alloc {tensor_memory_col_offset = 16 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    %c = ttng.tmem_alloc {tensor_memory_col_offset = 32 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %a, %true : tensor<128x16xf32, #blocked> -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    ttg.barrier local
    ttng.tmem_store %data, %b, %true : tensor<128x16xf32, #blocked> -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
    ttng.tmem_store %data, %b, %true : tensor<128x16xf32, #blocked> -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    %loaded = ttng.tmem_load %b : !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x16xf32, #blocked>
    ttng.tmem_store %loaded, %c, %true : tensor<128x16xf32, #blocked> -> !ttg.memdesc<128x16xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}
