// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --triton-nvidia-gpu-membar='compute-capability=100 ptx-version=87' | FileCheck %s --check-prefixes=CHECK,WAIT,TMEM
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --triton-nvidia-gpu-tmem-barrier-insertion | FileCheck %s --check-prefix=TMEM
// RUN: triton-opt %s -split-input-file --tritongpu-optimize-thread-regions --convert-scf-to-cf --triton-nvidia-gpu-membar='compute-capability=100 ptx-version=87' | FileCheck %s --check-prefix=GUARD

#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared_copy = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared_fp8 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared_padded_b = #ttg.shared_linear<{offset = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0], [0, 0], [16, 0], [32, 0]]}, alignment = 16>
#shared_padded_b32 = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0]]}, alignment = 16>
#shared_scales = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked_broadcast_warps = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked_scales = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear64 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0]], block = []}>
#tmem128 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem64 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
#tmem8 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 8, colStride = 1>
#tmem32 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 32, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#publish_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
!publish_data = tensor<128x8xi32, #blocked>
!publish_smem = !ttg.memdesc<128x8xi32, #publish_shared, #ttg.shared_memory, mutable>
!publish_tmem = !ttg.memdesc<128x8xi32, #tmem8, #ttng.tensor_memory, mutable>
!publish_bar = !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @alloc_then_alloc
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: tt.return
  tt.func @alloc_then_alloc(%arg0: tensor<128x128xf32, #blocked>) {
    %0 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @alloc_then_ld
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: tt.return
  tt.func @alloc_then_ld(%arg0: tensor<128x128xf32, #blocked>) {
    %0 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }

  // Reuse an existing store wait when Membar runs on previously lowered IR.
  // CHECK-LABEL: @existing_store_wait
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NOT: ttng.tmem_wait store
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: tt.return
  tt.func private @existing_store_wait(%data: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> attributes {noinline = true} {
    %tile = ttng.tmem_alloc %data {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_wait store
    %value = ttng.tmem_load %tile : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return %value : tensor<128x128xf32, #blocked>
  }

  // CHECK-LABEL: @alloc_then_st
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttng.tmem_store
  tt.func @alloc_then_st(%arg0: tensor<128x128xf32, #blocked>) {
    %true = arith.constant true
    %0 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %arg0, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // TMEM-LABEL: @alloc_then_mma
  // TMEM: ttng.tmem_alloc
  // TMEM-NEXT: ttng.tmem_wait store
  // TMEM-NEXT: ttg.barrier local
  // TMEM-NEXT: ttng.tc_gen5_mma
  tt.func @alloc_then_mma(%arg0: tensor<128x128xf32, #blocked>,
                          %arg1: !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
                          %arg2: !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>) {
    %false = arith.constant false
    %true = arith.constant true
    %0 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %arg1, %arg2, %0, %false, %true :
      !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // One rendezvous before the MMA covers the shared copy and the TMEM store.
  // CHECK-LABEL: @shared_copy_and_accumulator_before_mma
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttng.fence_async_shared
  // CHECK-NEXT: {{.*}} = ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tc_gen5_mma
  tt.func @shared_copy_and_accumulator_before_mma(%src: tensor<128x128x!tt.ptr<f16>, #blocked>,
                                                 %b: !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>,
                                                 %init: tensor<128x128xf32, #blocked>) {
    %true = arith.constant true
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory, mutable>
    ttg.async_copy_global_to_local %src, %a {contiguity = 8 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked> -> !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory, mutable>
    ttg.async_commit_group
    ttg.async_wait {num = 0 : i32}
    ttng.fence_async_shared {bCluster = false}
    %acc = ttng.tmem_alloc %init {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a, %b, %acc, %true, %true :
      !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory, mutable>,
      !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // Keep the assumption before the rendezvous required by the MMA.
  // CHECK-LABEL: @alloc_assume_then_mma
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: llvm.intr.assume
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tc_gen5_mma
  tt.func @alloc_assume_then_mma(%data: tensor<128x128xf32, #blocked>, %condition: i1,
                                %a: !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
                                %b: !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>) {
    %false = arith.constant false
    %true = arith.constant true
    %acc = ttng.tmem_alloc %data {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    llvm.intr.assume %condition : i1
    ttng.tc_gen5_mma %a, %b, %acc, %false, %true :
      !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @ld_then_alloc
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttng.tmem_alloc
  tt.func @ld_then_alloc(%arg0: tensor<128x128xf32, #blocked>) {
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %2 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @ld_then_ld
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: tt.return
  tt.func @ld_then_ld() {
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %2 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }

  // Same-thread overwrites wait immediately before the store, after arithmetic.
  // WAIT-LABEL: @ld_then_st
  // WAIT: ttng.tmem_load
  // WAIT-NEXT: arith.addf
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: tt.return
  tt.func @ld_then_st(%arg0: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> {
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %updated = arith.addf %arg0, %arg0 : tensor<128x128xf32, #blocked>
    ttng.tmem_store %updated, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return %1 : tensor<128x128xf32, #blocked>
  }

  // Shrinking #linear64 to one column broadcasts lane 16 within each warp.
  // The unique warp owners can overwrite after completing their loads.
  // WAIT-LABEL: @ld_then_st_broadcast_lanes
  // WAIT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: ttng.tmem_store
  tt.func @ld_then_st_broadcast_lanes(%data: tensor<64x1xf32, #linear64>) -> tensor<64x1xf32, #linear64> {
    %true = arith.constant true
    %zero = arith.constant dense<0.0> : tensor<64x1xf32, #linear64>
    %mem = ttng.tmem_alloc %zero {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<64x1xf32, #linear64>) -> !ttg.memdesc<64x1xf32, #tmem64, #ttng.tensor_memory, mutable>
    ttg.barrier local
    %loaded = ttng.tmem_load %mem : !ttg.memdesc<64x1xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<64x1xf32, #linear64>
    ttng.tmem_store %data, %mem, %true : tensor<64x1xf32, #linear64> -> !ttg.memdesc<64x1xf32, #tmem64, #ttng.tensor_memory, mutable>
    tt.return %loaded : tensor<64x1xf32, #linear64>
  }

  // Warps 0 and 4 access the same words and must rendezvous before overwriting.
  // WAIT-LABEL: @ld_then_st_broadcast_warps
  // WAIT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: ttng.tmem_store
  tt.func @ld_then_st_broadcast_warps(%data: tensor<128x1xf32, #blocked_broadcast_warps>) -> tensor<128x1xf32, #blocked_broadcast_warps> attributes {"ttg.num-warps" = 8 : i32} {
    %true = arith.constant true
    %zero = arith.constant dense<0.0> : tensor<128x1xf32, #blocked_broadcast_warps>
    %mem = ttng.tmem_alloc %zero {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x1xf32, #blocked_broadcast_warps>) -> !ttg.memdesc<128x1xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    %loaded = ttng.tmem_load %mem : !ttg.memdesc<128x1xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked_broadcast_warps>
    ttng.tmem_store %data, %mem, %true : tensor<128x1xf32, #blocked_broadcast_warps> -> !ttg.memdesc<128x1xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return %loaded : tensor<128x1xf32, #blocked_broadcast_warps>
  }

  // Complete the high-half store and low-half load before the same-warp store.
  // WAIT-LABEL: @wait_mixed_store_hazards
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: tt.return
  tt.func @wait_mixed_store_hazards(%data: tensor<128x128xf32, #blocked>, %half: tensor<128x64xf32, #blocked>) -> tensor<128x64xf32, #blocked> {
    %true = arith.constant true
    %parent = ttng.tmem_alloc %data {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %low = ttng.tmem_subslice %parent {offset = 0 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128>
    %high = ttng.tmem_subslice %parent {offset = 64 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128>
    ttng.tmem_store %half, %high, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128>
    ttg.barrier local
    %loaded = ttng.tmem_load %low : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #blocked>
    ttng.tmem_store %data, %parent, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return %loaded : tensor<128x64xf32, #blocked>
  }

  // WAIT-LABEL: @tmem_opaque_handoff
  // WAIT-NOT: {{ttng\.tmem_wait|ttg\.barrier}}
  // WAIT: {{.*}}tt.elementwise_inline_asm
  tt.func private @tmem_opaque_handoff() attributes {noinline = true} {
    %opaque = tt.elementwise_inline_asm "mov.u32 $0, 0;" {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32
    tt.return
  }

  // The load completes before either overwrite. The optional copy needs a
  // rendezvous with the loading warps.
  // WAIT-LABEL: @wait_same_thread_after_barrier
  // WAIT-NOT: ttng.tmem_wait load
  // WAIT: ttng.tmem_load
  // WAIT: ttng.tmem_wait load
  // WAIT-DAG: ttg.barrier local
  // WAIT-DAG: ttng.tmem_store
  // WAIT: ttng.tmem_copy
  // WAIT-NOT: ttg.barrier local
  // WAIT: tt.call @tmem_opaque_handoff
  // WAIT-NEXT: tt.return
  tt.func @wait_same_thread_after_barrier(%condition: i1, %init: tensor<128x128xf32, #blocked>, %data: tensor<128x64xf32, #blocked>, %shared: !ttg.memdesc<128x64xf32, #shared_copy, #ttg.shared_memory>) -> tensor<128x128xf32, #blocked> {
    %true = arith.constant true
    %parent = ttng.tmem_alloc %init {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %low = ttng.tmem_subslice %parent {offset = 0 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128>
    %high = ttng.tmem_subslice %parent {offset = 64 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128>
    %loaded = ttng.tmem_load %parent : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    ttg.barrier local
    %updated = arith.addf %data, %data : tensor<128x64xf32, #blocked>
    ttng.tmem_store %updated, %low, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128>
    cf.cond_br %condition, ^copy, ^exit
  ^copy:
    ttng.tmem_copy %shared, %high : !ttg.memdesc<128x64xf32, #shared_copy, #ttg.shared_memory>, !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128>
    cf.br ^exit
  ^exit:
    tt.call @tmem_opaque_handoff() : () -> ()
    tt.return %loaded : tensor<128x128xf32, #blocked>
  }

  // CHECK-LABEL: @ld_then_mma
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tc_gen5_mma
  tt.func @ld_then_mma(%arg0: !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
                       %arg1: !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>) {
    %false = arith.constant false
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    ttng.tc_gen5_mma %arg0, %arg1, %0, %false, %true :
      !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // Allocation padding is part of every access footprint.
  // WAIT-LABEL: @padded_tmem_store_then_load
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT: ttng.tmem_load
  tt.func @padded_tmem_store_then_load(%data: tensor<128x1xf32, #blocked>) {
    %true = arith.constant true
    %parent = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x8xf32, #tmem8, #ttng.tensor_memory, mutable>
    %d = ttg.memdesc_reinterpret %parent : !ttg.memdesc<128x8xf32, #tmem8, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem8, #ttng.tensor_memory, mutable>
    %old = ttng.tmem_subslice %parent {offset = 7 : i32} : !ttg.memdesc<128x8xf32, #tmem8, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem8, #ttng.tensor_memory, mutable, 128x8>
    ttng.tmem_store %data, %d, %true : tensor<128x1xf32, #blocked> -> !ttg.memdesc<128x1xf32, #tmem8, #ttng.tensor_memory, mutable>
    %loaded = ttng.tmem_load %old : !ttg.memdesc<128x1xf32, #tmem8, #ttng.tensor_memory, mutable, 128x8> -> tensor<128x1xf32, #blocked>
    tt.return
  }

  // MMA overwrites column 7 even though D has only one logical column.
  // WAIT-LABEL: @padded_tmem_load_then_mma
  // WAIT: ttng.tmem_load
  // WAIT: ttng.tmem_wait load
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: ttng.tc_gen5_mma
  tt.func @padded_tmem_load_then_mma(
      %a: !ttg.memdesc<128x64xf8E4M3FN, #shared_fp8, #ttg.shared_memory>,
      %b: !ttg.memdesc<64x1xf8E4M3FN, #shared_padded_b, #ttg.shared_memory>) {
    %false = arith.constant false
    %true = arith.constant true
    %parent = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x8xf32, #tmem8, #ttng.tensor_memory, mutable>
    %old = ttng.tmem_subslice %parent {offset = 7 : i32} : !ttg.memdesc<128x8xf32, #tmem8, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem8, #ttng.tensor_memory, mutable, 128x8>
    %loaded = ttng.tmem_load %old : !ttg.memdesc<128x1xf32, #tmem8, #ttng.tensor_memory, mutable, 128x8> -> tensor<128x1xf32, #blocked>
    %d = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem8, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a, %b, %d, %false, %true {is_async} :
      !ttg.memdesc<128x64xf8E4M3FN, #shared_fp8, #ttg.shared_memory>,
      !ttg.memdesc<64x1xf8E4M3FN, #shared_padded_b, #ttg.shared_memory>,
      !ttg.memdesc<128x1xf32, #tmem8, #ttng.tensor_memory, mutable>
    tt.return
  }

  // Byte 16 belongs to B's reserved N columns; K is packed two FP4 values per byte.
  // CHECK-LABEL: @padded_fp4_shared_store_then_mma
  // CHECK: ttg.local_store
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttng.tc_gen5_mma_scaled
  tt.func @padded_fp4_shared_store_then_mma(
      %data: tensor<1xi32, #blocked0>,
      %sa: !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>,
      %sb: !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>) {
    %false = arith.constant false
    %true = arith.constant true
    %a = ttg.local_alloc {allocation.offset = 2048 : i32} : () -> !ttg.memdesc<128x64xi8, #shared_fp8, #ttg.shared_memory, mutable>
    %d = ttng.tmem_alloc {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem8, #ttng.tensor_memory, mutable>
    %old = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<1xi32, #barrier, #ttg.shared_memory, mutable>
    ttg.local_store %data, %old : tensor<1xi32, #blocked0> -> !ttg.memdesc<1xi32, #barrier, #ttg.shared_memory, mutable>
    %b = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x1xi8, #shared_padded_b, #ttg.shared_memory, mutable>
    ttng.tc_gen5_mma_scaled %a, %b, %d, %sa, %sb, %false, %true lhs = e2m1 rhs = e2m1 {is_async} :
      !ttg.memdesc<128x64xi8, #shared_fp8, #ttg.shared_memory, mutable>,
      !ttg.memdesc<64x1xi8, #shared_padded_b, #ttg.shared_memory, mutable>,
      !ttg.memdesc<128x1xf32, #tmem8, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>,
      !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
    tt.return
  }

  // Scale column 1 holds rows 32-63, outside the scale subview.
  // WAIT-LABEL: @padded_scale_store_outside_subview
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: {{.*}}ttng.tmem_alloc
  // WAIT-NEXT: {{.*}}ttng.tmem_subslice
  // WAIT-NEXT: ttng.tc_gen5_mma_scaled
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: tt.return
  tt.func @padded_scale_store_outside_subview(
      %data: tensor<128x1xf32, #blocked>,
      %a: !ttg.memdesc<128x128xi8, #shared_fp8, #ttg.shared_memory>,
      %b: !ttg.memdesc<256x8xi8, #shared_padded_b32, #ttg.shared_memory>) {
    %false = arith.constant false
    %true = arith.constant true
    %sa = ttng.tmem_alloc {tensor_memory_col_offset = 256 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    %d = ttng.tmem_alloc {tensor_memory_col_offset = 384 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x16xf32, #tmem32, #ttng.tensor_memory, mutable>
    %old_parent = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x8xf32, #tmem8, #ttng.tensor_memory, mutable>
    %old = ttng.tmem_subslice %old_parent {offset = 1 : i32} : !ttg.memdesc<128x8xf32, #tmem8, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem8, #ttng.tensor_memory, mutable, 128x8>
    ttng.tmem_store %data, %old, %true : tensor<128x1xf32, #blocked> -> !ttg.memdesc<128x1xf32, #tmem8, #ttng.tensor_memory, mutable, 128x8>
    %parent = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<64x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    %sb = ttng.tmem_subslice %parent {offset = 0 : i32, dim = 0 : i32} : !ttg.memdesc<64x8xi8, #tmem_scales, #ttng.tensor_memory, mutable> -> !ttg.memdesc<32x8xi8, #tmem_scales, #ttng.tensor_memory, mutable, 64x8>
    ttng.tc_gen5_mma_scaled %a, %b, %d, %sa, %sb, %false, %true lhs = e2m1 rhs = e2m1 {is_async} :
      !ttg.memdesc<128x128xi8, #shared_fp8, #ttg.shared_memory>,
      !ttg.memdesc<256x8xi8, #shared_padded_b32, #ttg.shared_memory>,
      !ttg.memdesc<128x16xf32, #tmem32, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<32x8xi8, #tmem_scales, #ttng.tensor_memory, mutable, 64x8>
    tt.return
  }

  // An uninitialized allocation does not access reused storage. Wait only
  // before its first store.
  // CHECK-LABEL: @st_then_alloc
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: tt.return
  tt.func @st_then_alloc(%arg0: tensor<128x128xf32, #blocked>) {
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    ttng.tmem_store %arg0, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %arg0, %1, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // TMEM-LABEL: @st_then_ld
  // TMEM: ttng.tmem_store
  // TMEM-NEXT: ttng.tmem_wait store
  // TMEM-NEXT: ttng.tmem_load
  // TMEM-NEXT: ttng.tmem_wait load
  // TMEM-NEXT: tt.return
  tt.func @st_then_ld(%arg0: tensor<128x128xf32, #blocked>) {
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    ttng.tmem_store %arg0, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }

  // CHECK-LABEL: @st_then_st
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: tt.return
  tt.func @st_then_st(%arg0: tensor<128x128xf32, #blocked>) {
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    ttng.tmem_store %arg0, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %arg0, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @st_then_mma
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tc_gen5_mma
  // CHECK-NOT: ttng.tmem_wait
  // CHECK: tt.return
  tt.func @st_then_mma(%arg0: tensor<128x128xf32, #blocked>,
                       %arg1: !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
                       %arg2: !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>) {
    %false = arith.constant false
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    ttng.tmem_store %arg0, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %arg1, %arg2, %0, %false, %true :
      !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @mma_then_alloc
  // CHECK: ttng.tc_gen5_mma
  // CHECK-NEXT: ttng.tmem_alloc
  tt.func @mma_then_alloc(%arg0: !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
                          %arg1: !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>) {
    %false = arith.constant false
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    ttng.tc_gen5_mma %arg0, %arg1, %0, %false, %true :
      !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @mma_then_ld
  // CHECK: ttng.tc_gen5_mma
  // CHECK-NEXT: ttng.tmem_load
  tt.func @mma_then_ld(%arg0: !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
                       %arg1: !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>) {
    %false = arith.constant false
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    ttng.tc_gen5_mma %arg0, %arg1, %0, %false, %true :
      !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }

  // CHECK-LABEL: @mma_then_st
  // CHECK: ttng.tc_gen5_mma
  // CHECK-NEXT: ttng.tmem_store
  tt.func @mma_then_st(%arg0: tensor<128x128xf32, #blocked>,
                       %arg1: !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
                       %arg2: !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>) {
    %false = arith.constant false
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    ttng.tc_gen5_mma %arg1, %arg2, %0, %false, %true :
      !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %arg0, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @mma_then_mma
  // CHECK: ttng.tc_gen5_mma
  // CHECK-NEXT: ttng.tc_gen5_mma
  tt.func @mma_then_mma(%arg0: !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
                        %arg1: !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>) {
    %false = arith.constant false
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    ttng.tc_gen5_mma %arg0, %arg1, %0, %false, %true :
      !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %arg0, %arg1, %0, %false, %true :
      !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @ld_then_st_non_aliasing
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_store
  tt.func @ld_then_st_non_aliasing(%arg0: tensor<128x128xf32, #blocked>) {
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %2 = ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    ttng.tmem_store %arg0, %2, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // Slices of one allocation can be physically disjoint even though their
  // root allocation is the same.
  // CHECK-LABEL: @ld_then_st_disjoint_slices
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_store
  tt.func @ld_then_st_disjoint_slices(%data: tensor<128x64xf32, #blocked>) {
    %true = arith.constant true
    %parent = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %low = ttng.tmem_subslice %parent {offset = 0 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128>
    %high = ttng.tmem_subslice %parent {offset = 64 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128>
    %loaded = ttng.tmem_load %low : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #blocked>
    ttng.tmem_store %data, %high, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }

  // Both possible descriptors of arith.select must be included in the access.
  // CHECK-LABEL: @ld_selected_then_st
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttng.tmem_store
  tt.func @ld_selected_then_st(%condition: i1, %data: tensor<128x64xf32, #blocked>) {
    %true = arith.constant true
    %parent = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %low = ttng.tmem_subslice %parent {offset = 0 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128>
    %high = ttng.tmem_subslice %parent {offset = 64 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128>
    %selected = arith.select %condition, %low, %high : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128>
    %loaded = ttng.tmem_load %selected : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32, #blocked>
    ttng.tmem_store %data, %high, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable, 128x128>
    tt.return
  }

  // Unknown descriptor origins must not become empty footprints.
  // CHECK-LABEL: @ld_unknown_then_st
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_store
  tt.func @ld_unknown_then_st(%unknown: !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory>, %data: tensor<128x128xf32, #blocked>) {
    %true = arith.constant true
    %known = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %loaded = ttng.tmem_load %unknown : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    ttng.tmem_store %data, %known, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // Packed f16 elements occupy half as many physical columns as f32 elements.
  // Allocation reuse must compare those columns, not logical element indices.
  // CHECK-LABEL: @ld_packed_then_st_disjoint
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_store
  tt.func @ld_packed_then_st_disjoint(%data: tensor<128x64xf32, #blocked>) {
    %true = arith.constant true
    %packed = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem128, #ttng.tensor_memory, mutable>
    %loaded = ttng.tmem_load %packed : !ttg.memdesc<128x128xf16, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf16, #blocked>
    %other = ttng.tmem_alloc {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %other, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @ld_packed_then_st_overlap
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttng.tmem_store
  tt.func @ld_packed_then_st_overlap(%data: tensor<128x64xf32, #blocked>) {
    %true = arith.constant true
    %packed = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem128, #ttng.tensor_memory, mutable>
    %loaded = ttng.tmem_load %packed : !ttg.memdesc<128x128xf16, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf16, #blocked>
    %other = ttng.tmem_alloc {tensor_memory_col_offset = 32 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %other, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // A 64-row allocation uses one 16-row half of every warp's TMEM rows.
  // CHECK-LABEL: @ld_then_st_disjoint_row_groups
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_store
  tt.func @ld_then_st_disjoint_row_groups(%data: tensor<64x128xf32, #linear64>) {
    %true = arith.constant true
    %low = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<64x128xf32, #tmem64, #ttng.tensor_memory, mutable>
    %high = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 16 : i32} : () -> !ttg.memdesc<64x128xf32, #tmem64, #ttng.tensor_memory, mutable>
    %loaded = ttng.tmem_load %low : !ttg.memdesc<64x128xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #linear64>
    ttng.tmem_store %data, %high, %true : tensor<64x128xf32, #linear64> -> !ttg.memdesc<64x128xf32, #tmem64, #ttng.tensor_memory, mutable>
    tt.return
  }

  // The reused rows have different lane owners, but the same warp owners.
  // CHECK-LABEL: @ld_then_alloc_then_st_aliases_second_row
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttng.tmem_store
  tt.func @ld_then_alloc_then_st_aliases_second_row(%arg0: tensor<64x128xf32, #linear64>) {
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %2 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 16 : i32} : () -> !ttg.memdesc<64x128xf32, #tmem64, #ttng.tensor_memory, mutable>
    ttng.tmem_store %arg0, %2, %true : tensor<64x128xf32, #linear64> -> !ttg.memdesc<64x128xf32, #tmem64, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @alloc_then_alloc_partial_overlap
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttng.tmem_alloc
  tt.func @alloc_then_alloc_partial_overlap(%arg0: tensor<128x128xf32, #blocked>) {
    %0 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @st_then_mma_scaled_scale_operand
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tc_gen5_mma_scaled
  tt.func @st_then_mma_scaled_scale_operand(
      %arg0: tensor<128x1xi8, #blocked_scales>,
      %arg1: !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
      %arg2: !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>,
      %arg3: !ttg.memdesc<64x1xi8, #tmem_scales, #ttng.tensor_memory>) {
    %true = arith.constant true
    %d = ttng.tmem_alloc {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %a_scale = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x1xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    ttg.barrier local
    ttng.tmem_store %arg0, %a_scale, %true : tensor<128x1xi8, #blocked_scales> -> !ttg.memdesc<128x1xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma_scaled %arg1, %arg2, %d, %a_scale, %arg3, %true, %true lhs = e5m2 rhs = e5m2 :
      !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<128x1xi8, #tmem_scales, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<64x1xi8, #tmem_scales, #ttng.tensor_memory>
    tt.return
  }

  // Shared scale arguments have unknown addresses, but cannot alias TMEM.
  // CHECK-LABEL: @st_then_mma_scaled_shared_scales
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tc_gen5_mma_scaled
  tt.func @st_then_mma_scaled_shared_scales(
      %data: tensor<128x128xf32, #blocked>,
      %a: !ttg.memdesc<128x128xf8E5M2, #shared_fp8, #ttg.shared_memory>,
      %b: !ttg.memdesc<128x128xf8E5M2, #shared_fp8, #ttg.shared_memory>,
      %a_scale: !ttg.memdesc<128x4xi8, #shared_scales, #ttg.shared_memory>,
      %b_scale: !ttg.memdesc<128x4xi8, #shared_scales, #ttg.shared_memory>) {
    %false = arith.constant false
    %true = arith.constant true
    %d = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %other = ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %other, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma_scaled %a, %b, %d, %a_scale, %b_scale, %false, %true lhs = e5m2 rhs = e5m2 :
      !ttg.memdesc<128x128xf8E5M2, #shared_fp8, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf8E5M2, #shared_fp8, #ttg.shared_memory>,
      !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<128x4xi8, #shared_scales, #ttg.shared_memory>,
      !ttg.memdesc<128x4xi8, #shared_scales, #ttg.shared_memory>
    tt.return
  }

  // Complete both loads before storing %a, retaining their publication
  // barrier before the copy.
  // CHECK-LABEL: @ld_then_tmem_copy
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_load
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NOT: ttng.tmem_wait load
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_copy
  // CHECK-NOT: ttng.tmem_wait load
  // CHECK: tt.return
  tt.func @ld_then_tmem_copy(
      %arg0: !ttg.memdesc<128x128xf32, #shared_copy, #ttg.shared_memory>,
      %data: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> {
    %true = arith.constant true
    %0 = ttng.tmem_alloc %data {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %a = ttng.tmem_alloc %data {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %loaded = ttng.tmem_load %a : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %updated = arith.addf %loaded, %data : tensor<128x128xf32, #blocked>
    ttng.tmem_store %updated, %a, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_copy %arg0, %0 : !ttg.memdesc<128x128xf32, #shared_copy, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return %1 : tensor<128x128xf32, #blocked>
  }

  // CHECK-LABEL: @st_then_tmem_copy
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_copy
  // CHECK-NOT: ttng.tmem_wait
  // CHECK: tt.return
  tt.func @st_then_tmem_copy(
      %arg0: tensor<128x128xf32, #blocked>,
      %arg1: !ttg.memdesc<128x128xf32, #shared_copy, #ttg.shared_memory>) {
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    ttng.tmem_store %arg0, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_copy %arg1, %0 : !ttg.memdesc<128x128xf32, #shared_copy, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // TMEM-LABEL: @tmem_entry_c
  // TMEM-NOT: ttng.tmem_wait load
  // TMEM: {{.*}} = ttng.tmem_alloc
  // TMEM-NEXT: ttg.barrier local
  // TMEM-NEXT: tt.return
  tt.func private @tmem_entry_c() {
    %cst = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %alloc = ttng.tmem_alloc %cst {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    tt.return
  }

  // TMEM-LABEL: @tmem_entry_b
  // TMEM-NOT: {{ttng\.tmem_wait|ttg\.barrier}}
  // TMEM: tt.call @tmem_entry_c
  // TMEM-NEXT: tt.return
  tt.func private @tmem_entry_b() {
    tt.call @tmem_entry_c() : () -> ()
    tt.return
  }

  // Wait in the caller before its load, before the second call, and at return.
  // TMEM-LABEL: @tmem_entry_a
  // TMEM-NEXT: {{.*}} = ttng.tmem_alloc
  // TMEM-NEXT: tt.call @tmem_entry_b
  // TMEM-NEXT: ttng.tmem_wait store
  // TMEM-NEXT: {{.*}} = ttng.tmem_load
  // TMEM-NEXT: ttng.tmem_wait load
  // TMEM-NEXT: tt.call @tmem_entry_b
  // TMEM-NEXT: ttng.tmem_wait store
  // TMEM-NEXT: tt.return
  tt.func @tmem_entry_a() {
    %alloc = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.call @tmem_entry_b() : () -> ()
    %loaded = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.call @tmem_entry_b() : () -> ()
    tt.return
  }

  // Wait for the disjoint store before the MMA signals its completion barrier.
  // CHECK-LABEL: @wait_disjoint_mma_publication
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tc_gen5_mma
  // CHECK: ttng.wait_barrier
  // CHECK-NEXT: tt.return
  tt.func @wait_disjoint_mma_publication(
      %data: tensor<128x128xf32, #blocked>,
      %a: !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory, mutable>,
      %b: !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory, mutable>,
      %bar: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>) {
    %false = arith.constant false
    %true = arith.constant true
    %phase = arith.constant 0 : i32
    %d = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %other = ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %other, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a, %b, %d, %false, %true, %bar[%true] {is_async} :
      !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory, mutable>,
      !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory, mutable>,
      !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
    ttng.wait_barrier %bar, %phase deps %a, %b, %d : !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared_a, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared_b, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // Batch stores before the first load and both loads at the publication barrier.
  // WAIT-LABEL: @wait_disjoint_batches
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: ttng.tmem_load
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: ttng.arrive_barrier
  // WAIT-NEXT: tt.return
  tt.func @wait_disjoint_batches(%data: tensor<128x64xf32, #blocked>, %bar: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>) -> (tensor<128x64xf32, #blocked>, tensor<128x64xf32, #blocked>) {
    %true = arith.constant true
    %low = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %high = ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %low, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %high, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %a = ttng.tmem_load %low : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    ttg.barrier local
    %b = ttng.tmem_load %high : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
    tt.return %a, %b : tensor<128x64xf32, #blocked>, tensor<128x64xf32, #blocked>
  }

  // The same-warp load completes stores from both sides of the poll's barrier.
  // WAIT-LABEL: @wait_hoisted_preserves_later_store
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: %{{.*}} = tt.atomic_poll acquire
  // WAIT-NEXT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: tt.return
  tt.func @wait_hoisted_preserves_later_store(%data: tensor<128x64xf32, #blocked>, %ptr: !tt.ptr<i32>, %expected: i32) -> (tensor<128x64xf32, #blocked>, tensor<128x64xf32, #blocked>) {
    %true = arith.constant true
    %low = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %high = ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %low, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    ttng.tmem_store %data, %high, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %a = ttng.tmem_load %low : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    %b = ttng.tmem_load %high : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    tt.return %a, %b : tensor<128x64xf32, #blocked>, tensor<128x64xf32, #blocked>
  }

  // Complete the store after the poll, immediately before the same-warp load.
  // WAIT-LABEL: @wait_trailing_atomic_barrier
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: %{{.*}} = tt.atomic_poll acquire
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: tt.return
  tt.func @wait_trailing_atomic_barrier(%ptr: !tt.ptr<i32>, %expected: i32, %data: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> {
    %true = arith.constant true
    %mem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %mem, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    %loaded = ttng.tmem_load %mem : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return %loaded : tensor<128x128xf32, #blocked>
  }

  // Keep the same-warp corrections pending and complete them at publication.
  // WAIT-LABEL: @wait_disjoint_corrections
  // WAIT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: ttng.tmem_store
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: ttng.arrive_barrier
  tt.func @wait_disjoint_corrections(%data: tensor<128x64xf32, #blocked>, %bar: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>) {
    %true = arith.constant true
    %low = ttng.tmem_alloc %data {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %high = ttng.tmem_alloc %data {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %a = ttng.tmem_load %low : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    ttng.tmem_store %a, %low, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    %b = ttng.tmem_load %high : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    ttng.tmem_store %b, %high, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
    tt.return
  }

  // Both accesses complete before a poll's trailing barrier publishes them
  // to the arrival.
  // WAIT-LABEL: @wait_last_trailing_barrier
  // WAIT: ttng.tmem_load
  // WAIT-DAG: ttng.tmem_store
  // WAIT-DAG: ttng.tmem_wait load
  // WAIT: ttng.tmem_wait store
  // WAIT-NOT: ttg.barrier
  // WAIT: %{{.*}} = tt.atomic_poll acquire
  // WAIT-NOT: ttg.barrier
  // WAIT: ttng.arrive_barrier
  // WAIT-NEXT: tt.return
  tt.func @wait_last_trailing_barrier(%data: tensor<128x64xf32, #blocked>, %ptr: !tt.ptr<i32>, %expected: i32, %bar: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>) {
    %true = arith.constant true
    %low = ttng.tmem_alloc %data {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %high = ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %loaded = ttng.tmem_load %low : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    ttng.tmem_store %loaded, %high, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %first = tt.atomic_poll acquire, gpu, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    %last = tt.atomic_poll acquire, gpu, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
    tt.return
  }

  // Complete the initializer before branching without repeating its store wait.
  // Both paths reuse it without a rendezvous and complete loads before exit.
  // WAIT-LABEL: @wait_initializer_before_fork
  // WAIT: ttng.tmem_alloc
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: cf.cond_br
  // WAIT-NOT: ttng.tmem_wait store
  // WAIT-NOT: ttg.barrier local
  // WAIT: ttng.tmem_load
  // WAIT-NEXT: tt.store
  // WAIT-NEXT: cf.br
  // WAIT-NOT: ttng.tmem_wait store
  // WAIT-NOT: ttg.barrier local
  // WAIT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: tt.return
  tt.func @wait_initializer_before_fork(%condition: i1, %data: tensor<128x64xf32, #blocked>, %out: tensor<128x64x!tt.ptr<f32>, #blocked>) -> tensor<128x64xf32, #blocked> {
    %mem = ttng.tmem_alloc %data {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    cf.cond_br %condition, ^body, ^merge
  ^body:
    %value = ttng.tmem_load %mem : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    tt.store %out, %value : tensor<128x64x!tt.ptr<f32>, #blocked>
    cf.br ^merge
  ^merge:
    %result = ttng.tmem_load %mem : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    tt.return %result : tensor<128x64xf32, #blocked>
  }

  // A single wait at the merge covers the stores from either arm before
  // the same owners overwrite the low tile.
  // CHECK-LABEL: @wait_cfg_join
  // CHECK: cf.cond_br
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: cf.br ^[[JOIN:bb[0-9]+]]
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: cf.br ^[[JOIN]]
  // CHECK: ^[[JOIN]]:
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: tt.return
  tt.func @wait_cfg_join(%condition: i1, %data: tensor<128x64xf32, #blocked>) {
    %true = arith.constant true
    %low = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %high = ttng.tmem_alloc {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    cf.cond_br %condition, ^left, ^right
  ^left:
    ttng.tmem_store %data, %low, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    cf.br ^merge
  ^right:
    ttng.tmem_store %data, %high, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    cf.br ^merge
  ^merge:
    ttng.tmem_store %data, %low, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // The first store must complete before the same owners overwrite it.
  // The acquire poll is independent of TMEM completion.
  // CHECK-LABEL: @wait_barrier_before_branch
  // CHECK: cf.cond_br
  // CHECK: ttng.tmem_store
  // CHECK-DAG: ttng.tmem_wait store
  // CHECK-DAG: %{{.*}} = tt.atomic_poll acquire
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: tt.return
  tt.func @wait_barrier_before_branch(%condition: i1, %ptr: !tt.ptr<i32>, %expected: i32, %data: tensor<128x128xf32, #blocked>) {
    %true = arith.constant true
    %mem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    cf.cond_br %condition, ^left, ^merge
  ^left:
    ttng.tmem_store %data, %mem, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    cf.br ^merge
  ^merge:
    ttng.tmem_store %data, %mem, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // The header completes the initial and loop-carried stores. One wait on
  // exit completes the disjoint loads from every iteration.
  // CHECK-LABEL: @wait_cfg_backedge
  // CHECK: cf.br ^[[HEADER:bb[0-9]+]]
  // CHECK: ^[[HEADER]](
  // CHECK: ttng.tmem_wait store
  // CHECK-NEXT: cf.cond_br
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NEXT: %{{.*}} = arith.subi
  // CHECK-NEXT: cf.br ^[[HEADER]]
  // CHECK: ttng.tmem_wait load
  // CHECK-NEXT: tt.return
  tt.func @wait_cfg_backedge(%iterations: i32, %data: tensor<128x128xf32, #blocked>) {
    %true = arith.constant true
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %mem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %source = ttng.tmem_alloc %data {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    cf.br ^header(%iterations : i32)
  ^header(%remaining: i32):
    %again = arith.cmpi sgt, %remaining, %zero : i32
    cf.cond_br %again, ^body, ^exit
  ^body:
    %loaded = ttng.tmem_load %source : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    ttng.tmem_store %loaded, %mem, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %next = arith.subi %remaining, %one : i32
    cf.br ^header(%next : i32)
  ^exit:
    tt.return
  }

  // Global effects leave stores pending; opaque synchronization needs publication.
  // WAIT-LABEL: @wait_global_effects
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: ttng.prefetch_bulk
  // WAIT-NEXT: %{{.*}} = tt.load
  // WAIT-NEXT: tt.print
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: %{{.*}} = tt.elementwise_inline_asm
  // WAIT-NEXT: tt.return
  tt.func @wait_global_effects(%ptr: !tt.ptr<i32>, %value: f32) {
    %true = arith.constant true
    %c64 = arith.constant 64 : i32
    %data = tt.splat %value : f32 -> tensor<128x128xf32, #blocked>
    %mem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %mem, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.prefetch_bulk %ptr, %c64, %true : !tt.ptr<i32>, i32, i1
    %loaded = tt.load %ptr {isVolatile = true} : !tt.ptr<i32>
    tt.print "loaded" {hex = false, isSigned = array<i32: 1>} : %loaded : i32
    %opaque = tt.elementwise_inline_asm "bar.sync 0; mov.u32 $0, 0;" {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32
    tt.return
  }

  // The load waits for the TMEM store. The descriptor acquire can defer
  // its rendezvous to the first shared-memory access.
  // WAIT-LABEL: @wait_tensormap_acquire_barrier
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: %{{.*}} = ttg.global_scratch_alloc
  // WAIT-NEXT: ttng.tensormap_create
  // WAIT-NEXT: ttng.tensormap_fenceproxy_acquire
  // WAIT-NEXT: %{{.*}} = ttng.reinterpret_tensor_descriptor
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: %{{.*}} = ttng.tmem_load
  // CHECK-NEXT: ttg.barrier local
  // WAIT-NEXT: %{{.*}} = ttg.local_alloc
  // WAIT-NEXT: ttng.fence_async_shared
  // CHECK-NEXT: ttg.barrier local
  // WAIT-NEXT: ttng.async_tma_copy_local_to_global
  // WAIT-NEXT: ttng.async_tma_store_wait
  // WAIT: ttng.tmem_wait load
  tt.func @wait_tensormap_acquire_barrier(%out: !tt.ptr<f32>, %value: f32) {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant 32 : i32
    %c128 = arith.constant 128 : i32
    %c512 = arith.constant 512 : i64
    %data = tt.splat %value : f32 -> tensor<128x128xf32, #blocked>
    %mem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %mem, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %raw = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
    ttng.tensormap_create %raw, %out, [%c32, %c128], [%c128, %c128], [%c512], [%c1, %c1] {elem_type = 7 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<f32>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_fenceproxy_acquire %raw : !tt.ptr<i8>
    %desc = ttng.reinterpret_tensor_descriptor %raw : !tt.ptr<i8> to !tt.tensordesc<128x128xf32, #shared_copy>
    %loaded = ttng.tmem_load %mem : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %smem = ttg.local_alloc %loaded : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #shared_copy, #ttg.shared_memory, mutable>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %smem : !tt.tensordesc<128x128xf32, #shared_copy>, !ttg.memdesc<128x128xf32, #shared_copy, #ttg.shared_memory, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32, read_only}
    ttg.local_dealloc %smem : !ttg.memdesc<128x128xf32, #shared_copy, #ttg.shared_memory, mutable>
    tt.return
  }

  // Independent shared-memory synchronization leaves the store pending until
  // the TMEM load needs it. These empty copy groups are already complete.
  // WAIT-LABEL: @tmem_wait_after_shared_synchronization
  // WAIT: ttng.tmem_alloc
  // WAIT-NEXT: ttng.fence_async_shared
  // WAIT-NEXT: ttg.async_commit_group
  // WAIT-NEXT: ttg.async_wait
  // WAIT-NEXT: ttng.async_tma_store_wait
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: %{{.*}} = ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: tt.return
  tt.func @tmem_wait_after_shared_synchronization(%data: tensor<128x128xf32, #blocked>) {
    %mem = ttng.tmem_alloc %data {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.fence_async_shared {bCluster = false}
    ttg.async_commit_group
    ttg.async_wait {num = 0 : i32}
    ttng.async_tma_store_wait {pendings = 0 : i32, read_only}
    %loaded = ttng.tmem_load %mem : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }

  // The caller completes the disjoint load and the callee's store at return.
  // TMEM-LABEL: @tmem_call_disjoint
  // TMEM: ttng.tmem_load
  // TMEM-NEXT: tt.call @tmem_entry_b
  // TMEM-NEXT: ttng.tmem_wait load
  // TMEM-NEXT: ttng.tmem_wait store
  // TMEM-NEXT: tt.return
  tt.func @tmem_call_disjoint() {
    %alloc = ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %loaded = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.call @tmem_entry_b() : () -> ()
    tt.return
  }

  // WAIT-LABEL: @elementwise_inline_asm_tmem_effects
  // CHECK: ttng.tmem_alloc
  // WAIT: ttng.tmem_wait store
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: {{.*}}tt.elementwise_inline_asm
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: {{.*}}ttng.tmem_load
  tt.func @elementwise_inline_asm_tmem_effects(%data: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> {
    %mem = ttng.tmem_alloc %data {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %unused = tt.elementwise_inline_asm "mov.u32 $0, 0;" {constraints = "=r,r", pure = false, packed_element = 1 : i32} %mem : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> i32
    %result = ttng.tmem_load %mem : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return %result : tensor<128x128xf32, #blocked>
  }

  // WAIT-LABEL: @inline_asm_no_tmem_effects
  // WAIT: ttg.inline_asm
  // WAIT-NEXT: {{.*}}ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: tt.return
  tt.func @inline_asm_no_tmem_effects(%mem: !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>) -> tensor<128x128xf32, #blocked> {
    ttg.inline_asm "// opaque descriptor" {constraints = "r", pure = false} %mem : (!ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>) -> ()
    %result = ttng.tmem_load %mem : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return %result : tensor<128x128xf32, #blocked>
  }

  // TMEM-LABEL: @tmem_load_argument
  // TMEM: ttg.barrier all
  // TMEM-NEXT: {{.*}} = ttng.tmem_load
  // TMEM-NEXT: tt.return
  tt.func private @tmem_load_argument(%mem: !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>) -> tensor<128x128xf32, #blocked> attributes {noinline = true} {
    ttg.barrier all
    %loaded = ttng.tmem_load %mem : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return %loaded : tensor<128x128xf32, #blocked>
  }

  // The load stays pending through the global store until the opaque operation.
  // WAIT-LABEL: @tmem_call_load_then_publish
  // WAIT: tt.call @tmem_load_argument
  // WAIT-NEXT: tt.store
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: tt.call @tmem_opaque_handoff
  // WAIT-NEXT: tt.return
  tt.func private @tmem_call_load_then_publish(%mem: !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>, %out: !tt.ptr<f32>, %value: f32) -> tensor<128x128xf32, #blocked> attributes {noinline = true} {
    %loaded = tt.call @tmem_load_argument(%mem) : (!ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>) -> tensor<128x128xf32, #blocked>
    tt.store %out, %value : !tt.ptr<f32>
    tt.call @tmem_opaque_handoff() : () -> ()
    tt.return %loaded : tensor<128x128xf32, #blocked>
  }

  // The caller waits for the callee's load before overwriting the tile.
  // TMEM-LABEL: @tmem_call_load_then_store
  // TMEM: ttng.tmem_alloc
  // TMEM-NEXT: ttng.tmem_wait store
  // TMEM-NEXT: {{.*}} = tt.call @tmem_load_argument
  // TMEM-NEXT: ttng.tmem_wait load
  // TMEM-NEXT: ttng.tmem_store
  // TMEM-NEXT: ttng.tmem_wait store
  // TMEM-NEXT: tt.return
  tt.func @tmem_call_load_then_store(%data: tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked> {
    %true = arith.constant true
    %mem = ttng.tmem_alloc %data {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %loaded = tt.call @tmem_load_argument(%mem) : (!ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>) -> tensor<128x128xf32, #blocked>
    ttng.tmem_store %data, %mem, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return %loaded : tensor<128x128xf32, #blocked>
  }

  // TMEM completion converges the issuing lanes. The shared store still
  // needs a memory-ordering barrier before the peer can observe its arrival.
  // CHECK-LABEL: @shared_and_tmem_publication
  // CHECK: ttg.local_store
  // CHECK-DAG: ttng.tmem_wait store
  // CHECK-DAG: ttg.barrier{{( warp)?}} local
  // CHECK: ttng.arrive_barrier
  tt.func @shared_and_tmem_publication() -> !publish_data {
    %phase = arith.constant 0 : i32
    %bar = ttg.local_alloc : () -> !publish_bar
    %shared = ttg.local_alloc : () -> !publish_smem
    %mem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !publish_tmem
    ttng.init_barrier %bar, 1 : !publish_bar
    %out = ttg.warp_specialize(%mem, %shared, %bar)
    default {
      ttng.wait_barrier %bar, %phase : !publish_bar
      %s = ttg.local_load %shared : !publish_smem -> !publish_data
      %t = ttng.tmem_load %mem : !publish_tmem -> !publish_data
      %sum = arith.addi %s, %t : !publish_data
      ttg.warp_yield %sum : !publish_data
    }
    partition0(%tile: !publish_tmem, %buf: !publish_smem, %ready: !publish_bar) num_warps(4) {
      %true = arith.constant true
      %value = arith.constant dense<1> : !publish_data
      ttg.local_store %value, %buf : !publish_data -> !publish_smem
      ttng.tmem_store %value, %tile, %true : !publish_data -> !publish_tmem
      ttng.arrive_barrier %ready, 1 : !publish_bar
      ttg.warp_return
    } : (!publish_tmem, !publish_smem, !publish_bar) -> !publish_data
    tt.return %out : !publish_data
  }

  // Each warp reports its own completion when the full barrier precedes the
  // wait. A full barrier after the wait lets one thread signal for all warps.
  // CHECK-LABEL: @tmem_wait_after_cta_stays_distributed
  // CHECK: ttng.init_barrier {{.*}}, 8 :
  // CHECK-NEXT: ttng.init_barrier {{.*}}, 4 :
  // CHECK: default
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttg.barrier all
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttng.arrive_barrier {{.*}}, 4 {per_warp}
  // CHECK-NEXT: ttg.barrier all
  // CHECK-NEXT: ttng.arrive_barrier {{.*}}, 4 :
  // CHECK: partition0
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttng.arrive_barrier {{.*}}, 4 {per_warp}
  tt.func @tmem_wait_after_cta_stays_distributed(%value: !publish_data) -> !publish_data {
    %phase = arith.constant 0 : i32
    %true = arith.constant true
    %ready = ttg.local_alloc : () -> !publish_bar
    %done = ttg.local_alloc : () -> !publish_bar
    %mem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !publish_tmem
    ttng.init_barrier %ready, 2 : !publish_bar
    ttng.init_barrier %done, 1 : !publish_bar
    ttg.warp_specialize(%mem, %ready, %done)
    default {
      ttng.tmem_store %value, %mem, %true : !publish_data -> !publish_tmem
      ttg.barrier all
      ttng.arrive_barrier %ready, 1 : !publish_bar
      ttg.barrier all
      ttng.arrive_barrier %ready, 1 : !publish_bar
      ttng.wait_barrier %done, %phase : !publish_bar
      %next = arith.addi %value, %value : !publish_data
      ttng.tmem_store %next, %mem, %true : !publish_data -> !publish_tmem
      ttg.warp_yield
    }
    partition0(%tile: !publish_tmem, %full: !publish_bar, %empty: !publish_bar) num_warps(4) {
      %zero = arith.constant 0 : i32
      ttng.wait_barrier %full, %zero : !publish_bar
      ttg.barrier all
      %v = ttng.tmem_load %tile : !publish_tmem -> !publish_data
      ttng.arrive_barrier %empty, 1 : !publish_bar
      tt.print "tile" {hex = false, isSigned = array<i32: 1>} : %v : !publish_data
      ttg.warp_return
    } : (!publish_tmem, !publish_bar, !publish_bar) -> ()
    %loaded = ttng.tmem_load %mem : !publish_tmem -> !publish_data
    tt.return %loaded : !publish_data
  }

  // A one-column shift changes the warp owning each overlapping word.
  // The wait before the existing CTA barrier publishes to both later readers.
  // WAIT-LABEL: @st_shifted_warp_ownership
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: ttng.wait_barrier
  // WAIT-NEXT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: tt.return
  tt.func @st_shifted_warp_ownership(%data: tensor<128x2xf32, #blocked_broadcast_warps>, %ack: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>) -> (tensor<128x2xf32, #blocked_broadcast_warps>, tensor<128x2xf32, #blocked_broadcast_warps>) attributes {"ttg.num-warps" = 8 : i32} {
    %true = arith.constant true
    %phase = arith.constant 0 : i32
    %parent = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x4xf32, #tmem128, #ttng.tensor_memory, mutable>
    %a = ttng.tmem_subslice %parent {offset = 0 : i32, dim = 1 : i32} : !ttg.memdesc<128x4xf32, #tmem128, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x2xf32, #tmem128, #ttng.tensor_memory, mutable, 128x4>
    %b = ttng.tmem_subslice %parent {offset = 1 : i32, dim = 1 : i32} : !ttg.memdesc<128x4xf32, #tmem128, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x2xf32, #tmem128, #ttng.tensor_memory, mutable, 128x4>
    ttng.tmem_store %data, %b, %true : tensor<128x2xf32, #blocked_broadcast_warps> -> !ttg.memdesc<128x2xf32, #tmem128, #ttng.tensor_memory, mutable, 128x4>
    ttng.tmem_store %data, %a, %true : tensor<128x2xf32, #blocked_broadcast_warps> -> !ttg.memdesc<128x2xf32, #tmem128, #ttng.tensor_memory, mutable, 128x4>
    ttg.barrier local
    ttng.wait_barrier %ack, %phase : !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
    %same_warp = ttng.tmem_load %a : !ttg.memdesc<128x2xf32, #tmem128, #ttng.tensor_memory, mutable, 128x4> -> tensor<128x2xf32, #blocked_broadcast_warps>
    %cross_warp = ttng.tmem_load %b : !ttg.memdesc<128x2xf32, #tmem128, #ttng.tensor_memory, mutable, 128x4> -> tensor<128x2xf32, #blocked_broadcast_warps>
    tt.return %same_warp, %cross_warp : tensor<128x2xf32, #blocked_broadcast_warps>, tensor<128x2xf32, #blocked_broadcast_warps>
  }

  // Completing the full store for a same-owner overwrite must retain its
  // untouched columns for a later load from additional warps.
  // WAIT-LABEL: @waw_keeps_unwritten_columns
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: ttng.tmem_store
  // WAIT: ttg.barrier local
  // WAIT-NEXT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: tt.return
  tt.func @waw_keeps_unwritten_columns(%data: tensor<128x4xf32, #blocked_broadcast_warps>, %half: tensor<128x2xf32, #blocked_broadcast_warps>, %pred: i1) -> tensor<128x1xf32, #blocked_broadcast_warps> attributes {"ttg.num-warps" = 8 : i32} {
    %true = arith.constant true
    %parent = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x4xf32, #tmem128, #ttng.tensor_memory, mutable>
    %high = ttng.tmem_subslice %parent {offset = 2 : i32, dim = 1 : i32} : !ttg.memdesc<128x4xf32, #tmem128, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x2xf32, #tmem128, #ttng.tensor_memory, mutable, 128x4>
    %low = ttng.tmem_subslice %parent {offset = 0 : i32, dim = 1 : i32} : !ttg.memdesc<128x4xf32, #tmem128, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x1xf32, #tmem128, #ttng.tensor_memory, mutable, 128x4>
    ttng.tmem_store %data, %parent, %true : tensor<128x4xf32, #blocked_broadcast_warps> -> !ttg.memdesc<128x4xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %half, %high, %pred : tensor<128x2xf32, #blocked_broadcast_warps> -> !ttg.memdesc<128x2xf32, #tmem128, #ttng.tensor_memory, mutable, 128x4>
    %loaded = ttng.tmem_load %low : !ttg.memdesc<128x1xf32, #tmem128, #ttng.tensor_memory, mutable, 128x4> -> tensor<128x1xf32, #blocked_broadcast_warps>
    tt.return %loaded : tensor<128x1xf32, #blocked_broadcast_warps>
  }

  // Broadcasting one column across two warp groups has no unique warp owner.
  // Keep the RAW rendezvous even though the register layouts are identical.
  // WAIT-LABEL: @st_then_ld_broadcast_warps
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: tt.return
  tt.func @st_then_ld_broadcast_warps(%data: tensor<128x1xf32, #blocked_broadcast_warps>) -> tensor<128x1xf32, #blocked_broadcast_warps> attributes {"ttg.num-warps" = 8 : i32} {
    %true = arith.constant true
    %mem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %mem, %true : tensor<128x1xf32, #blocked_broadcast_warps> -> !ttg.memdesc<128x1xf32, #tmem128, #ttng.tensor_memory, mutable>
    %loaded = ttng.tmem_load %mem : !ttg.memdesc<128x1xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked_broadcast_warps>
    tt.return %loaded : tensor<128x1xf32, #blocked_broadcast_warps>
  }

  // An acquire can overlap the store, which must complete before publication.
  // WAIT-LABEL: @wait_acquire_before_arrival
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: ttng.wait_barrier
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: ttng.arrive_barrier
  // WAIT-NEXT: tt.return
  tt.func @wait_acquire_before_arrival(
      %data: tensor<128x128xf32, #blocked>,
      %ack: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>,
      %ready: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>,
      %pred: i1) {
    %true = arith.constant true
    %phase = arith.constant 0 : i32
    %mem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %mem, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.wait_barrier %ack, %phase, %pred : !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
    ttng.arrive_barrier %ready, 1 : !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
    tt.return
  }

  // Complete the first store after the acquire, before its same-warp overwrite.
  // WAIT-LABEL: @wait_acquire_saved_cta_waw
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: ttng.wait_barrier
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: tt.return
  tt.func @wait_acquire_saved_cta_waw(
      %data: tensor<128x128xf32, #blocked>,
      %ack: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>) {
    %true = arith.constant true
    %phase = arith.constant 0 : i32
    %mem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %mem, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    ttng.wait_barrier %ack, %phase : !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
    ttng.tmem_store %data, %mem, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // Each path completes the store at its first load or return.
  // WAIT-LABEL: @wait_acquire_before_cfg
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: ttng.wait_barrier
  // WAIT-NEXT: cf.cond_br
  // WAIT: ttng.tmem_wait store
  // WAIT-NEXT: tt.return
  // WAIT: ttng.tmem_wait store
  // WAIT-NEXT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: tt.return
  tt.func @wait_acquire_before_cfg(
      %data: tensor<128x128xf32, #blocked>,
      %ack: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>,
      %condition: i1) -> tensor<128x128xf32, #blocked> {
    %true = arith.constant true
    %phase = arith.constant 0 : i32
    %mem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %mem, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.wait_barrier %ack, %phase : !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
    cf.cond_br %condition, ^return_data, ^load_data
  ^return_data:
    tt.return %data : tensor<128x128xf32, #blocked>
  ^load_data:
    %loaded = ttng.tmem_load %mem : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return %loaded : tensor<128x128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#narrow = #ttng.tensor_memory_encoding<blockM = 128, blockN = 2, colStride = 1>
!data = tensor<128x2xi32, #blocked>
!pool = !ttg.memdesc<128x4xi32, #tmem, #ttng.tensor_memory, mutable>
!view = !ttg.memdesc<128x2xi32, #tmem, #ttng.tensor_memory, mutable, 128x4>
!peer = !ttg.memdesc<128x2xi32, #narrow, #ttng.tensor_memory>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100"} {
  tt.func private @publish(%flag: !tt.ptr<i32>) attributes {noinline = true} {
    %one = arith.constant 1 : i32
    %old = tt.atomic_rmw add, release, cta, %flag, %one : (!tt.ptr<i32>, i32) -> i32
    tt.return
  }

  // A distinct worker-visible allocation reuses the pool's first two columns.
  // Only the overlapping store must complete before the release atomic.
  // The two private loads can share a wait across the call to @publish.
  // GUARD-LABEL: @reused_storage
  // GUARD: ttng.tmem_store
  // GUARD-NEXT: ttng.tmem_wait store
  // GUARD-NEXT: ttg.barrier all
  // GUARD-NEXT: {{.*}}ttg.thread_predicate
  // GUARD: tt.atomic_rmw add, release
  // GUARD: ttng.tmem_store
  // GUARD-NEXT: ttg.barrier all
  // GUARD-NEXT: {{.*}}ttg.thread_predicate
  // GUARD-NEXT: cf.cond_br {{.*}}, {{.*}}, ^[[JOIN:bb[0-9]+]]
  // GUARD-NOT: ttng.tmem_wait store
  // GUARD: ^[[JOIN]]:
  // GUARD-NEXT: ttng.tmem_wait store
  // GUARD-NEXT: {{.*}}ttng.tmem_load
  // GUARD-NEXT: tt.call @publish
  // GUARD-NEXT: {{.*}}ttng.tmem_load
  tt.func public @reused_storage(%flag: !tt.ptr<i32>, %overlap: i1) {
    %one = arith.constant 1 : i32
    %true = arith.constant true
    %data = arith.constant dense<1> : !data
    %pool = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !pool
    %values = scf.if %overlap -> (!data) {
      %a = ttng.tmem_subslice %pool {offset = 0 : i32} : !pool -> !view
      ttng.tmem_store %data, %a, %true : !data -> !view
      %old = tt.atomic_rmw add, release, cta, %flag, %one : (!tt.ptr<i32>, i32) -> i32
      %v = ttng.tmem_load %a : !view -> !data
      scf.yield %v : !data
    } else {
      %b = ttng.tmem_subslice %pool {offset = 2 : i32} : !pool -> !view
      ttng.tmem_store %data, %b, %true : !data -> !view
      %old = tt.atomic_rmw add, release, cta, %flag, %one : (!tt.ptr<i32>, i32) -> i32
      %v = ttng.tmem_load %b : !view -> !data
      tt.call @publish(%flag) : (!tt.ptr<i32>) -> ()
      %after = ttng.tmem_load %b : !view -> !data
      %sum = arith.addi %v, %after : !data
      scf.yield %sum : !data
    }
    %peer = ttng.tmem_alloc %values {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (!data) -> !peer
    ttg.warp_specialize(%peer)
    default {
      ttg.warp_yield
    }
    partition0(%tile: !peer) num_warps(8) {
      %v = ttng.tmem_load %tile : !peer -> !data
      tt.print "tile" {hex = false, isSigned = array<i32: 1>} : %v : !data
      ttg.warp_return
    } : (!peer) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
!data = tensor<128x1xi32, #blocked>
!tile = !ttg.memdesc<128x1xi32, #tmem, #ttng.tensor_memory, mutable>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The first release lets the worker read. The second lets the producer overwrite.
  // GUARD-LABEL: @peer_handoff
  // GUARD: ttng.tmem_store
  // GUARD-NEXT: ttng.tmem_wait store
  // GUARD-NEXT: ttg.barrier all
  // GUARD-NEXT: {{.*}}ttg.thread_predicate
  // GUARD: tt.atomic_rmw exch, release
  // GUARD: partition0
  // GUARD: ttng.tmem_load
  // GUARD-NEXT: ttng.tmem_wait load
  // GUARD-NEXT: ttg.barrier all
  // GUARD-NEXT: {{.*}}ttg.thread_predicate
  // GUARD: tt.atomic_rmw add, release
  tt.func public @peer_handoff(%flag: !tt.ptr<i32>) {
    %tile = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !tile
    ttg.warp_specialize(%tile, %flag)
    default {
      %one = arith.constant 1 : i32
      %two = arith.constant 2 : i32
      %true = arith.constant true
      %a = arith.constant dense<1> : !data
      %b = arith.constant dense<2> : !data
      ttng.tmem_store %a, %tile, %true : !data -> !tile
      %old = tt.atomic_rmw exch, release, cta, %flag, %one : (!tt.ptr<i32>, i32) -> i32
      %done = tt.atomic_poll acquire, cta, %flag, %two : !tt.ptr<i32>, i32 -> i1
      ttng.tmem_store %b, %tile, %true : !data -> !tile
      ttg.warp_yield
    }
    partition0(%captured: !tile, %ready: !tt.ptr<i32>) num_warps(4) {
      %one = arith.constant 1 : i32
      %done = tt.atomic_poll acquire, cta, %ready, %one : !tt.ptr<i32>, i32 -> i1
      %v = ttng.tmem_load %captured : !tile -> !data
      %old = tt.atomic_rmw add, release, cta, %ready, %one : (!tt.ptr<i32>, i32) -> i32
      tt.print "tile" {hex = false, isSigned = array<i32: 1>} : %v : !data
      ttg.warp_return
    } : (!tile, !tt.ptr<i32>) -> ()
    tt.return
  }

  tt.func private @swap_tiles(%private: !tile, %worker: !tile) -> (!tile, !tile) attributes {noinline = true} {
    tt.return %worker, %private : !tile, !tile
  }

  // The worker captures the first result. Only its store needs a wait before release.
  // GUARD-LABEL: @publish_returned_tiles
  // GUARD: ttng.tmem_store
  // GUARD: ttng.tmem_wait store
  // GUARD: tt.atomic_rmw add, release
  // GUARD: ttng.tmem_store
  // GUARD-NOT: ttng.tmem_wait store
  // GUARD: tt.atomic_rmw add, release
  // GUARD: ttng.tmem_wait store
  // GUARD-NEXT: {{.*}}ttng.tmem_load
  tt.func public @publish_returned_tiles(%flag: !tt.ptr<i32>) {
    %true = arith.constant true
    %one = arith.constant 1 : i32
    %ones = arith.constant dense<1> : !data
    %a = ttng.tmem_alloc {tensor_memory_col_offset = 4 : i32, tensor_memory_row_offset = 0 : i32} : () -> !tile
    %b = ttng.tmem_alloc {tensor_memory_col_offset = 8 : i32, tensor_memory_row_offset = 0 : i32} : () -> !tile
    %worker, %private = tt.call @swap_tiles(%a, %b) : (!tile, !tile) -> (!tile, !tile)
    ttng.tmem_store %ones, %worker, %true : !data -> !tile
    %first = tt.atomic_rmw add, release, cta, %flag, %one : (!tt.ptr<i32>, i32) -> i32
    ttng.tmem_store %ones, %private, %true : !data -> !tile
    %second = tt.atomic_rmw add, release, cta, %flag, %one : (!tt.ptr<i32>, i32) -> i32
    %private_value = ttng.tmem_load %private : !tile -> !data
    tt.print "private" {hex = false, isSigned = array<i32: 1>} : %private_value : !data
    ttg.warp_specialize(%worker)
    default {
      ttg.warp_yield
    }
    partition0(%captured: !tile) num_warps(4) {
      %v = ttng.tmem_load %captured : !tile -> !data
      tt.print "worker" {hex = false, isSigned = array<i32: 1>} : %v : !data
      ttg.warp_return
    } : (!tile) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#row = #ttg.slice<{dim = 1, parent = #blocked}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
!data = tensor<128x1xi32, #blocked>
!ptrs = tensor<128x1x!tt.ptr<i32>, #blocked>
!tile = !ttg.memdesc<128x1xi32, #tmem, #ttng.tensor_memory, mutable>
!bar = !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Notifications on an unrelated counter let each private transfer batch
  // with the next transfer of the same kind.
  // GUARD-LABEL: @batch_private_accesses
  // GUARD: ttng.tmem_store
  // GUARD-NOT: ttng.tmem_wait store
  // GUARD: ttng.tmem_store
  // GUARD-NEXT: ttng.tmem_wait store
  // GUARD-NEXT: {{.*}}ttng.tmem_load
  // GUARD-NEXT: {{.*}}ttg.thread_predicate
  // GUARD-NOT: ttng.tmem_wait load
  // GUARD: ttng.tmem_load
  // GUARD: ttng.tmem_wait load
  // GUARD-NOT: ttng.tmem_wait
  // GUARD: tt.return
  tt.func public @batch_private_accesses(%flags: !tt.ptr<i32>, %counter: !tt.ptr<i32>) {
    %phase = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %true = arith.constant true
    %ones = arith.constant dense<1> : !data
    %rows = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #row>
    %offsets = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32, #row> -> !data
    %base = tt.splat %flags : !tt.ptr<i32> -> !ptrs
    %addresses = tt.addptr %base, %offsets : !ptrs, !data
    %bar = ttg.local_alloc : () -> !bar
    ttng.init_barrier %bar, 2 : !bar
    %a = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !tile
    %b = ttng.tmem_alloc {tensor_memory_col_offset = 4 : i32, tensor_memory_row_offset = 0 : i32} : () -> !tile
    ttng.tmem_store %ones, %a, %true : !data -> !tile
    %old = tt.atomic_rmw add, release, cta, %addresses, %ones : (!ptrs, !data) -> !data
    ttng.arrive_barrier %bar, 1 : !bar
    ttng.tmem_store %old, %b, %true : !data -> !tile
    %left = ttng.tmem_load %a : !tile -> !data
    %count = tt.atomic_rmw add, relaxed, cta, %counter, %one : (!tt.ptr<i32>, i32) -> i32
    ttng.arrive_barrier %bar, 1 : !bar
    %right = ttng.tmem_load %b : !tile -> !data
    %sum = arith.addi %left, %right : !data
    tt.print "tiles" {hex = false, isSigned = array<i32: 1>} : %sum : !data
    ttng.wait_barrier %bar, %phase : !bar
    tt.return
  }
}
