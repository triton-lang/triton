// RUN: triton-opt %s -triton-nvidia-gpu-tmem-barrier-insertion -test-print-membar -triton-nvidia-gpu-tmem-wait-insertion | FileCheck %s --check-prefixes=CHECK,WAIT
// RUN: triton-opt %s -triton-nvidia-gpu-tmem-wait-insertion | FileCheck %s --check-prefix=WAIT

#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared_copy = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared_fp8 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared_scales = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_scales = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear64 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0]], block = []}>
#tmem128 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem64 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @alloc_then_alloc
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier local
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
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: tt.return
  tt.func @alloc_then_ld(%arg0: tensor<128x128xf32, #blocked>) {
    %0 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }

  // CHECK-LABEL: @alloc_then_st
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_store
  tt.func @alloc_then_st(%arg0: tensor<128x128xf32, #blocked>) {
    %true = arith.constant true
    %0 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %arg0, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @alloc_then_mma
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tc_gen5_mma
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

  // CHECK-LABEL: @ld_then_alloc
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttg.barrier local
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

  // CHECK-LABEL: @ld_then_st
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: tt.return
  tt.func @ld_then_st(%arg0: tensor<128x128xf32, #blocked>) {
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    ttng.tmem_store %arg0, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
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

  // An uninitialized allocation does not access reused storage. Wait only
  // before its first store.
  // CHECK-LABEL: @st_then_alloc
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier local
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

  // CHECK-LABEL: @st_then_ld
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_load
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
  // CHECK-NEXT: ttg.barrier local
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
  // CHECK-NEXT: ttg.barrier local
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
  // CHECK-NEXT: ttg.barrier local
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

  // CHECK-LABEL: @ld_then_alloc_then_st_aliases_second_row
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttg.barrier local
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
  // CHECK-NEXT: ttg.barrier local
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

  // CHECK-LABEL: @ld_then_tmem_copy
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_copy
  // CHECK-NOT: ttng.tmem_wait
  // CHECK: tt.return
  tt.func @ld_then_tmem_copy(
      %arg0: !ttg.memdesc<128x128xf32, #shared_copy, #ttg.shared_memory>) {
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    ttng.tmem_copy %arg0, %0 : !ttg.memdesc<128x128xf32, #shared_copy, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
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

  // CHECK-LABEL: @tmem_entry_c
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: tt.return
  tt.func private @tmem_entry_c() {
    %cst = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %alloc = ttng.tmem_alloc %cst {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    tt.return
  }

  // CHECK-LABEL: @tmem_entry_b
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.call @tmem_entry_c
  tt.func private @tmem_entry_b() {
    tt.call @tmem_entry_c() : () -> ()
    tt.return
  }

  // The call supplies no entry barrier; wait before the caller's CTA barrier.
  // CHECK-LABEL: @tmem_entry_a
  // CHECK-NOT: ttg.barrier local
  // CHECK: tt.call @tmem_entry_b
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: tt.call @tmem_entry_b
  tt.func @tmem_entry_a() {
    tt.call @tmem_entry_b() : () -> ()
    %alloc = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    %loaded = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.call @tmem_entry_b() : () -> ()
    tt.return
  }

  // Wait for the disjoint store before the MMA signals its completion barrier.
  // CHECK-LABEL: @wait_disjoint_mma_publication
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
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

  // Batch disjoint stores before the local barrier and loads at the return.
  // CHECK-LABEL: @wait_disjoint_batches
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: tt.return
  tt.func @wait_disjoint_batches(%data: tensor<128x64xf32, #blocked>) -> (tensor<128x64xf32, #blocked>, tensor<128x64xf32, #blocked>) {
    %true = arith.constant true
    %low = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %high = ttng.tmem_alloc {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %low, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %high, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %a = ttng.tmem_load %low : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    %b = ttng.tmem_load %high : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    tt.return %a, %b : tensor<128x64xf32, #blocked>, tensor<128x64xf32, #blocked>
  }

  // Hoisting the wait for %low must leave %high's later store pending.
  // WAIT-LABEL: @wait_hoisted_preserves_later_store
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: %{{.*}} = tt.atomic_poll acquire
  // WAIT-NEXT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: ttg.barrier local
  // WAIT-NEXT: ttng.tmem_load
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: tt.return
  tt.func @wait_hoisted_preserves_later_store(%data: tensor<128x64xf32, #blocked>, %ptr: !tt.ptr<i32>, %expected: i32) -> (tensor<128x64xf32, #blocked>, tensor<128x64xf32, #blocked>) {
    %true = arith.constant true
    %low = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %high = ttng.tmem_alloc {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %low, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    ttng.tmem_store %data, %high, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %a = ttng.tmem_load %low : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    %b = ttng.tmem_load %high : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    tt.return %a, %b : tensor<128x64xf32, #blocked>, tensor<128x64xf32, #blocked>
  }

  // Reuse the acquire poll's trailing CTA barrier for the following TMEM load.
  // WAIT-LABEL: @wait_trailing_atomic_barrier
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: %{{.*}} = tt.atomic_poll acquire
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

  // Keep %low's store pending across the barrier between %high's load and store.
  // Wait for both stores before arrive_barrier.
  // CHECK-LABEL: @wait_disjoint_corrections
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: ttng.arrive_barrier
  tt.func @wait_disjoint_corrections(%data: tensor<128x64xf32, #blocked>, %bar: !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>) {
    %true = arith.constant true
    %low = ttng.tmem_alloc %data {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %high = ttng.tmem_alloc %data {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %a = ttng.tmem_load %low : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    ttng.tmem_store %a, %low, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %b = ttng.tmem_load %high : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    ttng.tmem_store %b, %high, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #ttg.shared_memory, mutable>
    tt.return
  }

  // Defer load and store waits to the last poll's trailing barrier, even without
  // later memory conflicts.
  // CHECK-LABEL: @wait_last_trailing_barrier
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NEXT: %{{.*}} = tt.atomic_poll acquire
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: %{{.*}} = tt.atomic_poll acquire
  // CHECK-NEXT: tt.return
  tt.func @wait_last_trailing_barrier(%data: tensor<128x64xf32, #blocked>, %ptr: !tt.ptr<i32>, %expected: i32) {
    %true = arith.constant true
    %low = ttng.tmem_alloc %data {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %high = ttng.tmem_alloc {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %loaded = ttng.tmem_load %low : !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked>
    ttng.tmem_store %loaded, %high, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem128, #ttng.tensor_memory, mutable>
    %first = tt.atomic_poll acquire, gpu, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    %last = tt.atomic_poll acquire, gpu, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    tt.return
  }

  // Complete the initializer before branching without repeating its store wait.
  // Each path keeps its rendezvous, and loads complete before leaving the block.
  // WAIT-LABEL: @wait_initializer_before_fork
  // WAIT: ttng.tmem_alloc
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: cf.cond_br
  // WAIT-NOT: ttng.tmem_wait store
  // CHECK: ttg.barrier local
  // WAIT: ttng.tmem_load
  // WAIT-NEXT: tt.store
  // WAIT-NEXT: ttng.tmem_wait load
  // WAIT-NEXT: cf.br
  // WAIT-NOT: ttng.tmem_wait store
  // CHECK: ttg.barrier local
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

  // Complete stores before branching. The merge keeps its rendezvous before
  // overwriting %low, which may have been written by the left branch.
  // CHECK-LABEL: @wait_cfg_join
  // CHECK: cf.cond_br
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: cf.br
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: cf.br
  // CHECK-NOT: ttng.tmem_wait store
  // CHECK: ttg.barrier local
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

  // Complete the first store before the poll's trailing barrier, even though
  // the conflicting store is after the merge.
  // CHECK-LABEL: @wait_barrier_before_branch
  // CHECK: cf.cond_br
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: %{{.*}} = tt.atomic_poll acquire
  // CHECK-NEXT: cf.br
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

  // Complete stores before the backedge; the next iteration keeps its
  // rendezvous before writing the same descriptor again.
  // CHECK-LABEL: @wait_cfg_backedge
  // CHECK: cf.br
  // CHECK-NOT: ttng.tmem_wait
  // CHECK: cf.cond_br
  // CHECK-NOT: ttng.tmem_wait
  // CHECK: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_store
  // CHECK-NEXT: %{{.*}} = arith.subi
  // CHECK-NEXT: ttng.tmem_wait store
  // CHECK-NEXT: cf.br
  // CHECK-NOT: ttng.tmem_wait
  // CHECK: tt.return
  tt.func @wait_cfg_backedge(%iterations: i32, %data: tensor<128x128xf32, #blocked>) {
    %true = arith.constant true
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %mem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    cf.br ^header(%iterations : i32)
  ^header(%remaining: i32):
    %again = arith.cmpi sgt, %remaining, %zero : i32
    cf.cond_br %again, ^body, ^exit
  ^body:
    ttng.tmem_store %data, %mem, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %next = arith.subi %remaining, %one : i32
    cf.br ^header(%next : i32)
  ^exit:
    tt.return
  }

  // Global effects leave stores pending; opaque synchronization completes them.
  // WAIT-LABEL: @wait_global_effects
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: %{{.*}} = tt.load
  // WAIT-NEXT: tt.print
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: %{{.*}} = tt.elementwise_inline_asm
  // WAIT-NEXT: tt.return
  tt.func @wait_global_effects(%ptr: !tt.ptr<i32>, %value: f32) {
    %true = arith.constant true
    %data = tt.splat %value : f32 -> tensor<128x128xf32, #blocked>
    %mem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttng.tmem_store %data, %mem, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %loaded = tt.load %ptr {isVolatile = true} : !tt.ptr<i32>
    tt.print "loaded" {hex = false, isSigned = array<i32: 1>} : %loaded : i32
    %opaque = tt.elementwise_inline_asm "bar.sync 0; mov.u32 $0, 0;" {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32
    tt.return
  }

  // Reuse the descriptor acquire's trailing barrier for the following load.
  // WAIT-LABEL: @wait_tensormap_acquire_barrier
  // WAIT: ttng.tmem_store
  // WAIT-NEXT: %{{.*}} = ttg.global_scratch_alloc
  // WAIT-NEXT: ttng.tensormap_create
  // WAIT-NEXT: ttng.tmem_wait store
  // WAIT-NEXT: ttng.tensormap_fenceproxy_acquire
  // WAIT-NEXT: %{{.*}} = ttng.reinterpret_tensor_descriptor
  // WAIT-NEXT: %{{.*}} = ttng.tmem_load
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

  // Tensor-memory offsets share one physical frame across functions. Calls
  // still complete loads even when the callee's footprint is disjoint.
  // CHECK-LABEL: @tmem_call_disjoint
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_wait load
  // CHECK-NEXT: tt.call @tmem_entry_b
  tt.func @tmem_call_disjoint() {
    %alloc = ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %loaded = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.call @tmem_entry_b() : () -> ()
    tt.return
  }
}
