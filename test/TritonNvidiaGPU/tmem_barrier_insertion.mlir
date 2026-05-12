// RUN: triton-opt %s -triton-nvidia-gpu-tmem-barrier-insertion | FileCheck %s

#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared_copy = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked_scales = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear64 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[16, 0], [32, 0]], block = []}>
#tmem128 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem64 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @alloc_then_alloc
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_alloc
  tt.func @alloc_then_alloc(%arg0: tensor<128x128xf32, #blocked>) {
    %0 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @alloc_then_ld
  // CHECK: ttng.tmem_alloc
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_load
  tt.func @alloc_then_ld(%arg0: tensor<128x128xf32, #blocked>) {
    %0 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }

  // CHECK-LABEL: @alloc_then_st
  // CHECK: ttng.tmem_alloc
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
  tt.func @ld_then_ld() {
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    %1 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %2 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }

  // CHECK-LABEL: @ld_then_st
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_store
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

  // CHECK-LABEL: @st_then_alloc
  // CHECK: ttng.tmem_store
  // CHECK-NEXT: ttng.tmem_alloc
  tt.func @st_then_alloc(%arg0: tensor<128x128xf32, #blocked>) {
    %true = arith.constant true
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    ttg.barrier local
    ttng.tmem_store %arg0, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @st_then_ld
  // CHECK: ttng.tmem_store
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
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_store
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
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tc_gen5_mma
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

  // CHECK-LABEL: @ld_then_alloc_then_st_aliases_second_row
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttng.tmem_alloc
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
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_alloc
  tt.func @alloc_then_alloc_partial_overlap(%arg0: tensor<128x128xf32, #blocked>) {
    %0 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_alloc %arg0 {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>
    tt.return
  }

  // CHECK-LABEL: @st_then_mma_scaled_scale_operand
  // CHECK: ttng.tmem_store
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

  // CHECK-LABEL: @ld_then_tmem_copy
  // CHECK: ttng.tmem_load
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_copy
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
  // CHECK-NEXT: ttg.barrier local
  // CHECK-NEXT: ttng.tmem_copy
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
}
