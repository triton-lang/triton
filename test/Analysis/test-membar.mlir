// RUN: triton-opt %s -split-input-file --mlir-disable-threading --convert-scf-to-cf -test-print-membar 2>&1 | FileCheck %s

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#sliceAd0 = #triton_gpu.slice<{dim = 0, parent = #AL}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#A_SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#A_SHARED_T = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#B_SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #triton_gpu.mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A_DOT = #triton_gpu.dot_op<{opIdx = 0, parent = #C}>
#B_DOT = #triton_gpu.dot_op<{opIdx = 1, parent = #C}>

module attributes {"triton_gpu.num-warps" = 4 : i32} {

// CHECK-LABEL: matmul_loop
// There shouldn't be any membar with the dot op encoding.
func.func @matmul_loop(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %a_ptr_init = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %b_ptr_init = tt.broadcast %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %a_ = tt.load %a_ptr, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
    %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A_DOT>
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B_DOT>
    %c = tt.dot %a, %b, %prev_c {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #A_DOT> * tensor<32x128xf16, #B_DOT> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  return
}

// CHECK-LABEL: raw_single_block
func.func @raw_single_block(%A : !tt.ptr<f16>) {
  %cst1 = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %cst2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %0 = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %1 = tt.load %0, %cst1, %cst2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
  %2 = triton_gpu.convert_layout %1 : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.convert_layout
  %3 = triton_gpu.convert_layout %2 : (tensor<128x32xf16, #A_SHARED>) -> tensor<128x32xf16, #A_SHARED>
  return
}

// CHECK-LABEL: war_single_block
func.func @war_single_block(%A : !tt.ptr<f16>) {
  %cst1 = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %cst2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %0 = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %1 = tt.load %0, %cst1, %cst2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
  %2 = triton_gpu.convert_layout %1 : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.convert_layout
  %3 = triton_gpu.convert_layout %2 : (tensor<128x32xf16, #A_SHARED>) -> tensor<128x32xf16, #AL>
  // CHECK: gpu.barrier
  // CHECK-NEXT: %4 = triton_gpu.convert_layout
  %4 = triton_gpu.convert_layout %1 : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A_SHARED>
  return
}

// CHECK-LABEL: scratch
func.func @scratch() {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %0 = tt.cat %cst0, %cst0 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.convert_layout
  %1 = triton_gpu.convert_layout %0 : (tensor<32x16xf16, #A_SHARED>) -> tensor<32x16xf16, #AL>
  %2 = tt.reduce %1 {redOp = 1 : i32, axis = 0 : i32} : tensor<32x16xf16, #AL> -> tensor<16xf16, #sliceAd0>
  return
}

// CHECK-LABEL: async_wait
func.func @async_wait() {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %0 = tt.cat %cst0, %cst0 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  triton_gpu.async_wait {num = 4 : i32}
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.convert_layout
  %1 = triton_gpu.convert_layout %0 : (tensor<32x16xf16, #A_SHARED>) -> tensor<32x16xf16, #AL>
  return
}

// CHECK-LABEL: alloc
func.func @alloc() {
  %0 = triton_gpu.alloc_tensor : tensor<16x16xf16, #A_SHARED>
  %1 = tt.cat %0, %0 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.convert_layout
  %2 = triton_gpu.convert_layout %1 : (tensor<32x16xf16, #A_SHARED>) -> tensor<32x16xf16, #AL>
  return
}

// CHECK-LABEL: extract_slice
func.func @extract_slice() {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<1x16x16xf16, #A_SHARED>
  %index = arith.constant 0 : i32
  %0 = triton_gpu.extract_slice %cst0[%index, 0, 0][1, 16, 16][1, 1, 1] : tensor<1x16x16xf16, #A_SHARED> to tensor<16x16xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.convert_layout
  %1 = triton_gpu.convert_layout %0 : (tensor<16x16xf16, #A_SHARED>) -> tensor<16x16xf16, #AL>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.convert_layout
  %2 = triton_gpu.convert_layout %1 : (tensor<16x16xf16, #AL>) -> tensor<16x16xf16, #A_SHARED>
  return
}

// CHECK-LABEL: trans
func.func @trans() {
  // CHECK-NOT: gpu.barrier
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #A_SHARED>
  %b = tt.trans %cst0 : (tensor<16x32xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED_T>
  return
}

// CHECK-LABEL: insert_slice_async_op
func.func @insert_slice_async_op(%A : !tt.ptr<f16>, %i1 : i1) {
  %a_ptr = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<16x16x!tt.ptr<f16>, #AL>
  %mask = tt.splat %i1 : (i1) -> tensor<16x16xi1, #AL>
  %other = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %tensor = triton_gpu.alloc_tensor : tensor<1x16x16xf16, #A_SHARED>
  %index = arith.constant 0 : i32
  %3 = triton_gpu.insert_slice_async %a_ptr, %tensor, %index, %mask, %other {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16x!tt.ptr<f16>, #AL> -> tensor<1x16x16xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %4 = tt.cat %3, %3 {axis = 0} : (tensor<1x16x16xf16, #A_SHARED>, tensor<1x16x16xf16, #A_SHARED>) -> tensor<2x16x16xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %5 = tt.cat %4, %4 {axis = 0} : (tensor<2x16x16xf16, #A_SHARED>, tensor<2x16x16xf16, #A_SHARED>) -> tensor<4x16x16xf16, #A_SHARED>
  return
}

// CHECK-LABEL: insert_slice_op
func.func @insert_slice_op(%A : !tt.ptr<f16>, %i1 : i1) {
  %a_ptr = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<16x16x!tt.ptr<f16>, #AL>
  %mask = tt.splat %i1 : (i1) -> tensor<16x16xi1, #AL>
  %other = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %tensor = arith.constant dense<0.000000e+00> : tensor<1x16x16xf16, #A_SHARED>
  %index = arith.constant 0 : index
  %2 = tt.load %a_ptr, %mask, %other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #AL>
  // CHECK: gpu.barrier
  // CHECK-NEXT: tensor.insert_slice
  %3 = tensor.insert_slice %2 into %tensor[%index, 0, 0][1, 16, 16][1, 1, 1]: tensor<16x16xf16, #AL> into tensor<1x16x16xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %4 = tt.cat %3, %3 {axis = 0} : (tensor<1x16x16xf16, #A_SHARED>, tensor<1x16x16xf16, #A_SHARED>) -> tensor<2x16x16xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %5 = tt.cat %4, %4 {axis = 0} : (tensor<2x16x16xf16, #A_SHARED>, tensor<2x16x16xf16, #A_SHARED>) -> tensor<4x16x16xf16, #A_SHARED>
  return
}

// If branch inserted a barrier for %cst0 and %cst1, but else didn't, then the barrier should be inserted in the parent region
// CHECK-LABEL: multi_blocks
func.func @multi_blocks(%i1 : i1) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  scf.if %i1 {
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.cat
    %0 = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    scf.yield
  } else {
    %cst2 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
    %cst3 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.cat
    %1 = tt.cat %cst2, %cst3 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    scf.yield
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %2 = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  return
}

// Both branches inserted a barrier for %cst0 and %cst1, then the barrier doesn't need to be inserted in the parent region
// CHECK-LABEL: multi_blocks_join_barrier
func.func @multi_blocks_join_barrier(%i1 : i1) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  scf.if %i1 {
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.cat
    %0 = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    scf.yield
  } else {
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.cat
    %1 = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    scf.yield
  }
  %a_ = triton_gpu.convert_layout %cst0 : (tensor<16x16xf16, #A_SHARED>) -> tensor<16x16xf16, #AL>
  return
}

// Read yielded tensor requires a barrier
// CHECK-LABEL: multi_blocks_yield
func.func @multi_blocks_yield(%i1 : i1) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  %a = scf.if %i1 -> (tensor<32x16xf16, #A_SHARED>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.cat
    %0 = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    scf.yield %0 : tensor<32x16xf16, #A_SHARED>
  } else {
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.cat
    %1 = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    scf.yield %1 : tensor<32x16xf16, #A_SHARED>
  }
  %a_ = triton_gpu.convert_layout %cst0 : (tensor<16x16xf16, #A_SHARED>) -> tensor<16x16xf16, #AL>
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %4 = tt.cat %a, %a {axis = 0} : (tensor<32x16xf16, #A_SHARED>, tensor<32x16xf16, #A_SHARED>) -> tensor<64x16xf16, #A_SHARED>
  return
}

// Even though the entry block doesn't have a barrier, the successors should have barriers
// CHECK-LABEL: multi_blocks_entry_no_shared
func.func @multi_blocks_entry_no_shared(%i1 : i1) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %a = scf.if %i1 -> (tensor<32x16xf16, #A_SHARED>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.cat
    %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
    %0 = tt.cat %cst1, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    scf.yield %0 : tensor<32x16xf16, #A_SHARED>
  } else {
    // CHECK-NOT: gpu.barrier
    // CHECK: arith.constant
    %cst1 = arith.constant dense<0.000000e+00> : tensor<32x16xf16, #A_SHARED>
    scf.yield %cst1 : tensor<32x16xf16, #A_SHARED>
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %1 = tt.cat %a, %a {axis = 0} : (tensor<32x16xf16, #A_SHARED>, tensor<32x16xf16, #A_SHARED>) -> tensor<64x16xf16, #A_SHARED>
  return
}

// Conservatively add a barrier as if the branch (%i1) is never taken
// CHECK-LABEL: multi_blocks_noelse
func.func @multi_blocks_noelse(%i1 : i1) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  scf.if %i1 {
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.cat
    %0 = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    scf.yield
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.convert_layout
  %1 = triton_gpu.convert_layout %cst0 : (tensor<16x16xf16, #A_SHARED>) -> tensor<16x16xf16, #AL>
  return
}

// Conservatively add a barrier as if the branch (%i2) is never taken
// CHECK-LABEL: multi_blocks_nested_scf
func.func @multi_blocks_nested_scf(%i1 : i1, %i2 : i1) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  scf.if %i1 {
    scf.if %i2 {
      // CHECK: gpu.barrier
      // CHECK-NEXT: tt.cat
      %0 = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
      scf.yield
    }
    scf.yield
  } else {
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.cat
    %1 = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    scf.yield
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.convert_layout
  %2 = triton_gpu.convert_layout %cst0 : (tensor<16x16xf16, #A_SHARED>) -> tensor<16x16xf16, #AL>
  return
}

// CHECK-LABEL: for
func.func @for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.cat
    %5 = tt.cat %a_shared, %b_shared {axis = 0} : (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) -> tensor<256x32xf16, #A_SHARED>
    scf.yield %b_shared, %a_shared, %a_shared : tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>
  }
  return
}

// Although a_shared and b_shared are synced before entering the loop,
// they are reassociated with aliases (c_shared) and thus require a barrier.
// CHECK-LABEL: for_alias
func.func @for_alias(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %0 = tt.cat %a_shared_init, %b_shared_init {axis = 0} : (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) -> tensor<256x32xf16, #A_SHARED>
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) {
    %cst1 = tt.cat %a_shared_init, %b_shared_init {axis = 0} : (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) -> tensor<256x32xf16, #AL>
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.cat
    %7 = tt.cat %a_shared, %b_shared {axis = 0} : (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) -> tensor<256x32xf16, #AL>
    scf.yield %c_shared, %a_shared, %b_shared : tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %9 = tt.cat %0, %0 {axis = 0} : (tensor<256x32xf16, #A_SHARED>, tensor<256x32xf16, #A_SHARED>) -> tensor<512x32xf16, #A_SHARED>
  return
}

// Although cst2 is not an argument of scf.yield, its memory is reused by cst1.
// So we need a barrier both before and after cst1
// CHECK-LABEL: for_reuse
func.func @for_reuse(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %0 = tt.cat %a_shared_init, %b_shared_init {axis = 0} : (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) -> tensor<256x32xf16, #A_SHARED>
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.cat
    %6 = tt.cat %a_shared_init, %b_shared_init {axis = 0} : (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) -> tensor<256x32xf16, #A_SHARED>
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.cat
    %7 = tt.cat %a_shared, %b_shared {axis = 0} : (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) -> tensor<256x32xf16, #A_SHARED>
    scf.yield %c_shared, %a_shared, %b_shared : tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %9 = tt.cat %0, %0 {axis = 0} : (tensor<256x32xf16, #A_SHARED>, tensor<256x32xf16, #A_SHARED>) -> tensor<512x32xf16, #A_SHARED>
  return
}

// CHECK-LABEL: for_reuse_nested
func.func @for_reuse_nested(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %0 = tt.cat %a_shared_init, %b_shared_init {axis = 0} : (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) -> tensor<256x32xf16, #A_SHARED>
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.cat
    %6 = tt.cat %a_shared_init, %b_shared_init {axis = 0} : (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) -> tensor<256x32xf16, #A_SHARED>
    %a_shared_next, %b_shared_next, %c_shared_next = scf.for %ivv = %lb to %ub step %step iter_args(%a_shared_nested = %a_shared_init, %b_shared_nested = %b_shared_init, %c_shared_nested = %c_shared_init) -> (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) {
      // CHECK: gpu.barrier
      // CHECK-NEXT:  tt.cat
      %12 = tt.cat %a_shared_nested, %b_shared_nested {axis = 0} : (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) -> tensor<256x32xf16, #A_SHARED>
      scf.yield %c_shared_nested, %a_shared_nested, %b_shared_nested : tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>
    }
    scf.yield %c_shared, %a_shared, %b_shared : tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT:  tt.cat
  %15 = tt.cat %0, %0 {axis = 0} : (tensor<256x32xf16, #A_SHARED>, tensor<256x32xf16, #A_SHARED>) -> tensor<512x32xf16, #A_SHARED>
  return
}

// repeatedly write to the same shared memory addresses
// CHECK-LABEL: for_for_if
func.func @for_for_if(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) {
    %c_shared_next = scf.for %jv = %lb to %ub step %step iter_args(%c_shared_next = %c_shared) -> (tensor<128x32xf16, #A_SHARED>) {
      %c_shared_next_next = scf.if %i1 -> tensor<128x32xf16, #A_SHARED> {
        // CHECK: gpu.barrier
        // CHECK-NEXT: arith.constant
        %cst0 = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
        scf.yield %cst0 : tensor<128x32xf16, #A_SHARED>
      } else {
        // CHECK: gpu.barrier
        // CHECK-NEXT: arith.constant
        %cst0 = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
        scf.yield %cst0 : tensor<128x32xf16, #A_SHARED>
      }
      scf.yield %c_shared_next_next : tensor<128x32xf16, #A_SHARED>
    }
    scf.yield %a_shared, %b_shared, %c_shared_next : tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>
  }
  return
}

// c_block_next can either be converted from c_shared_init or c_shared_next_next
// CHECK-LABEL: for_if_for
func.func @for_if_for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK: gpu.barrier
  %c_blocked = triton_gpu.convert_layout %c_shared_init : (tensor<128x32xf16, #A_SHARED>) -> tensor<128x32xf16, #AL> 

  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) {
    %c_shared_next_next = scf.if %i1 -> tensor<128x32xf16, #A_SHARED> {
      // CHECK: gpu.barrier
      // CHECK-NEXT: arith.constant
      %cst0 = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
      scf.yield %cst0 : tensor<128x32xf16, #A_SHARED>
    } else {
      %c_shared_ = scf.for %jv = %lb to %ub step %step iter_args(%c_shared_next = %c_shared) -> (tensor<128x32xf16, #A_SHARED>) {
        // CHECK: gpu.barrier
        // CHECK-NEXT: triton_gpu.convert_layout
        %c_blocked_next = triton_gpu.convert_layout %c_shared_next : (tensor<128x32xf16, #A_SHARED>) -> tensor<128x32xf16, #AL> 
        scf.yield %c_shared : tensor<128x32xf16, #A_SHARED>
      }
      scf.yield %c_shared_ : tensor<128x32xf16, #A_SHARED>
    }
    // CHECK-NOT: gpu.barrier
    %b_blocked_next = triton_gpu.convert_layout %b_shared: (tensor<128x32xf16, #A_SHARED>) -> tensor<128x32xf16, #AL> 
    scf.yield %a_shared, %b_shared, %c_shared_next_next : tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>
  }
  return
}

// CHECK-LABEL: cf_if
func.func @cf_if(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>
  cf.cond_br %i1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %0 = tt.cat %cst, %cst_0 {axis = 0 : i64} : (tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>, tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>) -> tensor<32x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>
  cf.br ^bb2
^bb2:  // 2 preds: ^bb0, ^bb1
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.convert_layout
  %1 = triton_gpu.convert_layout %cst : (tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>) -> tensor<16x16xf16, #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  return
}

func.func @cf_if_else(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>
  cf.cond_br %i1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %0 = tt.cat %cst, %cst_0 {axis = 0 : i64} : (tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>, tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>) -> tensor<32x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>
  cf.br ^bb3(%0 : tensor<32x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>)
^bb2:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %1 = tt.cat %cst, %cst_0 {axis = 0 : i64} : (tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>, tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>) -> tensor<32x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>
  cf.br ^bb3(%1 : tensor<32x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>)
^bb3(%2: tensor<32x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>):  // 2 preds: ^bb1, ^bb2
  cf.br ^bb4
^bb4:  // pred: ^bb3
  %3 = triton_gpu.convert_layout %cst : (tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>) -> tensor<16x16xf16, #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %4 = tt.cat %2, %2 {axis = 0 : i64} : (tensor<32x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>, tensor<32x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>) -> tensor<64x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>
  return
}

func.func @cf_if_else_return(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>
  cf.cond_br %i1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %0 = tt.cat %cst, %cst_0 {axis = 0 : i64} : (tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>, tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>) -> tensor<32x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>
  return
^bb2:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.cat
  %1 = tt.cat %cst, %cst_0 {axis = 0 : i64} : (tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>, tensor<16x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>) -> tensor<32x16xf16, #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>>
  return
}

}
