// RUN: triton-opt %s -split-input-file --mlir-disable-threading -test-print-membar 2>&1 | FileCheck %s

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#sliceAd0 = #triton_gpu.slice<{dim = 0, parent = #AL}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#A = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#B = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #triton_gpu.mma<{version = 2, warpsPerCTA = [4, 1]}>

// CHECK-LABEL: matmul_loop
func @matmul_loop(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
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
    %a_ = tt.load %a_ptr, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isOtherUnspecified = false, isVolatile = false} : tensor<128x32xf16, #AL>
    %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isOtherUnspecified = false, isVolatile = false} : tensor<32x128xf16, #BL>
    %b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>
    // CHECK: Membar 13
    %c = tt.dot %a, %b, %prev_c {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  return
}

// CHECK-LABEL: raw_single_block
func @raw_single_block(%A : !tt.ptr<f16>) {
  %cst1 = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %cst2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %a_ptr = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a1_ = tt.load %a_ptr, %cst1, %cst2 {cache = 1 : i32, evict = 1 : i32, isOtherUnspecified = false, isVolatile = false} : tensor<128x32xf16, #AL>
  %a1 = triton_gpu.convert_layout %a1_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>
  // CHECK: Membar 5
  %a2 = triton_gpu.convert_layout %a1 : (tensor<128x32xf16, #A>) -> tensor<128x32xf16, #A>
  return
}

// CHECK-LABEL: war_single_block
func @war_single_block(%A : !tt.ptr<f16>) {
  %cst1 = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %cst2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %a_ptr = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a1_ = tt.load %a_ptr, %cst1, %cst2 {cache = 1 : i32, evict = 1 : i32, isOtherUnspecified = false, isVolatile = false} : tensor<128x32xf16, #AL>
  %a1 = triton_gpu.convert_layout %a1_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>
  // CHECK: Membar 5
  %a2 = triton_gpu.convert_layout %a1 : (tensor<128x32xf16, #A>) -> tensor<128x32xf16, #AL>
  // a2's liveness range ends here, and a3 and a2 have the same address range.
  // So it makes sense to have a WAR dependency between a2 and a3.
  // CHECK-NEXT: Membar 7
  %a3 = triton_gpu.convert_layout %a1_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>
  return
}

// CHECK-LABEL: scratch
func @scratch() {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  // CHECK: Membar 1
  %a = tt.cat %cst0, %cst0 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
  // CHECK-NEXT: Membar 3
  %aa = triton_gpu.convert_layout %a : (tensor<32x16xf16, #A>) -> tensor<32x16xf16, #AL>
  %b = tt.reduce %aa {redOp = 1 : i32, axis = 0 : i32} : tensor<32x16xf16, #AL> -> tensor<16xf16, #sliceAd0>
  return
}

// CHECK-LABEL: async_wait
func @async_wait() {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  // CHECK: Membar 1
  %a = tt.cat %cst0, %cst0 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
  triton_gpu.async_wait {num = 4 : i32}
  // CHECK-NEXT: Membar 4
  %a_ = triton_gpu.convert_layout %a : (tensor<32x16xf16, #A>) -> tensor<32x16xf16, #AL>
  return
}

// CHECK-LABEL: alloc
func @alloc() {
  %cst0 = triton_gpu.alloc_tensor : tensor<16x16xf16, #A>
  %a = tt.cat %cst0, %cst0 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
  // CHECK: Membar 2
  %b = triton_gpu.convert_layout %a : (tensor<32x16xf16, #A>) -> tensor<32x16xf16, #AL>
  return
}

// CHECK-LABEL: extract_slice
func @extract_slice() {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<1x16x16xf16, #A>
  %index = arith.constant 0 : i32
  %cst1 = triton_gpu.extract_slice %cst0, %index { axis = 0 : i32 } : tensor<1x16x16xf16, #A> -> tensor<16x16xf16, #A>
  // CHECK: Membar 3
  %cst2 = triton_gpu.convert_layout %cst1 : (tensor<16x16xf16, #A>) -> tensor<16x16xf16, #AL>
  // CHECK-NEXT: Membar 5
  %cst3 = triton_gpu.convert_layout %cst2 : (tensor<16x16xf16, #AL>) -> tensor<16x16xf16, #A>
  return
}

// CHECK-LABEL: insert_slice_async
func @insert_slice_async(%A : !tt.ptr<f16>, %i1 : i1) {
  %a_ptr = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<16x16x!tt.ptr<f16>, #AL>
  %mask = tt.splat %i1 : (i1) -> tensor<16x16xi1, #AL>
  %other = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %tensor = triton_gpu.alloc_tensor : tensor<1x16x16xf16, #A>
  %index = arith.constant 0 : i32
  %a = triton_gpu.insert_slice_async %a_ptr, %tensor, %index, %mask, %other {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isOtherUnspecified = false, isVolatile = false} : tensor<16x16x!tt.ptr<f16>, #AL> -> tensor<1x16x16xf16, #A>
  %b = tt.cat %a, %a {axis = 0} : (tensor<1x16x16xf16, #A>, tensor<1x16x16xf16, #A>) -> tensor<2x16x16xf16, #A>
  // CHECK: Membar 7
  %c = tt.cat %b, %b {axis = 0} : (tensor<2x16x16xf16, #A>, tensor<2x16x16xf16, #A>) -> tensor<4x16x16xf16, #A>
  return
}

// If branch inserted a barrier for %cst0 and %cst1, but else didn't, then the barrier should be inserted in the parent region
// CHECK-LABEL: multi_blocks
func @multi_blocks(%i1 : i1) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  scf.if %i1 {
    // CHECK: Membar 2
    %a = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
    scf.yield
  } else {
    %cst2 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
    %cst3 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
    // CHECK-NEXT: Membar 7
    %b = tt.cat %cst2, %cst3 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
    scf.yield
  }
  // CHECK-NEXT: Membar 10
  %c = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
  return
}

// Both branches inserted a barrier for %cst0 and %cst1, then the barrier doesn't need to be inserted in the parent region
// CHECK-LABEL: multi_blocks_join_barrier
func @multi_blocks_join_barrier(%i1 : i1) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  scf.if %i1 {
    // CHECK: Membar 2
    %a = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
    scf.yield
  } else {
    // CHECK-NEXT: Membar 5
    %a = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
    scf.yield
  }
  %a_ = triton_gpu.convert_layout %cst0 : (tensor<16x16xf16, #A>) -> tensor<16x16xf16, #AL>
  return
}

// Read yielded tensor requires a barrier
// CHECK-LABEL: multi_blocks_yield
func @multi_blocks_yield(%i1 : i1) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  %a = scf.if %i1 -> (tensor<32x16xf16, #A>) {
    // CHECK: Membar 2
    %a = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
    scf.yield %a : tensor<32x16xf16, #A>
  } else {
    // CHECK-NEXT: Membar 5
    %b = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
    scf.yield %b : tensor<32x16xf16, #A>
  }
  %a_ = triton_gpu.convert_layout %cst0 : (tensor<16x16xf16, #A>) -> tensor<16x16xf16, #AL>
  // CHECK-NEXT: Membar 9
  %b = tt.cat %a, %a {axis = 0} : (tensor<32x16xf16, #A>, tensor<32x16xf16, #A>) -> tensor<64x16xf16, #A>
  return
}

// Conservatively add a barrier as if the branch (%i1) is never taken
// CHECK-LABEL: multi_blocks_noelse
func @multi_blocks_noelse(%i1 : i1) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  scf.if %i1 {
    // CHECK: Membar 2
    %a = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
    scf.yield
  }
  %a_ = triton_gpu.convert_layout %cst0 : (tensor<16x16xf16, #A>) -> tensor<16x16xf16, #AL>
  return
}

// Conservatively add a barrier as if the branch (%i2) is never taken
// CHECK-LABEL: multi_blocks_nested_scf
func @multi_blocks_nested_scf(%i1 : i1, %i2 : i1) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  scf.if %i1 {
    scf.if %i2 {
      // CHECK: Membar 2
      %b = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
      scf.yield
    }
    scf.yield
  } else {
    // CHECK-NEXT: Membar 6
    %b = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
    scf.yield
  }
  // CHECK-NEXT: Membar 9
  %a_ = triton_gpu.convert_layout %cst0 : (tensor<16x16xf16, #A>) -> tensor<16x16xf16, #AL>
  return
}

// CHECK-LABEL: for
func @for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) {
    // CHECK-NEXT: Membar 3
    %cst0 = tt.cat %a_shared, %b_shared {axis = 0} : (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) -> tensor<256x32xf16, #A>
    scf.yield %b_shared, %a_shared, %a_shared : tensor<128x32xf16, #A>, tensor<128x32xf16, #A>, tensor<128x32xf16, #A>
  }
  return
}

// Although a_shared and b_shared are synced before entering the loop,
// they are reassociated with aliases (c_shared) and thus require a barrier.
// CHECK-LABEL: for_alias
func @for_alias(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  // CHECK-NEXT: Membar 2
  %cst0 = tt.cat %a_shared_init, %b_shared_init {axis = 0} : (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) -> tensor<256x32xf16, #A>
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) {
    %cst1 = tt.cat %a_shared_init, %b_shared_init {axis = 0} : (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) -> tensor<256x32xf16, #A>
    // CHECK-NEXT: Membar 6
    %cst2 = tt.cat %a_shared, %b_shared {axis = 0} : (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) -> tensor<256x32xf16, #A>
    scf.yield %c_shared, %a_shared, %b_shared : tensor<128x32xf16, #A>, tensor<128x32xf16, #A>, tensor<128x32xf16, #A>
  }
  // CHECK-NEXT: Membar 9
  %cst3 = tt.cat %cst0, %cst0 {axis = 0} : (tensor<256x32xf16, #A>, tensor<256x32xf16, #A>) -> tensor<512x32xf16, #A>
  return
}
