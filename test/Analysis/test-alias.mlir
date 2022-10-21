// RUN: triton-opt %s --mlir-disable-threading -test-print-alias -split-input-file 2>&1 | FileCheck %s

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
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
    // CHECK: %4 -> %4
    %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isOtherUnspecified = false, isVolatile = false} : tensor<32x128xf16, #BL>
    // CHECK-NEXT: %6 -> %6 
    %b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>
    %c = tt.dot %a, %b, %prev_c {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  return
}

// CHECK-LABEL: alloc
func @alloc(%A : !tt.ptr<f16>) {
  // CHECK: %cst -> %cst
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  // CHECK: %0 -> %0
  %cst2 = triton_gpu.alloc_tensor : tensor<16x16xf16, #A>
  return
}

// CHECK-LABEL: convert
func @convert(%A : !tt.ptr<f16>) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  // CHECK: %0 -> %0
  %cst1 = triton_gpu.convert_layout %cst0 : (tensor<16x16xf16, #AL>) -> tensor<16x16xf16, #A>
  return
}

// CHECK-LABEL: insert_slice_async
func @insert_slice_async(%A : !tt.ptr<f16>, %i1 : i1) {
  %a_ptr = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<16x16x!tt.ptr<f16>, #AL>
  %mask = tt.splat %i1 : (i1) -> tensor<16x16xi1, #AL>
  %other = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  // CHECK: %cst_0 -> %cst_0
  %tensor = arith.constant dense<0.000000e+00> : tensor<1x16x16xf16, #A>
  %index = arith.constant 0 : i32
  // CHECK: %2 -> %cst_0
  %a = triton_gpu.insert_slice_async %a_ptr, %tensor, %index, %mask, %other {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isOtherUnspecified = false, isVolatile = false} : tensor<16x16x!tt.ptr<f16>, #AL> -> tensor<1x16x16xf16, #A>
  return
}

// CHECK-LABEL: extract_slice
func @extract_slice(%A : !tt.ptr<f16>) {
  // CHECK: %cst -> %cst
  %cst0 = arith.constant dense<0.000000e+00> : tensor<1x16x16xf16, #A>
  %index = arith.constant 0 : i32
  // CHECK-NEXT: %0 -> %cst
  %cst1 = triton_gpu.extract_slice %cst0, %index { axis = 0 : i32 } : tensor<1x16x16xf16, #A> -> tensor<16x16xf16, #A>
  return
}

// CHECK-LABEL: if_cat
func @if_cat(%i1 : i1) {
  // CHECK: %cst -> %cst
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  // CHECK: %cst_0 -> %cst_0
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  // CHECK: %0 -> %1,%1
  %cst2 = scf.if %i1 -> tensor<32x16xf16, #A> {
    // CHECK: %1 -> %1
    %a = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
    scf.yield %a : tensor<32x16xf16, #A>
  } else {
    // CHECK: %1 -> %1
    %b = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A>, tensor<16x16xf16, #A>) -> tensor<32x16xf16, #A>
    scf.yield %b : tensor<32x16xf16, #A>
  }
  return
}

// CHECK-LABEL: if_alias
func @if_alias(%i1 : i1) {
  // CHECK: %cst -> %cst
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  // CHECK-NEXT: %cst_0 -> %cst_0
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A>
  // CHECK-NEXT: %0 -> %cst,%cst_0
  %cst2 = scf.if %i1 -> tensor<16x16xf16, #A> {
    scf.yield %cst0 : tensor<16x16xf16, #A>
  } else {
    scf.yield %cst1 : tensor<16x16xf16, #A>
  }
  return
}

// CHECK-LABEL: for
func @for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  // CHECK: %cst -> %cst
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  // CHECK-NEXT: %cst_0 -> %cst_0
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  // CHECK-NEXT: %cst_1 -> %cst_1
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  // CHECK-NEXT: %arg6 -> %cst
  // CHECK-NEXT: %arg7 -> %cst_0
  // CHECK-NEXT: %arg8 -> %cst_1
  // CHECK-NEXT: %0#0 -> %cst,%cst_0
  // CHECK-NEXT: %0#1 -> %cst,%cst_0
  // CHECK-NEXT: %0#2 -> %cst,%cst_0
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) {
    scf.yield %b_shared, %a_shared, %a_shared : tensor<128x32xf16, #A>, tensor<128x32xf16, #A>, tensor<128x32xf16, #A>
  }
  return
}

// CHECK-LABEL: for_if
func @for_if(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  // CHECK: %cst -> %cst
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  // CHECK-NEXT: %cst_0 -> %cst_0
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  // CHECK-NEXT: %cst_1 -> %cst_1
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  // CHECK-NEXT: %arg7 -> %cst
  // CHECK-NEXT: %arg8 -> %cst_0
  // CHECK-NEXT: %arg9 -> %cst_1
  // CHECK-NEXT: %0#0 -> %cst,%cst_0
  // CHECK-NEXT: %0#1 -> %cst,%cst_0
  // CHECK-NEXT: %0#2 -> %cst,%cst_0
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) {
    scf.if %i1 {
      %index = arith.constant 8 : i32
      // CHECK-NEXT: %1 -> %cst,%cst_0
      %cst0 = triton_gpu.extract_slice %a_shared, %index { axis = 0 : i32 } : tensor<128x32xf16, #A> -> tensor<32xf16, #A>
      scf.yield
    }
    scf.yield %b_shared, %a_shared, %a_shared : tensor<128x32xf16, #A>, tensor<128x32xf16, #A>, tensor<128x32xf16, #A>
  }
  return
}

// CHECK-LABEL: for_if_for
func @for_if_for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  // CHECK: %cst -> %cst
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  // CHECK-NEXT: %cst_0 -> %cst_0
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  // CHECK-NEXT: %cst_1 -> %cst_1
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  // CHECK-NEXT: %arg7 -> %cst
  // CHECK-NEXT: %arg8 -> %cst_0
  // CHECK-NEXT: %arg9 -> %cst_1
  // CHECK-NEXT: %0#0 -> %cst
  // CHECK-NEXT: %0#1 -> %cst_0
  // CHECK-NEXT: %0#2 -> %cst_2,%cst_2
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A>, tensor<128x32xf16, #A>, tensor<128x32xf16, #A>) {
    // CHECK-NEXT: %arg11 -> %cst_1,%cst_2,%cst_2
    // CHECK-NEXT: %1 -> %cst_2,%cst_2
    %c_shared_next = scf.for %jv = %lb to %ub step %step iter_args(%c_shared_next = %c_shared) -> (tensor<128x32xf16, #A>) {
      // CHECK-NEXT: %2 -> %cst_2,%cst_2
      %c_shared_next_next = scf.if %i1 -> tensor<128x32xf16, #A> {
        // CHECK-NEXT: %cst_2 -> %cst_2
        %cst0 = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
        scf.yield %cst0 : tensor<128x32xf16, #A>
      } else {
        // CHECK-NEXT: %cst_2 -> %cst_2
        %cst0 = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
        scf.yield %cst0 : tensor<128x32xf16, #A>
      }
      scf.yield %c_shared_next_next : tensor<128x32xf16, #A>
    }
    scf.yield %a_shared, %b_shared, %c_shared_next : tensor<128x32xf16, #A>, tensor<128x32xf16, #A>, tensor<128x32xf16, #A>
  }
  return
}
