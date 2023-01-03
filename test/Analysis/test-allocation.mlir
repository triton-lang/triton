// RUN: triton-opt %s -split-input-file --mlir-disable-threading -test-print-allocation 2>&1 | FileCheck %s

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#sliceAd0 = #triton_gpu.slice<{dim = 0, parent = #AL}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#A_SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#B_SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #triton_gpu.mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A_DOT = #triton_gpu.dot_op<{opIdx = 0, parent = #C}>
#B_DOT = #triton_gpu.dot_op<{opIdx = 1, parent = #C}>

module attributes {"triton_gpu.num-warps" = 4 : i32} {

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
    %a_ = tt.load %a_ptr, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
    // CHECK: offset = 0, size = 4608
    %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A_DOT>
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    // CHECK-NEXT: offset = 0, size = 4224
    %b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B_DOT>

    %c = tt.dot %a, %b, %prev_c {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #A_DOT> * tensor<32x128xf16, #B_DOT> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  return
  // CHECK-NEXT: size = 4608
}

// Shared memory is available after a tensor's liveness range ends
// CHECK-LABEL: reusable
func @reusable(%A : !tt.ptr<f16>) {
  %cst1 = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %cst2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %cst3 = arith.constant dense<true> : tensor<32x128xi1, #AL>
  %cst4 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #AL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_ptr = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %b_ptr = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #AL>
  %a1_ = tt.load %a_ptr, %cst1, %cst2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
  // CHECK-NEXT: offset = 0, size = 4608
  %a1 = triton_gpu.convert_layout %a1_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A_DOT>
  %a2_ = tt.load %b_ptr, %cst3, %cst4 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #AL>
  // CHECK-NEXT: offset = 0, size = 1152
  %a2 = triton_gpu.convert_layout %a2_ : (tensor<32x128xf16, #AL>) -> tensor<32x128xf16, #B_DOT>
  %a3_ = tt.load %a_ptr, %cst1, %cst2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
  // CHECK-NEXT: offset = 0, size = 4608
  %a3 = triton_gpu.convert_layout %a3_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A_DOT>
  %c = tt.dot %a1, %a2, %c_init {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #A_DOT> * tensor<32x128xf16, #B_DOT> -> tensor<128x128xf32, #C>
  %a4_ = tt.load %b_ptr, %cst3, %cst4 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #AL>
  // CHECK-NEXT: offset = 0, size = 1152
  %a4 = triton_gpu.convert_layout %a4_ : (tensor<32x128xf16, #AL>) -> tensor<32x128xf16, #B_DOT>
  %c1 = tt.dot %a3, %a4, %c {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #A_DOT> * tensor<32x128xf16, #B_DOT> -> tensor<128x128xf32, #C>
  return
  // CHECK-NEXT: size = 4608
}

// A tensor's shared memory offset is larger than it needs to accommodate further tensors
// %cst0->%c
// %cst1->%cst4
// %cst3->%g->%h->%i
// CHECK-LABEL: preallocate
func @preallocate(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 512
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1024, size = 512
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1536, size = 512
  %cst2 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 2048, size = 1024
  %a = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 3072, size = 1024
  %b = tt.cat %cst0, %cst2 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 0, size = 1024
  %c = tt.cat %cst1, %cst2 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1024, size = 1024
  %cst4 = arith.constant dense<0.000000e+00> : tensor<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 6144, size = 2048
  %e = tt.cat %a, %cst4 {axis = 0} : (tensor<32x16xf16, #A_SHARED>, tensor<32x16xf16, #A_SHARED>) -> tensor<64x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 8192, size = 2048
  %d = tt.cat %b, %cst4 {axis = 0} : (tensor<32x16xf16, #A_SHARED>, tensor<32x16xf16, #A_SHARED>) -> tensor<64x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 10240, size = 2048
  %f = tt.cat %c, %cst4 {axis = 0} : (tensor<32x16xf16, #A_SHARED>, tensor<32x16xf16, #A_SHARED>) -> tensor<64x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 0, size = 2048
  %cst5 = arith.constant dense<0.000000e+00> : tensor<64x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 2048, size = 4096
  %g = tt.cat %e, %cst5 {axis = 0} : (tensor<64x16xf16, #A_SHARED>, tensor<64x16xf16, #A_SHARED>) -> tensor<128x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 2048, size = 4096
  %h = tt.cat %d, %cst5 {axis = 0} : (tensor<64x16xf16, #A_SHARED>, tensor<64x16xf16, #A_SHARED>) -> tensor<128x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 2048, size = 4096
  %i = tt.cat %f, %cst5 {axis = 0} : (tensor<64x16xf16, #A_SHARED>, tensor<64x16xf16, #A_SHARED>) -> tensor<128x16xf16, #A_SHARED>
  return
  // CHECK-NEXT: size = 12288
}

// Unused tensors are immediately released
// CHECK-LABEL: unused
func @unused(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 1024
  %cst0 = arith.constant dense<0.000000e+00> : tensor<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 0, size = 512
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 512, size = 512
  %cst2 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1024, size = 1024
  %a = tt.cat %cst1, %cst2 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  return
  // CHECK: size = 2048
}

// cst0 is alive through the entire function, it cannot be released before the end of the function
// CHECK-LABEL: longlive
func @longlive(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 512
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 512, size = 512
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1024, size = 512
  %cst2 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1536, size = 1024
  %a = tt.cat %cst1, %cst2 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 512, size = 512
  %cst3 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1024, size = 512
  %cst4 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1536, size = 1024
  %b = tt.cat %cst3, %cst4 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1536, size = 512
  %cst5 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1536, size = 512
  %cst6 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1536, size = 1024
  %c = tt.cat %cst3, %cst4 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 512, size = 1024
  %d = tt.cat %cst0, %cst0 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  return
  // CHECK-NEXT: size = 2560
}

// CHECK-LABEL: alloc
func @alloc(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 512
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  // CHECK-NEXT: offset = 0, size = 512
  %cst2 = triton_gpu.alloc_tensor : tensor<16x16xf16, #A_SHARED>
  return
  // CHECK-NEXT: size = 512
}

// CHECK-LABEL: scratch
func @scratch() {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  // CHECK: scratch offset = 0, size = 512
  %b = tt.reduce %cst0 {redOp = 1 : i32, axis = 0 : i32} : tensor<16x16xf16, #AL> -> tensor<16xf16, #sliceAd0>
  return
  // CHECK-NEXT: size = 512
}

// CHECK-LABEL: insert_slice_async
func @insert_slice_async(%A : !tt.ptr<f16>, %i1 : i1) {
  %a_ptr = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<16x16x!tt.ptr<f16>, #AL>
  %mask = tt.splat %i1 : (i1) -> tensor<16x16xi1, #AL>
  %other = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  // CHECK: offset = 0, size = 512
  %tensor = arith.constant dense<0.000000e+00> : tensor<1x16x16xf16, #A_SHARED>
  %index = arith.constant 0 : i32
  %a = triton_gpu.insert_slice_async %a_ptr, %tensor, %index, %mask, %other {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16x!tt.ptr<f16>, #AL> -> tensor<1x16x16xf16, #A_SHARED>
  return
  // CHECK-NEXT: size = 512
}

// CHECK-LABEL: extract_slice
func @extract_slice(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 512
  %cst0 = arith.constant dense<0.000000e+00> : tensor<1x16x16xf16, #A_SHARED>
  %index = arith.constant 0 : index
  %cst1 = tensor.extract_slice %cst0[%index, 0, 0][1, 16, 16][1,1,1] : tensor<1x16x16xf16, #A_SHARED> to tensor<16x16xf16, #A_SHARED>
  return
  // CHECK-NEXT: size = 512
}

// B0 -> (B1) -> B0
// Memory used by B1 can be reused by B0.
// CHECK-LABEL: if
func @if(%i1 : i1) {
  // CHECK: offset = 0, size = 512
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 512, size = 512
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  scf.if %i1 {
    // CHECK-NEXT: offset = 1024, size = 1024
    %a = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    // CHECK-NEXT: offset = 1024, size = 1024
    %b = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  }
  // CHECK-NEXT: offset = 0, size = 512
  %cst2 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 512, size = 512
  %cst3 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1024, size = 1024
  %a = tt.cat %cst2, %cst3 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  return
  // CHECK-NEXT: size = 2048
}

// B0 -> (B1) -> (B2) -> B0
// Memory used by B0 cannot be reused by B1 or B2.
// CHECK-LABEL: if_else
func @if_else(%i1 : i1) {
  // CHECK: offset = 0, size = 512
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 512, size = 512
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  scf.if %i1 {
    // CHECK-NEXT: offset = 1024, size = 1024
    %a = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    // CHECK-NEXT: offset = 1024, size = 1024
    %b = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  } else {
    // CHECK-NEXT: offset = 1024, size = 512
    %cst2 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
    // CHECK-NEXT: offset = 1536, size = 512
    %cst3 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
    // CHECK-NEXT: offset = 2048, size = 1024
    %a = tt.cat %cst2, %cst3 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  }
  // CHECK-NEXT: offset = 1024, size = 1024
  %a = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  return
  // CHECK-NEXT: size = 3072
}

// Block arguments and yields are memory aliases that do not trigger a new
// allocation.
// CHECK-LABEL: for
func @for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 8192
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 8192, size = 8192
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 16384, size = 8192
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) {
    scf.yield %b_shared, %a_shared, %a_shared : tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>
  }
  return
  // CHECK-NEXT: size = 24576
}

// CHECK-LABEL: for_if_slice
func @for_if_slice(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  // CHECK: offset = 0, size = 8192
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 8192, size = 8192
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 16384, size = 8192
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) {
    scf.if %i1 {
      %index = arith.constant 8 : index
      %cst0 = tensor.extract_slice %a_shared[%index, 0][1, 32][1, 1] : tensor<128x32xf16, #A_SHARED> to tensor<32xf16, #A_SHARED>
      scf.yield
    }
    scf.yield %b_shared, %a_shared, %a_shared : tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>
  }
  return
  // CHECK-NEXT: size = 24576
}

// a_shared_init, b_shared_init, and c_shared_init's liveness ranges are span over the entire function before cst2.
// So they cannot be reused by cst0 and cst1, but can be reused by cst2.
// CHECK-LABEL: for_if_for
func @for_if_for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  // CHECK: offset = 0, size = 8192
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 8192, size = 8192
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 16384, size = 8192
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) {
    %c_shared_next = scf.for %jv = %lb to %ub step %step iter_args(%c_shared_next = %c_shared) -> (tensor<128x32xf16, #A_SHARED>) {
      %c_shared_next_next = scf.if %i1 -> tensor<128x32xf16, #A_SHARED> {
        // CHECK-NEXT: offset = 24576, size = 8192
        %cst0 = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
        scf.yield %cst0 : tensor<128x32xf16, #A_SHARED>
      } else {
        // CHECK-NEXT: offset = 32768, size = 8192
        %cst1 = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
        scf.yield %cst1 : tensor<128x32xf16, #A_SHARED>
      }
      scf.yield %c_shared_next_next : tensor<128x32xf16, #A_SHARED>
    }
    scf.yield %a_shared, %b_shared, %c_shared_next : tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>
  }
  // CHECK-NEXT: offset = 0, size = 8192
  %cst2 = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  return
  // CHECK-NEXT: size = 40960
}

}
