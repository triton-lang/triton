// RUN: triton-opt %s --mlir-disable-threading -test-print-alias -split-input-file 2>&1 | FileCheck %s

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#A_SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#A_SHARED_T = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#B_SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #triton_gpu.mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A_DOT = #triton_gpu.dot_op<{opIdx = 0, parent = #C}>
#B_DOT = #triton_gpu.dot_op<{opIdx = 1, parent = #C}>

// CHECK-LABEL: matmul_loop
// There shouldn't be any aliasing with the dot op encoding.
tt.func @matmul_loop(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
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
    %c = tt.dot %a, %b, %prev_c {transA = false, transB = false, allowTF32 = true} : tensor<128x32xf16, #A_DOT> * tensor<32x128xf16, #B_DOT> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return
}

// CHECK-LABEL: alloc
tt.func @alloc(%A : !tt.ptr<f16>) {
  // CHECK: %cst -> %cst
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  // CHECK: %0 -> %0
  %cst2 = triton_gpu.alloc_tensor : tensor<16x16xf16, #A_SHARED>
  tt.return
}

// CHECK-LABEL: convert
tt.func @convert(%A : !tt.ptr<f16>) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  // CHECK: %0 -> %0
  %cst1 = triton_gpu.convert_layout %cst0 : (tensor<16x16xf16, #AL>) -> tensor<16x16xf16, #A_SHARED>
  tt.return
}

// CHECK-LABEL: trans
tt.func @trans(%A : !tt.ptr<f16>) {
  // CHECK: %cst -> %cst
  %tensor = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #A_SHARED>
  // CHECK: %0 -> %cst
  %b = tt.trans %tensor : (tensor<16x32xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED_T>
  tt.return
}

// CHECK-LABEL: insert_slice_async
tt.func @insert_slice_async(%A : !tt.ptr<f16>, %i1 : i1) {
  %a_ptr = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<16x16x!tt.ptr<f16>, #AL>
  %mask = tt.splat %i1 : (i1) -> tensor<16x16xi1, #AL>
  %other = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  // CHECK: %cst_0 -> %cst_0
  %tensor = arith.constant dense<0.000000e+00> : tensor<1x16x16xf16, #A_SHARED>
  %index = arith.constant 0 : i32
  // CHECK: %2 -> %cst_0
  %a = triton_gpu.insert_slice_async %a_ptr, %tensor, %index, %mask, %other {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16x!tt.ptr<f16>, #AL> -> tensor<1x16x16xf16, #A_SHARED>
  tt.return
}

// CHECK-LABEL: insert_slice
tt.func @insert_slice(%A : !tt.ptr<f16>, %i1 : i1) {
  %a_ptr = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<16x16x!tt.ptr<f16>, #AL>
  %mask = tt.splat %i1 : (i1) -> tensor<16x16xi1, #AL>
  %other = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  // CHECK: %cst_0 -> %cst_0
  %tensor = arith.constant dense<0.000000e+00> : tensor<1x16x16xf16, #A_SHARED>
  %index = arith.constant 0 : index
  %a = tt.load %a_ptr, %mask, %other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #AL>
  // CHECK: %inserted_slice -> %cst_0
  %b = tensor.insert_slice %a into %tensor[%index, 0, 0][1, 16, 16][1, 1, 1]: tensor<16x16xf16, #AL> into tensor<1x16x16xf16, #A_SHARED>
  tt.return
}

// CHECK-LABEL: extract_slice
tt.func @extract_slice(%A : !tt.ptr<f16>) {
  // CHECK: %cst -> %cst
  %cst0 = arith.constant dense<0.000000e+00> : tensor<1x16x16xf16, #A_SHARED>
  %index = arith.constant 0 : i32
  // CHECK-NEXT: %0 -> %cst
  %cst1 = triton_gpu.extract_slice %cst0[%index, 0, 0][1, 16, 16][1, 1, 1] : tensor<1x16x16xf16, #A_SHARED> to tensor<16x16xf16, #A_SHARED>
  tt.return
}

// CHECK-LABEL: if_cat
tt.func @if_cat(%i1 : i1) {
  // CHECK: %cst -> %cst
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK: %cst_0 -> %cst_0
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK: %0 -> %1,%1
  %cst2 = scf.if %i1 -> tensor<32x16xf16, #A_SHARED> {
    // CHECK: %1 -> %1
    %a = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    scf.yield %a : tensor<32x16xf16, #A_SHARED>
  } else {
    // CHECK: %1 -> %1
    %b = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    scf.yield %b : tensor<32x16xf16, #A_SHARED>
  }
  tt.return
}

// CHECK-LABEL: if_alias
tt.func @if_alias(%i1 : i1) {
  // CHECK: %cst -> %cst
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: %cst_0 -> %cst_0
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  // CHECK-NEXT: %0 -> %cst,%cst_0
  %cst2 = scf.if %i1 -> tensor<16x16xf16, #A_SHARED> {
    scf.yield %cst0 : tensor<16x16xf16, #A_SHARED>
  } else {
    scf.yield %cst1 : tensor<16x16xf16, #A_SHARED>
  }
  tt.return
}

// CHECK-LABEL: for
tt.func @for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  // CHECK: %cst -> %cst
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %cst_0 -> %cst_0
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %cst_1 -> %cst_1
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %arg6 -> %cst
  // CHECK-NEXT: %arg7 -> %cst_0
  // CHECK-NEXT: %arg8 -> %cst_1
  // CHECK-NEXT: %0#0 -> %cst,%cst_0
  // CHECK-NEXT: %0#1 -> %cst,%cst_0
  // CHECK-NEXT: %0#2 -> %cst,%cst_0
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) {
    scf.yield %b_shared, %a_shared, %a_shared : tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>
  }
  tt.return
}

// CHECK-LABEL: for_if
tt.func @for_if(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  // CHECK: %cst -> %cst
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %cst_0 -> %cst_0
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %cst_1 -> %cst_1
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %arg7 -> %cst
  // CHECK-NEXT: %arg8 -> %cst_0
  // CHECK-NEXT: %arg9 -> %cst_1
  // CHECK-NEXT: %0#0 -> %cst,%cst_0
  // CHECK-NEXT: %0#1 -> %cst,%cst_0
  // CHECK-NEXT: %0#2 -> %cst,%cst_0
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) {
    scf.if %i1 {
      %index = arith.constant 8 : i32
      // CHECK-NEXT: %1 -> %cst,%cst_0
      %cst0 = triton_gpu.extract_slice %a_shared[%index, 0][1, 32][1, 1] : tensor<128x32xf16, #A_SHARED> to tensor<32xf16, #A_SHARED>
      scf.yield
    }
    scf.yield %b_shared, %a_shared, %a_shared : tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>
  }
  tt.return
}

// CHECK-LABEL: for_for_if
tt.func @for_for_if(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  // CHECK: %cst -> %cst
  %a_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %cst_0 -> %cst_0
  %b_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %cst_1 -> %cst_1
  %c_shared_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %arg7 -> %cst
  // CHECK-NEXT: %arg8 -> %cst_0
  // CHECK-NEXT: %arg9 -> %cst_1
  // CHECK-NEXT: %0#0 -> %cst
  // CHECK-NEXT: %0#1 -> %cst_0
  // CHECK-NEXT: %0#2 -> %cst_2,%cst_2
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) {
    // CHECK-NEXT: %arg11 -> %cst_1,%cst_2,%cst_2
    // CHECK-NEXT: %1 -> %cst_2,%cst_2
    %c_shared_next = scf.for %jv = %lb to %ub step %step iter_args(%c_shared_next = %c_shared) -> (tensor<128x32xf16, #A_SHARED>) {
      // CHECK-NEXT: %2 -> %cst_2,%cst_2
      %c_shared_next_next = scf.if %i1 -> tensor<128x32xf16, #A_SHARED> {
        // CHECK-NEXT: %cst_2 -> %cst_2
        %cst0 = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
        scf.yield %cst0 : tensor<128x32xf16, #A_SHARED>
      } else {
        // CHECK-NEXT: %cst_2 -> %cst_2
        %cst0 = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A_SHARED>
        scf.yield %cst0 : tensor<128x32xf16, #A_SHARED>
      }
      scf.yield %c_shared_next_next : tensor<128x32xf16, #A_SHARED>
    }
    scf.yield %a_shared, %b_shared, %c_shared_next : tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>
  }
  tt.return
}

// CHECK-LABEL: cf_for
tt.func @cf_for(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16>, %arg4: !tt.ptr<f16>) {
  // CHECK: %cst -> %cst
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %cst_0 -> %cst_0
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #A_SHARED>
  gpu.barrier
  // CHECK-NEXT: %0 -> %0
  %0 = tt.cat %cst, %cst_0 {axis = 0 : i64} : (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) -> tensor<256x32xf16, #A_SHARED>
  // CHECK-NEXT: %cst_1 -> %cst_1
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %2 -> %cst,%cst_0,%cst_1
  // CHECK-NEXT: %3 -> %cst,%cst_0,%cst_1
  // CHECK-NEXT: %4 -> %cst,%cst_0,%cst_1
  cf.br ^bb1(%arg0, %cst, %cst_0, %cst_1 : index, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>)
^bb1(%1: index, %2: tensor<128x32xf16, #A_SHARED>, %3: tensor<128x32xf16, #A_SHARED>, %4: tensor<128x32xf16, #A_SHARED>):  // 2 preds: ^bb0, ^bb2
  %5 = arith.cmpi slt, %1, %arg1 : index
  cf.cond_br %5, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %6 = tt.cat %cst, %cst_0 {axis = 0 : i64} : (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) -> tensor<256x32xf16, #blocked>
  gpu.barrier
  %7 = tt.cat %2, %3 {axis = 0 : i64} : (tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>) -> tensor<256x32xf16, #blocked>
  %8 = arith.addi %1, %arg2 : index
  cf.br ^bb1(%8, %4, %2, %3 : index, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>, tensor<128x32xf16, #A_SHARED>)
^bb3:  // pred: ^bb1
  gpu.barrier
  // CHECK-NEXT: %9 -> %9
  %9 = tt.cat %0, %0 {axis = 0 : i64} : (tensor<256x32xf16, #A_SHARED>, tensor<256x32xf16, #A_SHARED>) -> tensor<512x32xf16, #A_SHARED>
  tt.return
}
