// RUN: triton-opt %s -split-input-file -tritongpu-stream-pipeline -canonicalize | FileCheck %s

// 4 warps
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#ALs0 = #triton_gpu.slice<{parent=#AL, dim=0}>
#BLs0 = #triton_gpu.slice<{parent=#BL, dim=0}>
#BLs1 = #triton_gpu.slice<{parent=#BL, dim=1}>
#C = #triton_gpu.mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>

// CHECK: tt.func @matmul_loop
// Prologue
// CHECK: %[[A0_LOAD:.*]] = tt.load
// CHECK: %[[A0_SHARED:.*]] = triton_gpu.convert_layout %[[A0_LOAD]]
// CHECK: %[[B0_LOAD:.*]] = tt.load
// CHECK: %[[B0_SHARED:.*]] = triton_gpu.convert_layout %[[B0_LOAD]]
// Restructured for-loop
// CHECK: %[[FOR_OUTPUT:.*]]:{{.*}} = scf.for {{.*}} iter_args({{.*}}, %[[AC_ARG:.*]] = %[[A0_SHARED]], %[[BC_ARG:.*]] = %[[B0_SHARED]], %[[AN_ARG:.*]] = %{{.*}}, %[[BN_ARG:.*]] = %{{.*}})
// CHECK:   %[[AN_LOAD:.*]] = tt.load %[[AN_ARG]]
// CHECK:   %[[BN_LOAD:.*]] = tt.load %[[BN_ARG]]
// CHECK:   %[[AC_CVT:.*]] = triton_gpu.convert_layout %[[AC_ARG]]
// CHECK:   %[[BC_CVT:.*]] = triton_gpu.convert_layout %[[BC_ARG]]
// CHECK:   %[[BC_CVT_0:.*]] = arith.mulf %[[BC_CVT]], %{{.*}}
// CHECK:   tt.dot %[[AC_CVT]], %[[BC_CVT_0]], {{.*}}
// CHECK:   %[[AN_SHARED:.*]] = triton_gpu.convert_layout %[[AN_LOAD]]
// CHECK:   %[[BN_SHARED:.*]] = triton_gpu.convert_layout %[[BN_LOAD]]
// CHECK:   scf.yield {{.*}}, %[[AN_SHARED]], %[[BN_SHARED]],
// Epilogue
// CHECK: %[[AO_SHARED:.*]] = triton_gpu.convert_layout %[[FOR_OUTPUT]]#1
// CHECK: %[[BO_SHARED:.*]] = triton_gpu.convert_layout %[[FOR_OUTPUT]]#2
// CHECK: %[[BO_SHARED_0:.*]] = arith.mulf %[[BO_SHARED]], %{{.*}}
// CHECK-NEXT: tt.dot %[[AO_SHARED]], %[[BO_SHARED_0]], {{.*}}

tt.func @matmul_loop(%lb : index, %ub : index, %step : index,
                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C> {
  // A ptrs
  %a_ptr_splat = tt.splat %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
  %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : (tensor<32xi32, #ALs0>) -> tensor<1x32xi32, #AL>
  %a_offs = tt.broadcast %a_tmp1 : (tensor<1x32xi32, #AL>) -> tensor<128x32xi32, #AL>
  %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
  // B ptrs
  %b_ptr_splat = tt.splat %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>
  %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
  %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : (tensor<128xi32, #BLs0>) -> tensor<1x128xi32, #BL>
  %b_offs = tt.broadcast %b_tmp1 : (tensor<1x128xi32, #BL>) -> tensor<32x128xi32, #BL>
  %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>


  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %b_scale = arith.constant dense<4.> : tensor<32x128xf16, #B>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %a_ = tt.load %a_ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
    %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>
    %b__ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b_ = triton_gpu.convert_layout %b__ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>
    %b = arith.mulf %b_, %b_scale: tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK: tt.func @matmul_loop_nested
// CHECK: scf.for
// Prologue
// CHECK: %[[A0_LOAD:.*]] = tt.load
// CHECK: %[[A0_SHARED:.*]] = triton_gpu.convert_layout %[[A0_LOAD]]
// CHECK: %[[B0_LOAD:.*]] = tt.load
// CHECK: %[[B0_SHARED:.*]] = triton_gpu.convert_layout %[[B0_LOAD]]
// Restructured for-loop
// CHECK: %[[FOR_OUTPUT:.*]]:{{.*}} = scf.for {{.*}} iter_args({{.*}}, %[[AC_ARG:.*]] = %[[A0_SHARED]], %[[BC_ARG:.*]] = %[[B0_SHARED]], %[[AN_ARG:.*]] = %{{.*}}, %[[BN_ARG:.*]] = %{{.*}})
// CHECK:   %[[AN_LOAD:.*]] = tt.load %[[AN_ARG]]
// CHECK:   %[[BN_LOAD:.*]] = tt.load %[[BN_ARG]]
// CHECK:   %[[AC_CVT:.*]] = triton_gpu.convert_layout %[[AC_ARG]]
// CHECK:   %[[BC_CVT:.*]] = triton_gpu.convert_layout %[[BC_ARG]]
// CHECK:   tt.dot %[[AC_CVT]], %[[BC_CVT]], {{.*}}
// CHECK:   %[[AN_SHARED:.*]] = triton_gpu.convert_layout %[[AN_LOAD]]
// CHECK:   %[[BN_SHARED:.*]] = triton_gpu.convert_layout %[[BN_LOAD]]
// CHECK:   scf.yield {{.*}}, %[[AN_SHARED]], %[[BN_SHARED]],
// Epilogue
// CHECK: %[[AO_SHARED:.*]] = triton_gpu.convert_layout %[[FOR_OUTPUT]]#1
// CHECK: %[[BO_SHARED:.*]] = triton_gpu.convert_layout %[[FOR_OUTPUT]]#2
// CHECK-NEXT: tt.dot %[[AO_SHARED]], %[[BO_SHARED]], {{.*}}

tt.func @matmul_loop_nested(%lb : index, %ub : index, %step : index,
                         %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                         %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C>{

  %c_start = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %loop1:1 = scf.for %iv0 = %lb to %ub step %step iter_args(%c_init = %c_start) -> (tensor<128x128xf32, #C>) {
    // A ptrs
    %a_ptr_splat = tt.splat %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
    %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
    %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : (tensor<32xi32, #ALs0>) -> tensor<1x32xi32, #AL>
    %a_offs = tt.broadcast %a_tmp1 : (tensor<1x32xi32, #AL>) -> tensor<128x32xi32, #AL>
    %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    // B ptrs
    %b_ptr_splat = tt.splat %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>
    %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
    %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : (tensor<128xi32, #BLs0>) -> tensor<1x128xi32, #BL>
    %b_offs = tt.broadcast %b_tmp1 : (tensor<1x128xi32, #BL>) -> tensor<32x128xi32, #BL>
    %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>

    %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
    %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
    %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
    %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>

    %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

    %loop2:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
      %a_ = tt.load %a_ptr, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
      %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>
      %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
      %b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>

      %c = tt.dot %a, %b, %prev_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

      %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
      %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
      scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
    }

    scf.yield %loop2#2 : tensor<128x128xf32, #C>
  }
  tt.return %loop1#0 : tensor<128x128xf32, #C>
}


// CHECK: tt.func @matmul_loop_single_pipeline
// Prologue
// CHECK: %[[A0_LOAD:.*]] = tt.load
// CHECK: %[[A0_SHARED:.*]] = triton_gpu.convert_layout %[[A0_LOAD]]
// CHECK: %[[B0_LOAD:.*]] = tt.load
// CHECK: %[[B0_SHARED:.*]] = triton_gpu.convert_layout %[[B0_LOAD]]
// Restructured for-loop
// CHECK: %[[FOR_OUTPUT:.*]]:{{.*}} = scf.for {{.*}} iter_args({{.*}}, %[[BC_ARG:.*]] = %[[B0_SHARED]], %[[BN_ARG:.*]] = %{{.*}})
// CHECK:   %[[BN_LOAD:.*]] = tt.load %[[BN_ARG]]
// CHECK:   %[[BC_DOT:.*]] = triton_gpu.convert_layout %[[BC_ARG]]
// CHECK:   tt.dot %[[A0_SHARED]], %[[BC_DOT]], {{.*}}
// CHECK:   %[[BN_SHARED:.*]] = triton_gpu.convert_layout %[[BN_LOAD]]
// CHECK:   scf.yield {{.*}}, %[[BN_SHARED]],
// Epilogue
// CHECK: %[[BO_SHARED:.*]] = triton_gpu.convert_layout %[[FOR_OUTPUT]]#1
// CHECK-NEXT: tt.dot %[[A0_SHARED]], %[[BO_SHARED]], {{.*}}

tt.func @matmul_loop_single_pipeline(%lb : index, %ub : index, %step : index,
                                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C> {
  // A ptrs
  %a_ptr_splat = tt.splat %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
  %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : (tensor<32xi32, #ALs0>) -> tensor<1x32xi32, #AL>
  %a_offs = tt.broadcast %a_tmp1 : (tensor<1x32xi32, #AL>) -> tensor<128x32xi32, #AL>
  %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
  // B ptrs
  %b_ptr_splat = tt.splat %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>
  %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
  %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : (tensor<128xi32, #BLs0>) -> tensor<1x128xi32, #BL>
  %b_offs = tt.broadcast %b_tmp1 : (tensor<1x128xi32, #BL>) -> tensor<32x128xi32, #BL>
  %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
  %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>

  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>
    %c = tt.dot %a, %b, %prev_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_b_ptr, %c : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#1 : tensor<128x128xf32, #C>
}

// CHECK: tt.func @lut_bmm_scalar
// Prologue
// CHECK: %[[A0_LOAD:.*]] = tt.load
// CHECK: %[[A0_SHARED:.*]] = triton_gpu.convert_layout %[[A0_LOAD]]
// Restructured for-loop
// CHECK: scf.for
// CHECK:   %[[AN_LOAD:.*]] = tt.load
// CHECK:   tt.load
// CHECK:   tt.load
// CHECK:   triton_gpu.convert_layout
// CHECK:   triton_gpu.convert_layout
// CHECK:   tt.dot 
// CHECK:   %[[AN_SHARED:.*]] = triton_gpu.convert_layout %[[AN_LOAD]]
// CHECK:   scf.yield {{.*}}, %[[AN_SHARED]]
// Epilogue
// CHECK: tt.load
// CHECK: tt.load
// CHECK: triton_gpu.convert_layout
// CHECK: triton_gpu.convert_layout
// CHECK: tt.dot

tt.func @lut_bmm_scalar(%77: i64 {tt.divisibility=16: i32},
                   %76: index,
                   %49: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: !tt.ptr<i64>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %79:3 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %75) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>) {
    %82 = tt.load %arg20 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #AL>
    %83 = tt.load %arg21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i64
    %84 = arith.muli %77, %83 : i64
    %85 = tt.splat %84 : (i64) -> tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BL>
    %88 = triton_gpu.convert_layout %82 : (tensor<16x16xf16, #AL>) -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : (tensor<16x16xf16, #BL>) -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32 : !tt.ptr<i64>, i32
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>
  }
  tt.return %79#0 : tensor<16x16xf32, #C>
}

// CHECK: tt.func @lut_bmm_vector
// Prologue
// CHECK: %[[A0_LOAD:.*]] = tt.load
// CHECK: %[[A0_SHARED:.*]] = triton_gpu.convert_layout %[[A0_LOAD]]
// Restructured for-loop
// CHECK: scf.for
// CHECK:   %[[AN_LOAD:.*]] = tt.load
// CHECK:   tt.load
// CHECK:   tt.load
// CHECK:   triton_gpu.convert_layout
// CHECK:   triton_gpu.convert_layout
// CHECK:   tt.dot 
// CHECK:   %[[AN_SHARED:.*]] = triton_gpu.convert_layout %[[AN_LOAD]]
// CHECK:   scf.yield {{.*}}, %[[AN_SHARED]]
// Epilogue
// CHECK: tt.load
// CHECK: tt.load
// CHECK: triton_gpu.convert_layout
// CHECK: triton_gpu.convert_layout
// CHECK: tt.dot
tt.func @lut_bmm_vector(%77: tensor<16x16xi64, #BL> {tt.divisibility=16: i32, tt.constancy=16: i32},
                   %76: index,
                   %49: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: tensor<16x!tt.ptr<i64>, #BLs1>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c1_i32_splat = tt.splat %c1_i32 : (i32) -> tensor<16xi32, #BLs1>
  %79:3 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %75) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>) {
    %82 = tt.load %arg20 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #AL>
    %83 = tt.load %arg21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16xi64, #BLs1>
    %84 = tt.expand_dims %83 {axis=1: i32}: (tensor<16xi64, #BLs1>) -> tensor<16x1xi64, #BL>
    %850 = tt.broadcast %84 : (tensor<16x1xi64, #BL>) -> tensor<16x16xi64, #BL>
    %85 = arith.muli %77, %850 : tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BL>
    %88 = triton_gpu.convert_layout %82 : (tensor<16x16xf16, #AL>) -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : (tensor<16x16xf16, #BL>) -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32_splat : tensor<16x!tt.ptr<i64>, #BLs1>, tensor<16xi32, #BLs1>
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>
  }
  tt.return %79#0 : tensor<16x16xf32, #C>
}

// CHECK: tt.func @post_load_inv
// CHECK: scf.for
// CHECK:   scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: triton_gpu.convert_layout
// CHECK-NEXT: triton_gpu.convert_layout
// CHECK-NEXT: tt.dot

tt.func @post_load_inv(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                       %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                       %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                       %arg3: i32 {tt.divisibility = 16 : i32},
                       %arg4: i32 {tt.divisibility = 16 : i32},
                       %arg5: i32 {tt.divisibility = 16 : i32},
                       %arg6: i32 {tt.divisibility = 16 : i32},
                       %arg7: i32 {tt.divisibility = 16 : i32},
                       %arg8: i32 {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #C> {
  %c0_index = arith.constant 0 : index
  %c1_index = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %84 = arith.constant 900 : index
  %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #C>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #AL>
  %50 = tt.splat %arg3 : (i32) -> tensor<1x32xi32, #AL>
  %59 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %81 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %66 = tt.splat %arg4 : (i32) -> tensor<32x1xi32, #AL>
  %60 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %82 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %85:3 = scf.for %arg9 = %c0_index to %84 step %c1_index iter_args(%arg10 = %cst, %arg11 = %59, %arg12 = %81) -> (tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>)  {
    %130 = arith.index_cast %arg9 : index to i32
    %107 = arith.muli %130, %c32_i32 : i32
    %108 = arith.subi %arg5, %107 : i32
    %109 = tt.splat %108 : (i32) -> tensor<1x32xi32, #AL>
    %110 = arith.cmpi "slt", %50, %109 : tensor<1x32xi32, #AL>
    %111 = tt.broadcast %110 : (tensor<1x32xi1, #AL>) -> tensor<32x32xi1, #AL>
    %112 = tt.load %arg11, %111, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #AL>
    %113 = tt.splat %108 : (i32) -> tensor<32x1xi32, #AL>
    %114 = arith.cmpi "slt", %66, %113 : tensor<32x1xi32, #AL>
    %115 = tt.broadcast %114 : (tensor<32x1xi1, #AL>) -> tensor<32x32xi1, #AL>
    %116 = tt.load %arg12, %115, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #AL>
    %117 = triton_gpu.convert_layout %112 : (tensor<32x32xf32, #AL>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>>
    %118 = triton_gpu.convert_layout %116 : (tensor<32x32xf32, #AL>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>>
    %119 = tt.dot %117, %118, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>> -> tensor<32x32xf32, #C>
    %131 = arith.index_cast %arg9 : index to i32
    %120 = arith.addi %131, %c1_i32 : i32
    %121 = arith.muli %120, %c32_i32 : i32
    %122 = tt.splat %121 : (i32) -> tensor<32x32xi32, #AL>
    %123 = tt.addptr %60, %122 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    %124 = arith.muli %121, %arg7 : i32
    %125 = tt.splat %124 : (i32) -> tensor<32x32xi32, #AL>
    %126 = tt.addptr %82, %125 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    scf.yield %119, %123, %126 : tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>
  }
  tt.return %85#0 : tensor<32x32xf32, #C>
}

// No stream pipeline
// CHECK: tt.func @cross_iter_dep
// CHECK: scf.for
// CHECK:   scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: tt.return
tt.func @cross_iter_dep(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                        %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                        %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                        %arg3: i32 {tt.divisibility = 16 : i32},
                        %arg4: i32 {tt.divisibility = 16 : i32},
                        %arg5: i32 {tt.divisibility = 16 : i32},
                        %arg6: i32 {tt.divisibility = 16 : i32},
                        %arg7: i32 {tt.divisibility = 16 : i32},
                        %arg8: i32 {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #C> {
  %c0_i32 = arith.constant 0 : index
  %118 = arith.constant 32 : index
  %c1_i32 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #C>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #AL>
  %78 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %110 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %112 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %113 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %116 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %65 = tt.splat %arg3 : (i32) -> tensor<1x32xi32, #AL>
  %88 = tt.splat %arg4 : (i32) -> tensor<32x1xi32, #AL>
  %80 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
  %119:5 = scf.for %arg9 = %c0_i32 to %118 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %78, %arg12 = %110, %arg13 = %113, %arg14 = %116) -> (tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>)  {
    %161 = arith.index_cast %arg9 : index to i32
    %141 = arith.muli %161, %c32_i32 : i32
    %142 = arith.subi %arg5, %141 : i32
    %143 = tt.splat %142 : (i32) -> tensor<1x32xi32, #AL>
    %144 = arith.cmpi "slt", %65, %143 : tensor<1x32xi32, #AL>
    %145 = tt.broadcast %144 : (tensor<1x32xi1, #AL>) -> tensor<32x32xi1, #AL>
    %146 = tt.load %arg11, %145, %cst_1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #AL>
    %147 = tt.splat %142 : (i32) -> tensor<32x1xi32, #AL>
    %148 = arith.cmpi "slt", %88, %147: tensor<32x1xi32, #AL>
    %149 = tt.broadcast %148 : (tensor<32x1xi1, #AL>) -> tensor<32x32xi1, #AL>
    %150 = tt.load %arg12, %149, %cst_1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #AL>
    %151 = triton_gpu.convert_layout %146 : (tensor<32x32xf32, #AL>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>>
    %152 = triton_gpu.convert_layout %150 : (tensor<32x32xf32, #AL>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>>
    %153 = tt.dot %151, %152, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>> -> tensor<32x32xf32, #C>
    %162 = arith.index_cast %arg9 : index to i32
    %154 = arith.addi %162, %c2_i32 : i32
    %155 = arith.muli %154, %c32_i32 : i32
    %156 = tt.splat %155 : (i32) -> tensor<32x32xi32, #AL>
    %157 = tt.addptr %80, %156 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    %158 = arith.muli %155, %arg7 : i32
    %159 = tt.splat %158 : (i32) -> tensor<32x32xi32, #AL>
    %160 = tt.addptr %112, %159 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    scf.yield %153, %arg13, %arg14, %157, %160 : tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>
  }
  tt.return %119#0 : tensor<32x32xf32, #C>
}

// CHECK: tt.func @matmul_mixed_kernel
// Prologue
// CHECK: %[[A0_LOAD:.*]] = tt.load
// CHECK: %[[A0_SHARED:.*]] = triton_gpu.convert_layout %[[A0_LOAD]]
// Restructured for-loop
// CHECK: scf.for
// CHECK:   %[[AN_LOAD:.*]] = tt.load
// CHECK:   tt.load
// CHECK:   triton_gpu.convert_layout
// CHECK:   tt.dot 
// CHECK:   %[[AN_SHARED:.*]] = triton_gpu.convert_layout %[[AN_LOAD]]
// CHECK:   scf.yield {{.*}}, %[[AN_SHARED]]
// Epilogue
// CHECK: tt.load
// CHECK: triton_gpu.convert_layout
// CHECK: tt.dot

#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
tt.func @matmul_mixed_kernel(%arg0: !tt.ptr<f8E4M3FNUZ> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #blocked>
    %cst_0 = arith.constant dense<32> : tensor<64x32xi32, #blocked1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked2>
    %c31_i32 = arith.constant 31 : i32
    %c63_i32 = arith.constant 63 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.addi %arg4, %c31_i32 : i32
    %4 = arith.divsi %3, %c32_i32 : i32
    %5 = arith.divsi %0, %4 : i32
    %6 = arith.subi %2, %5 : i32
    %7 = arith.cmpi "slt", %6, %c1_i32: i32
    %8 = arith.select %7, %6, %c1_i32 : i32
    %9 = arith.remsi %0, %8 : i32
    %10 = arith.addi %5, %9 : i32
    %11 = arith.remsi %0, %4 : i32
    %12 = arith.divsi %11, %8 : i32
    %13 = arith.muli %10, %c64_i32 : i32
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %17 = tt.splat %13 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %18 = tt.splat %13 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %19 = tt.splat %13 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %20 = arith.addi %17, %14 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %21 = arith.addi %18, %15 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %22 = arith.addi %19, %16 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %23 = tt.splat %arg3 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %24 = arith.remsi %20, %23 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %25 = arith.muli %12, %c32_i32 : i32
    %26 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %27 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %28 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %29 = tt.splat %25 : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %30 = tt.splat %25 : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %31 = tt.splat %25 : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %32 = arith.addi %29, %26 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %33 = arith.addi %30, %27 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %34 = arith.addi %31, %28 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %35 = tt.splat %arg4 : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %36 = arith.remsi %32, %35 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %37 = tt.expand_dims %24 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<64x1xi32, #blocked1>
    %38 = tt.splat %arg6 : (i32) -> tensor<64x1xi32, #blocked1>
    %39 = arith.muli %37, %38 : tensor<64x1xi32, #blocked1>
    %40 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %41 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %42 = tt.expand_dims %40 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x32xi32, #blocked1>
    %43 = tt.expand_dims %41 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x32xi32, #blocked1>
    %44 = tt.broadcast %39 : (tensor<64x1xi32, #blocked1>) -> tensor<64x32xi32, #blocked1>
    %45 = tt.broadcast %42 : (tensor<1x32xi32, #blocked1>) -> tensor<64x32xi32, #blocked1>
    %46 = arith.addi %44, %45 : tensor<64x32xi32, #blocked1>
    %47 = tt.splat %arg0 : (!tt.ptr<f8E4M3FNUZ>) -> tensor<64x32x!tt.ptr<f8E4M3FNUZ>, #blocked1>
    %48 = tt.addptr %47, %46 : tensor<64x32x!tt.ptr<f8E4M3FNUZ>, #blocked1>, tensor<64x32xi32, #blocked1>
    %49 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %50 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %51 = tt.expand_dims %49 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<32x1xi32, #blocked2>
    %52 = tt.expand_dims %50 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<32x1xi32, #blocked2>
    %53 = tt.splat %arg7 : (i32) -> tensor<32x1xi32, #blocked2>
    %54 = arith.muli %51, %53 : tensor<32x1xi32, #blocked2>
    %55 = tt.expand_dims %36 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x32xi32, #blocked2>
    %56 = tt.broadcast %54 : (tensor<32x1xi32, #blocked2>) -> tensor<32x32xi32, #blocked2>
    %57 = tt.broadcast %55 : (tensor<1x32xi32, #blocked2>) -> tensor<32x32xi32, #blocked2>
    %58 = arith.addi %56, %57 : tensor<32x32xi32, #blocked2>
    %59 = tt.splat %arg1 : (!tt.ptr<f16>) -> tensor<32x32x!tt.ptr<f16>, #blocked2>
    %60 = tt.addptr %59, %58 : tensor<32x32x!tt.ptr<f16>, #blocked2>, tensor<32x32xi32, #blocked2>
    %61 = arith.addi %arg5, %c31_i32 : i32
    %62 = arith.divsi %61, %c32_i32 : i32
    %63 = tt.fp_to_fp %cst_1 : tensor<64x32xf32, #blocked1> -> tensor<64x32xf8E4M3FNUZ, #blocked1>
    %64 = arith.muli %arg7, %c32_i32 : i32
    %65 = tt.splat %64 : (i32) -> tensor<32x32xi32, #blocked2>
    %66:3 = scf.for %arg9 = %c0_i32 to %62 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %48, %arg12 = %60) -> (tensor<64x32xf16, #blocked>, tensor<64x32x!tt.ptr<f8E4M3FNUZ>, #blocked1>, tensor<32x32x!tt.ptr<f16>, #blocked2>)  : i32 {
      %86 = arith.muli %arg9, %c32_i32 : i32
      %87 = arith.subi %arg5, %86 : i32
      %88 = tt.splat %87 : (i32) -> tensor<1x32xi32, #blocked1>
      %89 = arith.cmpi "slt", %43, %88 : tensor<1x32xi32, #blocked1>
      %90 = tt.broadcast %89 : (tensor<1x32xi1, #blocked1>) -> tensor<64x32xi1, #blocked1>
      %91 = tt.load %arg11, %90, %63 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x32xf8E4M3FNUZ, #blocked1>
      %92 = tt.splat %87 : (i32) -> tensor<32x1xi32, #blocked2>
      %93 = arith.cmpi "slt", %52, %92: tensor<32x1xi32, #blocked2>
      %94 = tt.broadcast %93 : (tensor<32x1xi1, #blocked2>) -> tensor<32x32xi1, #blocked2>
      %95 = tt.load %arg12, %94, %cst_2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf16, #blocked2>
      %96 = tt.fp_to_fp %91 : tensor<64x32xf8E4M3FNUZ, #blocked1> -> tensor<64x32xf16, #blocked1>
      %97 = triton_gpu.convert_layout %96 : (tensor<64x32xf16, #blocked1>) -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
      %98 = triton_gpu.convert_layout %95 : (tensor<32x32xf16, #blocked2>) -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>
      %99 = tt.dot %97, %98, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x32xf16, #blocked>
      %100 = tt.addptr %arg11, %cst_0 : tensor<64x32x!tt.ptr<f8E4M3FNUZ>, #blocked1>, tensor<64x32xi32, #blocked1>
      %101 = tt.addptr %arg12, %65 : tensor<32x32x!tt.ptr<f16>, #blocked2>, tensor<32x32xi32, #blocked2>
      scf.yield %99, %100, %101 : tensor<64x32xf16, #blocked>, tensor<64x32x!tt.ptr<f8E4M3FNUZ>, #blocked1>, tensor<32x32x!tt.ptr<f16>, #blocked2>
    }
    %67 = tt.expand_dims %21 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64x1xi32, #blocked2>
    %68 = tt.expand_dims %22 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64x1xi32, #blocked2>
    %69 = tt.splat %arg8 : (i32) -> tensor<64x1xi32, #blocked2>
    %70 = arith.muli %69, %67 : tensor<64x1xi32, #blocked2>
    %71 = tt.splat %arg2 : (!tt.ptr<f16>) -> tensor<64x1x!tt.ptr<f16>, #blocked2>
    %72 = tt.addptr %71, %70 : tensor<64x1x!tt.ptr<f16>, #blocked2>, tensor<64x1xi32, #blocked2>
    %73 = tt.expand_dims %33 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x32xi32, #blocked2>
    %74 = tt.expand_dims %34 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x32xi32, #blocked2>
    %75 = tt.broadcast %72 : (tensor<64x1x!tt.ptr<f16>, #blocked2>) -> tensor<64x32x!tt.ptr<f16>, #blocked2>
    %76 = tt.broadcast %73 : (tensor<1x32xi32, #blocked2>) -> tensor<64x32xi32, #blocked2>
    %77 = tt.addptr %75, %76 : tensor<64x32x!tt.ptr<f16>, #blocked2>, tensor<64x32xi32, #blocked2>
    %78 = tt.splat %arg3 : (i32) -> tensor<64x1xi32, #blocked2>
    %79 = arith.cmpi "slt", %68, %78 : tensor<64x1xi32, #blocked2>
    %80 = tt.splat %arg4 : (i32) -> tensor<1x32xi32, #blocked2>
    %81 = arith.cmpi "slt", %74, %80 : tensor<1x32xi32, #blocked2>
    %82 = tt.broadcast %79 : (tensor<64x1xi1, #blocked2>) -> tensor<64x32xi1, #blocked2>
    %83 = tt.broadcast %81 : (tensor<1x32xi1, #blocked2>) -> tensor<64x32xi1, #blocked2>
    %84 = arith.andi %82, %83 : tensor<64x32xi1, #blocked2>
    %85 = triton_gpu.convert_layout %66#0 : (tensor<64x32xf16, #blocked>) -> tensor<64x32xf16, #blocked2>
    tt.store %77, %85, %84 {cache = 1 : i32, evict = 1 : i32} : tensor<64x32xf16, #blocked2>
    tt.return
}
