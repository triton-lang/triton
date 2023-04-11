// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions 2>&1 | FileCheck %s

#layout0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#layout1 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#layout2 = #triton_gpu.mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1]}>

// CHECK: [[$target_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK: [[$row_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK: [[$col_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK: [[$col_layout_novec:#.*]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK-LABEL: cst
tt.func @cst() -> tensor<1024xi32, #layout1> {
  %cst = arith.constant dense<0> : tensor<1024xi32, #layout0>
  %1 = triton_gpu.convert_layout %cst : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: tt.return %cst : tensor<1024xi32, [[$target_layout]]>
  tt.return %1: tensor<1024xi32, #layout1>
}

// CHECK-LABEL: range
tt.func @range() -> tensor<1024xi32, #layout1> {
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %1 = triton_gpu.convert_layout %0 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: tt.return %0 : tensor<1024xi32, [[$target_layout]]>
  tt.return %1: tensor<1024xi32, #layout1>
}

// CHECK-LABEL: splat
tt.func @splat(%arg0: i32) -> tensor<1024xi32, #layout1> {
  %0 = tt.splat %arg0 : (i32) -> tensor<1024xi32, #layout0>
  %1 = triton_gpu.convert_layout %0 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: tt.return %0 : tensor<1024xi32, [[$target_layout]]>
  tt.return %1: tensor<1024xi32, #layout1>
}

// CHECK-LABEL: remat
tt.func @remat(%arg0: i32) -> tensor<1024xi32, #layout1> {
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %2 = arith.muli %0, %1 : tensor<1024xi32, #layout0>
  %3 = triton_gpu.convert_layout %2 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  %4 = tt.splat %arg0 : (i32) -> tensor<1024xi32, #layout0>
  %5 = triton_gpu.convert_layout %2 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  %6 = arith.addi %3, %5 : tensor<1024xi32, #layout1>
  tt.return %6: tensor<1024xi32, #layout1>
  // CHECK: %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[$target_layout]]>
  // CHECK: %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[$target_layout]]>
  // CHECK: %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[$target_layout]]>
  // CHECK: %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[$target_layout]]>
  // CHECK: %4 = arith.muli %0, %2 : tensor<1024xi32, [[$target_layout]]>
  // CHECK: %5 = arith.muli %1, %3 : tensor<1024xi32, [[$target_layout]]>
  // CHECK: %6 = arith.addi %4, %5 : tensor<1024xi32, [[$target_layout]]>
  // CHECK: tt.return %6 : tensor<1024xi32, [[$target_layout]]>
}

// Always rematerialize single value loads
// CHECK-LABEL: remat_single_value
tt.func @remat_single_value(%arg: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %0 = tt.splat %arg : (!tt.ptr<i32>) -> tensor<1x!tt.ptr<i32>, #layout1>
  %1 = tt.load %0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  %2 = triton_gpu.convert_layout %1 : (tensor<1xi32, #layout1>) -> tensor<1xi32, #layout0>
  %3 = triton_gpu.convert_layout %0 : (tensor<1x!tt.ptr<i32>, #layout1>) -> tensor<1x!tt.ptr<i32>, #layout0>
  tt.store %3, %2 : tensor<1xi32, #layout0>
  tt.return
}


module attributes {"triton_gpu.num-warps" = 4 : i32} {
tt.func @remat_fast_load(%arg: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %0 = tt.splat %arg : (!tt.ptr<i32>) -> tensor<16x!tt.ptr<i32>, #layout1>
  %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #layout1>
  %2 = tt.addptr %0, %1 : tensor<16x!tt.ptr<i32>, #layout1>, tensor<16xi32, #layout1>
  %3 = tt.load %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  %4 = triton_gpu.convert_layout %3 : (tensor<16xi32, #layout1>) -> tensor<16xi32, #layout0>
  %5 = triton_gpu.convert_layout %2 : (tensor<16x!tt.ptr<i32>, #layout1>) -> tensor<16x!tt.ptr<i32>, #layout0>
  tt.store %5, %4 : tensor<16xi32, #layout0>
  tt.return
}
}

// CHECK-LABEL: if
tt.func @if(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  // CHECK-NOT: triton_gpu.convert_layout
  %c32_i32 = arith.constant dense<32> : tensor<1024xi32, #layout1>
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = tt.splat %0 : (i32) -> tensor<1024xi32, #layout1>
  %2 = arith.muli %1, %c32_i32 : tensor<1024xi32, #layout1>
  %3 = arith.addi %2, %c32_i32 : tensor<1024xi32, #layout1>
  %4 = arith.cmpi sgt, %0, %arg0 : i32
  %5 = tt.splat %arg1 : (!tt.ptr<i32>) -> tensor<1024x!tt.ptr<i32>, #layout0>
  scf.if %4 {
    %6 = triton_gpu.convert_layout %2 : (tensor<1024xi32, #layout1>) -> tensor<1024xi32, #layout0>
    tt.store %5, %6 : tensor<1024xi32, #layout0>
  }
  tt.return
}

// CHECK-LABEL: if_convert_else_not
tt.func @if_convert_else_not(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %c32_i32 = arith.constant dense<32> : tensor<1024xi32, #layout0>
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = tt.splat %0 : (i32) -> tensor<1024xi32, #layout0>
  %9 = tt.splat %0 : (i32) -> tensor<1024xi32, #layout1>
  %2 = arith.muli %1, %c32_i32 : tensor<1024xi32, #layout0>
  %3 = arith.addi %2, %c32_i32 : tensor<1024xi32, #layout0>
  %4 = arith.cmpi sgt, %0, %arg0 : i32
  %5 = tt.splat %arg1 : (!tt.ptr<i32>) -> tensor<1024x!tt.ptr<i32>, #layout1>
  %8 = scf.if %4 -> tensor<1024xi32, #layout1> {
    %6 = triton_gpu.convert_layout %2 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
    scf.yield %6 : tensor<1024xi32, #layout1>
  } else {
    scf.yield %9 : tensor<1024xi32, #layout1>
  }
  // CHECK-NOT: triton_gpu.convert_layout
  tt.store %5, %8 : tensor<1024xi32, #layout1>
  tt.return
}

// CHECK-LABEL: if_not_else_convert
tt.func @if_not_else_convert(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %c32_i32 = arith.constant dense<32> : tensor<1024xi32, #layout0>
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = tt.splat %0 : (i32) -> tensor<1024xi32, #layout0>
  %9 = tt.splat %0 : (i32) -> tensor<1024xi32, #layout1>
  %2 = arith.muli %1, %c32_i32 : tensor<1024xi32, #layout0>
  %3 = arith.addi %2, %c32_i32 : tensor<1024xi32, #layout0>
  %4 = arith.cmpi sgt, %0, %arg0 : i32
  %5 = tt.splat %arg1 : (!tt.ptr<i32>) -> tensor<1024x!tt.ptr<i32>, #layout1>
  %8 = scf.if %4 -> tensor<1024xi32, #layout1> {
    scf.yield %9 : tensor<1024xi32, #layout1>
  } else {
    %7 = triton_gpu.convert_layout %3 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
    scf.yield %7 : tensor<1024xi32, #layout1>
  }
  // CHECK-NOT: triton_gpu.convert_layout
  tt.store %5, %8 : tensor<1024xi32, #layout1>
  tt.return
}

// CHECK-LABEL: if_else_both_convert
tt.func @if_else_both_convert(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %c32_i32 = arith.constant dense<32> : tensor<1024xi32, #layout0>
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = tt.splat %0 : (i32) -> tensor<1024xi32, #layout0>
  %2 = arith.muli %1, %c32_i32 : tensor<1024xi32, #layout0>
  %3 = arith.addi %2, %c32_i32 : tensor<1024xi32, #layout0>
  %4 = arith.cmpi sgt, %0, %arg0 : i32
  %5 = tt.splat %arg1 : (!tt.ptr<i32>) -> tensor<1024x!tt.ptr<i32>, #layout1>
  %8 = scf.if %4 -> tensor<1024xi32, #layout1> {
    %6 = triton_gpu.convert_layout %2 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
    scf.yield %6 : tensor<1024xi32, #layout1>
  } else {
    %7 = triton_gpu.convert_layout %3 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
    scf.yield %7 : tensor<1024xi32, #layout1>
  }
  // TODO(csigg): seems like the whole function is converted to layout1.
  // disabledCHECK: triton_gpu.convert_layout
  // CHECK-NOT: triton_gpu.convert_layout
  tt.store %5, %8 : tensor<1024xi32, #layout1>
  tt.return
}

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#slice1dim1 = #triton_gpu.slice<{dim = 1, parent = #blocked1}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#slice2dim0 = #triton_gpu.slice<{dim = 0, parent = #blocked2}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>

// CHECK-LABEL: transpose
tt.func @transpose(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: [[loaded_val:%.*]] = tt.load {{.*}}, {{%cst.*}}, {{%cst.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf32, [[$row_layout]]>
  // CHECK: [[cvt_val:%.*]] = triton_gpu.convert_layout [[loaded_val]] : (tensor<64x64xf32, [[$row_layout]]>) -> tensor<64x64xf32, [[$col_layout]]>
  // CHECK: tt.store {{.*}}, [[cvt_val]], {{%cst.*}} : tensor<64x64xf32, [[$col_layout]]>
  // CHECK: tt.return
  %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked1>
  %cst_0 = arith.constant dense<true> : tensor<64x64xi1, #blocked1>
  %00 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice1dim1>
  %01 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice2dim0>
  %1 = tt.expand_dims %00 {axis = 1 : i32} : (tensor<64xi32, #slice1dim1>) -> tensor<64x1xi32, #blocked1>
  %2 = tt.splat %arg1 : (i32) -> tensor<64x1xi32, #blocked1>
  %3 = arith.muli %1, %2 : tensor<64x1xi32, #blocked1>
  %4 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x1x!tt.ptr<f32>, #blocked1>
  %5 = tt.addptr %4, %3 : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
  %6 = tt.expand_dims %01 {axis = 0 : i32} : (tensor<64xi32, #slice2dim0>) -> tensor<1x64xi32, #blocked2>
  %7 = tt.broadcast %5 : (tensor<64x1x!tt.ptr<f32>, #blocked1>) -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %8 = tt.broadcast %6 : (tensor<1x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %9 = triton_gpu.convert_layout %8 : (tensor<64x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked1>
  %10 = tt.addptr %7, %9 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
  %11 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x1x!tt.ptr<f32>, #blocked1>
  %12 = tt.addptr %11, %1 : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
  %13 = tt.splat %arg3 : (i32) -> tensor<1x64xi32, #blocked2>
  %14 = arith.muli %6, %13 : tensor<1x64xi32, #blocked2>
  %15 = tt.broadcast %12 : (tensor<64x1x!tt.ptr<f32>, #blocked1>) -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %16 = tt.broadcast %14 : (tensor<1x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %17 = triton_gpu.convert_layout %16 : (tensor<64x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked1>
  %18 = tt.addptr %15, %17 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
  %19 = triton_gpu.convert_layout %10 : (tensor<64x64x!tt.ptr<f32>, #blocked1>) -> tensor<64x64x!tt.ptr<f32>, #blocked3>
  %20 = triton_gpu.convert_layout %cst_0 : (tensor<64x64xi1, #blocked1>) -> tensor<64x64xi1, #blocked3>
  %21 = triton_gpu.convert_layout %cst : (tensor<64x64xf32, #blocked1>) -> tensor<64x64xf32, #blocked3>
  %22 = tt.load %19, %20, %21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf32, #blocked3>
  %23 = triton_gpu.convert_layout %22 : (tensor<64x64xf32, #blocked3>) -> tensor<64x64xf32, #blocked1>
  %24 = triton_gpu.convert_layout %18 : (tensor<64x64x!tt.ptr<f32>, #blocked1>) -> tensor<64x64x!tt.ptr<f32>, #blocked4>
  %25 = triton_gpu.convert_layout %23 : (tensor<64x64xf32, #blocked1>) -> tensor<64x64xf32, #blocked4>
  %26 = triton_gpu.convert_layout %cst_0 : (tensor<64x64xi1, #blocked1>) -> tensor<64x64xi1, #blocked4>
  tt.store %24, %25, %26 : tensor<64x64xf32, #blocked4>
  tt.return
}

// CHECK-LABEL: loop
tt.func @loop(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) {
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: [[loop_ret:%.*]]:2 = scf.for {{.*}} -> (tensor<64x64xf32, [[$row_layout]]>, tensor<64x64x!tt.ptr<f32>, [[$row_layout]]>)
  // CHECK-NEXT: {{.*}} = tt.load {{.*}} : tensor<64x64xf32, [[$row_layout]]>
  // CHECK-NEXT: {{.*}} = arith.addf {{.*}} : tensor<64x64xf32, [[$row_layout]]>
  // CHECK-NEXT: {{.*}} = tt.addptr {{.*}} : tensor<64x64x!tt.ptr<f32>, [[$row_layout]]>, tensor<64x64xi32, [[$row_layout]]>
  // CHECK-NEXT: scf.yield {{.*}} : tensor<64x64xf32, [[$row_layout]]>, tensor<64x64x!tt.ptr<f32>, [[$row_layout]]>
  // CHECK-NEXT: }
  // CHECK-NEXT: {{.*}} = triton_gpu.convert_layout [[loop_ret]]#0 : (tensor<64x64xf32, [[$row_layout]]>) -> tensor<64x64xf32, [[$col_layout_novec]]>
  // CHECK-NOT: triton_gpu.convert_layout
  %cst = arith.constant dense<true> : tensor<64x64xi1, #blocked1>
  %cst_0 = arith.constant dense<64> : tensor<64x64xi32, #blocked1>
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked1>
  %00 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice1dim1>
  %01 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice2dim0>
  %1 = tt.expand_dims %00 {axis = 1 : i32} : (tensor<64xi32, #slice1dim1>) -> tensor<64x1xi32, #blocked1>
  %2 = tt.splat %arg1 : (i32) -> tensor<64x1xi32, #blocked1>
  %3 = arith.muli %1, %2 : tensor<64x1xi32, #blocked1>
  %4 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x1x!tt.ptr<f32>, #blocked1>
  %5 = tt.addptr %4, %3 : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
  %6 = tt.expand_dims %01 {axis = 0 : i32} : (tensor<64xi32, #slice2dim0>) -> tensor<1x64xi32, #blocked2>
  %7 = tt.broadcast %5 : (tensor<64x1x!tt.ptr<f32>, #blocked1>) -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %8 = tt.broadcast %6 : (tensor<1x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %9 = triton_gpu.convert_layout %8 : (tensor<64x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked1>
  %10 = tt.addptr %7, %9 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
  %11:2 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %cst_1, %arg7 = %10) -> (tensor<64x64xf32, #blocked1>, tensor<64x64x!tt.ptr<f32>, #blocked1>) {
    %23 = triton_gpu.convert_layout %arg7 : (tensor<64x64x!tt.ptr<f32>, #blocked1>) -> tensor<64x64x!tt.ptr<f32>, #blocked3>
    %24 = triton_gpu.convert_layout %cst : (tensor<64x64xi1, #blocked1>) -> tensor<64x64xi1, #blocked3>
    %25 = triton_gpu.convert_layout %cst_1 : (tensor<64x64xf32, #blocked1>) -> tensor<64x64xf32, #blocked3>
    %26 = tt.load %23, %24, %25 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf32, #blocked3>
    %27 = triton_gpu.convert_layout %26 : (tensor<64x64xf32, #blocked3>) -> tensor<64x64xf32, #blocked1>
    %28 = arith.addf %arg6, %27 : tensor<64x64xf32, #blocked1>
    %29 = tt.addptr %arg7, %cst_0 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
    scf.yield %28, %29 : tensor<64x64xf32, #blocked1>, tensor<64x64x!tt.ptr<f32>, #blocked1>
  }
  %12 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x1x!tt.ptr<f32>, #blocked1>
  %13 = tt.addptr %12, %1 : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
  %14 = tt.splat %arg3 : (i32) -> tensor<1x64xi32, #blocked2>
  %15 = arith.muli %6, %14 : tensor<1x64xi32, #blocked2>
  %16 = tt.broadcast %13 : (tensor<64x1x!tt.ptr<f32>, #blocked1>) -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %17 = tt.broadcast %15 : (tensor<1x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %18 = triton_gpu.convert_layout %17 : (tensor<64x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked1>
  %19 = tt.addptr %16, %18 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
  %20 = triton_gpu.convert_layout %19 : (tensor<64x64x!tt.ptr<f32>, #blocked1>) -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %21 = triton_gpu.convert_layout %11#0 : (tensor<64x64xf32, #blocked1>) -> tensor<64x64xf32, #blocked1>
  %22 = triton_gpu.convert_layout %cst : (tensor<64x64xi1, #blocked1>) -> tensor<64x64xi1, #blocked1>
  tt.store %20, %21, %22 : tensor<64x64xf32, #blocked1>
  tt.return
}

// CHECK-LABEL: vecadd
tt.func @vecadd(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
  // CHECK-NOT: triton_gpu.convert_layout
  %c256_i32 = arith.constant 256 : i32
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = arith.muli %0, %c256_i32 : i32
  %2 = tt.splat %1 : (i32) -> tensor<256xi32, #layout1>
  %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #layout1>
  %4 = tt.splat %1 : (i32) -> tensor<256xi32, #layout1>
  %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #layout1>
  %6 = tt.splat %1 : (i32) -> tensor<256xi32, #layout1>
  %7 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #layout1>
  %8 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #layout1>
  %9 = arith.addi %6, %7 : tensor<256xi32, #layout1>
  %10 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #layout1>
  %11 = arith.addi %4, %5 : tensor<256xi32, #layout1>
  %12 = tt.addptr %8, %9 : tensor<256x!tt.ptr<f32>, #layout1>, tensor<256xi32, #layout1>
  %13 = tt.load %12 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #layout1>
  %14 = triton_gpu.convert_layout %13 : (tensor<256xf32, #layout1>) -> tensor<256xf32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>>
  %15 = tt.addptr %10, %11 : tensor<256x!tt.ptr<f32>, #layout1>, tensor<256xi32, #layout1>
  %16 = tt.load %15 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #layout1>
  %17 = triton_gpu.convert_layout %16 : (tensor<256xf32, #layout1>) -> tensor<256xf32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>>
  %18 = arith.addf %14, %17 : tensor<256xf32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>>
  %19 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #layout1>
  %20 = arith.addi %2, %3 : tensor<256xi32, #layout1>
  %21 = tt.addptr %19, %20 : tensor<256x!tt.ptr<f32>, #layout1>, tensor<256xi32, #layout1>
  %22 = triton_gpu.convert_layout %18 : (tensor<256xf32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>>) -> tensor<256xf32, #layout1>
  tt.store %21, %22 : tensor<256xf32, #layout1>
  tt.return
}

// Select has args with different element types
// CHECK-LABEL: select
tt.func @select(%arg0: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) {
  // CHECK-NOT: triton_gpu.convert_layout
  %cst = arith.constant dense<30000> : tensor<1x1xi32, #blocked2>
  %cst_0 = arith.constant dense<30000> : tensor<1x512xi32, #blocked2>
  %c512 = arith.constant 512 : index
  %c30000 = arith.constant 30000 : index
  %c0 = arith.constant 0 : index
  %cst_1 = arith.constant dense<2048> : tensor<1x1xi32, #blocked2>
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<1x512xf64, #blocked2>
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #blocked0>
  %2 = triton_gpu.convert_layout %1 : (tensor<1xi32, #blocked0>) -> tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %3 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<1x1xi32, #blocked1>
  %4 = triton_gpu.convert_layout %3 : (tensor<1x1xi32, #blocked1>) -> tensor<1x1xi32, #blocked2>
  %5 = tt.splat %0 : (i32) -> tensor<1x1xi32, #blocked2>
  %6 = arith.addi %5, %4 : tensor<1x1xi32, #blocked2>
  %7 = "triton_gpu.cmpi"(%6, %cst_1) {predicate = 2 : i64} : (tensor<1x1xi32, #blocked2>, tensor<1x1xi32, #blocked2>) -> tensor<1x1xi1, #blocked2>
  %8 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked0>
  %9 = triton_gpu.convert_layout %8 : (tensor<512xi32, #blocked0>) -> tensor<512xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
  %10 = tt.expand_dims %9 {axis = 0 : i32} : (tensor<512xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x512xi32, #blocked2>
  %11 = arith.muli %6, %cst : tensor<1x1xi32, #blocked2>
  %12 = tt.broadcast %11 : (tensor<1x1xi32, #blocked2>) -> tensor<1x512xi32, #blocked2>
  %13 = tt.splat %arg0 : (!tt.ptr<f64>) -> tensor<1x512x!tt.ptr<f64>, #blocked2>
  %14 = tt.broadcast %7 : (tensor<1x1xi1, #blocked2>) -> tensor<1x512xi1, #blocked2>
  %15 = scf.for %arg3 = %c0 to %c30000 step %c512 iter_args(%arg4 = %cst_2) -> (tensor<1x512xf64, #blocked2>) {
    %16 = arith.index_cast %arg3 : index to i32
    %17 = tt.splat %16 : (i32) -> tensor<1x512xi32, #blocked2>
    %18 = arith.addi %17, %10 : tensor<1x512xi32, #blocked2>
    %19 = "triton_gpu.cmpi"(%18, %cst_0) {predicate = 2 : i64} : (tensor<1x512xi32, #blocked2>, tensor<1x512xi32, #blocked2>) -> tensor<1x512xi1, #blocked2>
    %20 = arith.addi %18, %12 : tensor<1x512xi32, #blocked2>
    %21 = tt.addptr %13, %20 : tensor<1x512x!tt.ptr<f64>, #blocked2>, tensor<1x512xi32, #blocked2>
    %22 = arith.andi %19, %14 : tensor<1x512xi1, #blocked2>
    %23 = triton_gpu.convert_layout %21 : (tensor<1x512x!tt.ptr<f64>, #blocked2>) -> tensor<1x512x!tt.ptr<f64>, #blocked3>
    %24 = triton_gpu.convert_layout %22 : (tensor<1x512xi1, #blocked2>) -> tensor<1x512xi1, #blocked3>
    %25 = tt.load %23, %24 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<1x512xf64, #blocked3>
    %26 = triton_gpu.convert_layout %25 : (tensor<1x512xf64, #blocked3>) -> tensor<1x512xf64, #blocked2>
    %27 = arith.andi %14, %19 : tensor<1x512xi1, #blocked2>
    %28 = "triton_gpu.cmpf"(%arg4, %26) {predicate = 4 : i64} : (tensor<1x512xf64, #blocked2>, tensor<1x512xf64, #blocked2>) -> tensor<1x512xi1, #blocked2>
    %29 = arith.andi %27, %28 : tensor<1x512xi1, #blocked2>
    %30 = "triton_gpu.select"(%29, %26, %arg4) : (tensor<1x512xi1, #blocked2>, tensor<1x512xf64, #blocked2>, tensor<1x512xf64, #blocked2>) -> tensor<1x512xf64, #blocked2>
    %31 = triton_gpu.convert_layout %21 : (tensor<1x512x!tt.ptr<f64>, #blocked2>) -> tensor<1x512x!tt.ptr<f64>, #blocked3>
    %32 = triton_gpu.convert_layout %30 : (tensor<1x512xf64, #blocked2>) -> tensor<1x512xf64, #blocked3>
    %33 = triton_gpu.convert_layout %27 : (tensor<1x512xi1, #blocked2>) -> tensor<1x512xi1, #blocked3>
    tt.store %31, %32, %33 : tensor<1x512xf64, #blocked3>
    scf.yield %30 : tensor<1x512xf64, #blocked2>
  }
  tt.return
}

// Make sure the following IR doesn't hang the compiler.
// CHECK-LABEL: long_func
tt.func public @long_func(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg10: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg13: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg14: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg15: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}) {
  %cst = arith.constant dense<1.000000e+00> : tensor<1024xf32, #blocked0>
  %cst_0 = arith.constant dense<5.000000e-04> : tensor<1024xf32, #blocked0>
  %cst_1 = arith.constant dense<0.999499976> : tensor<1024xf32, #blocked0>
  %cst_2 = arith.constant dense<1.000000e+04> : tensor<1024xf32, #blocked0>
  %cst_3 = arith.constant dense<5000> : tensor<1024xi32, #blocked0>
  %cst_4 = arith.constant dense<150> : tensor<1024xi32, #blocked0>
  %cst_5 = arith.constant dense<false> : tensor<1024xi1, #blocked0>
  %cst_6 = arith.constant dense<2> : tensor<1024xi32, #blocked0>
  %cst_7 = arith.constant dense<4999> : tensor<1024xi32, #blocked0>
  %cst_8 = arith.constant dense<2499> : tensor<1024xi32, #blocked0>
  %cst_9 = arith.constant dense<2500> : tensor<1024xi32, #blocked0>
  %cst_10 = arith.constant dense<0.91629076> : tensor<1024xf32, #blocked0>
  %c2499_i32 = arith.constant 2499 : i32
  %cst_11 = arith.constant dense<1024> : tensor<1024xi32, #blocked0>
  %c1024_i32 = arith.constant 1024 : i32
  %cst_12 = arith.constant dense<1> : tensor<1024xi32, #blocked0>
  %cst_13 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked0>
  %cst_14 = arith.constant dense<0> : tensor<1024xi32, #blocked0>
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = arith.muli %0, %c1024_i32 : i32
  %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked0>
  %3 = tt.splat %1 : (i32) -> tensor<1024xi32, #blocked0>
  %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked0>
  %5 = "triton_gpu.cmpi"(%4, %cst_11) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %6 = tt.splat %arg5 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %7 = tt.addptr %6, %4 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %8 = triton_gpu.convert_layout %7 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked1>
  %9 = triton_gpu.convert_layout %5 : (tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked1>
  %10 = tt.load %8, %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked1>
  %11 = triton_gpu.convert_layout %10 : (tensor<1024xf32, #blocked1>) -> tensor<1024xf32, #blocked0>
  %12 = tt.splat %arg7 : (!tt.ptr<i64>) -> tensor<1024x!tt.ptr<i64>, #blocked0>
  %13 = tt.addptr %12, %4 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %14 = triton_gpu.convert_layout %13 : (tensor<1024x!tt.ptr<i64>, #blocked0>) -> tensor<1024x!tt.ptr<i64>, #blocked2>
  %15 = triton_gpu.convert_layout %5 : (tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked2>
  %16 = tt.load %14, %15 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked2>
  %17 = triton_gpu.convert_layout %16 : (tensor<1024xi64, #blocked2>) -> tensor<1024xi64, #blocked0>
  %18 = tt.splat %arg8 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %19 = tt.addptr %18, %4 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %20 = triton_gpu.convert_layout %19 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked1>
  %21 = triton_gpu.convert_layout %5 : (tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked1>
  %22 = tt.load %20, %21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked1>
  %23 = triton_gpu.convert_layout %22 : (tensor<1024xf32, #blocked1>) -> tensor<1024xf32, #blocked0>
  %24 = arith.subf %cst_13, %11 : tensor<1024xf32, #blocked0>
  %25 = math.exp %24 : tensor<1024xf32, #blocked0>
  %26 = arith.sitofp %cst_12 : tensor<1024xi32, #blocked0> to tensor<1024xf32, #blocked0>
  %27 = arith.addf %25, %26 : tensor<1024xf32, #blocked0>
  %28 = arith.divf %26, %27 : tensor<1024xf32, #blocked0>
  %29 = tt.addptr %arg6, %c2499_i32 : !tt.ptr<f32>, i32
  %30 = tt.load %29 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
  %31 = arith.subf %11, %cst_10 : tensor<1024xf32, #blocked0>
  %32 = arith.subf %cst_13, %31 : tensor<1024xf32, #blocked0>
  %33 = math.exp %32 : tensor<1024xf32, #blocked0>
  %34 = arith.addf %33, %26 : tensor<1024xf32, #blocked0>
  %35 = arith.divf %26, %34 : tensor<1024xf32, #blocked0>
  %36 = tt.splat %30 : (f32) -> tensor<1024xf32, #blocked0>
  %37 = "triton_gpu.cmpf"(%36, %35) {predicate = 3 : i64} : (tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %38 = "triton_gpu.select"(%37, %cst_14, %cst_9) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %39 = "triton_gpu.select"(%37, %cst_8, %cst_7) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %40 = arith.subi %39, %38 : tensor<1024xi32, #blocked0>
  %41 = "triton_gpu.cmpi"(%40, %cst_14) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %42 = "triton_gpu.cmpi"(%41, %cst_5) {predicate = 1 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %43 = arith.remsi %40, %cst_6 : tensor<1024xi32, #blocked0>
  %44 = "triton_gpu.cmpi"(%43, %cst_14) {predicate = 1 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %45 = arith.divsi %40, %cst_6 : tensor<1024xi32, #blocked0>
  %46 = arith.subi %45, %cst_12 : tensor<1024xi32, #blocked0>
  %47 = "triton_gpu.select"(%44, %46, %45) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %48 = "triton_gpu.select"(%42, %47, %45) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %49 = arith.addi %38, %48 : tensor<1024xi32, #blocked0>
  %50 = "triton_gpu.cmpi"(%38, %39) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %51 = "triton_gpu.select"(%50, %49, %cst_14) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %52 = tt.splat %arg6 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %53 = tt.addptr %52, %51 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %54 = triton_gpu.convert_layout %53 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %55 = tt.load %54 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %56 = "triton_gpu.cmpf"(%55, %35) {predicate = 3 : i64} : (tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %57 = "triton_gpu.cmpi"(%56, %cst_5) {predicate = 0 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %58 = arith.andi %57, %50 : tensor<1024xi1, #blocked0>
  %59 = arith.addi %51, %cst_12 : tensor<1024xi32, #blocked0>
  %60 = "triton_gpu.select"(%58, %59, %38) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %61 = arith.andi %56, %50 : tensor<1024xi1, #blocked0>
  %62 = "triton_gpu.select"(%61, %51, %39) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %63 = "triton_gpu.cmpi"(%60, %62) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %64 = arith.subi %62, %60 : tensor<1024xi32, #blocked0>
  %65 = "triton_gpu.cmpi"(%64, %cst_14) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %66 = "triton_gpu.cmpi"(%65, %cst_5) {predicate = 1 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %67 = arith.remsi %64, %cst_6 : tensor<1024xi32, #blocked0>
  %68 = "triton_gpu.cmpi"(%67, %cst_14) {predicate = 1 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %69 = arith.divsi %64, %cst_6 : tensor<1024xi32, #blocked0>
  %70 = arith.subi %69, %cst_12 : tensor<1024xi32, #blocked0>
  %71 = "triton_gpu.select"(%68, %70, %69) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %72 = "triton_gpu.select"(%66, %71, %69) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %73 = arith.addi %60, %72 : tensor<1024xi32, #blocked0>
  %74 = "triton_gpu.select"(%63, %73, %cst_14) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %75 = tt.addptr %52, %74 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %76 = triton_gpu.convert_layout %75 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %77 = tt.load %76 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %78 = "triton_gpu.cmpf"(%77, %35) {predicate = 3 : i64} : (tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %79 = "triton_gpu.cmpi"(%78, %cst_5) {predicate = 0 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %80 = arith.andi %79, %63 : tensor<1024xi1, #blocked0>
  %81 = arith.addi %74, %cst_12 : tensor<1024xi32, #blocked0>
  %82 = "triton_gpu.select"(%80, %81, %60) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %83 = arith.andi %78, %63 : tensor<1024xi1, #blocked0>
  %84 = "triton_gpu.select"(%83, %74, %62) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %85 = "triton_gpu.cmpi"(%82, %84) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %86 = arith.subi %84, %82 : tensor<1024xi32, #blocked0>
  %87 = "triton_gpu.cmpi"(%86, %cst_14) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %88 = "triton_gpu.cmpi"(%87, %cst_5) {predicate = 1 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %89 = arith.remsi %86, %cst_6 : tensor<1024xi32, #blocked0>
  %90 = "triton_gpu.cmpi"(%89, %cst_14) {predicate = 1 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %91 = arith.divsi %86, %cst_6 : tensor<1024xi32, #blocked0>
  %92 = arith.subi %91, %cst_12 : tensor<1024xi32, #blocked0>
  %93 = "triton_gpu.select"(%90, %92, %91) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %94 = "triton_gpu.select"(%88, %93, %91) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %95 = arith.addi %82, %94 : tensor<1024xi32, #blocked0>
  %96 = "triton_gpu.select"(%85, %95, %cst_14) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %97 = tt.addptr %52, %96 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %98 = triton_gpu.convert_layout %97 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %99 = tt.load %98 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %100 = "triton_gpu.cmpf"(%99, %35) {predicate = 3 : i64} : (tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %101 = "triton_gpu.cmpi"(%100, %cst_5) {predicate = 0 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %102 = arith.andi %101, %85 : tensor<1024xi1, #blocked0>
  %103 = arith.addi %96, %cst_12 : tensor<1024xi32, #blocked0>
  %104 = "triton_gpu.select"(%102, %103, %82) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %105 = arith.andi %100, %85 : tensor<1024xi1, #blocked0>
  %106 = "triton_gpu.select"(%105, %96, %84) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %107 = "triton_gpu.cmpi"(%104, %106) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %108 = arith.subi %106, %104 : tensor<1024xi32, #blocked0>
  %109 = "triton_gpu.cmpi"(%108, %cst_14) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %110 = "triton_gpu.cmpi"(%109, %cst_5) {predicate = 1 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %111 = arith.remsi %108, %cst_6 : tensor<1024xi32, #blocked0>
  %112 = "triton_gpu.cmpi"(%111, %cst_14) {predicate = 1 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %113 = arith.divsi %108, %cst_6 : tensor<1024xi32, #blocked0>
  %114 = arith.subi %113, %cst_12 : tensor<1024xi32, #blocked0>
  %115 = "triton_gpu.select"(%112, %114, %113) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %116 = "triton_gpu.select"(%110, %115, %113) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %117 = arith.addi %104, %116 : tensor<1024xi32, #blocked0>
  %118 = "triton_gpu.select"(%107, %117, %cst_14) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %119 = tt.addptr %52, %118 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %120 = triton_gpu.convert_layout %119 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %121 = tt.load %120 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %122 = "triton_gpu.cmpf"(%121, %35) {predicate = 3 : i64} : (tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %123 = "triton_gpu.cmpi"(%122, %cst_5) {predicate = 0 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %124 = arith.andi %123, %107 : tensor<1024xi1, #blocked0>
  %125 = arith.addi %118, %cst_12 : tensor<1024xi32, #blocked0>
  %126 = "triton_gpu.select"(%124, %125, %104) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %127 = arith.andi %122, %107 : tensor<1024xi1, #blocked0>
  %128 = "triton_gpu.select"(%127, %118, %106) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %129 = "triton_gpu.cmpi"(%126, %128) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %130 = arith.subi %128, %126 : tensor<1024xi32, #blocked0>
  %131 = "triton_gpu.cmpi"(%130, %cst_14) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %132 = "triton_gpu.cmpi"(%131, %cst_5) {predicate = 1 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %133 = arith.remsi %130, %cst_6 : tensor<1024xi32, #blocked0>
  %134 = "triton_gpu.cmpi"(%133, %cst_14) {predicate = 1 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %135 = arith.divsi %130, %cst_6 : tensor<1024xi32, #blocked0>
  %136 = arith.subi %135, %cst_12 : tensor<1024xi32, #blocked0>
  %137 = "triton_gpu.select"(%134, %136, %135) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %138 = "triton_gpu.select"(%132, %137, %135) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %139 = arith.addi %126, %138 : tensor<1024xi32, #blocked0>
  %140 = "triton_gpu.select"(%129, %139, %cst_14) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %141 = tt.addptr %52, %140 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %142 = triton_gpu.convert_layout %141 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %143 = tt.load %142 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %144 = "triton_gpu.cmpf"(%143, %35) {predicate = 3 : i64} : (tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %145 = "triton_gpu.cmpi"(%144, %cst_5) {predicate = 0 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %146 = arith.andi %145, %129 : tensor<1024xi1, #blocked0>
  %147 = arith.addi %140, %cst_12 : tensor<1024xi32, #blocked0>
  %148 = "triton_gpu.select"(%146, %147, %126) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %149 = arith.andi %144, %129 : tensor<1024xi1, #blocked0>
  %150 = "triton_gpu.select"(%149, %140, %128) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %151 = "triton_gpu.cmpi"(%148, %150) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %152 = arith.subi %150, %148 : tensor<1024xi32, #blocked0>
  %153 = "triton_gpu.cmpi"(%152, %cst_14) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %154 = "triton_gpu.cmpi"(%153, %cst_5) {predicate = 1 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %155 = arith.remsi %152, %cst_6 : tensor<1024xi32, #blocked0>
  %156 = "triton_gpu.cmpi"(%155, %cst_14) {predicate = 1 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %157 = arith.divsi %152, %cst_6 : tensor<1024xi32, #blocked0>
  %158 = arith.subi %157, %cst_12 : tensor<1024xi32, #blocked0>
  %159 = "triton_gpu.select"(%156, %158, %157) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %160 = "triton_gpu.select"(%154, %159, %157) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %161 = arith.addi %148, %160 : tensor<1024xi32, #blocked0>
  %162 = "triton_gpu.select"(%151, %161, %cst_14) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %163 = tt.addptr %52, %162 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %164 = triton_gpu.convert_layout %163 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %165 = tt.load %164 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %166 = "triton_gpu.cmpf"(%165, %35) {predicate = 3 : i64} : (tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %167 = "triton_gpu.cmpi"(%166, %cst_5) {predicate = 0 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %168 = arith.andi %167, %151 : tensor<1024xi1, #blocked0>
  %169 = arith.addi %162, %cst_12 : tensor<1024xi32, #blocked0>
  %170 = "triton_gpu.select"(%168, %169, %148) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %171 = arith.andi %166, %151 : tensor<1024xi1, #blocked0>
  %172 = "triton_gpu.select"(%171, %162, %150) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %173 = "triton_gpu.cmpi"(%170, %172) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %174 = arith.subi %172, %170 : tensor<1024xi32, #blocked0>
  %175 = "triton_gpu.cmpi"(%174, %cst_14) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %176 = "triton_gpu.cmpi"(%175, %cst_5) {predicate = 1 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %177 = arith.remsi %174, %cst_6 : tensor<1024xi32, #blocked0>
  %178 = "triton_gpu.cmpi"(%177, %cst_14) {predicate = 1 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %179 = arith.divsi %174, %cst_6 : tensor<1024xi32, #blocked0>
  %180 = arith.subi %179, %cst_12 : tensor<1024xi32, #blocked0>
  %181 = "triton_gpu.select"(%178, %180, %179) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %182 = "triton_gpu.select"(%176, %181, %179) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %183 = arith.addi %170, %182 : tensor<1024xi32, #blocked0>
  %184 = "triton_gpu.select"(%173, %183, %cst_14) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %185 = tt.addptr %52, %184 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %186 = triton_gpu.convert_layout %185 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %187 = tt.load %186 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %188 = "triton_gpu.cmpf"(%187, %35) {predicate = 3 : i64} : (tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %189 = "triton_gpu.cmpi"(%188, %cst_5) {predicate = 0 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %190 = arith.andi %189, %173 : tensor<1024xi1, #blocked0>
  %191 = arith.addi %184, %cst_12 : tensor<1024xi32, #blocked0>
  %192 = "triton_gpu.select"(%190, %191, %170) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %193 = arith.andi %188, %173 : tensor<1024xi1, #blocked0>
  %194 = "triton_gpu.select"(%193, %184, %172) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %195 = "triton_gpu.cmpi"(%192, %194) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %196 = arith.subi %194, %192 : tensor<1024xi32, #blocked0>
  %197 = "triton_gpu.cmpi"(%196, %cst_14) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %198 = "triton_gpu.cmpi"(%197, %cst_5) {predicate = 1 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %199 = arith.remsi %196, %cst_6 : tensor<1024xi32, #blocked0>
  %200 = "triton_gpu.cmpi"(%199, %cst_14) {predicate = 1 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %201 = arith.divsi %196, %cst_6 : tensor<1024xi32, #blocked0>
  %202 = arith.subi %201, %cst_12 : tensor<1024xi32, #blocked0>
  %203 = "triton_gpu.select"(%200, %202, %201) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %204 = "triton_gpu.select"(%198, %203, %201) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %205 = arith.addi %192, %204 : tensor<1024xi32, #blocked0>
  %206 = "triton_gpu.select"(%195, %205, %cst_14) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %207 = tt.addptr %52, %206 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %208 = triton_gpu.convert_layout %207 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %209 = tt.load %208 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %210 = "triton_gpu.cmpf"(%209, %35) {predicate = 3 : i64} : (tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %211 = "triton_gpu.cmpi"(%210, %cst_5) {predicate = 0 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %212 = arith.andi %211, %195 : tensor<1024xi1, #blocked0>
  %213 = arith.addi %206, %cst_12 : tensor<1024xi32, #blocked0>
  %214 = "triton_gpu.select"(%212, %213, %192) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %215 = arith.andi %210, %195 : tensor<1024xi1, #blocked0>
  %216 = "triton_gpu.select"(%215, %206, %194) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %217 = "triton_gpu.cmpi"(%214, %216) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %218 = arith.subi %216, %214 : tensor<1024xi32, #blocked0>
  %219 = "triton_gpu.cmpi"(%218, %cst_14) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %220 = "triton_gpu.cmpi"(%219, %cst_5) {predicate = 1 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %221 = arith.remsi %218, %cst_6 : tensor<1024xi32, #blocked0>
  %222 = "triton_gpu.cmpi"(%221, %cst_14) {predicate = 1 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %223 = arith.divsi %218, %cst_6 : tensor<1024xi32, #blocked0>
  %224 = arith.subi %223, %cst_12 : tensor<1024xi32, #blocked0>
  %225 = "triton_gpu.select"(%222, %224, %223) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %226 = "triton_gpu.select"(%220, %225, %223) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %227 = arith.addi %214, %226 : tensor<1024xi32, #blocked0>
  %228 = "triton_gpu.select"(%217, %227, %cst_14) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %229 = tt.addptr %52, %228 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %230 = triton_gpu.convert_layout %229 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %231 = tt.load %230 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %232 = "triton_gpu.cmpf"(%231, %35) {predicate = 3 : i64} : (tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %233 = "triton_gpu.cmpi"(%232, %cst_5) {predicate = 0 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %234 = arith.andi %233, %217 : tensor<1024xi1, #blocked0>
  %235 = arith.addi %228, %cst_12 : tensor<1024xi32, #blocked0>
  %236 = "triton_gpu.select"(%234, %235, %214) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %237 = arith.andi %232, %217 : tensor<1024xi1, #blocked0>
  %238 = "triton_gpu.select"(%237, %228, %216) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %239 = "triton_gpu.cmpi"(%236, %238) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %240 = arith.subi %238, %236 : tensor<1024xi32, #blocked0>
  %241 = "triton_gpu.cmpi"(%240, %cst_14) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %242 = "triton_gpu.cmpi"(%241, %cst_5) {predicate = 1 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %243 = arith.remsi %240, %cst_6 : tensor<1024xi32, #blocked0>
  %244 = "triton_gpu.cmpi"(%243, %cst_14) {predicate = 1 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %245 = arith.divsi %240, %cst_6 : tensor<1024xi32, #blocked0>
  %246 = arith.subi %245, %cst_12 : tensor<1024xi32, #blocked0>
  %247 = "triton_gpu.select"(%244, %246, %245) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %248 = "triton_gpu.select"(%242, %247, %245) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %249 = arith.addi %236, %248 : tensor<1024xi32, #blocked0>
  %250 = "triton_gpu.select"(%239, %249, %cst_14) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %251 = tt.addptr %52, %250 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %252 = triton_gpu.convert_layout %251 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %253 = tt.load %252 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %254 = "triton_gpu.cmpf"(%253, %35) {predicate = 3 : i64} : (tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %255 = "triton_gpu.cmpi"(%254, %cst_5) {predicate = 0 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %256 = arith.andi %255, %239 : tensor<1024xi1, #blocked0>
  %257 = arith.addi %250, %cst_12 : tensor<1024xi32, #blocked0>
  %258 = "triton_gpu.select"(%256, %257, %236) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %259 = arith.andi %254, %239 : tensor<1024xi1, #blocked0>
  %260 = "triton_gpu.select"(%259, %250, %238) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %261 = "triton_gpu.cmpi"(%258, %260) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %262 = arith.subi %260, %258 : tensor<1024xi32, #blocked0>
  %263 = "triton_gpu.cmpi"(%262, %cst_14) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %264 = "triton_gpu.cmpi"(%263, %cst_5) {predicate = 1 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %265 = arith.remsi %262, %cst_6 : tensor<1024xi32, #blocked0>
  %266 = "triton_gpu.cmpi"(%265, %cst_14) {predicate = 1 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %267 = arith.divsi %262, %cst_6 : tensor<1024xi32, #blocked0>
  %268 = arith.subi %267, %cst_12 : tensor<1024xi32, #blocked0>
  %269 = "triton_gpu.select"(%266, %268, %267) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %270 = "triton_gpu.select"(%264, %269, %267) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %271 = arith.addi %258, %270 : tensor<1024xi32, #blocked0>
  %272 = "triton_gpu.select"(%261, %271, %cst_14) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %273 = tt.addptr %52, %272 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %274 = triton_gpu.convert_layout %273 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %275 = tt.load %274 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %276 = "triton_gpu.cmpf"(%275, %35) {predicate = 3 : i64} : (tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %277 = "triton_gpu.cmpi"(%276, %cst_5) {predicate = 0 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %278 = arith.andi %277, %261 : tensor<1024xi1, #blocked0>
  %279 = arith.addi %272, %cst_12 : tensor<1024xi32, #blocked0>
  %280 = "triton_gpu.select"(%278, %279, %258) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %281 = arith.andi %276, %261 : tensor<1024xi1, #blocked0>
  %282 = "triton_gpu.select"(%281, %272, %260) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %283 = "triton_gpu.cmpi"(%280, %282) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %284 = arith.subi %282, %280 : tensor<1024xi32, #blocked0>
  %285 = "triton_gpu.cmpi"(%284, %cst_14) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %286 = "triton_gpu.cmpi"(%285, %cst_5) {predicate = 1 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %287 = arith.remsi %284, %cst_6 : tensor<1024xi32, #blocked0>
  %288 = "triton_gpu.cmpi"(%287, %cst_14) {predicate = 1 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %289 = arith.divsi %284, %cst_6 : tensor<1024xi32, #blocked0>
  %290 = arith.subi %289, %cst_12 : tensor<1024xi32, #blocked0>
  %291 = "triton_gpu.select"(%288, %290, %289) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %292 = "triton_gpu.select"(%286, %291, %289) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %293 = arith.addi %280, %292 : tensor<1024xi32, #blocked0>
  %294 = "triton_gpu.select"(%283, %293, %cst_14) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %295 = tt.addptr %52, %294 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %296 = triton_gpu.convert_layout %295 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %297 = tt.load %296 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %298 = "triton_gpu.cmpf"(%297, %35) {predicate = 3 : i64} : (tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %299 = "triton_gpu.cmpi"(%298, %cst_5) {predicate = 0 : i64} : (tensor<1024xi1, #blocked0>, tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked0>
  %300 = arith.andi %299, %283 : tensor<1024xi1, #blocked0>
  %301 = arith.addi %294, %cst_12 : tensor<1024xi32, #blocked0>
  %302 = "triton_gpu.select"(%300, %301, %280) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %303 = arith.extsi %cst_12 : tensor<1024xi32, #blocked0> to tensor<1024xi64, #blocked0>
  %304 = "triton_gpu.cmpi"(%17, %303) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %305 = arith.fptosi %23 : tensor<1024xf32, #blocked0> to tensor<1024xi64, #blocked0>
  %306 = arith.extsi %cst_14 : tensor<1024xi32, #blocked0> to tensor<1024xi64, #blocked0>
  %307 = "triton_gpu.cmpi"(%306, %305) {predicate = 4 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %308 = arith.extsi %cst_4 : tensor<1024xi32, #blocked0> to tensor<1024xi64, #blocked0>
  %309 = "triton_gpu.cmpi"(%305, %308) {predicate = 4 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %310 = "triton_gpu.select"(%309, %306, %305) : (tensor<1024xi1, #blocked0>, tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi64, #blocked0>
  %311 = "triton_gpu.select"(%307, %306, %310) : (tensor<1024xi1, #blocked0>, tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi64, #blocked0>
  %312 = "triton_gpu.select"(%304, %311, %306) : (tensor<1024xi1, #blocked0>, tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi64, #blocked0>
  %313 = arith.extsi %cst_3 : tensor<1024xi32, #blocked0> to tensor<1024xi64, #blocked0>
  %314 = arith.muli %312, %313 : tensor<1024xi64, #blocked0>
  %315 = arith.extsi %302 : tensor<1024xi32, #blocked0> to tensor<1024xi64, #blocked0>
  %316 = arith.addi %315, %314 : tensor<1024xi64, #blocked0>
  %317 = arith.trunci %316 : tensor<1024xi64, #blocked0> to tensor<1024xi32, #blocked0>
  %318 = arith.extsi %317 : tensor<1024xi32, #blocked0> to tensor<1024xi64, #blocked0>
  %319 = tt.splat %arg9 : (!tt.ptr<f64>) -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %320 = tt.addptr %319, %318 : tensor<1024x!tt.ptr<f64>, #blocked0>, tensor<1024xi64, #blocked0>
  %321 = triton_gpu.convert_layout %320 : (tensor<1024x!tt.ptr<f64>, #blocked0>) -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %322 = tt.load %321 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf64, #blocked0>
  %323 = arith.extf %cst_2 : tensor<1024xf32, #blocked0> to tensor<1024xf64, #blocked0>
  %324 = "triton_gpu.cmpf"(%322, %323) {predicate = 2 : i64} : (tensor<1024xf64, #blocked0>, tensor<1024xf64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %325 = tt.splat %arg10 : (!tt.ptr<f64>) -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %326 = tt.addptr %325, %318 : tensor<1024x!tt.ptr<f64>, #blocked0>, tensor<1024xi64, #blocked0>
  %327 = triton_gpu.convert_layout %326 : (tensor<1024x!tt.ptr<f64>, #blocked0>) -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %328 = tt.load %327 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf64, #blocked0>
  %329 = arith.divf %328, %322 : tensor<1024xf64, #blocked0>
  %330 = arith.truncf %329 : tensor<1024xf64, #blocked0> to tensor<1024xf32, #blocked0>
  %331 = arith.mulf %330, %cst_1 : tensor<1024xf32, #blocked0>
  %332 = arith.mulf %35, %cst_0 : tensor<1024xf32, #blocked0>
  %333 = arith.addf %331, %332 : tensor<1024xf32, #blocked0>
  %334 = "triton_gpu.select"(%324, %333, %35) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %335 = tt.addptr %319, %317 : tensor<1024x!tt.ptr<f64>, #blocked0>, tensor<1024xi32, #blocked0>
  %336 = triton_gpu.convert_layout %335 : (tensor<1024x!tt.ptr<f64>, #blocked0>) -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %337 = tt.load %336 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf64, #blocked0>
  %338 = arith.extf %cst : tensor<1024xf32, #blocked0> to tensor<1024xf64, #blocked0>
  %339 = arith.mulf %337, %338 : tensor<1024xf64, #blocked0>
  %340 = tt.addptr %325, %317 : tensor<1024x!tt.ptr<f64>, #blocked0>, tensor<1024xi32, #blocked0>
  %341 = triton_gpu.convert_layout %340 : (tensor<1024x!tt.ptr<f64>, #blocked0>) -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %342 = tt.load %341 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf64, #blocked0>
  %343 = arith.mulf %342, %338 : tensor<1024xf64, #blocked0>
  %344 = tt.splat %arg11 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %345 = tt.addptr %344, %4 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %346 = triton_gpu.convert_layout %345 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked1>
  %347 = triton_gpu.convert_layout %28 : (tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked1>
  %348 = triton_gpu.convert_layout %5 : (tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked1>
  tt.store %346, %347, %348 : tensor<1024xf32, #blocked1>
  %349 = tt.splat %arg12 : (!tt.ptr<i32>) -> tensor<1024x!tt.ptr<i32>, #blocked0>
  %350 = tt.addptr %349, %4 : tensor<1024x!tt.ptr<i32>, #blocked0>, tensor<1024xi32, #blocked0>
  %351 = triton_gpu.convert_layout %350 : (tensor<1024x!tt.ptr<i32>, #blocked0>) -> tensor<1024x!tt.ptr<i32>, #blocked1>
  %352 = triton_gpu.convert_layout %317 : (tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked1>
  %353 = triton_gpu.convert_layout %5 : (tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked1>
  tt.store %351, %352, %353 : tensor<1024xi32, #blocked1>
  %354 = tt.splat %arg13 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %355 = tt.addptr %354, %4 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %356 = triton_gpu.convert_layout %355 : (tensor<1024x!tt.ptr<f32>, #blocked0>) -> tensor<1024x!tt.ptr<f32>, #blocked1>
  %357 = triton_gpu.convert_layout %334 : (tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked1>
  %358 = triton_gpu.convert_layout %5 : (tensor<1024xi1, #blocked0>) -> tensor<1024xi1, #blocked1>
  tt.store %356, %357, %358 : tensor<1024xf32, #blocked1>
  %359 = tt.splat %arg14 : (!tt.ptr<f64>) -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %360 = tt.addptr %359, %318 : tensor<1024x!tt.ptr<f64>, #blocked0>, tensor<1024xi64, #blocked0>
  %361 = triton_gpu.convert_layout %360 : (tensor<1024x!tt.ptr<f64>, #blocked0>) -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %362 = triton_gpu.convert_layout %339 : (tensor<1024xf64, #blocked0>) -> tensor<1024xf64, #blocked0>
  tt.store %361, %362 : tensor<1024xf64, #blocked0>
  %363 = tt.splat %arg15 : (!tt.ptr<f64>) -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %364 = tt.addptr %363, %318 : tensor<1024x!tt.ptr<f64>, #blocked0>, tensor<1024xi64, #blocked0>
  %365 = triton_gpu.convert_layout %364 : (tensor<1024x!tt.ptr<f64>, #blocked0>) -> tensor<1024x!tt.ptr<f64>, #blocked0>
  %366 = triton_gpu.convert_layout %343 : (tensor<1024xf64, #blocked0>) -> tensor<1024xf64, #blocked0>
  tt.store %365, %366 : tensor<1024xf64, #blocked0>
  tt.return
}

// A mnist model from torch inductor.
// Check if topological sort is working correct and there's no unnecessary convert
// CHECK-LABEL: mnist
tt.func public @mnist(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32) {
  // CHECK-NOT: triton_gpu.convert_layout
  %cst = arith.constant dense<10> : tensor<16x1xi32, #blocked2>
  %cst_0 = arith.constant dense<10> : tensor<1x16xi32, #blocked3>
  %c16_i32 = arith.constant 16 : i32
  %cst_1 = arith.constant dense<64> : tensor<16x1xi32, #blocked2>
  %cst_2 = arith.constant dense<0xFF800000> : tensor<16x16xf32, #blocked2>
  %cst_3 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked2>
  %cst_4 = arith.constant dense<0> : tensor<16x16xi32, #blocked2>
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = arith.muli %0, %c16_i32 : i32
  %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked0>
  %3 = triton_gpu.convert_layout %2 : (tensor<16xi32, #blocked0>) -> tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %4 = tt.expand_dims %3 {axis = 1 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<16x1xi32, #blocked1>
  %5 = triton_gpu.convert_layout %4 : (tensor<16x1xi32, #blocked1>) -> tensor<16x1xi32, #blocked2>
  %6 = tt.splat %1 : (i32) -> tensor<16x1xi32, #blocked2>
  %7 = arith.addi %6, %5 : tensor<16x1xi32, #blocked2>
  %8 = "triton_gpu.cmpi"(%7, %cst_1) {predicate = 2 : i64} : (tensor<16x1xi32, #blocked2>, tensor<16x1xi32, #blocked2>) -> tensor<16x1xi1, #blocked2>
  %9 = triton_gpu.convert_layout %2 : (tensor<16xi32, #blocked0>) -> tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
  %10 = tt.expand_dims %9 {axis = 0 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x16xi32, #blocked3>
  %11 = "triton_gpu.cmpi"(%10, %cst_0) {predicate = 2 : i64} : (tensor<1x16xi32, #blocked3>, tensor<1x16xi32, #blocked3>) -> tensor<1x16xi1, #blocked3>
  %12 = arith.muli %7, %cst : tensor<16x1xi32, #blocked2>
  %13 = tt.broadcast %10 : (tensor<1x16xi32, #blocked3>) -> tensor<16x16xi32, #blocked3>
  %14 = triton_gpu.convert_layout %13 : (tensor<16x16xi32, #blocked3>) -> tensor<16x16xi32, #blocked2>
  %15 = tt.broadcast %12 : (tensor<16x1xi32, #blocked2>) -> tensor<16x16xi32, #blocked2>
  %16 = arith.addi %14, %15 : tensor<16x16xi32, #blocked2>
  %17 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<16x16x!tt.ptr<f32>, #blocked2>
  %18 = tt.addptr %17, %16 : tensor<16x16x!tt.ptr<f32>, #blocked2>, tensor<16x16xi32, #blocked2>
  %19 = tt.broadcast %11 : (tensor<1x16xi1, #blocked3>) -> tensor<16x16xi1, #blocked3>
  %20 = triton_gpu.convert_layout %19 : (tensor<16x16xi1, #blocked3>) -> tensor<16x16xi1, #blocked2>
  %21 = tt.broadcast %8 : (tensor<16x1xi1, #blocked2>) -> tensor<16x16xi1, #blocked2>
  %22 = arith.andi %20, %21 : tensor<16x16xi1, #blocked2>
  %23 = triton_gpu.convert_layout %18 : (tensor<16x16x!tt.ptr<f32>, #blocked2>) -> tensor<16x16x!tt.ptr<f32>, #blocked4>
  %24 = triton_gpu.convert_layout %22 : (tensor<16x16xi1, #blocked2>) -> tensor<16x16xi1, #blocked4>
  %25 = tt.load %23, %24 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<16x16xf32, #blocked4>
  %26 = triton_gpu.convert_layout %25 : (tensor<16x16xf32, #blocked4>) -> tensor<16x16xf32, #blocked2>
  %27 = "triton_gpu.cmpf"(%cst_2, %26) {predicate = 4 : i64} : (tensor<16x16xf32, #blocked2>, tensor<16x16xf32, #blocked2>) -> tensor<16x16xi1, #blocked2>
  %28 = arith.andi %22, %27 : tensor<16x16xi1, #blocked2>
  %29 = "triton_gpu.select"(%28, %26, %cst_2) : (tensor<16x16xi1, #blocked2>, tensor<16x16xf32, #blocked2>, tensor<16x16xf32, #blocked2>) -> tensor<16x16xf32, #blocked2>
  %30 = tt.reduce %29 {axis = 1 : i32, redOp = 12 : i32} : tensor<16x16xf32, #blocked2> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
  %31 = triton_gpu.convert_layout %30 : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<16xf32, #blocked0>
  %32 = triton_gpu.convert_layout %31 : (tensor<16xf32, #blocked0>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %33 = tt.expand_dims %32 {axis = 1 : i32} : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<16x1xf32, #blocked1>
  %34 = triton_gpu.convert_layout %33 : (tensor<16x1xf32, #blocked1>) -> tensor<16x1xf32, #blocked2>
  %35 = arith.sitofp %cst_4 : tensor<16x16xi32, #blocked2> to tensor<16x16xf32, #blocked2>
  %36 = arith.addf %35, %cst_3 : tensor<16x16xf32, #blocked2>
  %37 = triton_gpu.convert_layout %18 : (tensor<16x16x!tt.ptr<f32>, #blocked2>) -> tensor<16x16x!tt.ptr<f32>, #blocked4>
  %38 = triton_gpu.convert_layout %22 : (tensor<16x16xi1, #blocked2>) -> tensor<16x16xi1, #blocked4>
  %39 = tt.load %37, %38 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<16x16xf32, #blocked4>
  %40 = triton_gpu.convert_layout %39 : (tensor<16x16xf32, #blocked4>) -> tensor<16x16xf32, #blocked2>
  %41 = tt.broadcast %34 : (tensor<16x1xf32, #blocked2>) -> tensor<16x16xf32, #blocked2>
  %42 = arith.subf %40, %41 : tensor<16x16xf32, #blocked2>
  %43 = math.exp %42 : tensor<16x16xf32, #blocked2>
  %44 = arith.addf %36, %43 : tensor<16x16xf32, #blocked2>
  %45 = "triton_gpu.select"(%22, %44, %36) : (tensor<16x16xi1, #blocked2>, tensor<16x16xf32, #blocked2>, tensor<16x16xf32, #blocked2>) -> tensor<16x16xf32, #blocked2>
  %46 = tt.reduce %45 {axis = 1 : i32, redOp = 2 : i32} : tensor<16x16xf32, #blocked2> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
  %47 = triton_gpu.convert_layout %46 : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<16xf32, #blocked0>
  %48 = triton_gpu.convert_layout %47 : (tensor<16xf32, #blocked0>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %49 = tt.expand_dims %48 {axis = 1 : i32} : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<16x1xf32, #blocked1>
  %50 = triton_gpu.convert_layout %49 : (tensor<16x1xf32, #blocked1>) -> tensor<16x1xf32, #blocked2>
  %51 = triton_gpu.convert_layout %18 : (tensor<16x16x!tt.ptr<f32>, #blocked2>) -> tensor<16x16x!tt.ptr<f32>, #blocked4>
  %52 = triton_gpu.convert_layout %22 : (tensor<16x16xi1, #blocked2>) -> tensor<16x16xi1, #blocked4>
  %53 = tt.load %51, %52 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<16x16xf32, #blocked4>
  %54 = triton_gpu.convert_layout %53 : (tensor<16x16xf32, #blocked4>) -> tensor<16x16xf32, #blocked2>
  %55 = arith.subf %54, %41 : tensor<16x16xf32, #blocked2>
  %56 = math.log %50 : tensor<16x1xf32, #blocked2>
  %57 = tt.broadcast %56 : (tensor<16x1xf32, #blocked2>) -> tensor<16x16xf32, #blocked2>
  %58 = arith.subf %55, %57 : tensor<16x16xf32, #blocked2>
  %59 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<16x16x!tt.ptr<f32>, #blocked2>
  %60 = tt.addptr %59, %16 : tensor<16x16x!tt.ptr<f32>, #blocked2>, tensor<16x16xi32, #blocked2>
  %61 = triton_gpu.convert_layout %60 : (tensor<16x16x!tt.ptr<f32>, #blocked2>) -> tensor<16x16x!tt.ptr<f32>, #blocked4>
  %62 = triton_gpu.convert_layout %58 : (tensor<16x16xf32, #blocked2>) -> tensor<16x16xf32, #blocked4>
  %63 = triton_gpu.convert_layout %22 : (tensor<16x16xi1, #blocked2>) -> tensor<16x16xi1, #blocked4>
  tt.store %61, %62, %63 : tensor<16x16xf32, #blocked4>
  tt.return
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 4], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
// cmpf and cmpi have different operands and result types
// CHECK-LABEL: cmp
tt.func public @cmp(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) {
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %c0 = arith.constant 0 : index
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant dense<-3.40282347E+38> : tensor<64x64xf32, #blocked2>
  %cst_0 = arith.constant dense<4194304> : tensor<64x1xi32, #blocked2>
  %cst_1 = arith.constant dense<12> : tensor<64x1xi32, #blocked2>
  %cst_2 = arith.constant dense<2048> : tensor<1x64xi32, #blocked3>
  %cst_3 = arith.constant dense<0> : tensor<64x64xi32, #blocked2>
  %cst_4 = arith.constant dense<2048> : tensor<64x1xi32, #blocked2>
  %cst_5 = arith.constant dense<49152> : tensor<64x1xi32, #blocked2>
  %cst_6 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked2>
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked0>
  %3 = triton_gpu.convert_layout %2 : (tensor<64xi32, #blocked0>) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %4 = tt.expand_dims %3 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<64x1xi32, #blocked1>
  %5 = triton_gpu.convert_layout %4 : (tensor<64x1xi32, #blocked1>) -> tensor<64x1xi32, #blocked2>
  %6 = tt.splat %1 : (i32) -> tensor<64x1xi32, #blocked2>
  %7 = arith.addi %6, %5 : tensor<64x1xi32, #blocked2>
  %8 = "triton_gpu.cmpi"(%7, %cst_5) {predicate = 2 : i64} : (tensor<64x1xi32, #blocked2>, tensor<64x1xi32, #blocked2>) -> tensor<64x1xi1, #blocked2>
  %9 = triton_gpu.convert_layout %2 : (tensor<64xi32, #blocked0>) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
  %10 = tt.expand_dims %9 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x64xi32, #blocked3>
  %11 = arith.remsi %7, %cst_4 : tensor<64x1xi32, #blocked2>
  %12 = arith.divsi %7, %cst_4 : tensor<64x1xi32, #blocked2>
  %13 = arith.sitofp %cst_3 : tensor<64x64xi32, #blocked2> to tensor<64x64xf32, #blocked2>
  %14 = arith.addf %13, %cst_6 : tensor<64x64xf32, #blocked2>
  %15 = arith.muli %7, %cst_4 : tensor<64x1xi32, #blocked2>
  %16 = tt.broadcast %15 : (tensor<64x1xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %17 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<64x64x!tt.ptr<f16>, #blocked2>
  %18 = tt.broadcast %8 : (tensor<64x1xi1, #blocked2>) -> tensor<64x64xi1, #blocked2>
  %19 = arith.muli %11, %cst_4 : tensor<64x1xi32, #blocked2>
  %20 = tt.broadcast %19 : (tensor<64x1xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %21 = arith.divsi %12, %cst_1 : tensor<64x1xi32, #blocked2>
  %22 = arith.muli %21, %cst_0 : tensor<64x1xi32, #blocked2>
  %23 = tt.broadcast %22 : (tensor<64x1xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %24 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<64x64x!tt.ptr<f32>, #blocked2>
  %25 = scf.for %arg6 = %c0 to %c2048 step %c64 iter_args(%arg7 = %14) -> (tensor<64x64xf32, #blocked2>) {
    %44 = arith.index_cast %arg6 : index to i32
    %45 = tt.splat %44 : (i32) -> tensor<1x64xi32, #blocked3>
    %46 = arith.addi %45, %10 : tensor<1x64xi32, #blocked3>
    %47 = "triton_gpu.cmpi"(%46, %cst_2) {predicate = 2 : i64} : (tensor<1x64xi32, #blocked3>, tensor<1x64xi32, #blocked3>) -> tensor<1x64xi1, #blocked3>
    %48 = tt.broadcast %46 : (tensor<1x64xi32, #blocked3>) -> tensor<64x64xi32, #blocked3>
    %49 = triton_gpu.convert_layout %48 : (tensor<64x64xi32, #blocked3>) -> tensor<64x64xi32, #blocked2>
    %50 = arith.addi %49, %16 : tensor<64x64xi32, #blocked2>
    %51 = tt.addptr %17, %50 : tensor<64x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xi32, #blocked2>
    %52 = tt.broadcast %47 : (tensor<1x64xi1, #blocked3>) -> tensor<64x64xi1, #blocked3>
    %53 = triton_gpu.convert_layout %52 : (tensor<64x64xi1, #blocked3>) -> tensor<64x64xi1, #blocked2>
    %54 = arith.andi %53, %18 : tensor<64x64xi1, #blocked2>
    %55 = triton_gpu.convert_layout %51 : (tensor<64x64x!tt.ptr<f16>, #blocked2>) -> tensor<64x64x!tt.ptr<f16>, #blocked4>
    %56 = triton_gpu.convert_layout %54 : (tensor<64x64xi1, #blocked2>) -> tensor<64x64xi1, #blocked4>
    %57 = tt.load %55, %56 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<64x64xf16, #blocked4>
    %58 = triton_gpu.convert_layout %57 : (tensor<64x64xf16, #blocked4>) -> tensor<64x64xf16, #blocked2>
    %59 = arith.extf %58 : tensor<64x64xf16, #blocked2> to tensor<64x64xf32, #blocked2>
    %60 = arith.addi %49, %20 : tensor<64x64xi32, #blocked2>
    %61 = arith.addi %60, %23 : tensor<64x64xi32, #blocked2>
    %62 = tt.addptr %24, %61 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %63 = triton_gpu.convert_layout %62 : (tensor<64x64x!tt.ptr<f32>, #blocked2>) -> tensor<64x64x!tt.ptr<f32>, #blocked5>
    %64 = triton_gpu.convert_layout %54 : (tensor<64x64xi1, #blocked2>) -> tensor<64x64xi1, #blocked5>
    %65 = tt.load %63, %64 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<64x64xf32, #blocked5>
    %66 = triton_gpu.convert_layout %65 : (tensor<64x64xf32, #blocked5>) -> tensor<64x64xf32, #blocked2>
    %67 = arith.addf %59, %66 : tensor<64x64xf32, #blocked2>
    %68 = "triton_gpu.cmpf"(%67, %67) {predicate = 13 : i64} : (tensor<64x64xf32, #blocked2>, tensor<64x64xf32, #blocked2>) -> tensor<64x64xi1, #blocked2>
    %69 = "triton_gpu.cmpf"(%67, %cst) {predicate = 2 : i64} : (tensor<64x64xf32, #blocked2>, tensor<64x64xf32, #blocked2>) -> tensor<64x64xi1, #blocked2>
    %70 = "triton_gpu.select"(%69, %67, %cst) : (tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>, tensor<64x64xf32, #blocked2>) -> tensor<64x64xf32, #blocked2>
    %71 = "triton_gpu.select"(%68, %67, %70) : (tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>, tensor<64x64xf32, #blocked2>) -> tensor<64x64xf32, #blocked2>
    %72 = math.exp %71 : tensor<64x64xf32, #blocked2>
    %73 = arith.addf %arg7, %72 : tensor<64x64xf32, #blocked2>
    %74 = "triton_gpu.select"(%54, %73, %arg7) : (tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>, tensor<64x64xf32, #blocked2>) -> tensor<64x64xf32, #blocked2>
    scf.yield %74 : tensor<64x64xf32, #blocked2>
  }
  %26 = tt.reduce %25 {axis = 1 : i32, redOp = 2 : i32} : tensor<64x64xf32, #blocked2> -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
  %27 = triton_gpu.convert_layout %26 : (tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64xf32, #blocked0>
  %28 = triton_gpu.convert_layout %27 : (tensor<64xf32, #blocked0>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %29 = tt.expand_dims %28 {axis = 1 : i32} : (tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<64x1xf32, #blocked1>
  %30 = triton_gpu.convert_layout %29 : (tensor<64x1xf32, #blocked1>) -> tensor<64x1xf32, #blocked2>
  %31 = arith.muli %7, %cst_4 : tensor<64x1xi32, #blocked2>
  %32 = tt.broadcast %31 : (tensor<64x1xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %33 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<64x64x!tt.ptr<f16>, #blocked2>
  %34 = tt.broadcast %8 : (tensor<64x1xi1, #blocked2>) -> tensor<64x64xi1, #blocked2>
  %35 = arith.muli %11, %cst_4 : tensor<64x1xi32, #blocked2>
  %36 = tt.broadcast %35 : (tensor<64x1xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %37 = arith.divsi %12, %cst_1 : tensor<64x1xi32, #blocked2>
  %38 = arith.muli %37, %cst_0 : tensor<64x1xi32, #blocked2>
  %39 = tt.broadcast %38 : (tensor<64x1xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %40 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<64x64x!tt.ptr<f32>, #blocked2>
  %41 = tt.broadcast %30 : (tensor<64x1xf32, #blocked2>) -> tensor<64x64xf32, #blocked2>
  %42 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x64x!tt.ptr<f32>, #blocked2>
  %43 = tt.splat %arg3 : (!tt.ptr<f16>) -> tensor<64x64x!tt.ptr<f16>, #blocked2>
  scf.for %arg6 = %c0 to %c2048 step %c64 {
    %44 = arith.index_cast %arg6 : index to i32
    %45 = tt.splat %44 : (i32) -> tensor<1x64xi32, #blocked3>
    %46 = arith.addi %45, %10 : tensor<1x64xi32, #blocked3>
    %47 = "triton_gpu.cmpi"(%46, %cst_2) {predicate = 2 : i64} : (tensor<1x64xi32, #blocked3>, tensor<1x64xi32, #blocked3>) -> tensor<1x64xi1, #blocked3>
    %48 = tt.broadcast %46 : (tensor<1x64xi32, #blocked3>) -> tensor<64x64xi32, #blocked3>
    %49 = triton_gpu.convert_layout %48 : (tensor<64x64xi32, #blocked3>) -> tensor<64x64xi32, #blocked2>
    %50 = arith.addi %49, %32 : tensor<64x64xi32, #blocked2>
    %51 = tt.addptr %33, %50 : tensor<64x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xi32, #blocked2>
    %52 = tt.broadcast %47 : (tensor<1x64xi1, #blocked3>) -> tensor<64x64xi1, #blocked3>
    %53 = triton_gpu.convert_layout %52 : (tensor<64x64xi1, #blocked3>) -> tensor<64x64xi1, #blocked2>
    %54 = arith.andi %53, %34 : tensor<64x64xi1, #blocked2>
    %55 = triton_gpu.convert_layout %51 : (tensor<64x64x!tt.ptr<f16>, #blocked2>) -> tensor<64x64x!tt.ptr<f16>, #blocked4>
    %56 = triton_gpu.convert_layout %54 : (tensor<64x64xi1, #blocked2>) -> tensor<64x64xi1, #blocked4>
    %57 = tt.load %55, %56 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<64x64xf16, #blocked4>
    %58 = triton_gpu.convert_layout %57 : (tensor<64x64xf16, #blocked4>) -> tensor<64x64xf16, #blocked2>
    %59 = arith.extf %58 : tensor<64x64xf16, #blocked2> to tensor<64x64xf32, #blocked2>
    %60 = arith.addi %49, %36 : tensor<64x64xi32, #blocked2>
    %61 = arith.addi %60, %39 : tensor<64x64xi32, #blocked2>
    %62 = tt.addptr %40, %61 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %63 = triton_gpu.convert_layout %62 : (tensor<64x64x!tt.ptr<f32>, #blocked2>) -> tensor<64x64x!tt.ptr<f32>, #blocked5>
    %64 = triton_gpu.convert_layout %54 : (tensor<64x64xi1, #blocked2>) -> tensor<64x64xi1, #blocked5>
    %65 = tt.load %63, %64 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<64x64xf32, #blocked5>
    %66 = triton_gpu.convert_layout %65 : (tensor<64x64xf32, #blocked5>) -> tensor<64x64xf32, #blocked2>
    %67 = arith.addf %59, %66 : tensor<64x64xf32, #blocked2>
    %68 = "triton_gpu.cmpf"(%67, %67) {predicate = 13 : i64} : (tensor<64x64xf32, #blocked2>, tensor<64x64xf32, #blocked2>) -> tensor<64x64xi1, #blocked2>
    %69 = "triton_gpu.cmpf"(%67, %cst) {predicate = 2 : i64} : (tensor<64x64xf32, #blocked2>, tensor<64x64xf32, #blocked2>) -> tensor<64x64xi1, #blocked2>
    %70 = "triton_gpu.select"(%69, %67, %cst) : (tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>, tensor<64x64xf32, #blocked2>) -> tensor<64x64xf32, #blocked2>
    %71 = "triton_gpu.select"(%68, %67, %70) : (tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>, tensor<64x64xf32, #blocked2>) -> tensor<64x64xf32, #blocked2>
    %72 = math.exp %71 : tensor<64x64xf32, #blocked2>
    %73 = arith.divf %72, %41 : tensor<64x64xf32, #blocked2>
    %74 = tt.addptr %42, %50 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %75 = triton_gpu.convert_layout %74 : (tensor<64x64x!tt.ptr<f32>, #blocked2>) -> tensor<64x64x!tt.ptr<f32>, #blocked5>
    %76 = triton_gpu.convert_layout %73 : (tensor<64x64xf32, #blocked2>) -> tensor<64x64xf32, #blocked5>
    %77 = triton_gpu.convert_layout %54 : (tensor<64x64xi1, #blocked2>) -> tensor<64x64xi1, #blocked5>
    tt.store %75, %76, %77 : tensor<64x64xf32, #blocked5>
    %78 = tt.addptr %43, %50 : tensor<64x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xi32, #blocked2>
    %79 = arith.truncf %73 : tensor<64x64xf32, #blocked2> to tensor<64x64xf16, #blocked2>
    %80 = triton_gpu.convert_layout %78 : (tensor<64x64x!tt.ptr<f16>, #blocked2>) -> tensor<64x64x!tt.ptr<f16>, #blocked4>
    %81 = triton_gpu.convert_layout %79 : (tensor<64x64xf16, #blocked2>) -> tensor<64x64xf16, #blocked4>
    %82 = triton_gpu.convert_layout %54 : (tensor<64x64xi1, #blocked2>) -> tensor<64x64xi1, #blocked4>
    tt.store %80, %81, %82 : tensor<64x64xf16, #blocked4>
  }
  tt.return
}

// -----

// Just make sure it doesn't crash on non-tensor types.
// CHECK-LABEL: if_no_tensor
tt.func public @if_no_tensor(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}) {
  // CHECK-NOT: triton_gpu.convert_layout
  %c-1_i64 = arith.constant -1 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c-1_i32 = arith.constant -1 : i32
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = tt.addptr %arg3, %0 : !tt.ptr<i64>, i32
  %2 = tt.load %1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i64
  %3 = arith.cmpi eq, %2, %c-1_i64 : i64
  %4 = arith.select %3, %c-1_i32, %arg2 : i32
  %5 = scf.if %3 -> (!tt.ptr<f32>) {
    scf.yield %arg0 : !tt.ptr<f32>
  } else {
    %10 = tt.addptr %arg0, %2 : !tt.ptr<f32>, i64
    scf.yield %10 : !tt.ptr<f32>
  }
  %6 = arith.extsi %4 : i32 to i64
  %7 = arith.cmpi slt, %2, %6 : i64
  %8 = tt.load %5, %7, %cst {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
  %9 = tt.addptr %arg1, %0 : !tt.ptr<f32>, i32
  tt.store %9, %8 {cache = 1 : i32, evict = 1 : i32} : f32
  tt.return
}

// -----

// Check if the SimplifyReduceCvt rewriter pattern doesn't hang.
// CHECK-LABEL: reduce_cvt
// CHECK-NOT: triton_gpu.convert_layout
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 2 : i32} {
  tt.func public @reduce_cvt1(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32) {
    %cst = arith.constant dense<0> : tensor<1x2xi32, #blocked>
    %cst_0 = arith.constant dense<2> : tensor<1x2xi32, #blocked>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #blocked1>
    %1 = triton_gpu.convert_layout %0 : (tensor<2xi32, #blocked1>) -> tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x2xi32, #blocked>
    %3 = "triton_gpu.cmpi"(%2, %cst_0) {predicate = 2 : i64} : (tensor<1x2xi32, #blocked>, tensor<1x2xi32, #blocked>) -> tensor<1x2xi1, #blocked>
    %4 = tt.reduce %cst {axis = 1 : i32, redOp = 1 : i32} : tensor<1x2xi32, #blocked> -> tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %5 = triton_gpu.convert_layout %4 : (tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<1xi32, #blocked1>
    %6 = triton_gpu.convert_layout %5 : (tensor<1xi32, #blocked1>) -> tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : (tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<1x1xi32, #blocked2>
    %8 = triton_gpu.convert_layout %7 : (tensor<1x1xi32, #blocked2>) -> tensor<1x1xi32, #blocked>
    %9 = tt.splat %arg0 : (!tt.ptr<i64>) -> tensor<1x2x!tt.ptr<i64>, #blocked>
    %10 = tt.addptr %9, %2 : tensor<1x2x!tt.ptr<i64>, #blocked>, tensor<1x2xi32, #blocked>
    %11 = tt.broadcast %8 : (tensor<1x1xi32, #blocked>) -> tensor<1x2xi32, #blocked>
    %12 = arith.extsi %11 : tensor<1x2xi32, #blocked> to tensor<1x2xi64, #blocked>
    %13 = triton_gpu.convert_layout %10 : (tensor<1x2x!tt.ptr<i64>, #blocked>) -> tensor<1x2x!tt.ptr<i64>, #blocked3>
    %14 = triton_gpu.convert_layout %12 : (tensor<1x2xi64, #blocked>) -> tensor<1x2xi64, #blocked3>
    %15 = triton_gpu.convert_layout %3 : (tensor<1x2xi1, #blocked>) -> tensor<1x2xi1, #blocked3>
    tt.store %13, %14, %15 {cache = 1 : i32, evict = 1 : i32} : tensor<1x2xi64, #blocked3>
    tt.return
  }
}

// -----

// Check if the SimplifyReduceCvt handles convert_layout lifted from the for loop.
// CHECK-LABEL: reduce_cvt2
// CHECK: tt.reduce
// CHECK-NEXT: triton_gpu.convert_layout
// CHECK-NOT: triton_gpu.convert_layout
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  tt.func public @reduce_cvt2(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x256xf32, #blocked>
    %c3136_i32 = arith.constant 3136 : index
    %c256_i32 = arith.constant 256 : index
    %c0_i32 = arith.constant 0 : index
    %cst_0 = arith.constant dense<3.136000e+03> : tensor<1x1xf32, #blocked>
    %cst_1 = arith.constant dense<50176> : tensor<1x256xi32, #blocked>
    %cst_2 = arith.constant dense<196> : tensor<1x1xi32, #blocked>
    %cst_3 = arith.constant dense<196> : tensor<1x256xi32, #blocked>
    %cst_4 = arith.constant dense<3136> : tensor<1x256xi32, #blocked>
    %cst_5 = arith.constant dense<256> : tensor<1x1xi32, #blocked>
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #blocked1>
    %2 = triton_gpu.convert_layout %1 : (tensor<1xi32, #blocked1>) -> tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<1x1xi32, #blocked2>
    %4 = triton_gpu.convert_layout %3 : (tensor<1x1xi32, #blocked2>) -> tensor<1x1xi32, #blocked>
    %5 = tt.splat %0 : (i32) -> tensor<1x1xi32, #blocked>
    %6 = arith.addi %5, %4 : tensor<1x1xi32, #blocked>
    %7 = "triton_gpu.cmpi"(%6, %cst_5) {predicate = 2 : i64} : (tensor<1x1xi32, #blocked>, tensor<1x1xi32, #blocked>) -> tensor<1x1xi1, #blocked>
    %8 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %9 = triton_gpu.convert_layout %8 : (tensor<256xi32, #blocked1>) -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %10 = tt.expand_dims %9 {axis = 0 : i32} : (tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x256xi32, #blocked>
    %11 = arith.muli %6, %cst_2 : tensor<1x1xi32, #blocked>
    %12 = tt.broadcast %11 : (tensor<1x1xi32, #blocked>) -> tensor<1x256xi32, #blocked>
    %13 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<1x256x!tt.ptr<f32>, #blocked>
    %14 = tt.broadcast %7 : (tensor<1x1xi1, #blocked>) -> tensor<1x256xi1, #blocked>
    %15 = scf.for %arg5 = %c0_i32 to %c3136_i32 step %c256_i32 iter_args(%arg6 = %cst) -> (tensor<1x256xf32, #blocked>) {
      %42 = arith.index_cast %arg5 : index to i32
      %43 = tt.splat %42 : (i32) -> tensor<1x256xi32, #blocked>
      %44 = arith.addi %43, %10 : tensor<1x256xi32, #blocked>
      %45 = "triton_gpu.cmpi"(%44, %cst_4) {predicate = 2 : i64} : (tensor<1x256xi32, #blocked>, tensor<1x256xi32, #blocked>) -> tensor<1x256xi1, #blocked>
      %46 = arith.remsi %44, %cst_3 : tensor<1x256xi32, #blocked>
      %47 = arith.divsi %44, %cst_3 : tensor<1x256xi32, #blocked>
      %48 = arith.addi %46, %12 : tensor<1x256xi32, #blocked>
      %49 = arith.muli %47, %cst_1 : tensor<1x256xi32, #blocked>
      %50 = arith.addi %48, %49 : tensor<1x256xi32, #blocked>
      %51 = tt.addptr %13, %50 : tensor<1x256x!tt.ptr<f32>, #blocked>, tensor<1x256xi32, #blocked>
      %52 = arith.andi %45, %14 : tensor<1x256xi1, #blocked>
      %53 = triton_gpu.convert_layout %51 : (tensor<1x256x!tt.ptr<f32>, #blocked>) -> tensor<1x256x!tt.ptr<f32>, #blocked3>
      %54 = triton_gpu.convert_layout %52 : (tensor<1x256xi1, #blocked>) -> tensor<1x256xi1, #blocked3>
      %55 = triton_gpu.convert_layout %cst : (tensor<1x256xf32, #blocked>) -> tensor<1x256xf32, #blocked3>
      %56 = tt.load %53, %54, %55 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<1x256xf32, #blocked3>
      %57 = triton_gpu.convert_layout %56 : (tensor<1x256xf32, #blocked3>) -> tensor<1x256xf32, #blocked>
      %58 = arith.addf %arg6, %57 : tensor<1x256xf32, #blocked>
      %59 = "triton_gpu.select"(%52, %58, %arg6) : (tensor<1x256xi1, #blocked>, tensor<1x256xf32, #blocked>, tensor<1x256xf32, #blocked>) -> tensor<1x256xf32, #blocked>
      scf.yield %59 : tensor<1x256xf32, #blocked>
    }
    %16 = tt.reduce %15 {axis = 1 : i32, redOp = 2 : i32} : tensor<1x256xf32, #blocked> -> tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %17 = triton_gpu.convert_layout %16 : (tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<1xf32, #blocked1>
    %18 = triton_gpu.convert_layout %17 : (tensor<1xf32, #blocked1>) -> tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %19 = tt.expand_dims %18 {axis = 1 : i32} : (tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<1x1xf32, #blocked2>
    %20 = triton_gpu.convert_layout %19 : (tensor<1x1xf32, #blocked2>) -> tensor<1x1xf32, #blocked>
    %21 = arith.divf %20, %cst_0 : tensor<1x1xf32, #blocked>
    %22 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<1x1x!tt.ptr<f32>, #blocked>
    %23 = tt.addptr %22, %6 : tensor<1x1x!tt.ptr<f32>, #blocked>, tensor<1x1xi32, #blocked>
    %24 = triton_gpu.convert_layout %23 : (tensor<1x1x!tt.ptr<f32>, #blocked>) -> tensor<1x1x!tt.ptr<f32>, #blocked>
    %25 = triton_gpu.convert_layout %21 : (tensor<1x1xf32, #blocked>) -> tensor<1x1xf32, #blocked>
    %26 = triton_gpu.convert_layout %7 : (tensor<1x1xi1, #blocked>) -> tensor<1x1xi1, #blocked>
    tt.store %24, %25, %26 {cache = 1 : i32, evict = 1 : i32} : tensor<1x1xf32, #blocked>
    tt.return
  }
}
