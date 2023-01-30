// RUN: triton-opt %s -split-input-file -tritongpu-combine 2>&1 | FileCheck %s

#layout0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#layout1 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// CHECK: [[target_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK: [[row_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK: [[col_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK: [[col_layout_novec:#.*]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK-LABEL: cst
func @cst() -> tensor<1024xi32, #layout1> {
  %cst = arith.constant dense<0> : tensor<1024xi32, #layout0>
  %1 = triton_gpu.convert_layout %cst : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: return %cst : tensor<1024xi32, [[target_layout]]>
  return %1: tensor<1024xi32, #layout1>
}

// CHECK-LABEL: range
func @range() -> tensor<1024xi32, #layout1> {
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %1 = triton_gpu.convert_layout %0 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: return %0 : tensor<1024xi32, [[target_layout]]>
  return %1: tensor<1024xi32, #layout1>
}

// CHECK-LABEL: splat
func @splat(%arg0: i32) -> tensor<1024xi32, #layout1> {
  %0 = tt.splat %arg0 : (i32) -> tensor<1024xi32, #layout0>
  %1 = triton_gpu.convert_layout %0 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: return %0 : tensor<1024xi32, [[target_layout]]>
  return %1: tensor<1024xi32, #layout1>
}

// CHECK-LABEL: remat
func @remat(%arg0: i32) -> tensor<1024xi32, #layout1> {
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %2 = arith.muli %0, %1 : tensor<1024xi32, #layout0>
  %3 = triton_gpu.convert_layout %2 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  %4 = tt.splat %arg0 : (i32) -> tensor<1024xi32, #layout0>
  %5 = triton_gpu.convert_layout %2 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  %6 = arith.addi %3, %5 : tensor<1024xi32, #layout1>
  return %6: tensor<1024xi32, #layout1>
  // CHECK: %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[target_layout]]>
  // CHECK: %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[target_layout]]>
  // CHECK: %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[target_layout]]>
  // CHECK: %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[target_layout]]>
  // CHECK: %4 = arith.muli %0, %2 : tensor<1024xi32, [[target_layout]]>
  // CHECK: %5 = arith.muli %1, %3 : tensor<1024xi32, [[target_layout]]>
  // CHECK: %6 = arith.addi %4, %5 : tensor<1024xi32, [[target_layout]]>
  // CHECK: return %6 : tensor<1024xi32, [[target_layout]]>
}

// CHECK-LABEL: if
func @if(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
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
  return
}

// CHECK-LABEL: if_convert_else_not
func @if_convert_else_not(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
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
  return
}

// CHECK-LABEL: if_not_else_convert
func @if_not_else_convert(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
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
  return
}

// CHECK-LABEL: if_else_both_convert
func @if_else_both_convert(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
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
  // CHECK: triton_gpu.convert_layout
  // CHECK-NOT: triton_gpu.convert_layout
  tt.store %5, %8 : tensor<1024xi32, #layout1>
  return
}

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#slice1dim1 = #triton_gpu.slice<{dim = 1, parent = #blocked1}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#slice2dim0 = #triton_gpu.slice<{dim = 0, parent = #blocked2}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>

// CHECK-LABEL: transpose
func @transpose(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: [[loaded_val:%.*]] = tt.load {{.*}}, {{%cst.*}}, {{%cst.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf32, [[row_layout]]>
  // CHECK: [[cvt_val:%.*]] = triton_gpu.convert_layout [[loaded_val]] : (tensor<64x64xf32, [[row_layout]]>) -> tensor<64x64xf32, [[col_layout]]>
  // CHECK: tt.store {{.*}}, [[cvt_val]], {{%cst.*}} : tensor<64x64xf32, [[col_layout]]>
  // CHECK: return
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
  return
}

// CHECK-LABEL: loop
func @loop(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) {
    // CHECK-NOT: triton_gpu.convert_layout
    // CHECK: [[loop_ret:%.*]]:2 = scf.for {{.*}} -> (tensor<64x64xf32, [[row_layout]]>, tensor<64x64x!tt.ptr<f32>, [[row_layout]]>)
    // CHECK-NEXT: {{.*}} = tt.load {{.*}} : tensor<64x64xf32, [[row_layout]]>
    // CHECK-NEXT: {{.*}} = arith.addf {{.*}} : tensor<64x64xf32, [[row_layout]]>
    // CHECK-NEXT: {{.*}} = tt.addptr {{.*}} : tensor<64x64x!tt.ptr<f32>, [[row_layout]]>, tensor<64x64xi32, [[row_layout]]>
    // CHECK-NEXT: scf.yield {{.*}} : tensor<64x64xf32, [[row_layout]]>, tensor<64x64x!tt.ptr<f32>, [[row_layout]]>
    // CHECK-NEXT: }
    // CHECK-NEXT: {{.*}} = triton_gpu.convert_layout [[loop_ret]]#0 : (tensor<64x64xf32, [[row_layout]]>) -> tensor<64x64xf32, [[col_layout_novec]]>
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
    return
}

// CHECK-LABEL: vecadd
func @vecadd(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
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
  return
}

// Select has args with different element types
// CHECK-LABEL: select
func @select(%arg0: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f64> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) {
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
  return
}