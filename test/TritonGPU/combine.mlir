// RUN: triton-opt %s -split-input-file -tritongpu-combine 2>&1 | FileCheck %s

#layout0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#layout1 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// CHECK: [[target_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK: [[row_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK: [[col_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK: [[col_layout_novec:#.*]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
func @cst() -> tensor<1024xi32, #layout1> {
  %cst = arith.constant dense<0> : tensor<1024xi32, #layout0>
  %1 = triton_gpu.convert_layout %cst : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: return %cst : tensor<1024xi32, [[target_layout]]>
  return %1: tensor<1024xi32, #layout1>
}

func @range() -> tensor<1024xi32, #layout1> {
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %1 = triton_gpu.convert_layout %0 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: return %0 : tensor<1024xi32, [[target_layout]]>
  return %1: tensor<1024xi32, #layout1>
}

func @splat(%arg0: i32) -> tensor<1024xi32, #layout1> {
  %0 = tt.splat %arg0 : (i32) -> tensor<1024xi32, #layout0>
  %1 = triton_gpu.convert_layout %0 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: return %0 : tensor<1024xi32, [[target_layout]]>
  return %1: tensor<1024xi32, #layout1>
}

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

// Make sure the following IR doesn't hang the compiler.
// CHECK-LABEL: long_func
func public @long_func(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
  %cst = arith.constant dense<1536> : tensor<1024xi32, #blocked1>
  %cst_0 = arith.constant dense<0> : tensor<1024xi32, #blocked0>
  %c1024_i32 = arith.constant 1024 : i32
  %cst_1 = arith.constant dense<1536> : tensor<1024xi32, #blocked0>
  %cst_2 = arith.constant dense<-2> : tensor<1024xi32, #blocked0>
  %cst_3 = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked0>
  %cst_4 = arith.constant dense<1> : tensor<1024xi32, #blocked0>
  %cst_5 = arith.constant dense<2> : tensor<1024xi32, #blocked0>
  %cst_6 = arith.constant dense<3> : tensor<1024xi32, #blocked0>
  %cst_7 = arith.constant dense<4> : tensor<1024xi32, #blocked0>
  %cst_8 = arith.constant dense<12> : tensor<1024xi32, #blocked0>
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = arith.muli %0, %c1024_i32 : i32
  %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked0>
  %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
  %4 = tt.splat %1 : (i32) -> tensor<1024xi32, #blocked0>
  %5 = tt.splat %1 : (i32) -> tensor<1024xi32, #blocked1>
  %6 = arith.addi %4, %2 : tensor<1024xi32, #blocked0>
  %7 = "triton_gpu.cmpi"(%6, %cst_1) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %8 = arith.remsi %6, %cst_8 : tensor<1024xi32, #blocked0>
  %9 = arith.divsi %6, %cst_7 : tensor<1024xi32, #blocked0>
  %10 = arith.remsi %9, %cst_6 : tensor<1024xi32, #blocked0>
  %11 = arith.remsi %6, %cst_7 : tensor<1024xi32, #blocked0>
  %12 = arith.divsi %6, %cst_8 : tensor<1024xi32, #blocked0>
  %13 = arith.addi %10, %cst_2 : tensor<1024xi32, #blocked0>
  %14 = arith.addi %11, %cst_2 : tensor<1024xi32, #blocked0>
  %15 = arith.addi %10, %cst_6 : tensor<1024xi32, #blocked0>
  %16 = arith.addi %11, %cst_6 : tensor<1024xi32, #blocked0>
  %17 = "triton_gpu.cmpi"(%13, %cst_0) {predicate = 4 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %18 = "triton_gpu.select"(%17, %13, %cst_0) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %19 = "triton_gpu.cmpi"(%14, %cst_0) {predicate = 4 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %20 = "triton_gpu.select"(%19, %14, %cst_0) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %21 = "triton_gpu.cmpi"(%15, %cst_6) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %22 = "triton_gpu.select"(%21, %15, %cst_6) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %23 = "triton_gpu.cmpi"(%16, %cst_7) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %24 = "triton_gpu.select"(%23, %16, %cst_7) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %25 = arith.subi %22, %cst_4 : tensor<1024xi32, #blocked0>
  %26 = "triton_gpu.cmpi"(%18, %25) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %27 = "triton_gpu.select"(%26, %18, %25) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %28 = arith.subi %24, %cst_4 : tensor<1024xi32, #blocked0>
  %29 = "triton_gpu.cmpi"(%20, %28) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %30 = "triton_gpu.select"(%29, %20, %28) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %31 = arith.muli %27, %cst_7 : tensor<1024xi32, #blocked0>
  %32 = arith.addi %30, %31 : tensor<1024xi32, #blocked0>
  %33 = arith.muli %12, %cst_8 : tensor<1024xi32, #blocked0>
  %34 = arith.addi %32, %33 : tensor<1024xi32, #blocked0>
  %35 = tt.splat %arg0 : (!tt.ptr<i64>) -> tensor<1024x!tt.ptr<i64>, #blocked0>
  %36 = tt.addptr %35, %34 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %37 = tt.load %36, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %38 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>, #blocked0>
  %39 = tt.addptr %38, %34 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %40 = tt.load %39, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %41 = arith.extsi %8 : tensor<1024xi32, #blocked0> to tensor<1024xi64, #blocked0>
  %42 = "triton_gpu.cmpi"(%37, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %43 = "triton_gpu.select"(%42, %40, %cst_3) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %44 = arith.addi %20, %cst_4 : tensor<1024xi32, #blocked0>
  %45 = "triton_gpu.cmpi"(%44, %28) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %46 = "triton_gpu.select"(%45, %44, %28) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %47 = arith.addi %46, %31 : tensor<1024xi32, #blocked0>
  %48 = arith.addi %47, %33 : tensor<1024xi32, #blocked0>
  %49 = tt.addptr %35, %48 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %50 = tt.load %49, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %51 = tt.addptr %38, %48 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %52 = tt.load %51, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %53 = "triton_gpu.cmpi"(%50, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %54 = "triton_gpu.cmpi"(%18, %22) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %55 = "triton_gpu.cmpi"(%44, %24) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %56 = arith.andi %54, %55 : tensor<1024xi1, #blocked0>
  %57 = arith.andi %56, %53 : tensor<1024xi1, #blocked0>
  %58 = arith.addf %43, %52 : tensor<1024xf32, #blocked0>
  %59 = "triton_gpu.select"(%57, %58, %43) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %60 = arith.addi %20, %cst_5 : tensor<1024xi32, #blocked0>
  %61 = "triton_gpu.cmpi"(%60, %28) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %62 = "triton_gpu.select"(%61, %60, %28) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %63 = arith.addi %62, %31 : tensor<1024xi32, #blocked0>
  %64 = arith.addi %63, %33 : tensor<1024xi32, #blocked0>
  %65 = tt.addptr %35, %64 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %66 = tt.load %65, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %67 = tt.addptr %38, %64 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %68 = tt.load %67, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %69 = "triton_gpu.cmpi"(%66, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %70 = "triton_gpu.cmpi"(%60, %24) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %71 = arith.andi %54, %70 : tensor<1024xi1, #blocked0>
  %72 = arith.andi %71, %69 : tensor<1024xi1, #blocked0>
  %73 = arith.addf %59, %68 : tensor<1024xf32, #blocked0>
  %74 = "triton_gpu.select"(%72, %73, %59) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %75 = arith.addi %20, %cst_6 : tensor<1024xi32, #blocked0>
  %76 = "triton_gpu.cmpi"(%75, %28) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %77 = "triton_gpu.select"(%76, %75, %28) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %78 = arith.addi %77, %31 : tensor<1024xi32, #blocked0>
  %79 = arith.addi %78, %33 : tensor<1024xi32, #blocked0>
  %80 = tt.addptr %35, %79 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %81 = tt.load %80, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %82 = tt.addptr %38, %79 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %83 = tt.load %82, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %84 = "triton_gpu.cmpi"(%81, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %85 = "triton_gpu.cmpi"(%75, %24) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %86 = arith.andi %54, %85 : tensor<1024xi1, #blocked0>
  %87 = arith.andi %86, %84 : tensor<1024xi1, #blocked0>
  %88 = arith.addf %74, %83 : tensor<1024xf32, #blocked0>
  %89 = "triton_gpu.select"(%87, %88, %74) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %90 = arith.addi %20, %cst_7 : tensor<1024xi32, #blocked0>
  %91 = "triton_gpu.cmpi"(%90, %28) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %92 = "triton_gpu.select"(%91, %90, %28) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %93 = arith.addi %92, %31 : tensor<1024xi32, #blocked0>
  %94 = arith.addi %93, %33 : tensor<1024xi32, #blocked0>
  %95 = tt.addptr %35, %94 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %96 = tt.load %95, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %97 = tt.addptr %38, %94 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %98 = tt.load %97, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %99 = "triton_gpu.cmpi"(%96, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %100 = "triton_gpu.cmpi"(%90, %24) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %101 = arith.andi %54, %100 : tensor<1024xi1, #blocked0>
  %102 = arith.andi %101, %99 : tensor<1024xi1, #blocked0>
  %103 = arith.addf %89, %98 : tensor<1024xf32, #blocked0>
  %104 = "triton_gpu.select"(%102, %103, %89) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %105 = arith.addi %18, %cst_4 : tensor<1024xi32, #blocked0>
  %106 = "triton_gpu.cmpi"(%105, %25) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %107 = "triton_gpu.select"(%106, %105, %25) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %108 = arith.muli %107, %cst_7 : tensor<1024xi32, #blocked0>
  %109 = arith.addi %30, %108 : tensor<1024xi32, #blocked0>
  %110 = arith.addi %109, %33 : tensor<1024xi32, #blocked0>
  %111 = tt.addptr %35, %110 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %112 = tt.load %111, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %113 = tt.addptr %38, %110 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %114 = tt.load %113, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %115 = "triton_gpu.cmpi"(%112, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %116 = "triton_gpu.cmpi"(%105, %22) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %117 = "triton_gpu.cmpi"(%20, %24) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %118 = arith.andi %116, %117 : tensor<1024xi1, #blocked0>
  %119 = arith.andi %118, %115 : tensor<1024xi1, #blocked0>
  %120 = arith.addf %104, %114 : tensor<1024xf32, #blocked0>
  %121 = "triton_gpu.select"(%119, %120, %104) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %122 = arith.addi %46, %108 : tensor<1024xi32, #blocked0>
  %123 = arith.addi %122, %33 : tensor<1024xi32, #blocked0>
  %124 = tt.addptr %35, %123 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %125 = tt.load %124, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %126 = tt.addptr %38, %123 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %127 = tt.load %126, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %128 = "triton_gpu.cmpi"(%125, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %129 = arith.andi %116, %55 : tensor<1024xi1, #blocked0>
  %130 = arith.andi %129, %128 : tensor<1024xi1, #blocked0>
  %131 = arith.addf %121, %127 : tensor<1024xf32, #blocked0>
  %132 = "triton_gpu.select"(%130, %131, %121) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %133 = arith.addi %62, %108 : tensor<1024xi32, #blocked0>
  %134 = arith.addi %133, %33 : tensor<1024xi32, #blocked0>
  %135 = tt.addptr %35, %134 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %136 = tt.load %135, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %137 = tt.addptr %38, %134 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %138 = tt.load %137, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %139 = "triton_gpu.cmpi"(%136, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %140 = arith.andi %116, %70 : tensor<1024xi1, #blocked0>
  %141 = arith.andi %140, %139 : tensor<1024xi1, #blocked0>
  %142 = arith.addf %132, %138 : tensor<1024xf32, #blocked0>
  %143 = "triton_gpu.select"(%141, %142, %132) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %144 = arith.addi %77, %108 : tensor<1024xi32, #blocked0>
  %145 = arith.addi %144, %33 : tensor<1024xi32, #blocked0>
  %146 = tt.addptr %35, %145 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %147 = tt.load %146, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %148 = tt.addptr %38, %145 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %149 = tt.load %148, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %150 = "triton_gpu.cmpi"(%147, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %151 = arith.andi %116, %85 : tensor<1024xi1, #blocked0>
  %152 = arith.andi %151, %150 : tensor<1024xi1, #blocked0>
  %153 = arith.addf %143, %149 : tensor<1024xf32, #blocked0>
  %154 = "triton_gpu.select"(%152, %153, %143) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %155 = arith.addi %92, %108 : tensor<1024xi32, #blocked0>
  %156 = arith.addi %155, %33 : tensor<1024xi32, #blocked0>
  %157 = tt.addptr %35, %156 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %158 = tt.load %157, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %159 = tt.addptr %38, %156 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %160 = tt.load %159, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %161 = "triton_gpu.cmpi"(%158, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %162 = arith.andi %116, %100 : tensor<1024xi1, #blocked0>
  %163 = arith.andi %162, %161 : tensor<1024xi1, #blocked0>
  %164 = arith.addf %154, %160 : tensor<1024xf32, #blocked0>
  %165 = "triton_gpu.select"(%163, %164, %154) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %166 = arith.addi %18, %cst_5 : tensor<1024xi32, #blocked0>
  %167 = "triton_gpu.cmpi"(%166, %25) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %168 = "triton_gpu.select"(%167, %166, %25) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %169 = arith.muli %168, %cst_7 : tensor<1024xi32, #blocked0>
  %170 = arith.addi %30, %169 : tensor<1024xi32, #blocked0>
  %171 = arith.addi %170, %33 : tensor<1024xi32, #blocked0>
  %172 = tt.addptr %35, %171 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %173 = tt.load %172, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %174 = tt.addptr %38, %171 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %175 = tt.load %174, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %176 = "triton_gpu.cmpi"(%173, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %177 = "triton_gpu.cmpi"(%166, %22) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %178 = arith.andi %177, %117 : tensor<1024xi1, #blocked0>
  %179 = arith.andi %178, %176 : tensor<1024xi1, #blocked0>
  %180 = arith.addf %165, %175 : tensor<1024xf32, #blocked0>
  %181 = "triton_gpu.select"(%179, %180, %165) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %182 = arith.addi %46, %169 : tensor<1024xi32, #blocked0>
  %183 = arith.addi %182, %33 : tensor<1024xi32, #blocked0>
  %184 = tt.addptr %35, %183 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %185 = tt.load %184, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %186 = tt.addptr %38, %183 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %187 = tt.load %186, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %188 = "triton_gpu.cmpi"(%185, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %189 = arith.andi %177, %55 : tensor<1024xi1, #blocked0>
  %190 = arith.andi %189, %188 : tensor<1024xi1, #blocked0>
  %191 = arith.addf %181, %187 : tensor<1024xf32, #blocked0>
  %192 = "triton_gpu.select"(%190, %191, %181) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %193 = arith.addi %62, %169 : tensor<1024xi32, #blocked0>
  %194 = arith.addi %193, %33 : tensor<1024xi32, #blocked0>
  %195 = tt.addptr %35, %194 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %196 = tt.load %195, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %197 = tt.addptr %38, %194 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %198 = tt.load %197, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %199 = "triton_gpu.cmpi"(%196, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %200 = arith.andi %177, %70 : tensor<1024xi1, #blocked0>
  %201 = arith.andi %200, %199 : tensor<1024xi1, #blocked0>
  %202 = arith.addf %192, %198 : tensor<1024xf32, #blocked0>
  %203 = "triton_gpu.select"(%201, %202, %192) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %204 = arith.addi %77, %169 : tensor<1024xi32, #blocked0>
  %205 = arith.addi %204, %33 : tensor<1024xi32, #blocked0>
  %206 = tt.addptr %35, %205 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %207 = tt.load %206, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %208 = tt.addptr %38, %205 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %209 = tt.load %208, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %210 = "triton_gpu.cmpi"(%207, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %211 = arith.andi %177, %85 : tensor<1024xi1, #blocked0>
  %212 = arith.andi %211, %210 : tensor<1024xi1, #blocked0>
  %213 = arith.addf %203, %209 : tensor<1024xf32, #blocked0>
  %214 = "triton_gpu.select"(%212, %213, %203) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %215 = arith.addi %92, %169 : tensor<1024xi32, #blocked0>
  %216 = arith.addi %215, %33 : tensor<1024xi32, #blocked0>
  %217 = tt.addptr %35, %216 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %218 = tt.load %217, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %219 = tt.addptr %38, %216 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %220 = tt.load %219, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %221 = "triton_gpu.cmpi"(%218, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %222 = arith.andi %177, %100 : tensor<1024xi1, #blocked0>
  %223 = arith.andi %222, %221 : tensor<1024xi1, #blocked0>
  %224 = arith.addf %214, %220 : tensor<1024xf32, #blocked0>
  %225 = "triton_gpu.select"(%223, %224, %214) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %226 = arith.addi %18, %cst_6 : tensor<1024xi32, #blocked0>
  %227 = "triton_gpu.cmpi"(%226, %25) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %228 = "triton_gpu.select"(%227, %226, %25) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %229 = arith.muli %228, %cst_7 : tensor<1024xi32, #blocked0>
  %230 = arith.addi %30, %229 : tensor<1024xi32, #blocked0>
  %231 = arith.addi %230, %33 : tensor<1024xi32, #blocked0>
  %232 = tt.addptr %35, %231 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %233 = tt.load %232, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %234 = tt.addptr %38, %231 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %235 = tt.load %234, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %236 = "triton_gpu.cmpi"(%233, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %237 = "triton_gpu.cmpi"(%226, %22) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %238 = arith.andi %237, %117 : tensor<1024xi1, #blocked0>
  %239 = arith.andi %238, %236 : tensor<1024xi1, #blocked0>
  %240 = arith.addf %225, %235 : tensor<1024xf32, #blocked0>
  %241 = "triton_gpu.select"(%239, %240, %225) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %242 = arith.addi %46, %229 : tensor<1024xi32, #blocked0>
  %243 = arith.addi %242, %33 : tensor<1024xi32, #blocked0>
  %244 = tt.addptr %35, %243 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %245 = tt.load %244, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %246 = tt.addptr %38, %243 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %247 = tt.load %246, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %248 = "triton_gpu.cmpi"(%245, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %249 = arith.andi %237, %55 : tensor<1024xi1, #blocked0>
  %250 = arith.andi %249, %248 : tensor<1024xi1, #blocked0>
  %251 = arith.addf %241, %247 : tensor<1024xf32, #blocked0>
  %252 = "triton_gpu.select"(%250, %251, %241) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %253 = arith.addi %62, %229 : tensor<1024xi32, #blocked0>
  %254 = arith.addi %253, %33 : tensor<1024xi32, #blocked0>
  %255 = tt.addptr %35, %254 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %256 = tt.load %255, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %257 = tt.addptr %38, %254 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %258 = tt.load %257, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %259 = "triton_gpu.cmpi"(%256, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %260 = arith.andi %237, %70 : tensor<1024xi1, #blocked0>
  %261 = arith.andi %260, %259 : tensor<1024xi1, #blocked0>
  %262 = arith.addf %252, %258 : tensor<1024xf32, #blocked0>
  %263 = "triton_gpu.select"(%261, %262, %252) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %264 = arith.addi %77, %229 : tensor<1024xi32, #blocked0>
  %265 = arith.addi %264, %33 : tensor<1024xi32, #blocked0>
  %266 = tt.addptr %35, %265 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %267 = tt.load %266, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %268 = tt.addptr %38, %265 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %269 = tt.load %268, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %270 = "triton_gpu.cmpi"(%267, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %271 = arith.andi %237, %85 : tensor<1024xi1, #blocked0>
  %272 = arith.andi %271, %270 : tensor<1024xi1, #blocked0>
  %273 = arith.addf %263, %269 : tensor<1024xf32, #blocked0>
  %274 = "triton_gpu.select"(%272, %273, %263) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %275 = arith.addi %92, %229 : tensor<1024xi32, #blocked0>
  %276 = arith.addi %275, %33 : tensor<1024xi32, #blocked0>
  %277 = tt.addptr %35, %276 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %278 = tt.load %277, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %279 = tt.addptr %38, %276 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %280 = tt.load %279, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %281 = "triton_gpu.cmpi"(%278, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %282 = arith.andi %237, %100 : tensor<1024xi1, #blocked0>
  %283 = arith.andi %282, %281 : tensor<1024xi1, #blocked0>
  %284 = arith.addf %274, %280 : tensor<1024xf32, #blocked0>
  %285 = "triton_gpu.select"(%283, %284, %274) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %286 = arith.addi %18, %cst_7 : tensor<1024xi32, #blocked0>
  %287 = "triton_gpu.cmpi"(%286, %25) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %288 = "triton_gpu.select"(%287, %286, %25) : (tensor<1024xi1, #blocked0>, tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi32, #blocked0>
  %289 = arith.muli %288, %cst_7 : tensor<1024xi32, #blocked0>
  %290 = arith.addi %30, %289 : tensor<1024xi32, #blocked0>
  %291 = arith.addi %290, %33 : tensor<1024xi32, #blocked0>
  %292 = tt.addptr %35, %291 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %293 = tt.load %292, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %294 = tt.addptr %38, %291 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %295 = tt.load %294, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %296 = "triton_gpu.cmpi"(%293, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %297 = "triton_gpu.cmpi"(%286, %22) {predicate = 2 : i64} : (tensor<1024xi32, #blocked0>, tensor<1024xi32, #blocked0>) -> tensor<1024xi1, #blocked0>
  %298 = arith.andi %297, %117 : tensor<1024xi1, #blocked0>
  %299 = arith.andi %298, %296 : tensor<1024xi1, #blocked0>
  %300 = arith.addf %285, %295 : tensor<1024xf32, #blocked0>
  %301 = "triton_gpu.select"(%299, %300, %285) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %302 = arith.addi %46, %289 : tensor<1024xi32, #blocked0>
  %303 = arith.addi %302, %33 : tensor<1024xi32, #blocked0>
  %304 = tt.addptr %35, %303 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %305 = tt.load %304, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %306 = tt.addptr %38, %303 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %307 = tt.load %306, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %308 = "triton_gpu.cmpi"(%305, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %309 = arith.andi %297, %55 : tensor<1024xi1, #blocked0>
  %310 = arith.andi %309, %308 : tensor<1024xi1, #blocked0>
  %311 = arith.addf %301, %307 : tensor<1024xf32, #blocked0>
  %312 = "triton_gpu.select"(%310, %311, %301) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %313 = arith.addi %62, %289 : tensor<1024xi32, #blocked0>
  %314 = arith.addi %313, %33 : tensor<1024xi32, #blocked0>
  %315 = tt.addptr %35, %314 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %316 = tt.load %315, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %317 = tt.addptr %38, %314 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %318 = tt.load %317, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %319 = "triton_gpu.cmpi"(%316, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %320 = arith.andi %297, %70 : tensor<1024xi1, #blocked0>
  %321 = arith.andi %320, %319 : tensor<1024xi1, #blocked0>
  %322 = arith.addf %312, %318 : tensor<1024xf32, #blocked0>
  %323 = "triton_gpu.select"(%321, %322, %312) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %324 = arith.addi %77, %289 : tensor<1024xi32, #blocked0>
  %325 = arith.addi %324, %33 : tensor<1024xi32, #blocked0>
  %326 = tt.addptr %35, %325 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %327 = tt.load %326, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %328 = tt.addptr %38, %325 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %329 = tt.load %328, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %330 = "triton_gpu.cmpi"(%327, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %331 = arith.andi %297, %85 : tensor<1024xi1, #blocked0>
  %332 = arith.andi %331, %330 : tensor<1024xi1, #blocked0>
  %333 = arith.addf %323, %329 : tensor<1024xf32, #blocked0>
  %334 = "triton_gpu.select"(%332, %333, %323) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %335 = arith.addi %92, %289 : tensor<1024xi32, #blocked0>
  %336 = arith.addi %335, %33 : tensor<1024xi32, #blocked0>
  %337 = tt.addptr %35, %336 : tensor<1024x!tt.ptr<i64>, #blocked0>, tensor<1024xi32, #blocked0>
  %338 = tt.load %337, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xi64, #blocked0>
  %339 = tt.addptr %38, %336 : tensor<1024x!tt.ptr<f32>, #blocked0>, tensor<1024xi32, #blocked0>
  %340 = tt.load %339, %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked0>
  %341 = "triton_gpu.cmpi"(%338, %41) {predicate = 0 : i64} : (tensor<1024xi64, #blocked0>, tensor<1024xi64, #blocked0>) -> tensor<1024xi1, #blocked0>
  %342 = arith.andi %297, %100 : tensor<1024xi1, #blocked0>
  %343 = arith.andi %342, %341 : tensor<1024xi1, #blocked0>
  %344 = arith.addf %334, %340 : tensor<1024xf32, #blocked0>
  %345 = "triton_gpu.select"(%343, %344, %334) : (tensor<1024xi1, #blocked0>, tensor<1024xf32, #blocked0>, tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked0>
  %346 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>, #blocked1>
  %347 = arith.addi %5, %3 : tensor<1024xi32, #blocked1>
  %348 = tt.addptr %346, %347 : tensor<1024x!tt.ptr<f32>, #blocked1>, tensor<1024xi32, #blocked1>
  %349 = triton_gpu.convert_layout %345 : (tensor<1024xf32, #blocked0>) -> tensor<1024xf32, #blocked1>
  %350 = "triton_gpu.cmpi"(%347, %cst) {predicate = 2 : i64} : (tensor<1024xi32, #blocked1>, tensor<1024xi32, #blocked1>) -> tensor<1024xi1, #blocked1>
  tt.store %348, %349, %350 : tensor<1024xf32, #blocked1>
  return
}

// A mnist model from torch inductor.
// Check if topological sort is working correct and there's no unnecessary convert
// CHECK-LABEL: mnist
func public @mnist(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32) {
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
  return
}
