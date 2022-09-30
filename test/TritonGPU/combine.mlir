// RUN: triton-opt %s -tritongpu-combine 2>&1 | FileCheck %s

#layout0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#layout1 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// CHECK: [[target_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK: [[row_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK: [[col_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK: [[col_layout_novec:#.*]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>

// CHECK-LABEL: loop
func @loop(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) {
    // CHECK-NOT: triton_gpu.convert_layout
    // CHECK: [[loop_ret:%.*]]:2 = scf.for {{.*}} -> (tensor<64x64xf32, [[row_layout]]>, tensor<64x64x!tt.ptr<f32>, [[row_layout]]>)
    // CHECK-NEXT: {{.*}} = tt.load {{.*}} : tensor<64x64xf32, [[row_layout]]>
    // CHECK-NEXT: {{.*}} = arith.addf {{.*}} : tensor<64x64xf32, [[row_layout]]>
    // CHECK-NEXT: {{.*}} = tt.addptr {{.*}} : tensor<64x64x!tt.ptr<f32>, [[row_layout]]>
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
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked0>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<64xi32, #blocked0>) -> tensor<64x1xi32, #blocked1>
    %2 = tt.splat %arg1 : (i32) -> tensor<64x1xi32, #blocked1>
    %3 = arith.muli %1, %2 : tensor<64x1xi32, #blocked1>
    %4 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x1x!tt.ptr<f32>, #blocked1>
    %5 = tt.addptr %4, %3 : tensor<64x1x!tt.ptr<f32>, #blocked1>
    %6 = tt.expand_dims %0 {axis = 0 : i32} : (tensor<64xi32, #blocked0>) -> tensor<1x64xi32, #blocked2>
    %7 = tt.broadcast %5 : (tensor<64x1x!tt.ptr<f32>, #blocked1>) -> tensor<64x64x!tt.ptr<f32>, #blocked1>
    %8 = tt.broadcast %6 : (tensor<1x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
    %9 = triton_gpu.convert_layout %8 : (tensor<64x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked1>
    %10 = tt.addptr %7, %9 : tensor<64x64x!tt.ptr<f32>, #blocked1>
    %11:2 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %cst_1, %arg7 = %10) -> (tensor<64x64xf32, #blocked1>, tensor<64x64x!tt.ptr<f32>, #blocked1>) {
      %23 = triton_gpu.convert_layout %arg7 : (tensor<64x64x!tt.ptr<f32>, #blocked1>) -> tensor<64x64x!tt.ptr<f32>, #blocked3>
      %24 = triton_gpu.convert_layout %cst : (tensor<64x64xi1, #blocked1>) -> tensor<64x64xi1, #blocked3>
      %25 = triton_gpu.convert_layout %cst_1 : (tensor<64x64xf32, #blocked1>) -> tensor<64x64xf32, #blocked3>
      %26 = tt.load %23, %24, %25 {cache = 1 : i32, evict = 1 : i32, isOtherUnspecified = false, isVolatile = false} : tensor<64x64xf32, #blocked3>
      %27 = triton_gpu.convert_layout %26 : (tensor<64x64xf32, #blocked3>) -> tensor<64x64xf32, #blocked1>
      %28 = arith.addf %arg6, %27 : tensor<64x64xf32, #blocked1>
      %29 = tt.addptr %arg7, %cst_0 : tensor<64x64x!tt.ptr<f32>, #blocked1>
      scf.yield %28, %29 : tensor<64x64xf32, #blocked1>, tensor<64x64x!tt.ptr<f32>, #blocked1>
    }
    %12 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x1x!tt.ptr<f32>, #blocked1>
    %13 = tt.addptr %12, %1 : tensor<64x1x!tt.ptr<f32>, #blocked1>
    %14 = tt.splat %arg3 : (i32) -> tensor<1x64xi32, #blocked2>
    %15 = arith.muli %6, %14 : tensor<1x64xi32, #blocked2>
    %16 = tt.broadcast %13 : (tensor<64x1x!tt.ptr<f32>, #blocked1>) -> tensor<64x64x!tt.ptr<f32>, #blocked1>
    %17 = tt.broadcast %15 : (tensor<1x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
    %18 = triton_gpu.convert_layout %17 : (tensor<64x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked1>
    %19 = tt.addptr %16, %18 : tensor<64x64x!tt.ptr<f32>, #blocked1>
    %20 = triton_gpu.convert_layout %19 : (tensor<64x64x!tt.ptr<f32>, #blocked1>) -> tensor<64x64x!tt.ptr<f32>, #blocked1>
    %21 = triton_gpu.convert_layout %11#0 : (tensor<64x64xf32, #blocked1>) -> tensor<64x64xf32, #blocked1>
    %22 = triton_gpu.convert_layout %cst : (tensor<64x64xi1, #blocked1>) -> tensor<64x64xi1, #blocked1>
    tt.store %20, %21, %22 : tensor<64x64xf32, #blocked1>
    return
}
