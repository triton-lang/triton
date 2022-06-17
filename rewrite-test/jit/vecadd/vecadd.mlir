module {
  func @add_kernel__Pfp32_Pfp32_Pfp32_i32_i32_i32__(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.broadcast %1 : (i32) -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.broadcast %arg3 : (i32) -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.broadcast %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %8 = tt.getelementptr %7, %4 : tensor<256x!tt.ptr<f32>>
    %9 = tt.broadcast %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %10 = tt.getelementptr %9, %4 : tensor<256x!tt.ptr<f32>>
    %cst = arith.constant 0.000000e+00 : f32
    %11 = tt.broadcast %cst : (f32) -> tensor<256xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %12 = arith.index_cast %c0_i32 : i32 to index
    %13 = arith.index_cast %arg4 : i32 to index
    %14 = arith.index_cast %c32_i32 : i32 to index
    %15:3 = scf.for %arg6 = %12 to %13 step %14 iter_args(%arg7 = %11, %arg8 = %8, %arg9 = %10) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>) {
      %cst_0 = arith.constant 0.000000e+00 : f32
      %18 = tt.broadcast %cst_0 : (f32) -> tensor<256xf32>
      %19 = tt.load %arg8, %6, %18 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
      %cst_1 = arith.constant 0.000000e+00 : f32
      %20 = tt.broadcast %cst_1 : (f32) -> tensor<256xf32>
      %21 = tt.load %arg9, %6, %20 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
      %22 = arith.addf %19, %21 : tensor<256xf32>
      %23 = arith.addf %arg7, %22 : tensor<256xf32>
      %24 = tt.broadcast %arg5 : (i32) -> tensor<256xi32>
      %25 = tt.getelementptr %arg8, %24 : tensor<256x!tt.ptr<f32>>
      %26 = tt.broadcast %arg5 : (i32) -> tensor<256xi32>
      %27 = tt.getelementptr %arg9, %26 : tensor<256x!tt.ptr<f32>>
      scf.yield %23, %25, %27 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>
    }
    %16 = tt.broadcast %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %17 = tt.getelementptr %16, %4 : tensor<256x!tt.ptr<f32>>
    tt.store %17, %15#0, %6, : tensor<256xf32>
    return
  }
}
module {
  func @add_kernel__Pfp32_Pfp32_Pfp32_i32_i32_i32__(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
    %3 = tt.broadcast %1 : (i32) -> tensor<256xi32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
    %4 = arith.addi %3, %2 : tensor<256xi32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
    %5 = tt.broadcast %arg3 : (i32) -> tensor<256xi32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
    %6 = "triton_gpu.cmpi"(%4, %5) {predicate = 2 : i64} : (tensor<256xi32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>, tensor<256xi32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>) -> tensor<256xi1, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
    %7 = tt.broadcast %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
    %8 = tt.getelementptr %7, %4 : tensor<256x!tt.ptr<f32>, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
    %9 = tt.broadcast %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
    %10 = tt.getelementptr %9, %4 : tensor<256x!tt.ptr<f32>, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
    %11 = arith.index_cast %arg4 : i32 to index
    %12:3 = scf.for %arg6 = %c0 to %11 step %c32 iter_args(%arg7 = %cst, %arg8 = %8, %arg9 = %10) -> (tensor<256xf32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>, tensor<256x!tt.ptr<f32>, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>, tensor<256x!tt.ptr<f32>, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>) {
      %15 = tt.load %arg8, %6, %cst {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
      %16 = tt.load %arg9, %6, %cst {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
      %17 = arith.addf %15, %16 : tensor<256xf32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
      %18 = arith.addf %arg7, %17 : tensor<256xf32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
      %19 = tt.broadcast %arg5 : (i32) -> tensor<256xi32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
      %20 = tt.getelementptr %arg8, %19 : tensor<256x!tt.ptr<f32>, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
      %21 = tt.getelementptr %arg9, %19 : tensor<256x!tt.ptr<f32>, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
      scf.yield %18, %20, %21 : tensor<256xf32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>, tensor<256x!tt.ptr<f32>, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>, tensor<256x!tt.ptr<f32>, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
    }
    %13 = tt.broadcast %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
    %14 = tt.getelementptr %13, %4 : tensor<256x!tt.ptr<f32>, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
    tt.store %14, %12#0, %6, : tensor<256xf32, #triton_gpu.blocked_layout<{threadTileSize = [1], warpTileSize = [32], blockTileSize = [128], order = [0]}>>
    return
  }
}
