module {
  tt.func public @_attn_fwd_tma(%arg0: f32, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg5: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg6: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg7: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<64xf32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<64xf32>
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant dense<-1.000000e+06> : tensor<64x64xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %c0_i32 = arith.constant 0 : i32
    %cst_3 = arith.constant 1.44269502 : f32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %arg3 : i32
    %3 = arith.remsi %1, %arg3 : i32
    %4 = arith.muli %3, %arg8 : i32
    %5 = arith.addi %2, %4 : i32
    %6 = arith.muli %0, %c64_i32 : i32
    %7 = arith.addi %5, %6 : i32
    %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %9 = tt.splat %6 : i32 -> tensor<64xi32>
    %10 = arith.addi %9, %8 : tensor<64xi32>
    %11 = arith.mulf %arg0, %cst_3 : f32
    %12 = tt.reinterpret_tensor_descriptor %arg4 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<64x64xf16>>
    %13 = tt.descriptor_load %12[%7, %c0_i32] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
    %14:4 = scf.for %arg9 = %c0_i32 to %6 step %c64_i32 iter_args(%arg10 = %cst, %arg11 = %cst_2, %arg12 = %cst_0, %arg13 = %5) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, i32)  : i32 {
      %31 = tt.reinterpret_tensor_descriptor %arg5 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<64x64xf16>>
      %32 = tt.descriptor_load %31[%arg13, %c0_i32] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
      %33 = tt.trans %32 {order = array<i32: 1, 0>} : tensor<64x64xf16> -> tensor<64x64xf16>
      %34 = tt.dot %13, %33, %cst_2, inputPrecision = tf32 : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
      %35 = "tt.reduce"(%34) <{axis = 1 : i32}> ({
      ^bb0(%arg14: f32, %arg15: f32):
        %58 = arith.maxnumf %arg14, %arg15 : f32
        tt.reduce.return %58 : f32
      }) : (tensor<64x64xf32>) -> tensor<64xf32>
      %36 = tt.splat %11 : f32 -> tensor<64xf32>
      %37 = arith.mulf %35, %36 : tensor<64xf32>
      %38 = arith.maxnumf %arg12, %37 : tensor<64xf32>
      %39 = tt.splat %11 : f32 -> tensor<64x64xf32>
      %40 = arith.mulf %34, %39 : tensor<64x64xf32>
      %41 = tt.expand_dims %38 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %42 = tt.broadcast %41 : tensor<64x1xf32> -> tensor<64x64xf32>
      %43 = arith.subf %40, %42 : tensor<64x64xf32>
      %44 = math.exp2 %43 : tensor<64x64xf32>
      %45 = "tt.reduce"(%44) <{axis = 1 : i32}> ({
      ^bb0(%arg14: f32, %arg15: f32):
        %58 = arith.addf %arg14, %arg15 : f32
        tt.reduce.return %58 : f32
      }) : (tensor<64x64xf32>) -> tensor<64xf32>
      %46 = arith.subf %arg12, %38 : tensor<64xf32>
      %47 = math.exp2 %46 : tensor<64xf32>
      %48 = arith.mulf %arg10, %47 : tensor<64xf32>
      %49 = arith.addf %48, %45 : tensor<64xf32>
      %50 = tt.expand_dims %47 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %51 = tt.broadcast %50 : tensor<64x1xf32> -> tensor<64x64xf32>
      %52 = arith.mulf %arg11, %51 : tensor<64x64xf32>
      %53 = tt.reinterpret_tensor_descriptor %arg6 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<64x64xf16>>
      %54 = tt.descriptor_load %53[%arg13, %c0_i32] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
      %55 = arith.truncf %44 : tensor<64x64xf32> to tensor<64x64xf16>
      %56 = tt.dot %55, %54, %52, inputPrecision = tf32 : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
      %57 = arith.addi %arg13, %c64_i32 : i32
      scf.yield %49, %56, %38, %57 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, i32
    } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
    %15 = arith.muli %0, %c64_i32 {tt.divisibility = dense<64> : tensor<1xi32>} : i32
    %16 = arith.addi %0, %c1_i32 : i32
    %17 = arith.muli %16, %c64_i32 : i32
    %18 = arith.addi %5, %15 : i32
    %19:4 = scf.for %arg9 = %15 to %17 step %c64_i32 iter_args(%arg10 = %14#0, %arg11 = %14#1, %arg12 = %14#2, %arg13 = %18) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, i32)  : i32 {
      %31 = tt.reinterpret_tensor_descriptor %arg5 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<64x64xf16>>
      %32 = tt.descriptor_load %31[%arg13, %c0_i32] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
      %33 = tt.trans %32 {order = array<i32: 1, 0>} : tensor<64x64xf16> -> tensor<64x64xf16>
      %34 = tt.dot %13, %33, %cst_2, inputPrecision = tf32 : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
      %35 = tt.expand_dims %10 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
      %36 = tt.expand_dims %8 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %37 = tt.splat %arg9 : i32 -> tensor<1x64xi32>
      %38 = arith.addi %37, %36 : tensor<1x64xi32>
      %39 = tt.broadcast %35 : tensor<64x1xi32> -> tensor<64x64xi32>
      %40 = tt.broadcast %38 : tensor<1x64xi32> -> tensor<64x64xi32>
      %41 = arith.cmpi sge, %39, %40 : tensor<64x64xi32>
      %42 = tt.splat %11 : f32 -> tensor<64x64xf32>
      %43 = arith.mulf %34, %42 : tensor<64x64xf32>
      %44 = arith.select %41, %cst_2, %cst_1 : tensor<64x64xi1>, tensor<64x64xf32>
      %45 = arith.addf %43, %44 : tensor<64x64xf32>
      %46 = "tt.reduce"(%45) <{axis = 1 : i32}> ({
      ^bb0(%arg14: f32, %arg15: f32):
        %65 = arith.maxnumf %arg14, %arg15 : f32
        tt.reduce.return %65 : f32
      }) : (tensor<64x64xf32>) -> tensor<64xf32>
      %47 = arith.maxnumf %arg12, %46 : tensor<64xf32>
      %48 = tt.expand_dims %47 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %49 = tt.broadcast %48 : tensor<64x1xf32> -> tensor<64x64xf32>
      %50 = arith.subf %45, %49 : tensor<64x64xf32>
      %51 = math.exp2 %50 : tensor<64x64xf32>
      %52 = "tt.reduce"(%51) <{axis = 1 : i32}> ({
      ^bb0(%arg14: f32, %arg15: f32):
        %65 = arith.addf %arg14, %arg15 : f32
        tt.reduce.return %65 : f32
      }) : (tensor<64x64xf32>) -> tensor<64xf32>
      %53 = arith.subf %arg12, %47 : tensor<64xf32>
      %54 = math.exp2 %53 : tensor<64xf32>
      %55 = arith.mulf %arg10, %54 : tensor<64xf32>
      %56 = arith.addf %55, %52 : tensor<64xf32>
      %57 = tt.expand_dims %54 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
      %58 = tt.broadcast %57 : tensor<64x1xf32> -> tensor<64x64xf32>
      %59 = arith.mulf %arg11, %58 : tensor<64x64xf32>
      %60 = tt.reinterpret_tensor_descriptor %arg6 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<64x64xf16>>
      %61 = tt.descriptor_load %60[%arg13, %c0_i32] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
      %62 = arith.truncf %51 : tensor<64x64xf32> to tensor<64x64xf16>
      %63 = tt.dot %62, %61, %59, inputPrecision = tf32 : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
      %64 = arith.addi %arg13, %c64_i32 : i32
      scf.yield %56, %63, %47, %64 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, i32
    } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
    %20 = math.log2 %19#0 : tensor<64xf32>
    %21 = arith.addf %19#2, %20 : tensor<64xf32>
    %22 = tt.expand_dims %19#0 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32>
    %23 = tt.broadcast %22 : tensor<64x1xf32> -> tensor<64x64xf32>
    %24 = arith.divf %19#1, %23 : tensor<64x64xf32>
    %25 = arith.muli %1, %arg8 : i32
    %26 = tt.addptr %arg1, %25 : !tt.ptr<f32>, i32
    %27 = tt.splat %26 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %28 = tt.addptr %27, %10 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %28, %21 : tensor<64x!tt.ptr<f32>>
    %29 = arith.truncf %24 : tensor<64x64xf32> to tensor<64x64xf16>
    %30 = tt.reinterpret_tensor_descriptor %arg7 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<64x64xf16>>
    tt.descriptor_store %30[%7, %c0_i32], %29 : !tt.tensordesc<tensor<64x64xf16>>, tensor<64x64xf16>
    tt.return
  }
}

