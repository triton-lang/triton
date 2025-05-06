module {
  tt.func public @attention_inner_loop_kernel_data_part(%arg0: !tt.tensordesc<tensor<128x128xf16>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<128x128xf16>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<128x128xf16>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<tensor<128x128xf16>>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg21: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: f32) attributes {noinline = false} {
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant dense<128> : tensor<128xi32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.descriptor_load %arg0[%1, %c0_i32] : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16>
    %3 = arith.addi %1, %c128_i32 : i32
    %4 = tt.descriptor_load %arg0[%3, %c0_i32] : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16>
    %5:6 = scf.for %arg25 = %c0_i32 to %arg23 step %c128_i32 iter_args(%arg26 = %cst_0, %arg27 = %cst_0, %arg28 = %cst_2, %arg29 = %cst_1, %arg30 = %cst_2, %arg31 = %cst_1) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>)  : i32 {
      %21 = tt.descriptor_load %arg5[%arg25, %c0_i32] : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16>
      %22 = tt.trans %21 {order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
      %23 = tt.descriptor_load %arg10[%arg25, %c0_i32] : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16>
      %24 = tt.dot %2, %22, %cst_0, inputPrecision = tf32 : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
      %25 = "tt.reduce"(%24) <{axis = 1 : i32}> ({
      ^bb0(%arg32: f32, %arg33: f32):
        %64 = arith.maxnumf %arg32, %arg33 : f32
        tt.reduce.return %64 : f32
      }) : (tensor<128x128xf32>) -> tensor<128xf32>
      %26 = tt.splat %arg24 : f32 -> tensor<128xf32>
      %27 = arith.mulf %25, %26 : tensor<128xf32>
      %28 = arith.maxnumf %arg29, %27 : tensor<128xf32>
      %29 = tt.splat %arg24 : f32 -> tensor<128x128xf32>
      %30 = arith.mulf %24, %29 : tensor<128x128xf32>
      %31 = tt.expand_dims %28 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %32 = tt.broadcast %31 : tensor<128x1xf32> -> tensor<128x128xf32>
      %33 = arith.subf %30, %32 : tensor<128x128xf32>
      %34 = math.exp2 %33 : tensor<128x128xf32>
      %35 = arith.subf %arg29, %28 : tensor<128xf32>
      %36 = math.exp2 %35 : tensor<128xf32>
      %37 = "tt.reduce"(%34) <{axis = 1 : i32}> ({
      ^bb0(%arg32: f32, %arg33: f32):
        %64 = arith.addf %arg32, %arg33 : f32
        tt.reduce.return %64 : f32
      }) : (tensor<128x128xf32>) -> tensor<128xf32>
      %38 = tt.expand_dims %36 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %39 = tt.broadcast %38 : tensor<128x1xf32> -> tensor<128x128xf32>
      %40 = arith.mulf %arg26, %39 : tensor<128x128xf32>
      %41 = tt.dot %4, %22, %cst_0, inputPrecision = tf32 : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
      %42 = "tt.reduce"(%41) <{axis = 1 : i32}> ({
      ^bb0(%arg32: f32, %arg33: f32):
        %64 = arith.maxnumf %arg32, %arg33 : f32
        tt.reduce.return %64 : f32
      }) : (tensor<128x128xf32>) -> tensor<128xf32>
      %43 = arith.mulf %42, %26 : tensor<128xf32>
      %44 = arith.maxnumf %arg31, %43 : tensor<128xf32>
      %45 = arith.mulf %41, %29 : tensor<128x128xf32>
      %46 = tt.expand_dims %44 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %47 = tt.broadcast %46 : tensor<128x1xf32> -> tensor<128x128xf32>
      %48 = arith.subf %45, %47 : tensor<128x128xf32>
      %49 = math.exp2 %48 : tensor<128x128xf32>
      %50 = arith.subf %arg31, %44 : tensor<128xf32>
      %51 = math.exp2 %50 : tensor<128xf32>
      %52 = "tt.reduce"(%49) <{axis = 1 : i32}> ({
      ^bb0(%arg32: f32, %arg33: f32):
        %64 = arith.addf %arg32, %arg33 : f32
        tt.reduce.return %64 : f32
      }) : (tensor<128x128xf32>) -> tensor<128xf32>
      %53 = tt.expand_dims %51 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
      %54 = tt.broadcast %53 : tensor<128x1xf32> -> tensor<128x128xf32>
      %55 = arith.mulf %arg27, %54 : tensor<128x128xf32>
      %56 = arith.truncf %34 : tensor<128x128xf32> to tensor<128x128xf16>
      %57 = tt.dot %56, %23, %40, inputPrecision = tf32 : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
      %58 = arith.mulf %arg28, %36 : tensor<128xf32>
      %59 = arith.addf %58, %37 : tensor<128xf32>
      %60 = arith.truncf %49 : tensor<128x128xf32> to tensor<128x128xf16>
      %61 = tt.dot %60, %23, %55, inputPrecision = tf32 : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
      %62 = arith.mulf %arg30, %51 : tensor<128xf32>
      %63 = arith.addf %62, %52 : tensor<128xf32>
      scf.yield %57, %61, %59, %28, %63, %44 : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>
    } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
    %6 = arith.truncf %5#0 : tensor<128x128xf32> to tensor<128x128xf16>
    tt.descriptor_store %arg15[%1, %c0_i32], %6 : !tt.tensordesc<tensor<128x128xf16>>, tensor<128x128xf16>
    %7 = tt.addptr %arg20, %1 : !tt.ptr<f16>, i32
    %8 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %9 = tt.splat %7 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>>
    %10 = tt.addptr %9, %8 : tensor<128x!tt.ptr<f16>>, tensor<128xi32>
    %11 = arith.truncf %5#2 : tensor<128xf32> to tensor<128xf16>
    tt.store %10, %11 : tensor<128x!tt.ptr<f16>>
    %12 = tt.addptr %arg21, %1 : !tt.ptr<f16>, i32
    %13 = tt.splat %12 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>>
    %14 = tt.addptr %13, %8 : tensor<128x!tt.ptr<f16>>, tensor<128xi32>
    %15 = arith.truncf %5#3 : tensor<128xf32> to tensor<128xf16>
    tt.store %14, %15 : tensor<128x!tt.ptr<f16>>
    %16 = arith.truncf %5#1 : tensor<128x128xf32> to tensor<128x128xf16>
    tt.descriptor_store %arg15[%3, %c0_i32], %16 : !tt.tensordesc<tensor<128x128xf16>>, tensor<128x128xf16>
    %17 = tt.addptr %10, %cst : tensor<128x!tt.ptr<f16>>, tensor<128xi32>
    %18 = arith.truncf %5#4 : tensor<128xf32> to tensor<128xf16>
    tt.store %17, %18 : tensor<128x!tt.ptr<f16>>
    %19 = tt.addptr %14, %cst : tensor<128x!tt.ptr<f16>>, tensor<128xi32>
    %20 = arith.truncf %5#5 : tensor<128xf32> to tensor<128xf16>
    tt.store %19, %20 : tensor<128x!tt.ptr<f16>>
    tt.return
  }
}