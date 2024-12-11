#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ortho_diag_test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<5.000000e-01> : tensor<1x64xf32, #blocked>
    %cst_0 = arith.constant dense<-7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_1 = arith.constant dense<-1.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_4 = arith.constant dense<7.52316385E-37> : tensor<1x64xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_5 = arith.constant dense<64> : tensor<64x1xi32, #blocked2>
    %cst_6 = arith.constant dense<7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_7 = arith.constant dense<0> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %7 = arith.muli %5, %cst_5 : tensor<64x1xi32, #blocked2>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked2>
    %9 = tt.addptr %8, %7 : tensor<64x1x!tt.ptr<f32>, #blocked2>, tensor<64x1xi32, #blocked2>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %11 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %13 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %14 = tt.broadcast %9 : tensor<64x1x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %15 = tt.broadcast %13 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
    %16 = tt.broadcast %12 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %17 = tt.addptr %14, %15 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %18 = tt.load %17 : tensor<64x64x!tt.ptr<f32>, #blocked2>
    %19 = ttg.convert_layout %18 : tensor<64x64xf32, #blocked2> -> tensor<64x64xf32, #blocked1>
    %20 = tt.broadcast %6 : tensor<64x1xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %21 = arith.cmpi eq, %20, %16 : tensor<64x64xi32, #blocked1>
    %22 = arith.uitofp %21 : tensor<64x64xi1, #blocked1> to tensor<64x64xf32, #blocked1>
    %23 = arith.xori %20, %16 : tensor<64x64xi32, #blocked1>
    %24 = arith.cmpi eq, %1, %cst_7 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %25:2 = scf.for %arg4 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg5 = %19, %arg6 = %22) -> (tensor<64x64xf32, #blocked1>, tensor<64x64xf32, #blocked1>)  : i32 {
      %36 = arith.mulf %arg5, %arg5 : tensor<64x64xf32, #blocked1>
      %37 = tt.gather %36[%23] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %38 = "tt.reduce"(%37) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %101 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %101 : f32
      }) : (tensor<64x64xf32, #blocked1>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %39 = arith.select %24, %cst_0, %38 : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %40:2 = "tt.reduce"(%39, %1) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: i32, %arg9: f32, %arg10: i32):
        %101 = arith.cmpf oeq, %arg7, %arg9 : f32
        %102 = arith.cmpi slt, %arg8, %arg10 : i32
        %103 = arith.andi %101, %102 : i1
        %104 = arith.cmpf ogt, %arg7, %arg9 : f32
        %105 = arith.ori %104, %103 : i1
        %106 = arith.select %105, %arg7, %arg9 : f32
        %107 = arith.select %105, %arg8, %arg10 : i32
        tt.reduce.return %106, %107 : f32, i32
      }) : (tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>) -> (f32, i32)
      %41 = tt.gather %arg5[%6] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>

      %42 = ttg.convert_layout %41 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked4>
      %43 = tt.reshape %42 : tensor<64x1xf32, #blocked4> -> tensor<1x64xf32, #blocked2>
      %44 = ttg.convert_layout %43 : tensor<1x64xf32, #blocked2> -> tensor<1x64xf32, #blocked>

      %45 = tt.splat %40#1 : i32 -> tensor<64x1xi32, #blocked1>
      %46 = arith.xori %6, %45 : tensor<64x1xi32, #blocked1>
      %47 = tt.gather %arg5[%46] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>

      %48 = ttg.convert_layout %47 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked4>
      %49 = tt.reshape %48 : tensor<64x1xf32, #blocked4> -> tensor<1x64xf32, #blocked2>
      %50 = ttg.convert_layout %49 : tensor<1x64xf32, #blocked2> -> tensor<1x64xf32, #blocked>

      %51 = tt.splat %40#1 : i32 -> tensor<1x64xi32, #blocked>
      %52 = tt.splat %40#1 : i32 -> tensor<1x64xi32, #blocked1>
      %53 = arith.xori %11, %51 : tensor<1x64xi32, #blocked>
      %54 = arith.xori %12, %52 : tensor<1x64xi32, #blocked1>
      %55 = tt.gather %44[%53] {axis = 1 : i32, efficient_layout} : (tensor<1x64xf32, #blocked>, tensor<1x64xi32, #blocked>) -> tensor<1x64xf32, #blocked>

      %56 = tt.splat %40#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %57 = arith.xori %0, %56 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %58 = arith.subi %0, %57 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %59 = arith.subf %44, %55 : tensor<1x64xf32, #blocked>
      %60 = arith.mulf %59, %cst : tensor<1x64xf32, #blocked>
      %61 = arith.sitofp %58 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %62 = arith.mulf %61, %cst_6 : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %63 = tt.expand_dims %62 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %64 = arith.addf %60, %63 : tensor<1x64xf32, #blocked>
      %65 = arith.cmpf olt, %64, %cst_3 : tensor<1x64xf32, #blocked>
      %66 = arith.select %65, %cst_1, %cst_2 : tensor<1x64xi1, #blocked>, tensor<1x64xf32, #blocked>
      %67 = arith.mulf %50, %50 : tensor<1x64xf32, #blocked>
      %68 = arith.addf %67, %cst_4 : tensor<1x64xf32, #blocked>
      %69 = arith.mulf %60, %60 : tensor<1x64xf32, #blocked>
      %70 = arith.addf %69, %68 : tensor<1x64xf32, #blocked>
      %71 = math.sqrt %70 : tensor<1x64xf32, #blocked>
      %72 = arith.mulf %66, %71 : tensor<1x64xf32, #blocked>
      %73 = arith.addf %64, %72 : tensor<1x64xf32, #blocked>
      %74 = arith.divf %50, %73 : tensor<1x64xf32, #blocked>
      %75 = arith.mulf %74, %74 : tensor<1x64xf32, #blocked>
      %76 = arith.addf %75, %cst_2 : tensor<1x64xf32, #blocked>
      %77 = math.rsqrt %76 : tensor<1x64xf32, #blocked>
      %78 = arith.mulf %74, %77 : tensor<1x64xf32, #blocked>
      %79 = tt.broadcast %54 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
      %80 = tt.gather %arg5[%79] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>

      %81 = ttg.convert_layout %77 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked1>
      %82 = tt.broadcast %81 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %85 = arith.mulf %82, %arg5 : tensor<64x64xf32, #blocked1>

      %86 = ttg.convert_layout %78 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked1>
      %87 = tt.broadcast %86 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %88 = arith.mulf %87, %80 : tensor<64x64xf32, #blocked1>
      %89 = arith.addf %85, %88 : tensor<64x64xf32, #blocked1>
      %90 = tt.trans %89 {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked5>
      %91 = ttg.convert_layout %90 : tensor<64x64xf32, #blocked5> -> tensor<64x64xf32, #blocked1>
      %92 = tt.gather %91[%79] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>

      %93 = arith.mulf %82, %91 : tensor<64x64xf32, #blocked1>

      %95 = arith.mulf %87, %92 : tensor<64x64xf32, #blocked1>
      %96 = arith.addf %93, %95 : tensor<64x64xf32, #blocked1>
      %97 = tt.gather %arg6[%79] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %98 = arith.mulf %82, %arg6 : tensor<64x64xf32, #blocked1>
      %99 = arith.mulf %87, %97 : tensor<64x64xf32, #blocked1>
      %100 = arith.addf %98, %99 : tensor<64x64xf32, #blocked1>
      scf.yield %96, %100 : tensor<64x64xf32, #blocked1>, tensor<64x64xf32, #blocked1>
    }
    %26 = tt.gather %25#0[%6] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>
    %27 = ttg.convert_layout %26 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked6>
    %28 = tt.reshape %27 : tensor<64x1xf32, #blocked6> -> tensor<64xf32, #blocked3>
    %29 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked2>
    %30 = tt.addptr %29, %7 : tensor<64x1x!tt.ptr<f32>, #blocked2>, tensor<64x1xi32, #blocked2>
    %31 = tt.broadcast %30 : tensor<64x1x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %32 = tt.addptr %31, %15 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %33 = ttg.convert_layout %25#1 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked2>
    tt.store %32, %33 : tensor<64x64x!tt.ptr<f32>, #blocked2>
    %34 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked3>
    %35 = tt.addptr %34, %2 : tensor<64x!tt.ptr<f32>, #blocked3>, tensor<64xi32, #blocked3>
    tt.store %35, %28 : tensor<64x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

