#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ortho_diag_test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<5.000000e-01> : tensor<1x64xf32, #blocked>
    %cst_0 = arith.constant dense<-7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_1 = arith.constant dense<-1.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_3 = arith.constant dense<7.52316385E-37> : tensor<1x64xf32, #blocked2>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_5 = arith.constant dense<64> : tensor<64x1xi32, #blocked3>
    %cst_6 = arith.constant dense<7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_7 = arith.constant dense<0> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked4>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xi32, #blocked3>
    %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %7 = arith.muli %5, %cst_5 : tensor<64x1xi32, #blocked3>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked3>
    %9 = tt.addptr %8, %7 : tensor<64x1x!tt.ptr<f32>, #blocked3>, tensor<64x1xi32, #blocked3>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %11 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %12 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %13 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %14 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %15 = tt.expand_dims %11 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x64xi32, #blocked3>
    %16 = tt.broadcast %9 : tensor<64x1x!tt.ptr<f32>, #blocked3> -> tensor<64x64x!tt.ptr<f32>, #blocked3>
    %17 = tt.broadcast %15 : tensor<1x64xi32, #blocked3> -> tensor<64x64xi32, #blocked3>
    %18 = tt.broadcast %14 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %19 = tt.addptr %16, %17 : tensor<64x64x!tt.ptr<f32>, #blocked3>, tensor<64x64xi32, #blocked3>
    %20 = tt.load %19 : tensor<64x64x!tt.ptr<f32>, #blocked3>
    %21 = ttg.convert_layout %20 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked1>
    %22 = tt.broadcast %6 : tensor<64x1xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %23 = arith.cmpi eq, %22, %18 : tensor<64x64xi32, #blocked1>
    %24 = arith.uitofp %23 : tensor<64x64xi1, #blocked1> to tensor<64x64xf32, #blocked1>
    %25 = arith.xori %22, %18 : tensor<64x64xi32, #blocked1>
    %26 = arith.cmpi eq, %1, %cst_7 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %27:2 = scf.for %arg4 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg5 = %21, %arg6 = %24) -> (tensor<64x64xf32, #blocked1>, tensor<64x64xf32, #blocked1>)  : i32 {
      %38 = arith.mulf %arg5, %arg5 : tensor<64x64xf32, #blocked1>
      %39 = tt.gather %38[%25] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %40 = "tt.reduce"(%39) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %98 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %98 : f32
      }) : (tensor<64x64xf32, #blocked1>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %41 = arith.select %26, %cst_0, %40 : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %42:2 = "tt.reduce"(%41, %1) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: i32, %arg9: f32, %arg10: i32):
        %98 = arith.cmpf oeq, %arg7, %arg9 : f32
        %99 = arith.cmpi slt, %arg8, %arg10 : i32
        %100 = arith.andi %98, %99 : i1
        %101 = arith.cmpf ogt, %arg7, %arg9 : f32
        %102 = arith.ori %101, %100 : i1
        %103 = arith.select %102, %arg7, %arg9 : f32
        %104 = arith.select %102, %arg8, %arg10 : i32
        tt.reduce.return %103, %104 : f32, i32
      }) : (tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>) -> (f32, i32)
      %43 = ttg.convert_layout %arg5 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked2>
      %44 = tt.gather %43[%12] {axis = 0 : i32, efficient_layout} : (tensor<64x64xf32, #blocked2>, tensor<1x64xi32, #blocked2>) -> tensor<1x64xf32, #blocked2>
      %45 = ttg.convert_layout %44 : tensor<1x64xf32, #blocked2> -> tensor<1x64xf32, #blocked>
      %46 = tt.splat %42#1 : i32 -> tensor<1x64xi32, #blocked2>
      %47 = tt.splat %42#1 : i32 -> tensor<1x64xi32, #blocked>
      %48 = tt.splat %42#1 : i32 -> tensor<1x64xi32, #blocked1>
      %49 = arith.xori %12, %46 : tensor<1x64xi32, #blocked2>
      %50 = arith.xori %13, %47 : tensor<1x64xi32, #blocked>
      %51 = arith.xori %14, %48 : tensor<1x64xi32, #blocked1>
      %52 = tt.gather %43[%49] {axis = 0 : i32, efficient_layout} : (tensor<64x64xf32, #blocked2>, tensor<1x64xi32, #blocked2>) -> tensor<1x64xf32, #blocked2>
      %53 = ttg.convert_layout %52 : tensor<1x64xf32, #blocked2> -> tensor<1x64xf32, #blocked>
      %54 = tt.gather %45[%50] {axis = 1 : i32, efficient_layout} : (tensor<1x64xf32, #blocked>, tensor<1x64xi32, #blocked>) -> tensor<1x64xf32, #blocked>
      %55 = tt.splat %42#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %56 = arith.xori %0, %55 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %57 = arith.subi %0, %56 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %58 = arith.subf %45, %54 : tensor<1x64xf32, #blocked>
      %59 = arith.mulf %58, %cst : tensor<1x64xf32, #blocked>
      %60 = arith.sitofp %57 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %61 = arith.mulf %60, %cst_6 : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %62 = tt.expand_dims %61 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %63 = arith.addf %59, %62 : tensor<1x64xf32, #blocked>
      %64 = arith.cmpf olt, %63, %cst_4 : tensor<1x64xf32, #blocked>
      %65 = arith.select %64, %cst_1, %cst_2 : tensor<1x64xi1, #blocked>, tensor<1x64xf32, #blocked>
      %66 = arith.mulf %52, %52 : tensor<1x64xf32, #blocked2>
      %67 = arith.addf %66, %cst_3 : tensor<1x64xf32, #blocked2>
      %68 = ttg.convert_layout %67 : tensor<1x64xf32, #blocked2> -> tensor<1x64xf32, #blocked>
      %69 = arith.mulf %59, %59 : tensor<1x64xf32, #blocked>
      %70 = arith.addf %69, %68 : tensor<1x64xf32, #blocked>
      %71 = math.sqrt %70 : tensor<1x64xf32, #blocked>
      %72 = arith.mulf %65, %71 : tensor<1x64xf32, #blocked>
      %73 = arith.addf %63, %72 : tensor<1x64xf32, #blocked>
      %74 = arith.divf %53, %73 : tensor<1x64xf32, #blocked>
      %75 = arith.mulf %74, %74 : tensor<1x64xf32, #blocked>
      %76 = arith.addf %75, %cst_2 : tensor<1x64xf32, #blocked>
      %77 = math.rsqrt %76 : tensor<1x64xf32, #blocked>
      %78 = arith.mulf %74, %77 : tensor<1x64xf32, #blocked>
      %79 = tt.broadcast %51 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
      %80 = tt.gather %arg5[%79] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %81 = tt.broadcast %77 : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %82 = ttg.convert_layout %81 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
      %83 = arith.mulf %82, %arg5 : tensor<64x64xf32, #blocked1>
      %84 = tt.broadcast %78 : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %85 = ttg.convert_layout %84 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
      %86 = arith.mulf %85, %80 : tensor<64x64xf32, #blocked1>
      %87 = arith.addf %83, %86 : tensor<64x64xf32, #blocked1>
      %88 = tt.trans %87 {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked2>
      %89 = ttg.convert_layout %88 : tensor<64x64xf32, #blocked2> -> tensor<64x64xf32, #blocked1>
      %90 = tt.gather %89[%79] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %91 = arith.mulf %82, %89 : tensor<64x64xf32, #blocked1>
      %92 = arith.mulf %85, %90 : tensor<64x64xf32, #blocked1>
      %93 = arith.addf %91, %92 : tensor<64x64xf32, #blocked1>
      %94 = tt.gather %arg6[%79] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %95 = arith.mulf %82, %arg6 : tensor<64x64xf32, #blocked1>
      %96 = arith.mulf %85, %94 : tensor<64x64xf32, #blocked1>
      %97 = arith.addf %95, %96 : tensor<64x64xf32, #blocked1>
      scf.yield %93, %97 : tensor<64x64xf32, #blocked1>, tensor<64x64xf32, #blocked1>
    }
    %28 = tt.gather %27#0[%6] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>
    %29 = ttg.convert_layout %28 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked5>
    %30 = tt.reshape %29 : tensor<64x1xf32, #blocked5> -> tensor<64xf32, #blocked4>
    %31 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked3>
    %32 = tt.addptr %31, %7 : tensor<64x1x!tt.ptr<f32>, #blocked3>, tensor<64x1xi32, #blocked3>
    %33 = tt.broadcast %32 : tensor<64x1x!tt.ptr<f32>, #blocked3> -> tensor<64x64x!tt.ptr<f32>, #blocked3>
    %34 = tt.addptr %33, %17 : tensor<64x64x!tt.ptr<f32>, #blocked3>, tensor<64x64xi32, #blocked3>
    %35 = ttg.convert_layout %27#1 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked3>
    tt.store %34, %35 : tensor<64x64x!tt.ptr<f32>, #blocked3>
    %36 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked4>
    %37 = tt.addptr %36, %2 : tensor<64x!tt.ptr<f32>, #blocked4>, tensor<64xi32, #blocked4>
    tt.store %37, %30 : tensor<64x!tt.ptr<f32>, #blocked4>
    tt.return
  }
}

