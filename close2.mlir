#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ortho_diag_test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<5.000000e-01> : tensor<1x64xf32, #blocked>
    %cst_0 = arith.constant dense<-7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_1 = arith.constant dense<-1.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_4 = arith.constant dense<64> : tensor<64x1xi32, #blocked2>
    %cst_5 = arith.constant dense<7.52316385E-37> : tensor<1x64xf32, #blocked>
    %cst_6 = arith.constant dense<7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_7 = arith.constant dense<0> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %7 = arith.muli %5, %cst_4 : tensor<64x1xi32, #blocked2>
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
        %126 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %126 : f32
      }) : (tensor<64x64xf32, #blocked1>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %39 = arith.select %24, %cst_0, %38 : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %40:2 = "tt.reduce"(%39, %1) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: i32, %arg9: f32, %arg10: i32):
        %126 = arith.cmpf oeq, %arg7, %arg9 : f32
        %127 = arith.cmpi slt, %arg8, %arg10 : i32
        %128 = arith.andi %126, %127 : i1
        %129 = arith.cmpf ogt, %arg7, %arg9 : f32
        %130 = arith.ori %129, %128 : i1
        %131 = arith.select %130, %arg7, %arg9 : f32
        %132 = arith.select %130, %arg8, %arg10 : i32
        tt.reduce.return %131, %132 : f32, i32
      }) : (tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>) -> (f32, i32)
      %41 = tt.gather %arg5[%6] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>

      %42 = ttg.convert_layout %41 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked4>
      %43 = ttg.convert_layout %41 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked5>
      %44 = tt.reshape %42 : tensor<64x1xf32, #blocked4> -> tensor<1x64xf32, #blocked2>
      %45 = tt.reshape %43 : tensor<64x1xf32, #blocked5> -> tensor<1x64xf32, #blocked6>
      %46 = ttg.convert_layout %44 : tensor<1x64xf32, #blocked2> -> tensor<1x64xf32, #blocked>
      %47 = ttg.convert_layout %45 : tensor<1x64xf32, #blocked6> -> tensor<1x64xf32, #blocked>

      %48 = tt.splat %40#1 : i32 -> tensor<64x1xi32, #blocked1>
      %49 = arith.xori %6, %48 : tensor<64x1xi32, #blocked1>
      %50 = tt.gather %arg5[%49] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>

      %51 = ttg.convert_layout %50 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked4>
      %52 = ttg.convert_layout %50 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked5>
      %53 = tt.reshape %51 : tensor<64x1xf32, #blocked4> -> tensor<1x64xf32, #blocked2>
      %54 = tt.reshape %52 : tensor<64x1xf32, #blocked5> -> tensor<1x64xf32, #blocked6>
      %55 = ttg.convert_layout %53 : tensor<1x64xf32, #blocked2> -> tensor<1x64xf32, #blocked>
      %56 = ttg.convert_layout %54 : tensor<1x64xf32, #blocked6> -> tensor<1x64xf32, #blocked>

      %57 = tt.splat %40#1 : i32 -> tensor<1x64xi32, #blocked>
      %58 = tt.splat %40#1 : i32 -> tensor<1x64xi32, #blocked1>
      %59 = arith.xori %11, %57 : tensor<1x64xi32, #blocked>
      %60 = arith.xori %12, %58 : tensor<1x64xi32, #blocked1>
      %61 = tt.gather %46[%59] {axis = 1 : i32, efficient_layout} : (tensor<1x64xf32, #blocked>, tensor<1x64xi32, #blocked>) -> tensor<1x64xf32, #blocked>
      %62 = tt.gather %47[%59] {axis = 1 : i32, efficient_layout} : (tensor<1x64xf32, #blocked>, tensor<1x64xi32, #blocked>) -> tensor<1x64xf32, #blocked>
      %63 = tt.splat %40#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %64 = arith.xori %0, %63 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %65 = arith.subi %0, %64 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %66 = arith.subf %46, %61 : tensor<1x64xf32, #blocked>
      %67 = arith.subf %47, %62 : tensor<1x64xf32, #blocked>
      %68 = arith.mulf %66, %cst : tensor<1x64xf32, #blocked>
      %69 = arith.mulf %67, %cst : tensor<1x64xf32, #blocked>
      %70 = arith.sitofp %65 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %71 = arith.mulf %70, %cst_6 : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %72 = tt.expand_dims %71 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %73 = arith.addf %68, %72 : tensor<1x64xf32, #blocked>
      %74 = arith.addf %69, %72 : tensor<1x64xf32, #blocked>
      %75 = arith.cmpf olt, %73, %cst_3 : tensor<1x64xf32, #blocked>
      %76 = arith.cmpf olt, %74, %cst_3 : tensor<1x64xf32, #blocked>
      %77 = arith.select %75, %cst_1, %cst_2 : tensor<1x64xi1, #blocked>, tensor<1x64xf32, #blocked>
      %78 = arith.select %76, %cst_1, %cst_2 : tensor<1x64xi1, #blocked>, tensor<1x64xf32, #blocked>
      %79 = arith.mulf %55, %55 : tensor<1x64xf32, #blocked>
      %80 = arith.mulf %56, %56 : tensor<1x64xf32, #blocked>
      %81 = arith.addf %79, %cst_5 : tensor<1x64xf32, #blocked>
      %82 = arith.addf %80, %cst_5 : tensor<1x64xf32, #blocked>
      %83 = arith.mulf %68, %68 : tensor<1x64xf32, #blocked>
      %84 = arith.mulf %69, %69 : tensor<1x64xf32, #blocked>
      %85 = arith.addf %83, %81 : tensor<1x64xf32, #blocked>
      %86 = arith.addf %84, %82 : tensor<1x64xf32, #blocked>
      %87 = math.sqrt %85 : tensor<1x64xf32, #blocked>
      %88 = math.sqrt %86 : tensor<1x64xf32, #blocked>
      %89 = arith.mulf %77, %87 : tensor<1x64xf32, #blocked>
      %90 = arith.mulf %78, %88 : tensor<1x64xf32, #blocked>
      %91 = arith.addf %73, %89 : tensor<1x64xf32, #blocked>
      %92 = arith.addf %74, %90 : tensor<1x64xf32, #blocked>
      %93 = arith.divf %55, %91 : tensor<1x64xf32, #blocked>
      %94 = arith.divf %56, %92 : tensor<1x64xf32, #blocked>
      %95 = arith.mulf %93, %93 : tensor<1x64xf32, #blocked>
      %96 = arith.mulf %94, %94 : tensor<1x64xf32, #blocked>
      %97 = arith.addf %95, %cst_2 : tensor<1x64xf32, #blocked>
      %98 = arith.addf %96, %cst_2 : tensor<1x64xf32, #blocked>
      %99 = math.rsqrt %97 : tensor<1x64xf32, #blocked>
      %100 = math.rsqrt %98 : tensor<1x64xf32, #blocked>
      %101 = arith.mulf %93, %99 : tensor<1x64xf32, #blocked>
      %102 = arith.mulf %94, %100 : tensor<1x64xf32, #blocked>
      %103 = tt.broadcast %60 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
      %104 = tt.gather %arg5[%103] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>

      %105 = ttg.convert_layout %99 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked1>
      %107 = ttg.convert_layout %100 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked1>
      %106 = tt.broadcast %105 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %108 = tt.broadcast %107 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %109 = arith.mulf %106, %arg5 : tensor<64x64xf32, #blocked1>

      %110 = ttg.convert_layout %101 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked1>
      %112 = ttg.convert_layout %102 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked1>
      %111 = tt.broadcast %110 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %113 = tt.broadcast %112 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %114 = arith.mulf %111, %104 : tensor<64x64xf32, #blocked1>
      %115 = arith.addf %109, %114 : tensor<64x64xf32, #blocked1>
      %116 = tt.trans %115 {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked7>
      %117 = ttg.convert_layout %116 : tensor<64x64xf32, #blocked7> -> tensor<64x64xf32, #blocked1>
      %118 = tt.gather %117[%103] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>

      %119 = arith.mulf %108, %117 : tensor<64x64xf32, #blocked1>

      %120 = arith.mulf %113, %118 : tensor<64x64xf32, #blocked1>
      %121 = arith.addf %119, %120 : tensor<64x64xf32, #blocked1>
      %122 = tt.gather %arg6[%103] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %123 = arith.mulf %108, %arg6 : tensor<64x64xf32, #blocked1>
      %124 = arith.mulf %113, %122 : tensor<64x64xf32, #blocked1>
      %125 = arith.addf %123, %124 : tensor<64x64xf32, #blocked1>
      scf.yield %121, %125 : tensor<64x64xf32, #blocked1>, tensor<64x64xf32, #blocked1>
    }
    %26 = tt.gather %25#0[%6] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>
    %27 = ttg.convert_layout %26 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked5>
    %28 = tt.reshape %27 : tensor<64x1xf32, #blocked5> -> tensor<64xf32, #blocked3>
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

