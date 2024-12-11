#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ortho_diag_test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<-7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<1x64xf32, #blocked1>
    %cst_1 = arith.constant dense<-1.000000e+00> : tensor<1x64xf32, #blocked1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<1x64xf32, #blocked1>
    %cst_3 = arith.constant dense<7.52316385E-37> : tensor<1x64xf32, #blocked1>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #blocked1>
    %cst_5 = arith.constant dense<64> : tensor<64x1xi32, #blocked2>
    %cst_6 = arith.constant dense<7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_7 = arith.constant dense<0> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %7 = arith.muli %5, %cst_5 : tensor<64x1xi32, #blocked2>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked2>
    %9 = tt.addptr %8, %7 : tensor<64x1x!tt.ptr<f32>, #blocked2>, tensor<64x1xi32, #blocked2>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %12 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %13 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %14 = tt.broadcast %9 : tensor<64x1x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %15 = tt.broadcast %11 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
    %16 = tt.broadcast %12 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %17 = tt.addptr %14, %15 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %18 = tt.load %17 : tensor<64x64x!tt.ptr<f32>, #blocked2>
    %19 = ttg.convert_layout %18 : tensor<64x64xf32, #blocked2> -> tensor<64x64xf32, #blocked1>
    %20 = tt.broadcast %6 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %21 = arith.cmpi eq, %20, %16 : tensor<64x64xi32, #blocked>
    %22 = arith.uitofp %21 : tensor<64x64xi1, #blocked> to tensor<64x64xf32, #blocked>
    %23 = arith.xori %20, %16 : tensor<64x64xi32, #blocked>
    %24 = arith.cmpi eq, %0, %cst_7 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %25:2 = scf.for %arg4 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg5 = %19, %arg6 = %22) -> (tensor<64x64xf32, #blocked1>, tensor<64x64xf32, #blocked>)  : i32 {
      %37 = arith.mulf %arg5, %arg5 : tensor<64x64xf32, #blocked1>
      %38 = ttg.convert_layout %37 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
      %39 = tt.gather %38[%23] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
      %40 = "tt.reduce"(%39) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %105 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %105 : f32
      }) : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %41 = arith.select %24, %cst, %40 : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %42:2 = "tt.reduce"(%41, %0) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: i32, %arg9: f32, %arg10: i32):
        %105 = arith.cmpf oeq, %arg7, %arg9 : f32
        %106 = arith.cmpi slt, %arg8, %arg10 : i32
        %107 = arith.andi %105, %106 : i1
        %108 = arith.cmpf ogt, %arg7, %arg9 : f32
        %109 = arith.ori %108, %107 : i1
        %110 = arith.select %109, %arg7, %arg9 : f32
        %111 = arith.select %109, %arg8, %arg10 : i32
        tt.reduce.return %110, %111 : f32, i32
      }) : (tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) -> (f32, i32)
      %43 = ttg.convert_layout %arg5 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
      %44 = tt.gather %43[%6] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked>
      %45 = ttg.convert_layout %44 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked4>
      %46 = tt.reshape %45 : tensor<64x1xf32, #blocked4> -> tensor<1x64xf32, #blocked1>
      %47 = tt.splat %42#1 : i32 -> tensor<64x1xi32, #blocked>
      %48 = arith.xori %6, %47 : tensor<64x1xi32, #blocked>
      %49 = tt.gather %43[%48] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked>
      %50 = ttg.convert_layout %49 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked4>
      %51 = tt.reshape %50 : tensor<64x1xf32, #blocked4> -> tensor<1x64xf32, #blocked1>
      %52 = tt.splat %42#1 : i32 -> tensor<1x64xi32, #blocked>
      %53 = tt.splat %42#1 : i32 -> tensor<1x64xi32, #blocked1>
      %54 = arith.xori %12, %52 : tensor<1x64xi32, #blocked>
      %55 = arith.xori %13, %53 : tensor<1x64xi32, #blocked1>
      %56 = tt.gather %46[%55] {axis = 1 : i32} : (tensor<1x64xf32, #blocked1>, tensor<1x64xi32, #blocked1>) -> tensor<1x64xf32, #blocked1>
      %57 = tt.splat %42#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %58 = arith.xori %1, %57 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %59 = arith.subi %1, %58 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %60 = arith.subf %46, %56 : tensor<1x64xf32, #blocked1>
      %61 = arith.mulf %60, %cst_0 : tensor<1x64xf32, #blocked1>
      %62 = arith.sitofp %59 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %63 = arith.mulf %62, %cst_6 : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %64 = tt.expand_dims %63 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xf32, #blocked1>
      %65 = arith.addf %61, %64 : tensor<1x64xf32, #blocked1>
      %66 = arith.cmpf olt, %65, %cst_4 : tensor<1x64xf32, #blocked1>
      %67 = arith.select %66, %cst_1, %cst_2 : tensor<1x64xi1, #blocked1>, tensor<1x64xf32, #blocked1>
      %68 = arith.mulf %51, %51 : tensor<1x64xf32, #blocked1>
      %69 = arith.addf %68, %cst_3 : tensor<1x64xf32, #blocked1>
      %70 = arith.mulf %61, %61 : tensor<1x64xf32, #blocked1>
      %71 = arith.addf %70, %69 : tensor<1x64xf32, #blocked1>
      %72 = math.sqrt %71 : tensor<1x64xf32, #blocked1>
      %73 = arith.mulf %67, %72 : tensor<1x64xf32, #blocked1>
      %74 = arith.addf %65, %73 : tensor<1x64xf32, #blocked1>
      %75 = arith.divf %51, %74 : tensor<1x64xf32, #blocked1>
      %76 = arith.mulf %75, %75 : tensor<1x64xf32, #blocked1>
      %77 = arith.addf %76, %cst_2 : tensor<1x64xf32, #blocked1>
      %78 = math.rsqrt %77 : tensor<1x64xf32, #blocked1>
      %79 = arith.mulf %75, %78 : tensor<1x64xf32, #blocked1>
      %80 = tt.broadcast %54 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
      %81 = tt.broadcast %55 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
      %82 = tt.gather %43[%80] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
      %83 = ttg.convert_layout %78 : tensor<1x64xf32, #blocked1> -> tensor<1x64xf32, #blocked5>
      %84 = tt.broadcast %83 : tensor<1x64xf32, #blocked5> -> tensor<64x64xf32, #blocked5>
      %85 = ttg.convert_layout %78 : tensor<1x64xf32, #blocked1> -> tensor<1x64xf32, #blocked>
      %86 = tt.broadcast %85 : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %87 = tt.broadcast %78 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %88 = arith.mulf %87, %arg5 : tensor<64x64xf32, #blocked1>
      %89 = ttg.convert_layout %79 : tensor<1x64xf32, #blocked1> -> tensor<1x64xf32, #blocked>
      %90 = tt.broadcast %89 : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %91 = tt.broadcast %79 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %92 = arith.mulf %90, %82 : tensor<64x64xf32, #blocked>
      %93 = ttg.convert_layout %92 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
      %94 = arith.addf %88, %93 : tensor<64x64xf32, #blocked1>
      %95 = tt.trans %94 {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked5>
      %96 = tt.gather %95[%81] {axis = 1 : i32} : (tensor<64x64xf32, #blocked5>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %97 = arith.mulf %84, %95 : tensor<64x64xf32, #blocked5>
      %98 = ttg.convert_layout %97 : tensor<64x64xf32, #blocked5> -> tensor<64x64xf32, #blocked1>
      %99 = arith.mulf %91, %96 : tensor<64x64xf32, #blocked1>
      %100 = arith.addf %98, %99 : tensor<64x64xf32, #blocked1>
      %101 = tt.gather %arg6[%80] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
      %102 = arith.mulf %86, %arg6 : tensor<64x64xf32, #blocked>
      %103 = arith.mulf %90, %101 : tensor<64x64xf32, #blocked>
      %104 = arith.addf %102, %103 : tensor<64x64xf32, #blocked>
      scf.yield %100, %104 : tensor<64x64xf32, #blocked1>, tensor<64x64xf32, #blocked>
    }
    %26 = ttg.convert_layout %25#0 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
    %27 = tt.gather %26[%6] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked>
    %28 = ttg.convert_layout %27 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked4>
    %29 = tt.reshape %28 : tensor<64x1xf32, #blocked4> -> tensor<64xf32, #blocked3>
    %30 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked2>
    %31 = tt.addptr %30, %7 : tensor<64x1x!tt.ptr<f32>, #blocked2>, tensor<64x1xi32, #blocked2>
    %32 = tt.broadcast %31 : tensor<64x1x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %33 = tt.addptr %32, %15 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %34 = ttg.convert_layout %25#1 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked2>
    tt.store %33, %34 : tensor<64x64x!tt.ptr<f32>, #blocked2>
    %35 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked3>
    %36 = tt.addptr %35, %2 : tensor<64x!tt.ptr<f32>, #blocked3>, tensor<64xi32, #blocked3>
    tt.store %36, %29 : tensor<64x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

