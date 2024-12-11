#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ortho_diag_test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<7.52316385E-37> : tensor<64x1xf32, #blocked>
    %cst_0 = arith.constant dense<-7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant dense<5.000000e-01> : tensor<64x1xf32, #blocked1>
    %cst_2 = arith.constant dense<7.52316385E-37> : tensor<64x1xf32, #blocked1>
    %cst_3 = arith.constant dense<-1.000000e+00> : tensor<64x1xf32, #blocked1>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<64x1xf32, #blocked1>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<64x1xf32, #blocked1>
    %cst_6 = arith.constant dense<64> : tensor<64x1xi32, #blocked2>
    %cst_7 = arith.constant dense<0> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked5>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.expand_dims %5 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %8 = tt.expand_dims %6 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %9 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<64x1xi32, #blocked4>
    %10 = arith.muli %7, %cst_6 : tensor<64x1xi32, #blocked2>
    %11 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked2>
    %12 = tt.addptr %11, %10 : tensor<64x1x!tt.ptr<f32>, #blocked2>, tensor<64x1xi32, #blocked2>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %15 = tt.expand_dims %13 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %16 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %17 = tt.expand_dims %14 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi32, #blocked4>
    %18 = tt.broadcast %12 : tensor<64x1x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %19 = tt.broadcast %15 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
    %20 = tt.broadcast %16 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %21 = tt.broadcast %17 : tensor<1x64xi32, #blocked4> -> tensor<64x64xi32, #blocked4>
    %22 = tt.addptr %18, %19 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %23 = tt.load %22 : tensor<64x64x!tt.ptr<f32>, #blocked2>
    %24 = ttg.convert_layout %23 : tensor<64x64xf32, #blocked2> -> tensor<64x64xf32, #blocked3>
    %25 = tt.broadcast %8 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %26 = tt.broadcast %9 : tensor<64x1xi32, #blocked4> -> tensor<64x64xi32, #blocked4>
    %27 = arith.cmpi eq, %26, %21 : tensor<64x64xi32, #blocked4>
    %28 = arith.uitofp %27 : tensor<64x64xi1, #blocked4> to tensor<64x64xf32, #blocked4>
    %29 = arith.xori %25, %20 : tensor<64x64xi32, #blocked>
    %30 = arith.cmpi eq, %0, %cst_7 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %31:2 = scf.for %arg4 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg5 = %24, %arg6 = %28) -> (tensor<64x64xf32, #blocked3>, tensor<64x64xf32, #blocked4>)  : i32 {
      %43 = arith.mulf %arg5, %arg5 : tensor<64x64xf32, #blocked3>
      %44 = ttg.convert_layout %43 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked>
      %45 = tt.gather %44[%29] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
      %46 = "tt.reduce"(%45) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %116 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %116 : f32
      }) : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %47 = arith.select %30, %cst_0, %46 : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %48:2 = "tt.reduce"(%47, %0) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: i32, %arg9: f32, %arg10: i32):
        %116 = arith.cmpf oeq, %arg7, %arg9 : f32
        %117 = arith.cmpi slt, %arg8, %arg10 : i32
        %118 = arith.andi %116, %117 : i1
        %119 = arith.cmpf ogt, %arg7, %arg9 : f32
        %120 = arith.ori %119, %118 : i1
        %121 = arith.select %120, %arg7, %arg9 : f32
        %122 = arith.select %120, %arg8, %arg10 : i32
        tt.reduce.return %121, %122 : f32, i32
      }) : (tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) -> (f32, i32)
      %49 = ttg.convert_layout %arg5 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked>
      %50 = tt.gather %49[%8] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked>
      %51 = ttg.convert_layout %50 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked1>
      %52 = tt.splat %48#1 : i32 -> tensor<64x1xi32, #blocked>
      %53 = arith.xori %8, %52 : tensor<64x1xi32, #blocked>
      %54 = tt.gather %49[%53] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked>
      %55 = ttg.convert_layout %54 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked1>
      %56 = tt.splat %48#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %57 = tt.splat %48#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
      %58 = tt.splat %48#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %59 = arith.xori %1, %56 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %60 = arith.xori %2, %57 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
      %61 = arith.xori %3, %58 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %62 = tt.expand_dims %59 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xi32, #blocked3>
      %63 = tt.expand_dims %60 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<64x1xi32, #blocked4>
      %64 = tt.expand_dims %61 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
      %65 = tt.gather %50[%64] {axis = 0 : i32} : (tensor<64x1xf32, #blocked>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>
      %66 = arith.subi %3, %61 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %67 = tt.expand_dims %66 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
      %68 = arith.subf %51, %65 : tensor<64x1xf32, #blocked1>
      %69 = arith.mulf %68, %cst_1 : tensor<64x1xf32, #blocked1>
      %70 = arith.sitofp %67 : tensor<64x1xi32, #blocked1> to tensor<64x1xf32, #blocked1>
      %71 = arith.mulf %70, %cst_2 : tensor<64x1xf32, #blocked1>
      %72 = arith.addf %69, %71 : tensor<64x1xf32, #blocked1>
      %73 = arith.cmpf olt, %72, %cst_5 : tensor<64x1xf32, #blocked1>
      %74 = arith.select %73, %cst_3, %cst_4 : tensor<64x1xi1, #blocked1>, tensor<64x1xf32, #blocked1>
      %75 = arith.mulf %54, %54 : tensor<64x1xf32, #blocked>
      %76 = arith.addf %75, %cst : tensor<64x1xf32, #blocked>
      %77 = ttg.convert_layout %76 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked1>
      %78 = arith.mulf %69, %69 : tensor<64x1xf32, #blocked1>
      %79 = arith.addf %78, %77 : tensor<64x1xf32, #blocked1>
      %80 = math.sqrt %79 : tensor<64x1xf32, #blocked1>
      %81 = arith.mulf %74, %80 : tensor<64x1xf32, #blocked1>
      %82 = arith.addf %72, %81 : tensor<64x1xf32, #blocked1>
      %83 = arith.divf %55, %82 : tensor<64x1xf32, #blocked1>
      %84 = arith.mulf %83, %83 : tensor<64x1xf32, #blocked1>
      %85 = arith.addf %84, %cst_4 : tensor<64x1xf32, #blocked1>
      %86 = math.rsqrt %85 : tensor<64x1xf32, #blocked1>
      %87 = arith.mulf %83, %86 : tensor<64x1xf32, #blocked1>
      %88 = tt.broadcast %62 : tensor<64x1xi32, #blocked3> -> tensor<64x64xi32, #blocked3>
      %89 = tt.broadcast %63 : tensor<64x1xi32, #blocked4> -> tensor<64x64xi32, #blocked4>
      %90 = tt.gather %arg5[%89] {axis = 0 : i32} : (tensor<64x64xf32, #blocked3>, tensor<64x64xi32, #blocked4>) -> tensor<64x64xf32, #blocked4>
      %91 = ttg.convert_layout %86 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked4>
      %92 = tt.broadcast %91 : tensor<64x1xf32, #blocked4> -> tensor<64x64xf32, #blocked4>
      %93 = ttg.convert_layout %86 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked>
      %94 = tt.broadcast %93 : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %95 = ttg.convert_layout %86 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked3>
      %96 = tt.broadcast %95 : tensor<64x1xf32, #blocked3> -> tensor<64x64xf32, #blocked3>
      %97 = arith.mulf %96, %arg5 : tensor<64x64xf32, #blocked3>
      %98 = ttg.convert_layout %87 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked4>
      %99 = tt.broadcast %98 : tensor<64x1xf32, #blocked4> -> tensor<64x64xf32, #blocked4>
      %100 = ttg.convert_layout %87 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked3>
      %101 = tt.broadcast %100 : tensor<64x1xf32, #blocked3> -> tensor<64x64xf32, #blocked3>
      %102 = arith.mulf %99, %90 : tensor<64x64xf32, #blocked4>
      %103 = ttg.convert_layout %102 : tensor<64x64xf32, #blocked4> -> tensor<64x64xf32, #blocked3>
      %104 = arith.addf %97, %103 : tensor<64x64xf32, #blocked3>
      %105 = tt.trans %104 {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked>
      %106 = ttg.convert_layout %105 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
      %107 = tt.gather %106[%88] {axis = 0 : i32} : (tensor<64x64xf32, #blocked3>, tensor<64x64xi32, #blocked3>) -> tensor<64x64xf32, #blocked3>
      %108 = arith.mulf %94, %105 : tensor<64x64xf32, #blocked>
      %109 = ttg.convert_layout %108 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
      %110 = arith.mulf %101, %107 : tensor<64x64xf32, #blocked3>
      %111 = arith.addf %109, %110 : tensor<64x64xf32, #blocked3>
      %112 = tt.gather %arg6[%89] {axis = 0 : i32} : (tensor<64x64xf32, #blocked4>, tensor<64x64xi32, #blocked4>) -> tensor<64x64xf32, #blocked4>
      %113 = arith.mulf %92, %arg6 : tensor<64x64xf32, #blocked4>
      %114 = arith.mulf %99, %112 : tensor<64x64xf32, #blocked4>
      %115 = arith.addf %113, %114 : tensor<64x64xf32, #blocked4>
      scf.yield %111, %115 : tensor<64x64xf32, #blocked3>, tensor<64x64xf32, #blocked4>
    }
    %32 = ttg.convert_layout %31#0 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked>
    %33 = tt.gather %32[%8] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked>
    %34 = ttg.convert_layout %33 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked1>
    %35 = tt.reshape %34 : tensor<64x1xf32, #blocked1> -> tensor<64xf32, #blocked5>
    %36 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked2>
    %37 = tt.addptr %36, %10 : tensor<64x1x!tt.ptr<f32>, #blocked2>, tensor<64x1xi32, #blocked2>
    %38 = tt.broadcast %37 : tensor<64x1x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %39 = tt.addptr %38, %19 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %40 = ttg.convert_layout %31#1 : tensor<64x64xf32, #blocked4> -> tensor<64x64xf32, #blocked2>
    tt.store %39, %40 : tensor<64x64x!tt.ptr<f32>, #blocked2>
    %41 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked5>
    %42 = tt.addptr %41, %4 : tensor<64x!tt.ptr<f32>, #blocked5>, tensor<64xi32, #blocked5>
    tt.store %42, %35 : tensor<64x!tt.ptr<f32>, #blocked5>
    tt.return
  }
}

