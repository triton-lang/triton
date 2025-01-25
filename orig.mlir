#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#loc = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":156:0)
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_persistent_fused(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":156:0), %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":156:0), %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":156:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":156:0), %arg4: i32 {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":156:0), %arg5: i32 {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":156:0), %arg6: i32 {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":156:0), %arg7: i32 {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":156:0), %arg8: i32 {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":156:0)) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32 loc(#loc1)
    %c3_i32 = arith.constant 3 : i32 loc(#loc1)
    %false = arith.constant false loc(#loc1)
    %cst = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc1)
    %cst_0 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc1)
    %c256_i32 = arith.constant 256 : i32 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c8_i32 = arith.constant 8 : i32 loc(#loc1)
    %c-1_i32 = arith.constant -1 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %c132_i32 = arith.constant 132 : i32 loc(#loc1)
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1> loc(#loc1)
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked> loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %c127_i32 = arith.constant 127 : i32 loc(#loc1)
    %c255_i32 = arith.constant 255 : i32 loc(#loc1)
    %c63_i32 = arith.constant 63 : i32 loc(#loc1)
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma> loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.addi %arg3, %c127_i32 : i32 loc(#loc78)
    %2 = arith.divsi %1, %c128_i32 : i32 loc(#loc79)
    %3 = arith.addi %arg4, %c255_i32 : i32 loc(#loc80)
    %4 = arith.divsi %3, %c256_i32 : i32 loc(#loc81)
    %5 = arith.addi %arg5, %c63_i32 : i32 loc(#loc82)
    %6 = arith.divsi %5, %c64_i32 : i32 loc(#loc83)
    %7 = arith.muli %2, %4 : i32 loc(#loc8)
    %8 = arith.divsi %7, %c132_i32 : i32 loc(#loc9)
    %9 = arith.remsi %7, %c132_i32 : i32 loc(#loc10)
    %10 = arith.cmpi slt, %0, %9 : i32 loc(#loc11)
    %11 = scf.if %10 -> (i32) {
      %122 = arith.addi %8, %c1_i32 : i32 loc(#loc13)
      scf.yield %122 : i32 loc(#loc13)
    } else {
      scf.yield %8 : i32 loc(#loc1)
    } loc(#loc12)
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc14)
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc14)
    %14 = arith.muli %4, %c8_i32 : i32 loc(#loc15)
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc16)
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc16)
    %17 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc17)
    %18 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> loc(#loc17)
    %19 = arith.muli %6, %11 : i32 loc(#loc18)
    %20 = ttg.local_alloc  : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> loc(#loc19)
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> loc(#loc20)
    %22 = arith.cmpi sgt, %19, %c0_i32 : i32 loc(#loc21)
    %23 = arith.subi %6, %c1_i32 : i32 loc(#loc22)
    %24 = arith.divsi %0, %14 : i32 loc(#loc23)
    %25 = arith.muli %24, %c8_i32 : i32 loc(#loc24)
    %26 = arith.subi %2, %25 : i32 loc(#loc25)
    %27 = arith.minsi %26, %c8_i32 : i32 loc(#loc26)
    %28 = arith.remsi %0, %27 : i32 loc(#loc27)
    %29 = arith.addi %25, %28 : i32 loc(#loc28)
    %30 = arith.remsi %0, %14 : i32 loc(#loc29)
    %31 = arith.divsi %30, %27 : i32 loc(#loc30)
    %32 = arith.muli %29, %c128_i32 : i32 loc(#loc31)
    %33 = arith.muli %31, %c256_i32 : i32 loc(#loc32)
    %34 = tt.splat %32 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc33)
    %35 = arith.addi %34, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc33)
    %36 = tt.splat %33 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc34)
    %37 = arith.addi %36, %17 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc34)
    %38 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc35)
    %39 = arith.cmpi slt, %35, %38 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc35)
    %40 = arith.select %39, %35, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc36)
    %41 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc37)
    %42 = arith.cmpi slt, %37, %41 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc37)
    %43 = arith.select %42, %37, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc38)
    %44 = tt.expand_dims %40 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1> loc(#loc39)
    %45 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1> loc(#loc40)
    %46 = arith.muli %44, %45 : tensor<128x1xi32, #blocked1> loc(#loc40)
    %47 = tt.expand_dims %12 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1> loc(#loc41)
    %48 = tt.broadcast %46 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc42)
    %49 = tt.broadcast %47 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc42)
    %50 = arith.addi %48, %49 : tensor<128x64xi32, #blocked1> loc(#loc42)
    %51 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1> loc(#loc43)
    %52 = tt.addptr %51, %50 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1> loc(#loc43)
    %53 = tt.expand_dims %13 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc44)
    %54 = tt.expand_dims %43 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked> loc(#loc45)
    %55 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked> loc(#loc46)
    %56 = arith.muli %54, %55 : tensor<1x256xi32, #blocked> loc(#loc46)
    %57 = tt.broadcast %53 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc47)
    %58 = tt.broadcast %56 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc47)
    %59 = arith.addi %57, %58 : tensor<64x256xi32, #blocked> loc(#loc47)
    %60 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked> loc(#loc48)
    %61 = tt.addptr %60, %59 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked> loc(#loc48)
    %62 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1> loc(#loc49)
    %63 = arith.cmpi slt, %47, %62 : tensor<1x64xi32, #blocked1> loc(#loc49)
    %64 = tt.broadcast %63 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1> loc(#loc19)
    %65 = ttg.memdesc_subview %20[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64> loc(#loc19)
    %66 = tt.splat %22 : i1 -> tensor<128x64xi1, #blocked1> loc(#loc21)
    %67 = arith.andi %66, %64 : tensor<128x64xi1, #blocked1> loc(#loc21)
    %68 = ttg.async_copy_global_to_local %52, %65 mask %67 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable, 3x128x64> loc(#loc19)
    %69 = ttg.async_commit_group %68 loc(#loc19)
    %70 = tt.splat %arg5 : i32 -> tensor<64x1xi32, #blocked> loc(#loc50)
    %71 = arith.cmpi slt, %53, %70 : tensor<64x1xi32, #blocked> loc(#loc50)
    %72 = tt.broadcast %71 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked> loc(#loc20)
    %73 = ttg.memdesc_subview %21[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc20)
    %74 = tt.splat %22 : i1 -> tensor<64x256xi1, #blocked> loc(#loc21)
    %75 = arith.andi %74, %72 : tensor<64x256xi1, #blocked> loc(#loc21)
    %76 = ttg.async_copy_global_to_local %61, %73 mask %75 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc20)
    %77 = ttg.async_commit_group %76 loc(#loc20)
    %78 = arith.cmpi sgt, %19, %c1_i32 : i32 loc(#loc21)
    %79 = arith.cmpi ne, %23, %c0_i32 : i32 loc(#loc84)
    %80 = arith.extui %79 : i1 to i32 loc(#loc51)
    %81 = arith.cmpi eq, %80, %c0_i32 : i32 loc(#loc53)
    %82:5 = scf.if %81 -> (i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) {
      %122 = arith.addi %0, %c132_i32 : i32 loc(#loc55)
      %123 = arith.divsi %122, %14 : i32 loc(#loc23)
      %124 = arith.muli %123, %c8_i32 : i32 loc(#loc24)
      %125 = arith.subi %2, %124 : i32 loc(#loc25)
      %126 = arith.minsi %125, %c8_i32 : i32 loc(#loc26)
      %127 = arith.remsi %122, %126 : i32 loc(#loc27)
      %128 = arith.addi %124, %127 : i32 loc(#loc28)
      %129 = arith.remsi %122, %14 : i32 loc(#loc29)
      %130 = arith.divsi %129, %126 : i32 loc(#loc30)
      %131 = arith.muli %128, %c128_i32 : i32 loc(#loc31)
      %132 = arith.muli %130, %c256_i32 : i32 loc(#loc32)
      %133 = tt.splat %131 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc33)
      %134 = arith.addi %133, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc33)
      %135 = tt.splat %132 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc34)
      %136 = arith.addi %135, %17 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc34)
      %137 = arith.cmpi slt, %134, %38 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc35)
      %138 = arith.select %137, %134, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc36)
      %139 = arith.cmpi slt, %136, %41 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc37)
      %140 = arith.select %139, %136, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc38)
      scf.yield %122, %128, %130, %138, %140 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc38)
    } else {
      scf.yield %0, %29, %31, %40, %43 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc1)
    } loc(#loc54)
    %83 = arith.muli %80, %c64_i32 : i32 loc(#loc56)
    %84 = tt.splat %83 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc57)
    %85 = tt.splat %83 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc57)
    %86 = arith.addi %84, %12 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc57)
    %87 = arith.addi %85, %13 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc57)
    %88 = tt.expand_dims %82#3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1> loc(#loc39)
    %89 = arith.muli %88, %45 : tensor<128x1xi32, #blocked1> loc(#loc40)
    %90 = tt.expand_dims %86 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1> loc(#loc41)
    %91 = tt.broadcast %89 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc42)
    %92 = tt.broadcast %90 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc42)
    %93 = arith.addi %91, %92 : tensor<128x64xi32, #blocked1> loc(#loc42)
    %94 = tt.addptr %51, %93 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1> loc(#loc43)
    %95 = tt.expand_dims %87 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc44)
    %96 = tt.expand_dims %82#4 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked> loc(#loc45)
    %97 = arith.muli %96, %55 : tensor<1x256xi32, #blocked> loc(#loc46)
    %98 = tt.broadcast %95 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc47)
    %99 = tt.broadcast %97 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc47)
    %100 = arith.addi %98, %99 : tensor<64x256xi32, #blocked> loc(#loc47)
    %101 = tt.addptr %60, %100 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked> loc(#loc48)
    %102 = arith.subi %arg5, %83 : i32 loc(#loc58)
    %103 = tt.splat %102 : i32 -> tensor<1x64xi32, #blocked1> loc(#loc49)
    %104 = arith.cmpi slt, %47, %103 : tensor<1x64xi32, #blocked1> loc(#loc49)
    %105 = tt.broadcast %104 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1> loc(#loc19)
    %106 = ttg.memdesc_subview %20[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64> loc(#loc19)
    %107 = tt.splat %78 : i1 -> tensor<128x64xi1, #blocked1> loc(#loc21)
    %108 = arith.andi %107, %105 : tensor<128x64xi1, #blocked1> loc(#loc21)
    %109 = ttg.async_copy_global_to_local %94, %106 mask %108 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable, 3x128x64> loc(#loc19)
    %110 = ttg.async_commit_group %109 loc(#loc19)
    %111 = tt.splat %102 : i32 -> tensor<64x1xi32, #blocked> loc(#loc50)
    %112 = arith.cmpi slt, %53, %111 : tensor<64x1xi32, #blocked> loc(#loc50)
    %113 = tt.broadcast %112 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked> loc(#loc20)
    %114 = ttg.memdesc_subview %21[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc20)
    %115 = tt.splat %78 : i1 -> tensor<64x256xi1, #blocked> loc(#loc21)
    %116 = arith.andi %115, %113 : tensor<64x256xi1, #blocked> loc(#loc21)
    %117 = ttg.async_copy_global_to_local %101, %114 mask %116 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc20)
    %118 = ttg.async_commit_group %117 loc(#loc20)
    %119:18 = scf.for %arg9 = %c0_i32 to %19 step %c1_i32 iter_args(%arg10 = %80, %arg11 = %82#0, %arg12 = %82#1, %arg13 = %82#2, %arg14 = %cst_3, %arg15 = %82#3, %arg16 = %82#4, %arg17 = %false, %arg18 = %c1_i32, %arg19 = %c-1_i32, %arg20 = %77, %arg21 = %118, %arg22 = %c0_i32, %arg23 = %80, %arg24 = %29, %arg25 = %82#1, %arg26 = %31, %arg27 = %82#2) -> (i32, i32, i32, i32, tensor<128x256xf32, #mma>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i1, i32, i32, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32, i32, i32)  : i32 {
      %122 = arith.subi %19, %c2_i32 : i32 loc(#loc21)
      %123 = arith.cmpi slt, %arg9, %122 : i32 loc(#loc21)
      %124 = arith.cmpi eq, %arg10, %23 : i32 loc(#loc52)
      %125 = arith.addi %arg10, %c1_i32 : i32 loc(#loc59)
      %126 = arith.select %124, %c0_i32, %125 : i32 loc(#loc51)
      %127 = arith.cmpi eq, %126, %c0_i32 : i32 loc(#loc53)
      %128:5 = scf.if %127 -> (i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) {
        %178 = arith.addi %arg11, %c132_i32 : i32 loc(#loc55)
        %179 = arith.divsi %178, %14 : i32 loc(#loc23)
        %180 = arith.muli %179, %c8_i32 : i32 loc(#loc24)
        %181 = arith.subi %2, %180 : i32 loc(#loc25)
        %182 = arith.minsi %181, %c8_i32 : i32 loc(#loc26)
        %183 = arith.remsi %178, %182 : i32 loc(#loc27)
        %184 = arith.addi %180, %183 : i32 loc(#loc28)
        %185 = arith.remsi %178, %14 : i32 loc(#loc29)
        %186 = arith.divsi %185, %182 : i32 loc(#loc30)
        %187 = arith.muli %184, %c128_i32 : i32 loc(#loc31)
        %188 = arith.muli %186, %c256_i32 : i32 loc(#loc32)
        %189 = tt.splat %187 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc33)
        %190 = arith.addi %189, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc33)
        %191 = tt.splat %188 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc34)
        %192 = arith.addi %191, %17 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc34)
        %193 = arith.cmpi slt, %190, %38 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc35)
        %194 = arith.select %193, %190, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc36)
        %195 = arith.cmpi slt, %192, %41 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc37)
        %196 = arith.select %195, %192, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc38)
        scf.yield %178, %184, %186, %194, %196 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc38)
      } else {
        scf.yield %arg11, %arg12, %arg13, %arg15, %arg16 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc1)
      } loc(#loc54)
      %129 = arith.addi %arg19, %c1_i32 : i32 loc(#loc21)
      %130 = arith.cmpi slt, %129, %c3_i32 : i32 loc(#loc21)
      %131 = arith.select %130, %129, %c0_i32 : i32 loc(#loc21)
      %132 = ttg.memdesc_subview %20[%131, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64> loc(#loc19)
      %133 = ttg.async_wait %arg20 {num = 2 : i32} loc(#loc19)
      %134 = ttg.memdesc_subview %21[%131, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc20)
      %135 = ttng.warp_group_dot %132, %134, %arg14, %arg17 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64> * !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> -> tensor<128x256xf32, #mma> loc(#loc60)
      %136:3 = ttng.warp_group_dot_wait %135, %132, %134 {pendings = 1 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc60)
      %137 = arith.cmpi ne, %arg22, %23 : i32 loc(#loc85)
      %138 = arith.addi %arg18, %c1_i32 : i32 loc(#loc21)
      %139 = arith.cmpi slt, %138, %c3_i32 : i32 loc(#loc21)
      %140 = arith.select %139, %138, %c0_i32 : i32 loc(#loc21)
      %141 = arith.muli %126, %c64_i32 : i32 loc(#loc56)
      %142 = tt.splat %141 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc57)
      %143 = tt.splat %141 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc57)
      %144 = arith.addi %142, %12 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc57)
      %145 = arith.addi %143, %13 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc57)
      %146 = tt.expand_dims %128#3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1> loc(#loc39)
      %147 = arith.muli %146, %45 : tensor<128x1xi32, #blocked1> loc(#loc40)
      %148 = tt.expand_dims %144 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1> loc(#loc41)
      %149 = tt.broadcast %147 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc42)
      %150 = tt.broadcast %148 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc42)
      %151 = arith.addi %149, %150 : tensor<128x64xi32, #blocked1> loc(#loc42)
      %152 = tt.addptr %51, %151 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1> loc(#loc43)
      %153 = tt.expand_dims %145 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc44)
      %154 = tt.expand_dims %128#4 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked> loc(#loc45)
      %155 = arith.muli %154, %55 : tensor<1x256xi32, #blocked> loc(#loc46)
      %156 = tt.broadcast %153 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc47)
      %157 = tt.broadcast %155 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc47)
      %158 = arith.addi %156, %157 : tensor<64x256xi32, #blocked> loc(#loc47)
      %159 = tt.addptr %60, %158 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked> loc(#loc48)
      %160 = arith.subi %arg5, %141 : i32 loc(#loc58)
      %161 = tt.splat %160 : i32 -> tensor<1x64xi32, #blocked1> loc(#loc49)
      %162 = arith.cmpi slt, %47, %161 : tensor<1x64xi32, #blocked1> loc(#loc49)
      %163 = tt.broadcast %162 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1> loc(#loc19)
      %164 = ttg.memdesc_subview %20[%140, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64> loc(#loc19)
      %165 = tt.splat %123 : i1 -> tensor<128x64xi1, #blocked1> loc(#loc21)
      %166 = arith.andi %165, %163 : tensor<128x64xi1, #blocked1> loc(#loc21)
      %167 = ttg.async_copy_global_to_local %152, %164 mask %166 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable, 3x128x64> loc(#loc19)
      %168 = ttg.async_commit_group %167 loc(#loc19)
      %169 = tt.splat %160 : i32 -> tensor<64x1xi32, #blocked> loc(#loc50)
      %170 = arith.cmpi slt, %53, %169 : tensor<64x1xi32, #blocked> loc(#loc50)
      %171 = tt.broadcast %170 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked> loc(#loc20)
      %172 = ttg.memdesc_subview %21[%140, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc20)
      %173 = tt.splat %123 : i1 -> tensor<64x256xi1, #blocked> loc(#loc21)
      %174 = arith.andi %173, %171 : tensor<64x256xi1, #blocked> loc(#loc21)
      %175 = ttg.async_copy_global_to_local %159, %172 mask %174 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc20)
      %176 = ttg.async_commit_group %175 loc(#loc20)
      %177 = arith.cmpi eq, %arg22, %23 : i32 loc(#loc61)
      scf.if %177 {
        %178:3 = ttng.warp_group_dot_wait %136#0, %132, %134 {pendings = 0 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc60)
        %179 = arith.muli %arg24, %c128_i32 : i32 loc(#loc63)
        %180 = tt.splat %179 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc64)
        %181 = arith.addi %180, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc64)
        %182 = arith.muli %arg26, %c256_i32 : i32 loc(#loc65)
        %183 = tt.splat %182 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> loc(#loc66)
        %184 = arith.addi %183, %18 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> loc(#loc66)
        %185 = tt.expand_dims %181 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2> loc(#loc67)
        %186 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked2> loc(#loc68)
        %187 = arith.muli %186, %185 : tensor<128x1xi32, #blocked2> loc(#loc68)
        %188 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2> loc(#loc69)
        %189 = tt.addptr %188, %187 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2> loc(#loc69)
        %190 = tt.expand_dims %184 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2> loc(#loc70)
        %191 = tt.broadcast %189 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x256x!tt.ptr<f16>, #blocked2> loc(#loc71)
        %192 = tt.broadcast %190 : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2> loc(#loc71)
        %193 = tt.addptr %191, %192 : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2> loc(#loc71)
        %194 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked2> loc(#loc72)
        %195 = arith.cmpi slt, %185, %194 : tensor<128x1xi32, #blocked2> loc(#loc72)
        %196 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked2> loc(#loc73)
        %197 = arith.cmpi slt, %190, %196 : tensor<1x256xi32, #blocked2> loc(#loc73)
        %198 = tt.broadcast %195 : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2> loc(#loc74)
        %199 = tt.broadcast %197 : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2> loc(#loc74)
        %200 = arith.andi %198, %199 : tensor<128x256xi1, #blocked2> loc(#loc74)
        %201 = arith.truncf %178#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma> loc(#loc75)
        %202 = ttg.convert_layout %201 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked2> loc(#loc76)
        tt.store %193, %202, %200 : tensor<128x256x!tt.ptr<f16>, #blocked2> loc(#loc76)
      } loc(#loc62)
      scf.yield %126, %128#0, %128#1, %128#2, %136#0, %128#3, %128#4, %137, %140, %131, %arg21, %176, %arg23, %126, %arg25, %128#1, %arg27, %128#2 : i32, i32, i32, i32, tensor<128x256xf32, #mma>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i1, i32, i32, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32, i32, i32 loc(#loc21)
    } loc(#loc21)
    %120 = ttng.warp_group_dot_wait %119#4 {pendings = 0 : i32} : tensor<128x256xf32, #mma> loc(#loc21)
    %121 = ttg.async_wait  {num = 0 : i32} loc(#loc21)
    ttg.local_dealloc %20 : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> loc(#loc21)
    ttg.local_dealloc %21 : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> loc(#loc21)
    tt.return loc(#loc77)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":167:30)
#loc3 = loc("/root/.pyenv/versions/3.11.8/lib/python3.11/site-packages/triton/language/standard.py":40:22)
#loc4 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":168:27)
#loc5 = loc("/root/.pyenv/versions/3.11.8/lib/python3.11/site-packages/triton/language/standard.py":40:28)
#loc6 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":169:27)
#loc7 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":170:25)
#loc8 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":171:28)
#loc9 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":173:32)
#loc10 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":174:31)
#loc11 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":174:19)
#loc12 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":174:7)
#loc13 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":175:24)
#loc14 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":180:35)
#loc15 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":182:38)
#loc16 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":186:27)
#loc17 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":187:27)
#loc18 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":191:32)
#loc19 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":213:20)
#loc20 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":214:20)
#loc21 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":191:22)
#loc22 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":192:38)
#loc23 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":195:34)
#loc24 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":196:37)
#loc25 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":197:43)
#loc26 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":197:56)
#loc27 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":198:45)
#loc28 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":198:35)
#loc29 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":199:31)
#loc30 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":199:52)
#loc31 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":201:30)
#loc32 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":202:30)
#loc33 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":203:32)
#loc34 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":204:32)
#loc35 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":205:41)
#loc36 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":205:53)
#loc37 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":206:41)
#loc38 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":206:53)
#loc39 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":210:34)
#loc40 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":210:45)
#loc41 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":210:64)
#loc42 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":210:57)
#loc43 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":210:26)
#loc44 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":211:33)
#loc45 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":211:64)
#loc46 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":211:75)
#loc47 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":211:56)
#loc48 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":211:26)
#loc49 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":213:60)
#loc50 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":214:60)
#loc51 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":192:44)
#loc52 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":192:28)
#loc53 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":193:17)
#loc54 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":193:11)
#loc55 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":194:23)
#loc56 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":209:22)
#loc57 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":209:37)
#loc58 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":213:64)
#loc59 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":192:49)
#loc60 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":215:35)
#loc61 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":217:17)
#loc62 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":217:11)
#loc63 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":218:30)
#loc64 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":218:45)
#loc65 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":219:30)
#loc66 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":219:45)
#loc67 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":220:49)
#loc68 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":220:41)
#loc69 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":220:29)
#loc70 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":220:80)
#loc71 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":220:60)
#loc72 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":221:41)
#loc73 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":221:66)
#loc74 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":221:47)
#loc75 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":225:35)
#loc76 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":226:29)
#loc77 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":191:4)
#loc78 = loc(callsite(#loc3 at #loc4))
#loc79 = loc(callsite(#loc5 at #loc4))
#loc80 = loc(callsite(#loc3 at #loc6))
#loc81 = loc(callsite(#loc5 at #loc6))
#loc82 = loc(callsite(#loc3 at #loc7))
#loc83 = loc(callsite(#loc5 at #loc7))
#loc84 = loc(fused[#loc51, #loc52])
#loc85 = loc(fused[#loc60, #loc61])

