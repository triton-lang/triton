#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg24: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg25: f32, %arg26: i32, %arg27: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg28: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<1x64xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128x64xf32, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked1>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %cst_5 = arith.constant dense<0.127517432> : tensor<128x64xf32, #blocked>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %c63_i32 = arith.constant 63 : i32
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked1>
    %c262144_i32 = arith.constant 262144 : i32
    %cst_8 = arith.constant dense<true> : tensor<128x128xi1, #blocked1>
    %cst_9 = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #blocked2>
    %c2_i32 = arith.constant 2 : i32
    %cst_10 = arith.constant dense<1.000000e+00> : tensor<128xf32, #blocked3>
    %cst_11 = arith.constant dense<0xFF800000> : tensor<128xf32, #blocked3>
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_12 = arith.constant dense<8192> : tensor<128xi32, #blocked3>
    %cst_13 = arith.constant dense<0x7F800000> : tensor<128xf32, #blocked3>
    %c8192_i32 = arith.constant 8192 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi sge, %arg5, %c0_i32 : i32
    llvm.intr.assume %0 : i1
    %1 = arith.cmpi sge, %arg6, %c0_i32 : i32
    llvm.intr.assume %1 : i1
    %2 = arith.cmpi sge, %arg7, %c0_i32 : i32
    llvm.intr.assume %2 : i1
    llvm.intr.assume %true : i1
    %3 = arith.cmpi sge, %arg8, %c0_i32 : i32
    llvm.intr.assume %3 : i1
    %4 = arith.cmpi sge, %arg9, %c0_i32 : i32
    llvm.intr.assume %4 : i1
    %5 = arith.cmpi sge, %arg10, %c0_i32 : i32
    llvm.intr.assume %5 : i1
    llvm.intr.assume %true : i1
    %6 = arith.cmpi sge, %arg17, %c0_i32 : i32
    llvm.intr.assume %6 : i1
    %7 = arith.cmpi sge, %arg18, %c0_i32 : i32
    llvm.intr.assume %7 : i1
    %8 = arith.cmpi sge, %arg19, %c0_i32 : i32
    llvm.intr.assume %8 : i1
    %9 = arith.cmpi sge, %arg20, %c0_i32 : i32
    llvm.intr.assume %9 : i1
    %10 = arith.cmpi sge, %arg11, %c0_i32 : i32
    llvm.intr.assume %10 : i1
    %11 = arith.cmpi sge, %arg12, %c0_i32 : i32
    llvm.intr.assume %11 : i1
    %12 = arith.cmpi sge, %arg13, %c0_i32 : i32
    llvm.intr.assume %12 : i1
    llvm.intr.assume %true : i1
    %13 = arith.cmpi sge, %arg14, %c0_i32 : i32
    llvm.intr.assume %13 : i1
    %14 = arith.cmpi sge, %arg15, %c0_i32 : i32
    llvm.intr.assume %14 : i1
    %15 = arith.cmpi sge, %arg16, %c0_i32 : i32
    llvm.intr.assume %15 : i1
    llvm.intr.assume %true : i1
    %16 = tt.get_program_id x : i32
    %17 = tt.get_program_id y : i32
    %18 = tt.get_program_id z : i32
    %19 = arith.muli %16, %c128_i32 : i32
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked3>
    %21 = tt.splat %19 : i32 -> tensor<128xi32, #blocked3>
    %22 = arith.addi %21, %20 : tensor<128xi32, #blocked3>
    %23 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3>
    %24 = tt.addptr %arg23, %18 : !tt.ptr<i32>, i32
    %25 = tt.addptr %24, %c1_i32 : !tt.ptr<i32>, i32
    %26 = tt.addptr %arg24, %18 : !tt.ptr<i32>, i32
    %27 = tt.addptr %26, %c1_i32 : !tt.ptr<i32>, i32
    scf.while (%arg29 = %c0_i32) : (i32) -> () {
      %28 = arith.cmpi slt, %arg29, %c1_i32 : i32
      scf.condition(%28)
    } do {
      %28 = tt.load %24 : !tt.ptr<i32>
      %29 = tt.load %25 : !tt.ptr<i32>
      %30 = arith.subi %29, %28 : i32
      %31 = arith.cmpi sle, %19, %30 : i32
      %32 = tt.load %26 : !tt.ptr<i32>
      %33 = tt.load %27 : !tt.ptr<i32>
      %34 = arith.subi %33, %32 : i32
      scf.if %31 {
        %35 = arith.addi %34, %c63_i32 : i32
        %36 = arith.divsi %35, %c64_i32 : i32
        %37 = arith.addi %16, %c1_i32 : i32
        %38 = arith.muli %37, %c128_i32 : i32
        %39 = arith.addi %38, %34 : i32
        %40 = arith.subi %39, %30 : i32
        %41 = arith.addi %40, %c63_i32 : i32
        %42 = arith.divsi %41, %c64_i32 : i32
        %43 = arith.minsi %36, %42 : i32
        %44 = arith.cmpi sle, %43, %c0_i32 : i32
        %45 = arith.cmpi sgt, %43, %c0_i32 : i32
        scf.if %44 {
          %46 = arith.muli %18, %arg14 : i32
          %47 = tt.addptr %arg4, %46 : !tt.ptr<f16>, i32
          %48 = arith.muli %17, %arg15 : i32
          %49 = tt.addptr %47, %48 : !tt.ptr<f16>, i32
          %50 = arith.muli %28, %arg16 : i32
          %51 = tt.addptr %49, %50 : !tt.ptr<f16>, i32
          %52 = ttg.convert_layout %22 : tensor<128xi32, #blocked3> -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
          %53 = tt.expand_dims %52 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<128x1xi32, #blocked4>
          %54 = ttg.convert_layout %53 : tensor<128x1xi32, #blocked4> -> tensor<128x1xi32, #blocked2>
          %55 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #blocked2>
          %56 = arith.muli %54, %55 : tensor<128x1xi32, #blocked2>
          %57 = tt.splat %51 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
          %58 = tt.addptr %57, %56 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
          %59 = ttg.convert_layout %20 : tensor<128xi32, #blocked3> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
          %60 = tt.expand_dims %59 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x128xi32, #blocked5>
          %61 = ttg.convert_layout %60 : tensor<1x128xi32, #blocked5> -> tensor<1x128xi32, #blocked1>
          %62 = tt.broadcast %58 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x128x!tt.ptr<f16>, #blocked2>
          %63 = ttg.convert_layout %62 : tensor<128x128x!tt.ptr<f16>, #blocked2> -> tensor<128x128x!tt.ptr<f16>, #blocked1>
          %64 = tt.broadcast %61 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
          %65 = tt.addptr %63, %64 : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>
          %66 = tt.splat %30 : i32 -> tensor<128x1xi32, #blocked2>
          %67 = arith.cmpi slt, %54, %66 : tensor<128x1xi32, #blocked2>
          %68 = tt.broadcast %67 : tensor<128x1xi1, #blocked2> -> tensor<128x128xi1, #blocked2>
          %69 = ttg.convert_layout %68 : tensor<128x128xi1, #blocked2> -> tensor<128x128xi1, #blocked1>
          tt.store %65, %cst_7, %69 : tensor<128x128x!tt.ptr<f16>, #blocked1>
          %70 = arith.muli %18, %c262144_i32 : i32
          %71 = tt.addptr %arg3, %70 : !tt.ptr<f32>, i32
          %72 = arith.muli %17, %c8192_i32 : i32
          %73 = tt.addptr %71, %72 : !tt.ptr<f32>, i32
          %74 = tt.splat %73 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked3>
          %75 = tt.addptr %74, %22 : tensor<128x!tt.ptr<f32>, #blocked3>, tensor<128xi32, #blocked3>
          %76 = arith.cmpi slt, %22, %cst_12 : tensor<128xi32, #blocked3>
          tt.store %75, %cst_13, %76 : tensor<128x!tt.ptr<f32>, #blocked3>
        }
        scf.if %45 {
          %46 = arith.divsi %17, %c4_i32 : i32
          %47 = arith.cmpi slt, %34, %c64_i32 : i32
          %48 = scf.if %47 -> (i32) {
            %160 = arith.subi %c64_i32, %34 : i32
            scf.yield %160 : i32
          } else {
            %160 = arith.remsi %34, %c64_i32 : i32
            %161 = arith.cmpi ne, %160, %c0_i32 : i32
            %162 = scf.if %161 -> (i32) {
              scf.yield %160 : i32
            } else {
              scf.yield %c0_i32 : i32
            }
            scf.yield %162 : i32
          }
          %49 = arith.muli %18, %arg5 : i32
          %50 = tt.addptr %arg0, %49 : !tt.ptr<f16>, i32
          %51 = arith.muli %17, %arg6 : i32
          %52 = tt.addptr %50, %51 : !tt.ptr<f16>, i32
          %53 = arith.muli %28, %arg7 : i32
          %54 = tt.addptr %52, %53 : !tt.ptr<f16>, i32
          %55 = ttg.convert_layout %22 : tensor<128xi32, #blocked3> -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
          %56 = tt.expand_dims %55 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<128x1xi32, #blocked4>
          %57 = ttg.convert_layout %56 : tensor<128x1xi32, #blocked4> -> tensor<128x1xi32, #blocked2>
          %58 = tt.splat %arg7 : i32 -> tensor<128x1xi32, #blocked2>
          %59 = arith.muli %57, %58 : tensor<128x1xi32, #blocked2>
          %60 = tt.splat %54 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
          %61 = tt.addptr %60, %59 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
          %62 = ttg.convert_layout %20 : tensor<128xi32, #blocked3> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
          %63 = tt.expand_dims %62 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x128xi32, #blocked5>
          %64 = ttg.convert_layout %63 : tensor<1x128xi32, #blocked5> -> tensor<1x128xi32, #blocked1>
          %65 = tt.broadcast %61 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x128x!tt.ptr<f16>, #blocked2>
          %66 = ttg.convert_layout %65 : tensor<128x128x!tt.ptr<f16>, #blocked2> -> tensor<128x128x!tt.ptr<f16>, #blocked1>
          %67 = tt.broadcast %64 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
          %68 = tt.addptr %66, %67 : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>
          %69 = arith.muli %18, %arg8 : i32
          %70 = tt.addptr %arg1, %69 : !tt.ptr<f16>, i32
          %71 = arith.muli %46, %arg9 : i32
          %72 = tt.addptr %70, %71 : !tt.ptr<f16>, i32
          %73 = arith.muli %32, %arg10 : i32
          %74 = tt.addptr %72, %73 : !tt.ptr<f16>, i32
          %75 = ttg.convert_layout %20 : tensor<128xi32, #blocked3> -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
          %76 = tt.expand_dims %75 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<128x1xi32, #blocked4>
          %77 = ttg.convert_layout %76 : tensor<128x1xi32, #blocked4> -> tensor<128x1xi32, #blocked2>
          %78 = tt.splat %74 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
          %79 = tt.addptr %78, %77 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
          %80 = ttg.convert_layout %23 : tensor<64xi32, #blocked3> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
          %81 = tt.expand_dims %80 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x64xi32, #blocked5>
          %82 = ttg.convert_layout %81 : tensor<1x64xi32, #blocked5> -> tensor<1x64xi32, #blocked>
          %83 = tt.splat %arg10 : i32 -> tensor<1x64xi32, #blocked>
          %84 = arith.muli %82, %83 : tensor<1x64xi32, #blocked>
          %85 = tt.broadcast %79 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x64x!tt.ptr<f16>, #blocked2>
          %86 = ttg.convert_layout %85 : tensor<128x64x!tt.ptr<f16>, #blocked2> -> tensor<128x64x!tt.ptr<f16>, #blocked>
          %87 = tt.broadcast %84 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
          %88 = tt.addptr %86, %87 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
          %89 = arith.muli %18, %arg11 : i32
          %90 = tt.addptr %arg2, %89 : !tt.ptr<f16>, i32
          %91 = arith.muli %46, %arg12 : i32
          %92 = tt.addptr %90, %91 : !tt.ptr<f16>, i32
          %93 = arith.muli %32, %arg13 : i32
          %94 = tt.addptr %92, %93 : !tt.ptr<f16>, i32
          %95 = ttg.convert_layout %23 : tensor<64xi32, #blocked3> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
          %96 = tt.expand_dims %95 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<64x1xi32, #blocked4>
          %97 = ttg.convert_layout %96 : tensor<64x1xi32, #blocked4> -> tensor<64x1xi32, #blocked2>
          %98 = tt.splat %arg13 : i32 -> tensor<64x1xi32, #blocked2>
          %99 = arith.muli %97, %98 : tensor<64x1xi32, #blocked2>
          %100 = tt.splat %94 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked2>
          %101 = tt.addptr %100, %99 : tensor<64x1x!tt.ptr<f16>, #blocked2>, tensor<64x1xi32, #blocked2>
          %102 = tt.broadcast %101 : tensor<64x1x!tt.ptr<f16>, #blocked2> -> tensor<64x128x!tt.ptr<f16>, #blocked2>
          %103 = ttg.convert_layout %102 : tensor<64x128x!tt.ptr<f16>, #blocked2> -> tensor<64x128x!tt.ptr<f16>, #blocked1>
          %104 = tt.broadcast %64 : tensor<1x128xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
          %105 = tt.addptr %103, %104 : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
          %106 = tt.splat %30 : i32 -> tensor<128x1xi32, #blocked2>
          %107 = arith.cmpi slt, %57, %106 : tensor<128x1xi32, #blocked2>
          %108 = tt.broadcast %107 : tensor<128x1xi1, #blocked2> -> tensor<128x128xi1, #blocked2>
          %109 = ttg.convert_layout %108 : tensor<128x128xi1, #blocked2> -> tensor<128x128xi1, #blocked1>
          %110 = tt.load %68, %109, %cst_7 : tensor<128x128x!tt.ptr<f16>, #blocked1>
          %111 = arith.cmpi eq, %48, %c0_i32 : i32
          %112 = arith.remsi %30, %c128_i32 : i32
          %113 = arith.cmpi eq, %112, %c0_i32 : i32
          %114 = arith.andi %111, %113 : i1
          %115 = arith.xori %114, %true : i1
          %116 = arith.extui %115 : i1 to i32
          %117 = arith.addi %116, %c2_i32 : i32
          %118 = arith.minsi %117, %43 : i32
          %119 = arith.subi %43, %118 : i32
          %120 = arith.muli %43, %c64_i32 : i32
          %121 = arith.cmpi sgt, %119, %c0_i32 : i32
          %122:5 = scf.if %121 -> (tensor<128xf32, #blocked3>, tensor<128xf32, #blocked3>, tensor<128x128xf32, #blocked1>, i32, i32) {
            %160 = arith.muli %119, %c64_i32 : i32
            %161 = arith.muli %arg10, %c64_i32 : i32
            %162 = tt.splat %161 : i32 -> tensor<128x64xi32, #blocked>
            %163 = arith.muli %arg13, %c64_i32 : i32
            %164 = tt.splat %163 : i32 -> tensor<64x128xi32, #blocked1>
            %165:5 = scf.for %arg29 = %c0_i32 to %160 step %c64_i32 iter_args(%arg30 = %cst_6, %arg31 = %cst_10, %arg32 = %cst_11, %arg33 = %88, %arg34 = %105) -> (tensor<128x128xf32, #blocked1>, tensor<128xf32, #blocked3>, tensor<128xf32, #blocked3>, tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked1>)  : i32 {
              %166 = tt.load %arg33 : tensor<128x64x!tt.ptr<f16>, #blocked>
              %167 = ttg.convert_layout %110 : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked6}>>
              %168 = ttg.convert_layout %166 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked6}>>
              %169 = ttg.convert_layout %cst_4 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked6>
              %170 = tt.dot %167, %168, %169, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked6}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked6}>> -> tensor<128x64xf32, #blocked6>
              %171 = ttg.convert_layout %170 : tensor<128x64xf32, #blocked6> -> tensor<128x64xf32, #blocked>
              %172 = arith.mulf %171, %cst_5 : tensor<128x64xf32, #blocked>
              %173 = arith.addf %172, %cst_4 : tensor<128x64xf32, #blocked>
              %174 = "tt.reduce"(%173) <{axis = 1 : i32}> ({
              ^bb0(%arg35: f32, %arg36: f32):
                %205 = arith.maxnumf %arg35, %arg36 : f32
                tt.reduce.return %205 : f32
              }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
              %175 = ttg.convert_layout %174 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked3>
              %176 = arith.maxnumf %arg32, %175 : tensor<128xf32, #blocked3>
              %177 = ttg.convert_layout %176 : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked4}>>
              %178 = tt.expand_dims %177 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<128x1xf32, #blocked4>
              %179 = ttg.convert_layout %178 : tensor<128x1xf32, #blocked4> -> tensor<128x1xf32, #blocked2>
              %180 = tt.broadcast %179 : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
              %181 = ttg.convert_layout %180 : tensor<128x64xf32, #blocked2> -> tensor<128x64xf32, #blocked>
              %182 = arith.subf %173, %181 : tensor<128x64xf32, #blocked>
              %183 = math.exp2 %182 : tensor<128x64xf32, #blocked>
              %184 = "tt.reduce"(%183) <{axis = 1 : i32}> ({
              ^bb0(%arg35: f32, %arg36: f32):
                %205 = arith.addf %arg35, %arg36 : f32
                tt.reduce.return %205 : f32
              }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
              %185 = ttg.convert_layout %184 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked3>
              %186 = arith.subf %arg32, %176 : tensor<128xf32, #blocked3>
              %187 = math.exp2 %186 : tensor<128xf32, #blocked3>
              %188 = ttg.convert_layout %187 : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked4}>>
              %189 = tt.expand_dims %188 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<128x1xf32, #blocked4>
              %190 = ttg.convert_layout %189 : tensor<128x1xf32, #blocked4> -> tensor<128x1xf32, #blocked2>
              %191 = tt.broadcast %190 : tensor<128x1xf32, #blocked2> -> tensor<128x128xf32, #blocked2>
              %192 = ttg.convert_layout %191 : tensor<128x128xf32, #blocked2> -> tensor<128x128xf32, #blocked1>
              %193 = arith.mulf %arg30, %192 : tensor<128x128xf32, #blocked1>
              %194 = tt.load %arg34 : tensor<64x128x!tt.ptr<f16>, #blocked1>
              %195 = arith.mulf %arg31, %187 : tensor<128xf32, #blocked3>
              %196 = arith.addf %195, %185 : tensor<128xf32, #blocked3>
              %197 = arith.truncf %183 : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
              %198 = ttg.convert_layout %197 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked7}>>
              %199 = ttg.convert_layout %194 : tensor<64x128xf16, #blocked1> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked7}>>
              %200 = ttg.convert_layout %193 : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #blocked7>
              %201 = tt.dot %198, %199, %200, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked7}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked7}>> -> tensor<128x128xf32, #blocked7>
              %202 = ttg.convert_layout %201 : tensor<128x128xf32, #blocked7> -> tensor<128x128xf32, #blocked1>
              %203 = tt.addptr %arg33, %162 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
              %204 = tt.addptr %arg34, %164 : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
              scf.yield %202, %196, %176, %203, %204 : tensor<128x128xf32, #blocked1>, tensor<128xf32, #blocked3>, tensor<128xf32, #blocked3>, tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked1>
            }
            scf.yield %165#2, %165#1, %165#0, %160, %120 : tensor<128xf32, #blocked3>, tensor<128xf32, #blocked3>, tensor<128x128xf32, #blocked1>, i32, i32
          } else {
            scf.yield %cst_11, %cst_10, %cst_6, %c0_i32, %120 : tensor<128xf32, #blocked3>, tensor<128xf32, #blocked3>, tensor<128x128xf32, #blocked1>, i32, i32
          }
          gpu.barrier
          %123 = arith.cmpi sgt, %118, %c0_i32 : i32
          %124:3 = scf.if %123 -> (tensor<128xf32, #blocked3>, tensor<128xf32, #blocked3>, tensor<128x128xf32, #blocked1>) {
            %160 = arith.subi %30, %34 : i32
            %161 = tt.splat %160 : i32 -> tensor<64xi32, #blocked3>
            %162 = arith.addi %23, %161 : tensor<64xi32, #blocked3>
            %163 = arith.muli %119, %c64_i32 : i32
            %164 = arith.muli %163, %arg10 : i32
            %165 = tt.splat %164 : i32 -> tensor<128x64xi32, #blocked>
            %166 = tt.addptr %88, %165 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
            %167 = arith.muli %163, %arg13 : i32
            %168 = tt.splat %167 : i32 -> tensor<64x128xi32, #blocked1>
            %169 = tt.addptr %105, %168 : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
            %170 = tt.splat %34 : i32 -> tensor<1x64xi32, #blocked>
            %171 = arith.cmpi ne, %48, %c0_i32 : i32
            %172 = tt.broadcast %57 : tensor<128x1xi32, #blocked2> -> tensor<128x64xi32, #blocked2>
            %173 = ttg.convert_layout %172 : tensor<128x64xi32, #blocked2> -> tensor<128x64xi32, #blocked>
            %174 = tt.splat %34 : i32 -> tensor<64x1xi32, #blocked2>
            %175 = arith.muli %arg10, %c64_i32 : i32
            %176 = tt.splat %175 : i32 -> tensor<128x64xi32, #blocked>
            %177 = arith.muli %arg13, %c64_i32 : i32
            %178 = tt.splat %177 : i32 -> tensor<64x128xi32, #blocked1>
            %179:5 = scf.for %arg29 = %122#3 to %122#4 step %c64_i32 iter_args(%arg30 = %122#2, %arg31 = %122#1, %arg32 = %122#0, %arg33 = %166, %arg34 = %169) -> (tensor<128x128xf32, #blocked1>, tensor<128xf32, #blocked3>, tensor<128xf32, #blocked3>, tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked1>)  : i32 {
              %180 = tt.splat %arg29 : i32 -> tensor<64xi32, #blocked3>
              %181 = arith.addi %180, %23 : tensor<64xi32, #blocked3>
              %182 = ttg.convert_layout %181 : tensor<64xi32, #blocked3> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
              %183 = tt.expand_dims %182 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x64xi32, #blocked5>
              %184 = ttg.convert_layout %183 : tensor<1x64xi32, #blocked5> -> tensor<1x64xi32, #blocked>
              %185 = arith.cmpi slt, %184, %170 : tensor<1x64xi32, #blocked>
              %186 = tt.broadcast %185 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
              %187 = tt.load %arg33, %186, %cst_2 : tensor<128x64x!tt.ptr<f16>, #blocked>
              %188 = arith.addi %arg29, %c64_i32 : i32
              %189 = arith.cmpi eq, %188, %122#4 : i32
              %190 = arith.andi %189, %171 : i1
              %191 = scf.if %190 -> (tensor<128x64xf32, #blocked>) {
                %243 = tt.splat %arg29 : i32 -> tensor<1x64xi32, #blocked>
                %244 = arith.addi %243, %82 : tensor<1x64xi32, #blocked>
                %245 = arith.cmpi slt, %244, %170 : tensor<1x64xi32, #blocked>
                %246 = arith.select %245, %cst_0, %cst : tensor<1x64xi1, #blocked>, tensor<1x64xf32, #blocked>
                %247 = tt.broadcast %246 : tensor<1x64xf32, #blocked> -> tensor<128x64xf32, #blocked>
                scf.yield %247 : tensor<128x64xf32, #blocked>
              } else {
                scf.yield %cst_4 : tensor<128x64xf32, #blocked>
              }
              %192 = arith.addi %180, %162 : tensor<64xi32, #blocked3>
              %193 = ttg.convert_layout %192 : tensor<64xi32, #blocked3> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
              %194 = tt.expand_dims %193 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x64xi32, #blocked5>
              %195 = ttg.convert_layout %194 : tensor<1x64xi32, #blocked5> -> tensor<1x64xi32, #blocked>
              %196 = tt.broadcast %195 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
              %197 = arith.cmpi sge, %173, %196 : tensor<128x64xi32, #blocked>
              %198 = arith.select %197, %191, %cst_1 : tensor<128x64xi1, #blocked>, tensor<128x64xf32, #blocked>
              %199 = ttg.convert_layout %110 : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked6}>>
              %200 = ttg.convert_layout %187 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked6}>>
              %201 = ttg.convert_layout %cst_4 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked6>
              %202 = tt.dot %199, %200, %201, inputPrecision = tf32 : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked6}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked6}>> -> tensor<128x64xf32, #blocked6>
              %203 = ttg.convert_layout %202 : tensor<128x64xf32, #blocked6> -> tensor<128x64xf32, #blocked>
              %204 = arith.mulf %203, %cst_5 : tensor<128x64xf32, #blocked>
              %205 = arith.addf %198, %204 : tensor<128x64xf32, #blocked>
              %206 = "tt.reduce"(%205) <{axis = 1 : i32}> ({
              ^bb0(%arg35: f32, %arg36: f32):
                %243 = arith.maxnumf %arg35, %arg36 : f32
                tt.reduce.return %243 : f32
              }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
              %207 = ttg.convert_layout %206 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked3>
              %208 = arith.maxnumf %arg32, %207 : tensor<128xf32, #blocked3>
              %209 = ttg.convert_layout %208 : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked4}>>
              %210 = tt.expand_dims %209 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<128x1xf32, #blocked4>
              %211 = ttg.convert_layout %210 : tensor<128x1xf32, #blocked4> -> tensor<128x1xf32, #blocked2>
              %212 = tt.broadcast %211 : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
              %213 = ttg.convert_layout %212 : tensor<128x64xf32, #blocked2> -> tensor<128x64xf32, #blocked>
              %214 = arith.subf %205, %213 : tensor<128x64xf32, #blocked>
              %215 = math.exp2 %214 : tensor<128x64xf32, #blocked>
              %216 = "tt.reduce"(%215) <{axis = 1 : i32}> ({
              ^bb0(%arg35: f32, %arg36: f32):
                %243 = arith.addf %arg35, %arg36 : f32
                tt.reduce.return %243 : f32
              }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
              %217 = ttg.convert_layout %216 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked3>
              %218 = arith.subf %arg32, %208 : tensor<128xf32, #blocked3>
              %219 = math.exp2 %218 : tensor<128xf32, #blocked3>
              %220 = ttg.convert_layout %219 : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked4}>>
              %221 = tt.expand_dims %220 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<128x1xf32, #blocked4>
              %222 = ttg.convert_layout %221 : tensor<128x1xf32, #blocked4> -> tensor<128x1xf32, #blocked2>
              %223 = tt.broadcast %222 : tensor<128x1xf32, #blocked2> -> tensor<128x128xf32, #blocked2>
              %224 = ttg.convert_layout %223 : tensor<128x128xf32, #blocked2> -> tensor<128x128xf32, #blocked1>
              %225 = arith.mulf %arg30, %224 : tensor<128x128xf32, #blocked1>
              %226 = ttg.convert_layout %181 : tensor<64xi32, #blocked3> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
              %227 = tt.expand_dims %226 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<64x1xi32, #blocked4>
              %228 = ttg.convert_layout %227 : tensor<64x1xi32, #blocked4> -> tensor<64x1xi32, #blocked2>
              %229 = arith.cmpi slt, %228, %174 : tensor<64x1xi32, #blocked2>
              %230 = tt.broadcast %229 : tensor<64x1xi1, #blocked2> -> tensor<64x128xi1, #blocked2>
              %231 = ttg.convert_layout %230 : tensor<64x128xi1, #blocked2> -> tensor<64x128xi1, #blocked1>
              %232 = tt.load %arg34, %231, %cst_3 : tensor<64x128x!tt.ptr<f16>, #blocked1>
              %233 = arith.mulf %arg31, %219 : tensor<128xf32, #blocked3>
              %234 = arith.addf %233, %217 : tensor<128xf32, #blocked3>
              %235 = arith.truncf %215 : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
              %236 = ttg.convert_layout %235 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked7}>>
              %237 = ttg.convert_layout %232 : tensor<64x128xf16, #blocked1> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked7}>>
              %238 = ttg.convert_layout %225 : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #blocked7>
              %239 = tt.dot %236, %237, %238, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked7}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked7}>> -> tensor<128x128xf32, #blocked7>
              %240 = ttg.convert_layout %239 : tensor<128x128xf32, #blocked7> -> tensor<128x128xf32, #blocked1>
              %241 = tt.addptr %arg33, %176 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
              %242 = tt.addptr %arg34, %178 : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
              scf.yield %240, %234, %208, %241, %242 : tensor<128x128xf32, #blocked1>, tensor<128xf32, #blocked3>, tensor<128xf32, #blocked3>, tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked1>
            }
            scf.yield %179#2, %179#1, %179#0 : tensor<128xf32, #blocked3>, tensor<128xf32, #blocked3>, tensor<128x128xf32, #blocked1>
          } else {
            scf.yield %122#0, %122#1, %122#2 : tensor<128xf32, #blocked3>, tensor<128xf32, #blocked3>, tensor<128x128xf32, #blocked1>
          }
          %125 = ttg.convert_layout %124#1 : tensor<128xf32, #blocked3> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked4}>>
          %126 = tt.expand_dims %125 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<128x1xf32, #blocked4>
          %127 = ttg.convert_layout %126 : tensor<128x1xf32, #blocked4> -> tensor<128x1xf32, #blocked2>
          %128 = arith.divf %cst_9, %127 : tensor<128x1xf32, #blocked2>
          %129 = tt.broadcast %128 : tensor<128x1xf32, #blocked2> -> tensor<128x128xf32, #blocked2>
          %130 = ttg.convert_layout %129 : tensor<128x128xf32, #blocked2> -> tensor<128x128xf32, #blocked1>
          %131 = arith.mulf %124#2, %130 : tensor<128x128xf32, #blocked1>
          %132 = arith.subi %30, %34 : i32
          %133 = arith.truncf %131 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
          %134 = arith.cmpi sgt, %132, %19 : i32
          %135 = arith.cmpi slt, %132, %38 : i32
          %136 = arith.andi %134, %135 : i1
          %137 = scf.if %136 -> (tensor<128x128xf16, #blocked1>) {
            %160 = tt.splat %132 : i32 -> tensor<128x1xi32, #blocked2>
            %161 = arith.cmpi sge, %57, %160 : tensor<128x1xi32, #blocked2>
            %162 = tt.broadcast %161 : tensor<128x1xi1, #blocked2> -> tensor<128x128xi1, #blocked2>
            %163 = ttg.convert_layout %162 : tensor<128x128xi1, #blocked2> -> tensor<128x128xi1, #blocked1>
            %164 = arith.select %163, %133, %cst_7 : tensor<128x128xi1, #blocked1>, tensor<128x128xf16, #blocked1>
            scf.yield %164 : tensor<128x128xf16, #blocked1>
          } else {
            scf.yield %133 : tensor<128x128xf16, #blocked1>
          }
          %138 = arith.muli %18, %c262144_i32 : i32
          %139 = tt.addptr %arg3, %138 : !tt.ptr<f32>, i32
          %140 = arith.muli %17, %c8192_i32 : i32
          %141 = tt.addptr %139, %140 : !tt.ptr<f32>, i32
          %142 = tt.splat %141 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked3>
          %143 = tt.addptr %142, %22 : tensor<128x!tt.ptr<f32>, #blocked3>, tensor<128xi32, #blocked3>
          %144 = arith.subi %38, %30 : i32
          %145 = arith.cmpi sgt, %144, %c0_i32 : i32
          scf.if %145 {
            %160 = arith.subi %c128_i32, %144 : i32
            %161 = tt.splat %160 : i32 -> tensor<128xi32, #blocked3>
            %162 = arith.cmpi slt, %20, %161 : tensor<128xi32, #blocked3>
            %163 = math.log2 %124#1 : tensor<128xf32, #blocked3>
            %164 = arith.addf %124#0, %163 : tensor<128xf32, #blocked3>
            tt.store %143, %164, %162 : tensor<128x!tt.ptr<f32>, #blocked3>
          } else {
            %160 = math.log2 %124#1 : tensor<128xf32, #blocked3>
            %161 = arith.addf %124#0, %160 : tensor<128xf32, #blocked3>
            tt.store %143, %161 : tensor<128x!tt.ptr<f32>, #blocked3>
          }
          %146 = arith.muli %18, %arg14 : i32
          %147 = tt.addptr %arg4, %146 : !tt.ptr<f16>, i32
          %148 = arith.muli %17, %arg15 : i32
          %149 = tt.addptr %147, %148 : !tt.ptr<f16>, i32
          %150 = arith.muli %28, %arg16 : i32
          %151 = tt.addptr %149, %150 : !tt.ptr<f16>, i32
          %152 = tt.splat %arg16 : i32 -> tensor<128x1xi32, #blocked2>
          %153 = arith.muli %57, %152 : tensor<128x1xi32, #blocked2>
          %154 = tt.splat %151 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
          %155 = tt.addptr %154, %153 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
          %156 = tt.broadcast %155 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x128x!tt.ptr<f16>, #blocked2>
          %157 = ttg.convert_layout %156 : tensor<128x128x!tt.ptr<f16>, #blocked2> -> tensor<128x128x!tt.ptr<f16>, #blocked1>
          %158 = tt.addptr %157, %67 : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>
          %159 = scf.if %145 -> (tensor<128x128xi1, #blocked1>) {
            scf.yield %109 : tensor<128x128xi1, #blocked1>
          } else {
            scf.yield %cst_8 : tensor<128x128xi1, #blocked1>
          }
          tt.store %158, %137, %159 : tensor<128x128x!tt.ptr<f16>, #blocked1>
        }
      }
      scf.yield %c1_i32 : i32
    }
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(tritongpu-coalesce, tritongpu-remove-layout-conversions, tritongpu-optimize-thread-locality, tritonamdgpu-accelerate-matmul{arch-generation-name=gfx942 kPack=1 matrix-instruction-size=0}, tritongpu-remove-layout-conversions, tritonamdgpu-optimize-epilogue, tritongpu-optimize-dot-operands{hoist-layout-conversion=true}, tt.func(tritonamdgpu-hoist-layout-conversions), tritongpu-fuse-nested-loops, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, triton-licm, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, tritonamdgpu-stream-pipeline{global_prefetch=0 local_prefetch=0 num_stages=2 use_async_copy=false}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, tritongpu-optimize-dot-operands{hoist-layout-conversion=true}, tritongpu-remove-layout-conversions, tritongpu-reduce-data-duplication, tt.func(tritonamdgpu-in-thread-transpose), tritongpu-remove-layout-conversions, tritonamdgpu-reorder-instructions, tritonamdgpu-block-pingpong{num-stages=2}, tt.func(tritonamdgpu-canonicalize-pointers), canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, tritonamdgpu-convert-buffer-ops{arch-generation-name=gfx942}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, symbol-dce)",
      disable_threading: false,
      verify_each: true
    }
  }
#-}
