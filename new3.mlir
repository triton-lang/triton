#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_0 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c132_i32 = arith.constant 132 : i32
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %6 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = arith.addi %arg3, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.addi %arg4, %c255_i32 : i32
    %10 = arith.divsi %9, %c256_i32 : i32
    %11 = arith.addi %arg5, %c63_i32 : i32
    %12 = arith.divsi %11, %c64_i32 : i32
    %13 = arith.muli %8, %10 : i32
    %14 = arith.muli %10, %c8_i32 : i32
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %17 = arith.subi %13, %0 : i32
    %18 = arith.ceildivsi %17, %c132_i32 : i32
    %19 = arith.maxsi %12, %c1_i32 : i32
    %20 = arith.muli %18, %19 : i32
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %22 = ttg.local_alloc  : () -> !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable>
    %23 = arith.cmpi sgt, %20, %c0_i32 : i32
    %24 = arith.divsi %0, %14 : i32
    %25 = arith.muli %24, %c8_i32 : i32
    %26 = arith.subi %8, %25 : i32
    %27 = arith.minsi %26, %c8_i32 : i32
    %28 = arith.remsi %0, %27 : i32
    %29 = arith.addi %25, %28 : i32
    %30 = arith.remsi %0, %14 : i32
    %31 = arith.divsi %30, %27 : i32
    %32 = arith.muli %29, %c128_i32 : i32
    %33 = arith.muli %31, %c256_i32 : i32
    %34 = tt.splat %32 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %35 = arith.addi %34, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %36 = tt.splat %33 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %37 = arith.addi %36, %4 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %38 = arith.cmpi slt, %35, %5 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %39 = arith.select %38, %35, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %40 = arith.cmpi slt, %37, %6 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %41 = arith.select %40, %37, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %42 = tt.expand_dims %39 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %43 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %44 = arith.muli %42, %43 : tensor<128x1xi32, #blocked1>
    %45 = tt.expand_dims %15 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %46 = tt.broadcast %44 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %47 = tt.broadcast %45 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %48 = arith.addi %46, %47 : tensor<128x64xi32, #blocked1>
    %49 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %50 = tt.addptr %49, %48 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %51 = tt.expand_dims %16 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %52 = tt.expand_dims %41 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %53 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked>
    %54 = arith.muli %52, %53 : tensor<1x256xi32, #blocked>
    %55 = tt.broadcast %51 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %56 = tt.broadcast %54 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %57 = arith.addi %55, %56 : tensor<64x256xi32, #blocked>
    %58 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked>
    %59 = tt.addptr %58, %57 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %60 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1>
    %61 = arith.cmpi slt, %45, %60 : tensor<1x64xi32, #blocked1>
    %62 = tt.broadcast %61 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %63 = ttg.memdesc_subview %21[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %64 = tt.splat %23 : i1 -> tensor<128x64xi1, #blocked1>
    %65 = arith.andi %64, %62 : tensor<128x64xi1, #blocked1>
    %66 = ttg.async_copy_global_to_local %50, %63 mask %65 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %67 = ttg.async_commit_group %66
    %68 = tt.splat %arg5 : i32 -> tensor<64x1xi32, #blocked>
    %69 = arith.cmpi slt, %51, %68 : tensor<64x1xi32, #blocked>
    %70 = tt.broadcast %69 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %71 = ttg.memdesc_subview %22[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    %72 = tt.splat %23 : i1 -> tensor<64x256xi1, #blocked>
    %73 = arith.andi %72, %70 : tensor<64x256xi1, #blocked>
    %74 = ttg.async_copy_global_to_local %59, %71 mask %73 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
    %75 = ttg.async_commit_group %74
    %76 = arith.cmpi sgt, %20, %c1_i32 : i32
    %77 = arith.remsi %c1_i32, %19 : i32
    %78 = arith.cmpi eq, %77, %c0_i32 : i32
    %79 = arith.cmpi ne, %77, %c0_i32 : i32
    %80 = arith.extui %79 : i1 to i32
    %81:5 = scf.if %78 -> (i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32) {
      %121 = arith.addi %0, %c132_i32 : i32
      %122 = arith.divsi %121, %14 : i32
      %123 = arith.muli %122, %c8_i32 : i32
      %124 = arith.subi %8, %123 : i32
      %125 = arith.minsi %124, %c8_i32 : i32
      %126 = arith.remsi %121, %125 : i32
      %127 = arith.addi %123, %126 : i32
      %128 = arith.remsi %121, %14 : i32
      %129 = arith.divsi %128, %125 : i32
      %130 = arith.muli %127, %c128_i32 : i32
      %131 = arith.muli %129, %c256_i32 : i32
      %132 = tt.splat %130 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %133 = arith.addi %132, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %134 = tt.splat %131 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %135 = arith.addi %134, %4 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %136 = arith.cmpi slt, %133, %5 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %137 = arith.select %136, %133, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %138 = arith.cmpi slt, %135, %6 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %139 = arith.select %138, %135, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      scf.yield %130, %131, %137, %139, %121 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32
    } else {
      scf.yield %32, %33, %39, %41, %0 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32
    }
    %82 = arith.muli %80, %c64_i32 : i32
    %83 = tt.splat %82 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %84 = tt.splat %82 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %85 = arith.addi %83, %15 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %86 = arith.addi %84, %16 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %87 = tt.expand_dims %81#2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %88 = arith.muli %87, %43 : tensor<128x1xi32, #blocked1>
    %89 = tt.expand_dims %85 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %90 = tt.broadcast %88 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %91 = tt.broadcast %89 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %92 = arith.addi %90, %91 : tensor<128x64xi32, #blocked1>
    %93 = tt.addptr %49, %92 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %94 = tt.expand_dims %86 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %95 = tt.expand_dims %81#3 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %96 = arith.muli %95, %53 : tensor<1x256xi32, #blocked>
    %97 = tt.broadcast %94 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %98 = tt.broadcast %96 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %99 = arith.addi %97, %98 : tensor<64x256xi32, #blocked>
    %100 = tt.addptr %58, %99 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %101 = arith.subi %arg5, %82 : i32
    %102 = tt.splat %101 : i32 -> tensor<1x64xi32, #blocked1>
    %103 = arith.cmpi slt, %45, %102 : tensor<1x64xi32, #blocked1>
    %104 = tt.broadcast %103 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %105 = ttg.memdesc_subview %21[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %106 = tt.splat %76 : i1 -> tensor<128x64xi1, #blocked1>
    %107 = arith.andi %106, %104 : tensor<128x64xi1, #blocked1>
    %108 = ttg.async_copy_global_to_local %93, %105 mask %107 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %109 = ttg.async_commit_group %108
    %110 = tt.splat %101 : i32 -> tensor<64x1xi32, #blocked>
    %111 = arith.cmpi slt, %51, %110 : tensor<64x1xi32, #blocked>
    %112 = tt.broadcast %111 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %113 = ttg.memdesc_subview %22[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    %114 = tt.splat %76 : i1 -> tensor<64x256xi1, #blocked>
    %115 = arith.andi %114, %112 : tensor<64x256xi1, #blocked>
    %116 = ttg.async_copy_global_to_local %100, %113 mask %115 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
    %117 = ttg.async_commit_group %116
    %lol = arith.subi %12, %c1_i32 : i32
    %118:16 = scf.for %arg9 = %c0_i32 to %20 step %c1_i32 iter_args(
      %arg10 = %81#4, %arg11 = %cst_3, %arg12 = %81#0, %arg13 = %81#1,
      %arg14 = %81#2, %arg15 = %81#3, %arg16 = %c1_i32, %arg17 = %c-1_i32,
      %arg18 = %80, %arg19 = %c0_i32, %arg21 = %75, %arg22 = %117, %arg23 = %32, %arg24 = %81#0, %arg25 = %33, %arg26 = %81#1) -> (i32, tensor<128x256xf32, #mma>, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, i32, i32, i32, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32)  : i32 {
      %121 = arith.subi %20, %c2_i32 : i32
      %122 = arith.cmpi slt, %arg9, %121 : i32
      %rollover = arith.cmpi eq, %arg18, %lol : i32
      %123 = arith.addi %arg18, %c1_i32 : i32
      %126 = arith.select %rollover, %c0_i32, %123 : i32
      %125 = arith.cmpi eq, %126, %c0_i32 : i32
      %127:5 = scf.if %125 -> (i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32) {
        %178 = arith.addi %arg10, %c132_i32 : i32
        %179 = arith.divsi %178, %14 : i32
        %180 = arith.muli %179, %c8_i32 : i32
        %181 = arith.subi %8, %180 : i32
        %182 = arith.minsi %181, %c8_i32 : i32
        %183 = arith.remsi %178, %182 : i32
        %184 = arith.addi %180, %183 : i32
        %185 = arith.remsi %178, %14 : i32
        %186 = arith.divsi %185, %182 : i32
        %187 = arith.muli %184, %c128_i32 : i32
        %188 = arith.muli %186, %c256_i32 : i32
        %189 = tt.splat %187 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %190 = arith.addi %189, %3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %191 = tt.splat %188 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %192 = arith.addi %191, %4 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %193 = arith.cmpi slt, %190, %5 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %194 = arith.select %193, %190, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %195 = arith.cmpi slt, %192, %6 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %196 = arith.select %195, %192, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        scf.yield %187, %188, %194, %196, %178 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32
      } else {
        scf.yield %arg12, %arg13, %arg14, %arg15, %arg10 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32
      }
      %128 = arith.addi %arg17, %c1_i32 : i32
      %129 = arith.cmpi slt, %128, %c3_i32 : i32
      %130 = arith.select %129, %128, %c0_i32 : i32
      %131 = arith.cmpi ne, %arg19, %c0_i32 : i32
      %132 = ttg.memdesc_subview %21[%130, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %133 = ttg.async_wait %arg21 {num = 2 : i32}
      %134 = ttg.memdesc_subview %22[%130, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %135 = ttng.warp_group_dot %132, %134, %arg11, %131 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> * !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> tensor<128x256xf32, #mma>
      %136:3 = ttng.warp_group_dot_wait %135, %132, %134 {pendings = 1 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %137 = arith.addi %arg16, %c1_i32 : i32
      %138 = arith.cmpi slt, %137, %c3_i32 : i32
      %139 = arith.select %138, %137, %c0_i32 : i32
      %140 = arith.muli %126, %c64_i32 : i32
      %141 = tt.splat %140 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %142 = tt.splat %140 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %143 = arith.addi %141, %15 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %144 = arith.addi %142, %16 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %145 = tt.expand_dims %127#2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %146 = arith.muli %145, %43 : tensor<128x1xi32, #blocked1>
      %147 = tt.expand_dims %143 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
      %148 = tt.broadcast %146 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %149 = tt.broadcast %147 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %150 = arith.addi %148, %149 : tensor<128x64xi32, #blocked1>
      %151 = tt.addptr %49, %150 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %152 = tt.expand_dims %144 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %153 = tt.expand_dims %127#3 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
      %154 = arith.muli %153, %53 : tensor<1x256xi32, #blocked>
      %155 = tt.broadcast %152 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %156 = tt.broadcast %154 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %157 = arith.addi %155, %156 : tensor<64x256xi32, #blocked>
      %158 = tt.addptr %58, %157 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
      %159 = arith.subi %arg5, %140 : i32
      %160 = tt.splat %159 : i32 -> tensor<1x64xi32, #blocked1>
      %161 = arith.cmpi slt, %45, %160 : tensor<1x64xi32, #blocked1>
      %162 = tt.broadcast %161 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
      %163 = ttg.memdesc_subview %21[%139, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %164 = tt.splat %122 : i1 -> tensor<128x64xi1, #blocked1>
      %165 = arith.andi %164, %162 : tensor<128x64xi1, #blocked1>
      %166 = ttg.async_copy_global_to_local %151, %163 mask %165 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
      %167 = ttg.async_commit_group %166
      %168 = tt.splat %159 : i32 -> tensor<64x1xi32, #blocked>
      %169 = arith.cmpi slt, %51, %168 : tensor<64x1xi32, #blocked>
      %170 = tt.broadcast %169 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
      %171 = ttg.memdesc_subview %22[%139, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %172 = tt.splat %122 : i1 -> tensor<64x256xi1, #blocked>
      %173 = arith.andi %172, %170 : tensor<64x256xi1, #blocked>
      %174 = ttg.async_copy_global_to_local %158, %171 mask %173 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
      %175 = ttg.async_commit_group %174
      %176 = arith.subi %19, %c1_i32 : i32
      %177 = arith.cmpi eq, %arg19, %176 : i32
      scf.if %177 {
        %178:3 = ttng.warp_group_dot_wait %136#0, %132, %134 {pendings = 0 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
        %179 = tt.splat %arg23 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %180 = arith.addi %179, %1 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %181 = tt.splat %arg25 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %182 = arith.addi %181, %2 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %183 = tt.expand_dims %180 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
        %184 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked2>
        %185 = arith.muli %184, %183 : tensor<128x1xi32, #blocked2>
        %186 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
        %187 = tt.addptr %186, %185 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
        %188 = tt.expand_dims %182 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
        %189 = tt.broadcast %187 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x256x!tt.ptr<f16>, #blocked2>
        %190 = tt.broadcast %188 : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
        %191 = tt.addptr %189, %190 : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
        %192 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked2>
        %193 = arith.cmpi slt, %183, %192 : tensor<128x1xi32, #blocked2>
        %194 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked2>
        %195 = arith.cmpi slt, %188, %194 : tensor<1x256xi32, #blocked2>
        %196 = tt.broadcast %193 : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
        %197 = tt.broadcast %195 : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
        %198 = arith.andi %196, %197 : tensor<128x256xi1, #blocked2>
        %199 = arith.truncf %178#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
        %200 = ttg.convert_layout %199 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked2>
        tt.store %191, %200, %198 : tensor<128x256x!tt.ptr<f16>, #blocked2>
      }
      scf.yield %127#4, %136#0, %127#0, %127#1,
                 %127#2, %127#3, %139, %130,
                 %126, %arg18, %arg22, %175, %arg24, %127#0, %arg26, %127#1 : i32, tensor<128x256xf32, #mma>, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, i32, i32, i32, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32
    }
    %119 = ttng.warp_group_dot_wait %118#1 {pendings = 0 : i32} : tensor<128x256xf32, #mma>
    %120 = ttg.async_wait  {num = 0 : i32}
    ttg.local_dealloc %21 : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable>
    tt.return
  }
}

