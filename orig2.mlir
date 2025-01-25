#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_persistent_fused(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %false = arith.constant false
    %cst = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_0 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c132_i32 = arith.constant 132 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.addi %arg5, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.divsi %7, %c132_i32 : i32
    %9 = arith.remsi %7, %c132_i32 : i32
    %10 = arith.cmpi slt, %0, %9 : i32
    %11 = scf.if %10 -> (i32) {
      %122 = arith.addi %8, %c1_i32 : i32
      scf.yield %122 : i32
    } else {
      scf.yield %8 : i32
    }
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = arith.muli %4, %c8_i32 : i32
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %17 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %18 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %19 = arith.muli %6, %11 : i32
    %20 = ttg.local_alloc  : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %21 = ttg.local_alloc  : () -> !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable>
    %22 = arith.cmpi sgt, %19, %c0_i32 : i32
    %23 = arith.subi %6, %c1_i32 : i32
    %24 = arith.divsi %0, %14 : i32
    %25 = arith.muli %24, %c8_i32 : i32
    %26 = arith.subi %2, %25 : i32
    %27 = arith.minsi %26, %c8_i32 : i32
    %28 = arith.remsi %0, %27 : i32
    %29 = arith.addi %25, %28 : i32
    %30 = arith.remsi %0, %14 : i32
    %31 = arith.divsi %30, %27 : i32
    %32 = arith.muli %29, %c128_i32 : i32
    %33 = arith.muli %31, %c256_i32 : i32
    %34 = tt.splat %32 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %35 = arith.addi %34, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %36 = tt.splat %33 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %37 = arith.addi %36, %17 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %38 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %39 = arith.cmpi slt, %35, %38 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %40 = arith.select %39, %35, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %41 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %42 = arith.cmpi slt, %37, %41 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %43 = arith.select %42, %37, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %44 = tt.expand_dims %40 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %45 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %46 = arith.muli %44, %45 : tensor<128x1xi32, #blocked1>
    %47 = tt.expand_dims %12 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %48 = tt.broadcast %46 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %49 = tt.broadcast %47 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %50 = arith.addi %48, %49 : tensor<128x64xi32, #blocked1>
    %51 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %52 = tt.addptr %51, %50 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %53 = tt.expand_dims %13 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %54 = tt.expand_dims %43 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %55 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked>
    %56 = arith.muli %54, %55 : tensor<1x256xi32, #blocked>
    %57 = tt.broadcast %53 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %58 = tt.broadcast %56 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %59 = arith.addi %57, %58 : tensor<64x256xi32, #blocked>
    %60 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked>
    %61 = tt.addptr %60, %59 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %62 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1>
    %63 = arith.cmpi slt, %47, %62 : tensor<1x64xi32, #blocked1>
    %64 = tt.broadcast %63 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %65 = ttg.memdesc_subview %20[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %66 = tt.splat %22 : i1 -> tensor<128x64xi1, #blocked1>
    %67 = arith.andi %66, %64 : tensor<128x64xi1, #blocked1>
    %68 = ttg.async_copy_global_to_local %52, %65 mask %67 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %69 = ttg.async_commit_group %68
    %70 = tt.splat %arg5 : i32 -> tensor<64x1xi32, #blocked>
    %71 = arith.cmpi slt, %53, %70 : tensor<64x1xi32, #blocked>
    %72 = tt.broadcast %71 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %73 = ttg.memdesc_subview %21[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    %74 = tt.splat %22 : i1 -> tensor<64x256xi1, #blocked>
    %75 = arith.andi %74, %72 : tensor<64x256xi1, #blocked>
    %76 = ttg.async_copy_global_to_local %61, %73 mask %75 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
    %77 = ttg.async_commit_group %76
    %78 = arith.cmpi sgt, %19, %c1_i32 : i32
    %79 = arith.cmpi ne, %23, %c0_i32 : i32
    %80 = arith.extui %79 : i1 to i32
    %81 = arith.cmpi eq, %80, %c0_i32 : i32
    %82:5 = scf.if %81 -> (i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) {
      %122 = arith.addi %0, %c132_i32 : i32
      %123 = arith.divsi %122, %14 : i32
      %124 = arith.muli %123, %c8_i32 : i32
      %125 = arith.subi %2, %124 : i32
      %126 = arith.minsi %125, %c8_i32 : i32
      %127 = arith.remsi %122, %126 : i32
      %128 = arith.addi %124, %127 : i32
      %129 = arith.remsi %122, %14 : i32
      %130 = arith.divsi %129, %126 : i32
      %131 = arith.muli %128, %c128_i32 : i32
      %132 = arith.muli %130, %c256_i32 : i32
      %133 = tt.splat %131 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %134 = arith.addi %133, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %135 = tt.splat %132 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %136 = arith.addi %135, %17 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %137 = arith.cmpi slt, %134, %38 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %138 = arith.select %137, %134, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %139 = arith.cmpi slt, %136, %41 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %140 = arith.select %139, %136, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      scf.yield %122, %128, %130, %138, %140 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    } else {
      scf.yield %0, %29, %31, %40, %43 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    }
    %83 = arith.muli %80, %c64_i32 : i32
    %84 = tt.splat %83 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %85 = tt.splat %83 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %86 = arith.addi %84, %12 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %87 = arith.addi %85, %13 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %88 = tt.expand_dims %82#3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %89 = arith.muli %88, %45 : tensor<128x1xi32, #blocked1>
    %90 = tt.expand_dims %86 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %91 = tt.broadcast %89 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %92 = tt.broadcast %90 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %93 = arith.addi %91, %92 : tensor<128x64xi32, #blocked1>
    %94 = tt.addptr %51, %93 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %95 = tt.expand_dims %87 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %96 = tt.expand_dims %82#4 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %97 = arith.muli %96, %55 : tensor<1x256xi32, #blocked>
    %98 = tt.broadcast %95 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %99 = tt.broadcast %97 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %100 = arith.addi %98, %99 : tensor<64x256xi32, #blocked>
    %101 = tt.addptr %60, %100 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %102 = arith.subi %arg5, %83 : i32
    %103 = tt.splat %102 : i32 -> tensor<1x64xi32, #blocked1>
    %104 = arith.cmpi slt, %47, %103 : tensor<1x64xi32, #blocked1>
    %105 = tt.broadcast %104 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %106 = ttg.memdesc_subview %20[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %107 = tt.splat %78 : i1 -> tensor<128x64xi1, #blocked1>
    %108 = arith.andi %107, %105 : tensor<128x64xi1, #blocked1>
    %109 = ttg.async_copy_global_to_local %94, %106 mask %108 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %110 = ttg.async_commit_group %109
    %111 = tt.splat %102 : i32 -> tensor<64x1xi32, #blocked>
    %112 = arith.cmpi slt, %53, %111 : tensor<64x1xi32, #blocked>
    %113 = tt.broadcast %112 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %114 = ttg.memdesc_subview %21[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    %115 = tt.splat %78 : i1 -> tensor<64x256xi1, #blocked>
    %116 = arith.andi %115, %113 : tensor<64x256xi1, #blocked>
    %117 = ttg.async_copy_global_to_local %101, %114 mask %116 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
    %118 = ttg.async_commit_group %117
    %119:18 = scf.for %arg9 = %c0_i32 to %19 step %c1_i32 iter_args(%arg10 = %80, %arg11 = %82#0, %arg12 = %82#1, %arg13 = %82#2, %arg14 = %cst_3, %arg15 = %82#3, %arg16 = %82#4, %arg17 = %false, %arg18 = %c1_i32, %arg19 = %c-1_i32, %arg20 = %77, %arg21 = %118, %arg22 = %c0_i32, %arg23 = %80, %arg24 = %29, %arg25 = %82#1, %arg26 = %31, %arg27 = %82#2) -> (i32, i32, i32, i32, tensor<128x256xf32, #mma>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i1, i32, i32, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32, i32, i32)  : i32 {
      %122 = arith.subi %19, %c2_i32 : i32
      %123 = arith.cmpi slt, %arg9, %122 : i32
      %124 = arith.cmpi eq, %arg10, %23 : i32
      %125 = arith.addi %arg10, %c1_i32 : i32
      %126 = arith.select %124, %c0_i32, %125 : i32
      %127 = arith.cmpi eq, %126, %c0_i32 : i32
      %128:5 = scf.if %127 -> (i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) {
        %178 = arith.addi %arg11, %c132_i32 : i32
        %179 = arith.divsi %178, %14 : i32
        %180 = arith.muli %179, %c8_i32 : i32
        %181 = arith.subi %2, %180 : i32
        %182 = arith.minsi %181, %c8_i32 : i32
        %183 = arith.remsi %178, %182 : i32
        %184 = arith.addi %180, %183 : i32
        %185 = arith.remsi %178, %14 : i32
        %186 = arith.divsi %185, %182 : i32
        %187 = arith.muli %184, %c128_i32 : i32
        %188 = arith.muli %186, %c256_i32 : i32
        %189 = tt.splat %187 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %190 = arith.addi %189, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %191 = tt.splat %188 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %192 = arith.addi %191, %17 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %193 = arith.cmpi slt, %190, %38 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %194 = arith.select %193, %190, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %195 = arith.cmpi slt, %192, %41 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %196 = arith.select %195, %192, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        scf.yield %178, %184, %186, %194, %196 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      } else {
        scf.yield %arg11, %arg12, %arg13, %arg15, %arg16 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      }
      %129 = arith.addi %arg19, %c1_i32 : i32
      %130 = arith.cmpi slt, %129, %c3_i32 : i32
      %131 = arith.select %130, %129, %c0_i32 : i32
      %132 = ttg.memdesc_subview %20[%131, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %133 = ttg.async_wait %arg20 {num = 2 : i32}
      %134 = ttg.memdesc_subview %21[%131, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %135 = ttng.warp_group_dot %132, %134, %arg14, %arg17 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> * !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> tensor<128x256xf32, #mma>
      %136:3 = ttng.warp_group_dot_wait %135, %132, %134 {pendings = 1 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %137 = arith.cmpi ne, %arg22, %23 : i32
      %138 = arith.addi %arg18, %c1_i32 : i32
      %139 = arith.cmpi slt, %138, %c3_i32 : i32
      %140 = arith.select %139, %138, %c0_i32 : i32
      %141 = arith.muli %126, %c64_i32 : i32
      %142 = tt.splat %141 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %143 = tt.splat %141 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %144 = arith.addi %142, %12 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %145 = arith.addi %143, %13 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %146 = tt.expand_dims %128#3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %147 = arith.muli %146, %45 : tensor<128x1xi32, #blocked1>
      %148 = tt.expand_dims %144 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
      %149 = tt.broadcast %147 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %150 = tt.broadcast %148 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %151 = arith.addi %149, %150 : tensor<128x64xi32, #blocked1>
      %152 = tt.addptr %51, %151 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %153 = tt.expand_dims %145 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %154 = tt.expand_dims %128#4 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
      %155 = arith.muli %154, %55 : tensor<1x256xi32, #blocked>
      %156 = tt.broadcast %153 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %157 = tt.broadcast %155 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %158 = arith.addi %156, %157 : tensor<64x256xi32, #blocked>
      %159 = tt.addptr %60, %158 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
      %160 = arith.subi %arg5, %141 : i32
      %161 = tt.splat %160 : i32 -> tensor<1x64xi32, #blocked1>
      %162 = arith.cmpi slt, %47, %161 : tensor<1x64xi32, #blocked1>
      %163 = tt.broadcast %162 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
      %164 = ttg.memdesc_subview %20[%140, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %165 = tt.splat %123 : i1 -> tensor<128x64xi1, #blocked1>
      %166 = arith.andi %165, %163 : tensor<128x64xi1, #blocked1>
      %167 = ttg.async_copy_global_to_local %152, %164 mask %166 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
      %168 = ttg.async_commit_group %167
      %169 = tt.splat %160 : i32 -> tensor<64x1xi32, #blocked>
      %170 = arith.cmpi slt, %53, %169 : tensor<64x1xi32, #blocked>
      %171 = tt.broadcast %170 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
      %172 = ttg.memdesc_subview %21[%140, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %173 = tt.splat %123 : i1 -> tensor<64x256xi1, #blocked>
      %174 = arith.andi %173, %171 : tensor<64x256xi1, #blocked>
      %175 = ttg.async_copy_global_to_local %159, %172 mask %174 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
      %176 = ttg.async_commit_group %175
      %177 = arith.cmpi eq, %arg22, %23 : i32
      scf.if %177 {
        %178:3 = ttng.warp_group_dot_wait %136#0, %132, %134 {pendings = 0 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
        %179 = arith.muli %arg24, %c128_i32 : i32
        %180 = tt.splat %179 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %181 = arith.addi %180, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %182 = arith.muli %arg26, %c256_i32 : i32
        %183 = tt.splat %182 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %184 = arith.addi %183, %18 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %185 = tt.expand_dims %181 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
        %186 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked2>
        %187 = arith.muli %186, %185 : tensor<128x1xi32, #blocked2>
        %188 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
        %189 = tt.addptr %188, %187 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
        %190 = tt.expand_dims %184 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
        %191 = tt.broadcast %189 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x256x!tt.ptr<f16>, #blocked2>
        %192 = tt.broadcast %190 : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
        %193 = tt.addptr %191, %192 : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
        %194 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked2>
        %195 = arith.cmpi slt, %185, %194 : tensor<128x1xi32, #blocked2>
        %196 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked2>
        %197 = arith.cmpi slt, %190, %196 : tensor<1x256xi32, #blocked2>
        %198 = tt.broadcast %195 : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
        %199 = tt.broadcast %197 : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
        %200 = arith.andi %198, %199 : tensor<128x256xi1, #blocked2>
        %201 = arith.truncf %178#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
        %202 = ttg.convert_layout %201 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked2>
        tt.store %193, %202, %200 : tensor<128x256x!tt.ptr<f16>, #blocked2>
      }
      scf.yield %126, %128#0, %128#1, %128#2, %136#0, %128#3, %128#4, %137, %140, %131, %arg21, %176, %arg23, %126, %arg25, %128#1, %arg27, %128#2 : i32, i32, i32, i32, tensor<128x256xf32, #mma>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i1, i32, i32, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32, i32, i32
    }
    %120 = ttng.warp_group_dot_wait %119#4 {pendings = 0 : i32} : tensor<128x256xf32, #mma>
    %121 = ttg.async_wait  {num = 0 : i32}
    ttg.local_dealloc %20 : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %21 : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable>
    tt.return
  }
}

