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
    %20 = arith.subi %6, %c1_i32 : i32
    %21 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %22 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %23 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked>
    %24 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked>
    %25 = tt.expand_dims %12 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %26 = tt.expand_dims %13 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %27 = ttg.local_alloc  : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %28 = ttg.local_alloc  : () -> !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable>
    %29 = arith.cmpi sgt, %19, %c0_i32 : i32
    %30 = arith.divsi %0, %14 : i32
    %31 = arith.muli %30, %c8_i32 : i32
    %32 = arith.subi %2, %31 : i32
    %33 = arith.minsi %32, %c8_i32 : i32
    %34 = arith.remsi %0, %33 : i32
    %35 = arith.addi %31, %34 : i32
    %36 = arith.remsi %0, %14 : i32
    %37 = arith.divsi %36, %33 : i32
    %38 = arith.muli %35, %c128_i32 : i32
    %39 = arith.muli %37, %c256_i32 : i32
    %40 = tt.splat %38 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %41 = arith.addi %40, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %42 = tt.splat %39 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %43 = arith.addi %42, %17 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %44 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %45 = arith.cmpi slt, %41, %44 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %46 = arith.select %45, %41, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %47 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %48 = arith.cmpi slt, %43, %47 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %49 = arith.select %48, %43, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %50 = tt.expand_dims %46 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %51 = arith.muli %50, %21 : tensor<128x1xi32, #blocked1>
    %52 = tt.broadcast %51 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %53 = tt.broadcast %25 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %54 = arith.addi %52, %53 : tensor<128x64xi32, #blocked1>
    %55 = tt.addptr %22, %54 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %56 = tt.expand_dims %49 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %57 = arith.muli %56, %23 : tensor<1x256xi32, #blocked>
    %58 = tt.broadcast %26 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %59 = tt.broadcast %57 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %60 = arith.addi %58, %59 : tensor<64x256xi32, #blocked>
    %61 = tt.addptr %24, %60 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %62 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1>
    %63 = arith.cmpi slt, %25, %62 : tensor<1x64xi32, #blocked1>
    %64 = tt.broadcast %63 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %65 = ttg.memdesc_subview %27[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %66 = tt.splat %29 : i1 -> tensor<128x64xi1, #blocked1>
    %67 = arith.andi %66, %64 : tensor<128x64xi1, #blocked1>
    %68 = ttg.async_copy_global_to_local %55, %65 mask %67 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %69 = ttg.async_commit_group %68
    %70 = tt.splat %arg5 : i32 -> tensor<64x1xi32, #blocked>
    %71 = arith.cmpi slt, %26, %70 : tensor<64x1xi32, #blocked>
    %72 = tt.broadcast %71 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %73 = ttg.memdesc_subview %28[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    %74 = tt.splat %29 : i1 -> tensor<64x256xi1, #blocked>
    %75 = arith.andi %74, %72 : tensor<64x256xi1, #blocked>
    %76 = ttg.async_copy_global_to_local %61, %73 mask %75 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
    %77 = ttg.async_commit_group %76
    %78 = arith.cmpi sgt, %19, %c1_i32 : i32
    %79 = arith.cmpi ne, %20, %c0_i32 : i32
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
      %137 = arith.cmpi slt, %134, %44 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %138 = arith.select %137, %134, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %139 = arith.cmpi slt, %136, %47 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %140 = arith.select %139, %136, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      scf.yield %122, %128, %130, %138, %140 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    } else {
      scf.yield %0, %35, %37, %46, %49 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    }
    %83 = arith.muli %80, %c64_i32 : i32
    %84 = tt.splat %83 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %85 = tt.splat %83 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %86 = arith.addi %84, %12 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %87 = arith.addi %85, %13 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %88 = tt.expand_dims %82#3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %89 = arith.muli %88, %21 : tensor<128x1xi32, #blocked1>
    %90 = tt.expand_dims %86 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %91 = tt.broadcast %89 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %92 = tt.broadcast %90 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %93 = arith.addi %91, %92 : tensor<128x64xi32, #blocked1>
    %94 = tt.addptr %22, %93 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %95 = tt.expand_dims %87 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %96 = tt.expand_dims %82#4 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %97 = arith.muli %96, %23 : tensor<1x256xi32, #blocked>
    %98 = tt.broadcast %95 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %99 = tt.broadcast %97 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %100 = arith.addi %98, %99 : tensor<64x256xi32, #blocked>
    %101 = tt.addptr %24, %100 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %102 = arith.subi %arg5, %83 : i32
    %103 = tt.splat %102 : i32 -> tensor<1x64xi32, #blocked1>
    %104 = arith.cmpi slt, %25, %103 : tensor<1x64xi32, #blocked1>
    %105 = tt.broadcast %104 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %106 = ttg.memdesc_subview %27[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %107 = tt.splat %78 : i1 -> tensor<128x64xi1, #blocked1>
    %108 = arith.andi %107, %105 : tensor<128x64xi1, #blocked1>
    %109 = ttg.async_copy_global_to_local %94, %106 mask %108 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %110 = ttg.async_commit_group %109
    %111 = tt.splat %102 : i32 -> tensor<64x1xi32, #blocked>
    %112 = arith.cmpi slt, %26, %111 : tensor<64x1xi32, #blocked>
    %113 = tt.broadcast %112 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %114 = ttg.memdesc_subview %28[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    %115 = tt.splat %78 : i1 -> tensor<64x256xi1, #blocked>
    %116 = arith.andi %115, %113 : tensor<64x256xi1, #blocked>
    %117 = ttg.async_copy_global_to_local %101, %114 mask %116 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
    %118 = ttg.async_commit_group %117
    %119:18 = scf.for %arg9 = %c0_i32 to %19 step %c1_i32 iter_args(%arg10 = %80, %arg11 = %82#0, %arg12 = %82#1, %arg13 = %82#2, %arg14 = %cst_3, %arg15 = %82#3, %arg16 = %82#4, %arg17 = %false, %arg18 = %c1_i32, %arg19 = %c-1_i32, %arg20 = %77, %arg21 = %118, %arg22 = %c0_i32, %arg23 = %80, %arg24 = %35, %arg25 = %82#1, %arg26 = %37, %arg27 = %82#2) -> (i32, i32, i32, i32, tensor<128x256xf32, #mma>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i1, i32, i32, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32, i32, i32)  : i32 {
      %122 = arith.subi %19, %c2_i32 : i32
      %123 = arith.cmpi slt, %arg9, %122 : i32
      %124 = arith.cmpi eq, %arg10, %20 : i32
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
        %193 = arith.cmpi slt, %190, %44 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %194 = arith.select %193, %190, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %195 = arith.cmpi slt, %192, %47 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %196 = arith.select %195, %192, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        scf.yield %178, %184, %186, %194, %196 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      } else {
        scf.yield %arg11, %arg12, %arg13, %arg15, %arg16 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      }
      %129 = arith.addi %arg19, %c1_i32 : i32
      %130 = arith.cmpi slt, %129, %c3_i32 : i32
      %131 = arith.select %130, %129, %c0_i32 : i32
      %132 = ttg.memdesc_subview %27[%131, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %133 = ttg.async_wait %arg20 {num = 2 : i32}
      %134 = ttg.memdesc_subview %28[%131, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %135 = ttng.warp_group_dot %132, %134, %arg14, %arg17 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> * !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> tensor<128x256xf32, #mma>
      %136:3 = ttng.warp_group_dot_wait %135, %132, %134 {pendings = 1 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %137 = arith.addi %arg18, %c1_i32 : i32
      %138 = arith.cmpi slt, %137, %c3_i32 : i32
      %139 = arith.select %138, %137, %c0_i32 : i32
      %140 = arith.muli %126, %c64_i32 : i32
      %141 = tt.splat %140 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %142 = tt.splat %140 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %143 = arith.addi %141, %12 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %144 = arith.addi %142, %13 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %145 = tt.expand_dims %128#3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %146 = arith.muli %145, %21 : tensor<128x1xi32, #blocked1>
      %147 = tt.expand_dims %143 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
      %148 = tt.broadcast %146 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %149 = tt.broadcast %147 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %150 = arith.addi %148, %149 : tensor<128x64xi32, #blocked1>
      %151 = tt.addptr %22, %150 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %152 = tt.expand_dims %144 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %153 = tt.expand_dims %128#4 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
      %154 = arith.muli %153, %23 : tensor<1x256xi32, #blocked>
      %155 = tt.broadcast %152 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %156 = tt.broadcast %154 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %157 = arith.addi %155, %156 : tensor<64x256xi32, #blocked>
      %158 = tt.addptr %24, %157 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
      %159 = arith.subi %arg5, %140 : i32
      %160 = tt.splat %159 : i32 -> tensor<1x64xi32, #blocked1>
      %161 = arith.cmpi slt, %25, %160 : tensor<1x64xi32, #blocked1>
      %162 = tt.broadcast %161 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
      %163 = ttg.memdesc_subview %27[%139, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %164 = tt.splat %123 : i1 -> tensor<128x64xi1, #blocked1>
      %165 = arith.andi %164, %162 : tensor<128x64xi1, #blocked1>
      %166 = ttg.async_copy_global_to_local %151, %163 mask %165 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
      %167 = ttg.async_commit_group %166
      %168 = tt.splat %159 : i32 -> tensor<64x1xi32, #blocked>
      %169 = arith.cmpi slt, %26, %168 : tensor<64x1xi32, #blocked>
      %170 = tt.broadcast %169 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
      %171 = ttg.memdesc_subview %28[%139, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %172 = tt.splat %123 : i1 -> tensor<64x256xi1, #blocked>
      %173 = arith.andi %172, %170 : tensor<64x256xi1, #blocked>
      %174 = ttg.async_copy_global_to_local %158, %171 mask %173 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
      %175 = ttg.async_commit_group %174
      %176 = arith.cmpi eq, %arg22, %20 : i32
      %177 = arith.cmpi ne, %arg22, %20 : i32
      scf.if %176 {
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
      scf.yield %126, %128#0, %128#1, %128#2, %136#0, %128#3, %128#4, %177, %139, %131, %arg21, %175, %arg23, %126, %arg25, %128#1, %arg27, %128#2 : i32, i32, i32, i32, tensor<128x256xf32, #mma>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i1, i32, i32, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32, i32, i32
    }
    %120 = ttng.warp_group_dot_wait %119#4 {pendings = 0 : i32} : tensor<128x256xf32, #mma>
    %121 = ttg.async_wait  {num = 0 : i32}
    ttg.local_dealloc %27 : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %28 : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable>
    tt.return
  }
}

