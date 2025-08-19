#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
module attributes {"nvws.warp-specialized" = true, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gemm_cpasync_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc(unknown), %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc(unknown), %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc(unknown), %arg3: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg4: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg5: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg6: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg7: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg8: i32 {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_0 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c84_i32 = arith.constant 84 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x256xf16, #blocked>
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %true = arith.constant true
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.addi %arg5, %c127_i32 : i32
    %6 = arith.divsi %5, %c128_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.divsi %7, %c84_i32 : i32
    %9 = arith.remsi %7, %c84_i32 : i32
    %10 = arith.cmpi slt, %0, %9 : i32
    %11 = scf.if %10 -> (i32) {
      %20 = arith.addi %8, %c1_i32 : i32
      scf.yield %20 : i32
    } else {
      scf.yield %8 : i32
    }
    %12 = arith.subi %0, %c84_i32 : i32
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %16 = arith.muli %4, %c8_i32 : i32
    %17 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %18 = arith.muli %6, %11 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %19:7 = scf.for %arg9 = %c0_i32 to %18 step %c1_i32 iter_args(%arg10 = %c-1_i32, %arg11 = %12, %arg12 = %12, %arg13 = %13, %arg14 = %17, %arg15 = %false, %arg16 = %token) -> (i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i1, !ttg.async.token)  : i32 {
      %20 = arith.subi %6, %c1_i32 : i32
      %21 = arith.cmpi eq, %arg10, %20 : i32
      %22 = arith.addi %arg10, %c1_i32 : i32
      %23 = arith.select %21, %c0_i32, %22 : i32
      %24 = arith.cmpi eq, %23, %c0_i32 : i32
      %25:3 = scf.if %24 -> (i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) {
        %66 = arith.addi %arg11, %c84_i32 : i32
        %67 = arith.divsi %66, %16 : i32
        %68 = arith.muli %67, %c8_i32 : i32
        %69 = arith.subi %2, %68 : i32
        %70 = arith.minsi %69, %c8_i32 : i32
        %71 = arith.remsi %66, %70 : i32
        %72 = arith.addi %68, %71 : i32
        %73 = arith.remsi %66, %16 : i32
        %74 = arith.divsi %73, %70 : i32
        %75 = arith.muli %72, %c128_i32 : i32
        %76 = arith.muli %74, %c256_i32 : i32
        %77 = tt.splat %75 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %78 = arith.addi %77, %13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %79 = tt.splat %76 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %80 = arith.addi %79, %17 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %81 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %82 = arith.cmpi slt, %78, %81 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %83 = arith.select %82, %78, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %84 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %85 = arith.cmpi slt, %80, %84 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %86 = arith.select %85, %80, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        scf.yield %66, %83, %86 : i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      } else {
        scf.yield %arg11, %arg13, %arg14 : i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      }
      %26 = arith.muli %23, %c128_i32 : i32
      %27 = tt.splat %26 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %28 = tt.splat %26 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %29 = arith.addi %27, %15 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %30 = arith.addi %28, %14 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %31 = tt.expand_dims %25#1 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %32 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
      %33 = arith.muli %31, %32 : tensor<128x1xi32, #blocked1>
      %34 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
      %35 = tt.broadcast %33 : tensor<128x1xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
      %36 = tt.broadcast %34 : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
      %37 = arith.addi %35, %36 : tensor<128x128xi32, #blocked1>
      %38 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked1>
      %39 = tt.addptr %38, %37 : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>
      %40 = tt.expand_dims %30 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %41 = tt.splat %arg7 : i32 -> tensor<128x1xi32, #blocked>
      %42 = arith.muli %40, %41 : tensor<128x1xi32, #blocked>
      %43 = tt.expand_dims %25#2 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
      %44 = tt.broadcast %42 : tensor<128x1xi32, #blocked> -> tensor<128x256xi32, #blocked>
      %45 = tt.broadcast %43 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
      %46 = arith.addi %44, %45 : tensor<128x256xi32, #blocked>
      %47 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked>
      %48 = tt.addptr %47, %46 : tensor<128x256x!tt.ptr<f16>, #blocked>, tensor<128x256xi32, #blocked>
      %49 = tt.expand_dims %15 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
      %50 = arith.subi %arg5, %26 : i32
      %51 = tt.splat %50 : i32 -> tensor<1x128xi32, #blocked1>
      %52 = arith.cmpi slt, %49, %51 : tensor<1x128xi32, #blocked1>
      %53 = tt.broadcast %52 : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
      %54 = tt.load %39, %53, %cst_1 : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %55 = ttg.local_alloc %54 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %56 = tt.expand_dims %14 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %57 = tt.splat %50 : i32 -> tensor<128x1xi32, #blocked>
      %58 = arith.cmpi slt, %56, %57 : tensor<128x1xi32, #blocked>
      %59 = tt.broadcast %58 : tensor<128x1xi1, #blocked> -> tensor<128x256xi1, #blocked>
      %60 = tt.load %48, %59, %cst_2 : tensor<128x256x!tt.ptr<f16>, #blocked>
      %61 = ttg.local_alloc %60 : (tensor<128x256xf16, #blocked>) -> !ttg.memdesc<128x256xf16, #shared, #smem>
      %62 = ttng.tc_gen5_mma %55, %61, %result[%arg16], %arg15, %true : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x256xf16, #shared, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %63 = arith.cmpi eq, %23, %20 : i32
      %64 = arith.cmpi ne, %23, %20 : i32
      %65:2 = scf.if %63 -> (i32, !ttg.async.token) {
        %66 = arith.addi %arg12, %c84_i32 : i32
        %67 = arith.divsi %66, %16 : i32
        %68 = arith.muli %67, %c8_i32 : i32
        %69 = arith.subi %2, %68 : i32
        %70 = arith.minsi %69, %c8_i32 : i32
        %71 = arith.remsi %66, %70 : i32
        %72 = arith.addi %68, %71 : i32
        %73 = arith.remsi %66, %16 : i32
        %74 = arith.divsi %73, %70 : i32
        %75 = arith.muli %72, %c128_i32 : i32
        %76 = tt.splat %75 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %77 = arith.addi %76, %14 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %78 = arith.muli %74, %c256_i32 : i32
        %79 = tt.splat %78 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %80 = arith.addi %79, %17 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %81 = tt.expand_dims %77 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
        %82 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked>
        %83 = arith.muli %82, %81 : tensor<128x1xi32, #blocked>
        %84 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
        %85 = tt.addptr %84, %83 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
        %86 = tt.expand_dims %80 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
        %87 = tt.broadcast %85 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x256x!tt.ptr<f16>, #blocked>
        %88 = tt.broadcast %86 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
        %89 = tt.addptr %87, %88 : tensor<128x256x!tt.ptr<f16>, #blocked>, tensor<128x256xi32, #blocked>
        %90 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked>
        %91 = arith.cmpi slt, %81, %90 : tensor<128x1xi32, #blocked>
        %92 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked>
        %93 = arith.cmpi slt, %86, %92 : tensor<1x256xi32, #blocked>
        %94 = tt.broadcast %91 : tensor<128x1xi1, #blocked> -> tensor<128x256xi1, #blocked>
        %95 = tt.broadcast %93 : tensor<1x256xi1, #blocked> -> tensor<128x256xi1, #blocked>
        %96 = arith.andi %94, %95 : tensor<128x256xi1, #blocked>
        %result_3, %token_4 = ttng.tmem_load %result[%62] : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked2>
        %97 = arith.truncf %result_3 : tensor<128x256xf32, #blocked2> to tensor<128x256xf16, #blocked2>
        %98 = ttg.convert_layout %97 : tensor<128x256xf16, #blocked2> -> tensor<128x256xf16, #blocked>
        tt.store %89, %98, %96 : tensor<128x256x!tt.ptr<f16>, #blocked>
        scf.yield %66, %token_4 : i32, !ttg.async.token
      } else {
        scf.yield %arg12, %62 : i32, !ttg.async.token
      }
      scf.yield %23, %25#0, %65#0, %25#1, %25#2, %64, %65#1 : i32, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i1, !ttg.async.token
    }
    tt.return
  }
}

