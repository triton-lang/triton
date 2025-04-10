#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_tma(%arg0: f32, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg5: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg6: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg7: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<1.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_2 = arith.constant 1.44269502 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_3 = arith.constant dense<-1.000000e+06> : tensor<64x64xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %arg3 : i32
    %3 = arith.remsi %1, %arg3 : i32
    %4 = arith.muli %3, %arg8 : i32
    %5 = arith.addi %2, %4 : i32
    %6 = arith.muli %0, %c64_i32 : i32
    %7 = arith.addi %5, %6 : i32
    %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %10 = tt.splat %6 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %11 = tt.splat %6 : i32 -> tensor<64xi32, #blocked1>
    %12 = arith.addi %10, %8 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = arith.addi %11, %9 : tensor<64xi32, #blocked1>
    %14 = arith.mulf %arg0, %cst_2 : f32
    %15 = tt.reinterpret_tensor_descriptor %arg4 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<64x64xf16, #shared>>
    %16 = tt.descriptor_load %15[%7, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked2>
    %17 = ttg.local_alloc %16 : (tensor<64x64xf16, #blocked2>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %18 = tt.reinterpret_tensor_descriptor %arg5 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<64x64xf16, #shared>>
    %19 = tt.splat %14 : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %20 = tt.splat %14 : f32 -> tensor<64x64xf32, #blocked>
    %21 = tt.reinterpret_tensor_descriptor %arg6 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<64x64xf16, #shared>>
    %22 = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %23 = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst_1, %23, %true : tensor<64x64xf32, #blocked> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %24:3 = scf.for %arg9 = %c0_i32 to %6 step %c64_i32 iter_args(%arg10 = %cst, %arg11 = %cst_0, %arg12 = %5) -> (tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32)  : i32 {
      %54 = tt.descriptor_load %18[%arg12, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked2>
      %55 = ttg.local_alloc %54 : (tensor<64x64xf16, #blocked2>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %56 = ttg.memdesc_trans %55 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      ttng.tc_gen5_mma %17, %56, %22, %false, %true : (!ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %57 = ttng.tmem_load %22 : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #blocked>
      %58 = "tt.reduce"(%57) <{axis = 1 : i32}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %80 = arith.maxnumf %arg13, %arg14 : f32
        tt.reduce.return %80 : f32
      }) : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %59 = arith.mulf %58, %19 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %60 = arith.maxnumf %arg11, %59 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %61 = arith.mulf %57, %20 : tensor<64x64xf32, #blocked>
      %62 = tt.expand_dims %60 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %63 = tt.broadcast %62 : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %64 = arith.subf %61, %63 : tensor<64x64xf32, #blocked>
      %65 = math.exp2 %64 : tensor<64x64xf32, #blocked>
      %66 = "tt.reduce"(%65) <{axis = 1 : i32}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %80 = arith.addf %arg13, %arg14 : f32
        tt.reduce.return %80 : f32
      }) : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %67 = arith.subf %arg11, %60 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %68 = math.exp2 %67 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %69 = arith.mulf %arg10, %68 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %70 = arith.addf %69, %66 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %71 = tt.expand_dims %68 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %72 = tt.broadcast %71 : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %73 = ttng.tmem_load %23 : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #blocked>
      %74 = arith.mulf %73, %72 : tensor<64x64xf32, #blocked>
      %75 = tt.descriptor_load %21[%arg12, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked2>
      %76 = ttg.local_alloc %75 : (tensor<64x64xf16, #blocked2>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %77 = arith.truncf %65 : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
      %78 = ttg.local_alloc %77 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      ttng.tmem_store %74, %23, %true : tensor<64x64xf32, #blocked> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %78, %76, %23, %true, %true : (!ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %79 = arith.addi %arg12, %c64_i32 : i32
      scf.yield %70, %60, %79 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32
    } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
    %25 = ttng.tmem_load %23 : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #blocked>
    %26 = arith.muli %0, %c64_i32 {tt.divisibility = dense<64> : tensor<1xi32>} : i32
    %27 = arith.addi %0, %c1_i32 : i32
    %28 = arith.muli %27, %c64_i32 : i32
    %29 = arith.addi %5, %26 : i32
    %30 = tt.reinterpret_tensor_descriptor %arg5 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<64x64xf16, #shared>>
    %31 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %32 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %33 = tt.expand_dims %32 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %34 = tt.broadcast %31 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %35 = tt.splat %14 : f32 -> tensor<64x64xf32, #blocked>
    %36 = tt.reinterpret_tensor_descriptor %arg6 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<64x64xf16, #shared>>
    %37 = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %38 = ttng.tmem_alloc : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %25, %38, %true : tensor<64x64xf32, #blocked> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %39:3 = scf.for %arg9 = %26 to %28 step %c64_i32 iter_args(%arg10 = %24#0, %arg11 = %24#1, %arg12 = %29) -> (tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32)  : i32 {
      %54 = tt.descriptor_load %30[%arg12, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked2>
      %55 = ttg.local_alloc %54 : (tensor<64x64xf16, #blocked2>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %56 = ttg.memdesc_trans %55 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      ttng.tc_gen5_mma %17, %56, %37, %false, %true : (!ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %57 = tt.splat %arg9 : i32 -> tensor<1x64xi32, #blocked>
      %58 = arith.addi %57, %33 : tensor<1x64xi32, #blocked>
      %59 = tt.broadcast %58 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
      %60 = arith.cmpi sge, %34, %59 : tensor<64x64xi32, #blocked>
      %61 = ttng.tmem_load %37 : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #blocked>
      %62 = arith.mulf %61, %35 : tensor<64x64xf32, #blocked>
      %63 = arith.select %60, %cst_1, %cst_3 : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %64 = arith.addf %62, %63 : tensor<64x64xf32, #blocked>
      %65 = "tt.reduce"(%64) <{axis = 1 : i32}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %85 = arith.maxnumf %arg13, %arg14 : f32
        tt.reduce.return %85 : f32
      }) : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %66 = arith.maxnumf %arg11, %65 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %67 = tt.expand_dims %66 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %68 = tt.broadcast %67 : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %69 = arith.subf %64, %68 : tensor<64x64xf32, #blocked>
      %70 = math.exp2 %69 : tensor<64x64xf32, #blocked>
      %71 = "tt.reduce"(%70) <{axis = 1 : i32}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %85 = arith.addf %arg13, %arg14 : f32
        tt.reduce.return %85 : f32
      }) : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %72 = arith.subf %arg11, %66 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %73 = math.exp2 %72 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %74 = arith.mulf %arg10, %73 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %75 = arith.addf %74, %71 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %76 = tt.expand_dims %73 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %77 = tt.broadcast %76 : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %78 = ttng.tmem_load %38 : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #blocked>
      %79 = arith.mulf %78, %77 : tensor<64x64xf32, #blocked>
      %80 = tt.descriptor_load %36[%arg12, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked2>
      %81 = ttg.local_alloc %80 : (tensor<64x64xf16, #blocked2>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %82 = arith.truncf %70 : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
      %83 = ttg.local_alloc %82 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      ttng.tmem_store %79, %38, %true : tensor<64x64xf32, #blocked> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %83, %81, %38, %true, %true : (!ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %84 = arith.addi %arg12, %c64_i32 : i32
      scf.yield %75, %66, %84 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32
    } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
    %40 = ttng.tmem_load %38 : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #blocked>
    %41 = math.log2 %39#0 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %42 = arith.addf %39#1, %41 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %43 = tt.expand_dims %39#0 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
    %44 = tt.broadcast %43 : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
    %45 = arith.divf %40, %44 : tensor<64x64xf32, #blocked>
    %46 = arith.muli %1, %arg8 : i32
    %47 = tt.addptr %arg1, %46 : !tt.ptr<f32>, i32
    %48 = tt.splat %47 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
    %49 = tt.addptr %48, %13 : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
    %50 = ttg.convert_layout %42 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked1>
    tt.store %49, %50 : tensor<64x!tt.ptr<f32>, #blocked1>
    %51 = arith.truncf %45 : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %52 = ttg.convert_layout %51 : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #blocked2>
    %53 = tt.reinterpret_tensor_descriptor %arg7 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<64x64xf16, #shared>>
    tt.descriptor_store %53[%7, %c0_i32], %52 : !tt.tensordesc<tensor<64x64xf16, #shared>>, tensor<64x64xf16, #blocked2>
    tt.return
  }
}

