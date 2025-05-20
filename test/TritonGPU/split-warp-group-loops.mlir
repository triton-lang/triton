// RUN: triton-opt %s -split-input-file  --tritongpu-split-warp-group-loops | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {nvws.mma = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 4 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @mmav5_aref_as_barrier(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %c128_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 128 : i32
    %c0_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 0 : i32
    %c1_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 1 : i32
    %c127_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 127 : i32
    %cst = arith.constant {groups = [@nvws.mma]} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.get_program_id x {groups = [@nvws.mma]} : i32
    %1 = arith.addi %arg3, %c127_i32 {groups = [@nvws.mma]} : i32
    %2 = arith.divsi %1, %c128_i32 {groups = [@nvws.mma]} : i32
    %3 = arith.remsi %0, %2 {groups = [@nvws.mma]} : i32
    %4 = arith.divsi %0, %2 {groups = [@nvws.mma]} : i32
    %5 = arith.muli %3, %c128_i32 {groups = [@nvws.mma]} : i32
    %6 = arith.muli %4, %c128_i32 {groups = [@nvws.mma]} : i32
    %7 = arith.addi %arg5, %c127_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %8 = arith.divsi %7, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    // CHECK: %[[MEMACC:.*]] = ttng.tmem_alloc
    // CHECK: %[[ACC:.*]] = ttg.memdesc_subview %[[MEMACC]]
    %9 = ttng.tmem_alloc %cst {groups = [@nvws.mma]} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-DAG: %[[AREF:.*]] = ttng.aref_create
    // CHECK: ttng.warp_group start_warp(4) num_warps(1)
    // CHECK: scf.for
    %10 = scf.for %arg7 = %c0_i32 to %8 step %c1_i32 iter_args(%arg8 = %c0_i32) -> (i32)  : i32 {
      %29 = tt.reinterpret_tensor_descriptor %arg0 {groups = [@nvws.tma_load]} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
      %30 = tt.descriptor_load %29[%5, %arg8] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #blocked1>
      %31 = ttg.local_alloc %30 {groups = [@nvws.mma]} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
      %32 = tt.reinterpret_tensor_descriptor %arg1 {groups = [@nvws.tma_load]} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
      %33 = tt.descriptor_load %32[%6, %arg8] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #blocked1>
      %34 = ttg.local_alloc %33 {groups = [@nvws.mma]} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
      %35 = ttg.memdesc_trans %34 {groups = [@nvws.mma], order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
      // CHECK: %[[A:.*]]:2 = ttng.aref_get.enter
      // CHECK: %[[B:.*]] = ttg.memdesc_trans %[[A]]#1
      // CHECK: ttng.tc_gen5_mma %[[A]]#0, %[[B]], %[[ACC]], %true, %true {groups = [@nvws.mma], is_async}
       ttng.tc_gen5_mma %31, %35, %9, %true, %true {groups = [@nvws.mma]} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %36 = arith.addi %arg8, %c128_i32 {groups = [@nvws.tma_load]} : i32
      scf.yield %36 : i32
    } {groups = [@nvws.mma, @nvws.tma_load], groups.0 = [@nvws.tma_load]}
    // CHECK: } {groups = [@nvws.mma]}
    // CHECK: ttng.tmem_load
    // CHECK: tt.store
    // CHECK: }
    %11 = ttng.tmem_load %9 {groups = [@nvws.mma]} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %12 = tt.fp_to_fp %11 {groups = [@nvws.mma]}, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
    %13 = tt.make_range {end = 128 : i32, groups = [@nvws.mma], start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %14 = tt.make_range {end = 128 : i32, groups = [@nvws.mma], start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %15 = tt.splat %5 {groups = [@nvws.mma]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %16 = arith.addi %15, %13 {groups = [@nvws.mma]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %17 = tt.splat %6 {groups = [@nvws.mma]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %18 = arith.addi %17, %14 {groups = [@nvws.mma]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %19 = tt.expand_dims %16 {axis = 1 : i32, groups = [@nvws.mma]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %20 = tt.splat %arg6 {groups = [@nvws.mma]} : i32 -> tensor<128x1xi32, #blocked2>
    %21 = arith.muli %20, %19 {groups = [@nvws.mma]} : tensor<128x1xi32, #blocked2>
    %22 = tt.splat %arg2 {groups = [@nvws.mma]} : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2>
    %23 = tt.addptr %22, %21 {groups = [@nvws.mma]} : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<128x1xi32, #blocked2>
    %24 = tt.expand_dims %18 {axis = 0 : i32, groups = [@nvws.mma]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
    %25 = tt.broadcast %23 {groups = [@nvws.mma]} : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>
    %26 = tt.broadcast %24 {groups = [@nvws.mma]} : tensor<1x128xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
    %27 = tt.addptr %25, %26 {groups = [@nvws.mma]} : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<128x128xi32, #blocked2>
    %28 = ttg.convert_layout %12 {groups = [@nvws.mma]} : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked2>
    tt.store %27, %28 {groups = [@nvws.mma]} : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {nvws.mma = {num_warps = 4 : i32, start_warp = 4 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 0 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_kernel_tma_persistent(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %c114_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 114 : i32
    %c1_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 128 : i32
    %c127_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 127 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = tt.get_program_id x {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %1 = arith.addi %arg3, %c127_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %2 = arith.divsi %1, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %3 = arith.addi %arg4, %c127_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %4 = arith.divsi %3, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %5 = arith.addi %arg5, %c127_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %6 = arith.divsi %5, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %7 = arith.muli %2, %4 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %8 = arith.divsi %7, %c114_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %9 = arith.remsi %7, %c114_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %10 = arith.cmpi slt, %0, %9 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %11 = scf.if %10 -> (i32) {
      %16 = arith.addi %8, %c1_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      scf.yield %16 : i32
    } else {
      scf.yield %8 : i32
    } {groups = [@nvws.mma, @nvws.tma_load]}
    %12 = arith.subi %0, %c114_i32 : i32
    %13 = arith.muli %4, %c8_i32 : i32
    %14 = arith.muli %6, %11 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    // CHECK: ttng.aref_create
    // CHECK: ttng.warp_group start_warp(0) num_warps(4)
    // CHECK: scf.for
    // CHECK: } {groups = [@nvws.tma_load]}
    // CHECK: }
    // CHECK: ttng.warp_group start_warp(4) num_warps(4)
    // CHECK: scf.for
    // CHECK: } {groups = [@nvws.mma]}
    // CHECK: }
    %15:7 = scf.for %arg6 = %c0_i32 to %14 step %c1_i32 iter_args(%arg7 = %c-1_i32, %arg8 = %12, %arg9 = %c0_i32, %arg10 = %c0_i32, %arg11 = %cst, %arg12 = %c-1_i32, %arg13 = %false) -> (i32, i32, i32, i32, tensor<128x128xf32, #mma>, i32, i1)  : i32 {
      %16 = arith.subi %6, %c1_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %17 = arith.cmpi eq, %arg7, %16 {groups = [@nvws.tma_load]} : i32
      %18 = arith.addi %arg7, %c1_i32 {groups = [@nvws.tma_load]} : i32
      %19 = arith.select %17, %c0_i32, %18 {groups = [@nvws.tma_load]} : i32
      %20 = arith.cmpi eq, %19, %c0_i32 {groups = [@nvws.tma_load]} : i32
      %21:3 = scf.if %20 -> (i32, i32, i32) {
        %36 = arith.addi %arg8, %c114_i32 {groups = [@nvws.tma_load]} : i32
        %37 = arith.divsi %36, %13 {groups = [@nvws.tma_load]} : i32
        %38 = arith.muli %37, %c8_i32 {groups = [@nvws.tma_load]} : i32
        %39 = arith.subi %2, %38 {groups = [@nvws.tma_load]} : i32
        %40 = arith.minsi %39, %c8_i32 {groups = [@nvws.tma_load]} : i32
        %41 = arith.remsi %36, %40 {groups = [@nvws.tma_load]} : i32
        %42 = arith.addi %38, %41 {groups = [@nvws.tma_load]} : i32
        %43 = arith.remsi %36, %13 {groups = [@nvws.tma_load]} : i32
        %44 = arith.divsi %43, %40 {groups = [@nvws.tma_load]} : i32
        %45 = arith.muli %42, %c128_i32 {groups = [@nvws.tma_load], tt.divisibility = dense<128> : tensor<1xi32>} : i32
        %46 = arith.muli %44, %c128_i32 {groups = [@nvws.tma_load], tt.divisibility = dense<128> : tensor<1xi32>} : i32
        scf.yield %36, %45, %46 : i32, i32, i32
      } else {
        scf.yield %arg8, %arg9, %arg10 : i32, i32, i32
      } {groups = [@nvws.tma_load]}
      %22 = arith.muli %19, %c128_i32 {groups = [@nvws.tma_load]} : i32
      %23 = tt.reinterpret_tensor_descriptor %arg0 {groups = [@nvws.tma_load]} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
      %24 = tt.descriptor_load %23[%21#1, %22] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #blocked>
      %25 = ttg.local_alloc %24 {groups = [@nvws.mma]} : (tensor<128x128xf8E4M3FN, #blocked>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
      %26 = tt.reinterpret_tensor_descriptor %arg1 {groups = [@nvws.tma_load]} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
      %27 = tt.descriptor_load %26[%21#2, %22] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #blocked>
      %28 = ttg.local_alloc %27 {groups = [@nvws.mma]} : (tensor<128x128xf8E4M3FN, #blocked>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
      %29 = ttg.memdesc_trans %28 {groups = [@nvws.mma], order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
      %30 = ttng.warp_group_dot %25, %29, %arg11, %arg13 {groups = [@nvws.mma], inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem> * !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem> -> tensor<128x128xf32, #mma>
      %31 = arith.cmpi eq, %arg12, %16 {groups = [@nvws.mma]} : i32
      %32 = arith.addi %arg12, %c1_i32 {groups = [@nvws.mma]} : i32
      %33 = arith.select %31, %c0_i32, %32 {groups = [@nvws.mma]} : i32
      %34 = arith.cmpi eq, %33, %16 {groups = [@nvws.mma]} : i32
      %35 = arith.cmpi ne, %33, %16 {groups = [@nvws.mma]} : i32
      scf.if %34 {
        %36 = arith.addi %arg6, %c1_i32 : i32
        %37 = arith.divsi %36, %6 : i32
        %38 = arith.muli %37, %c114_i32 : i32
        %39 = arith.addi %12, %38 : i32
        %40 = arith.divsi %39, %13 : i32
        %41 = arith.muli %40, %c8_i32 : i32
        %42 = arith.subi %2, %41 : i32
        %43 = arith.minsi %42, %c8_i32 : i32
        %44 = arith.remsi %39, %43 : i32
        %45 = arith.addi %41, %44 : i32
        %46 = arith.remsi %39, %13 : i32
        %47 = arith.divsi %46, %43 : i32
        %48 = arith.muli %45, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
        %49 = arith.muli %47, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
        %50 = tt.fp_to_fp %30, rounding = rtne : tensor<128x128xf32, #mma> -> tensor<128x128xf8E4M3FN, #mma>
        %51 = ttg.convert_layout %50 : tensor<128x128xf8E4M3FN, #mma> -> tensor<128x128xf8E4M3FN, #blocked>
        %52 = tt.reinterpret_tensor_descriptor %arg2 : !tt.ptr<f32> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
        tt.descriptor_store %52[%48, %49], %51 {groups = [@nvws.mma]} : !tt.tensordesc<tensor<128x128xf8E4M3FN>>, tensor<128x128xf8E4M3FN, #blocked>
      } {groups = [@nvws.mma]}
      scf.yield %19, %21#0, %21#1, %21#2, %30, %33, %35 : i32, i32, i32, i32, tensor<128x128xf32, #mma>, i32, i1
    } {groups = [@nvws.mma, @nvws.tma_load], groups.0 = [@nvws.tma_load], groups.1 = [@nvws.tma_load], groups.2 = [@nvws.tma_load], groups.3 = [@nvws.tma_load], groups.4 = [@nvws.mma], groups.5 = [@nvws.mma], groups.6 = [@nvws.mma]}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {nvws.mma = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 4 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_persistent_tma_ws_cooperative_kernel(%arg0: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg1: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg2: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg3: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant {groups = [@nvws.mma]} false
    %true = arith.constant {groups = [@nvws.mma]} true
    %c127_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 127 : i32
    %c8_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 8 : i32
    %c128_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 128 : i32
    %c0_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 0 : i32
    %c63_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 63 : i32
    %c1_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 1 : i32
    %c64_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 64 : i32
    %cst = arith.constant {groups = [@nvws.mma]} dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = arith.addi %arg4, %c127_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %1 = arith.divsi %0, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %2 = arith.addi %arg5, %c127_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %3 = arith.divsi %2, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %4 = arith.muli %1, %3 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %5 = tt.get_program_id x {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %6 = tt.get_num_programs x {groups = [@nvws.mma, @nvws.tma_load]} : i32
    // CHECK: ttng.warp_group start_warp(4) num_warps(4)
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: } {groups = [@nvws.tma_load]}
    // CHECK  } {groups = [@nvws.tma_load]}
    // CHECK: }
    // CHECK: ttng.warp_group start_warp(0) num_warps(4)
    // CHECK: scf.for %arg{{[1-9]+}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[NORM_IDX_OUTER:.*]] = {{.*}})
    // CHECK: %[[NORM_IDX_INNER_INIT:.*]] = arith.muli %[[NORM_IDX_OUTER]], {{.*}}
    // CHECK: %[[INNER_FOR_RESULT:.*]]:3 = scf.for %[[K_ITER:.*]] = %c0_i32 to {{.*}} step {{.*}} iter_args(%arg{{.*}} = {{.*}}, %arg{{.*}} = %false, %[[NORM_IDX_INNER:.*]] = %[[NORM_IDX_INNER_INIT]])
    // CHECK: aref_get.enter  {{.*}}[%[[NORM_IDX_INNER]]]
    // CHECK: %[[K_ITER_DIV:.*]] = arith.divsi %[[K_ITER]]
    // CHECK: %[[RELEASE_COND:.*]] = arith.cmpi sge, %[[K_ITER_DIV]]
    // CHECK: scf.if %[[RELEASE_COND]] {
    // CHECK:    arith.subi %[[NORM_IDX_INNER]]
    // CHECK:     ttng.aref_get.exit
    // CHECK: } {groups = [@nvws.mma]}
    // CHECK: } {groups = [@nvws.mma]}
    // CHECK: %[[NUM_RELEASE:.*]] = arith.minsi
    // CHECK: scf.for  %arg{{[1-9]+}} = {{.*}} to %[[NUM_RELEASE]] step %c1_i32
    // CHECK:  arith.subi
    // CHECK:  arith.subi
    // CHECK:  ttng.aref_get.exit
    // CHECK: } {groups = [@nvws.mma]}
    // CHECK: %[[NEXT_NORM_IDX_OUTER:.*]] = arith.addi %[[NORM_IDX_OUTER]], {{.*}}
    // CHECK: scf.yield {groups = [@nvws.mma]} %[[NEXT_NORM_IDX_OUTER]]
    // CHECK  } {groups = [@nvws.mma]}
    // CHECK: }
    scf.for %arg7 = %5 to %4 step %6  : i32 {
      %7 = arith.muli %3, %c8_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %8 = arith.divsi %arg7, %7 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %9 = arith.muli %8, %c8_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %10 = arith.subi %1, %9 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %11 = arith.minsi %10, %c8_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %12 = arith.remsi %arg7, %7 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %13 = arith.remsi %12, %11 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %14 = arith.addi %9, %13 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %15 = arith.divsi %12, %11 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %16 = arith.muli %14, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %17 = arith.muli %15, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %18 = arith.addi %arg6, %c63_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %19 = arith.divsi %18, %c64_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %20:3 = scf.for %arg8 = %c0_i32 to %19 step %c1_i32 iter_args(%arg9 = %cst, %arg10 = %c0_i32, %arg11 = %false) -> (tensor<128x128xf32, #mma>, i32, i1)  : i32 {
        %46 = tt.reinterpret_tensor_descriptor %arg0 {groups = [@nvws.tma_load]} : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<128x64xf16>>
        %47 = tt.descriptor_load %46[%16, %arg10] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked>
        %48 = ttg.local_alloc %47 {groups = [@nvws.mma]} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %49 = tt.reinterpret_tensor_descriptor %arg1 {groups = [@nvws.tma_load]} : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<64x128xf16>>
        %50 = tt.descriptor_load %49[%arg10, %17] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<64x128xf16>> -> tensor<64x128xf16, #blocked1>
        %51 = ttg.local_alloc %50 {groups = [@nvws.mma]} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
        %52 = ttng.warp_group_dot %48, %51, %arg9, %arg11 {groups = [@nvws.mma], inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<128x128xf32, #mma>
        %53 = arith.addi %arg10, %c64_i32 {groups = [@nvws.tma_load]} : i32
        scf.yield %52, %53, %true : tensor<128x128xf32, #mma>, i32, i1
      } {groups = [@nvws.mma, @nvws.tma_load], groups.0 = [@nvws.mma], groups.1 = [@nvws.tma_load], groups.2 = [@nvws.mma]}
      %21 = arith.truncf %20#0 {groups = [@nvws.mma]} : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
      %22 = tt.make_range {end = 128 : i32, groups = [@nvws.mma], start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %23 = tt.make_range {end = 128 : i32, groups = [@nvws.mma], start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %24 = tt.splat %16 {groups = [@nvws.mma]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %25 = arith.addi %24, %22 {groups = [@nvws.mma]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %26 = tt.splat %17 {groups = [@nvws.mma]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %27 = arith.addi %26, %23 {groups = [@nvws.mma]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %28 = tt.expand_dims %25 {axis = 1 : i32, groups = [@nvws.mma]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
      %29 = tt.splat %arg5 {groups = [@nvws.mma]} : i32 -> tensor<128x1xi32, #blocked2>
      %30 = arith.muli %28, %29 {groups = [@nvws.mma]} : tensor<128x1xi32, #blocked2>
      %31 = tt.splat %arg3 {groups = [@nvws.mma]} : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2>
      %32 = tt.addptr %31, %30 {groups = [@nvws.mma]} : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<128x1xi32, #blocked2>
      %33 = tt.expand_dims %27 {axis = 0 : i32, groups = [@nvws.mma]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
      %34 = tt.broadcast %32 {groups = [@nvws.mma]} : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>
      %35 = tt.broadcast %33 {groups = [@nvws.mma]} : tensor<1x128xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
      %36 = tt.addptr %34, %35 {groups = [@nvws.mma]} : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<128x128xi32, #blocked2>
      %37 = tt.splat %arg4 {groups = [@nvws.mma]} : i32 -> tensor<128x1xi32, #blocked2>
      %38 = arith.cmpi slt, %28, %37 {groups = [@nvws.mma]} : tensor<128x1xi32, #blocked2>
      %39 = tt.splat %arg5 {groups = [@nvws.mma]} : i32 -> tensor<1x128xi32, #blocked2>
      %40 = arith.cmpi slt, %33, %39 {groups = [@nvws.mma]} : tensor<1x128xi32, #blocked2>
      %41 = tt.broadcast %38 {groups = [@nvws.mma]} : tensor<128x1xi1, #blocked2> -> tensor<128x128xi1, #blocked2>
      %42 = tt.broadcast %40 {groups = [@nvws.mma]} : tensor<1x128xi1, #blocked2> -> tensor<128x128xi1, #blocked2>
      %43 = arith.andi %41, %42 {groups = [@nvws.mma]} : tensor<128x128xi1, #blocked2>
      %44 = tt.fp_to_fp %21 {groups = [@nvws.mma]}, rounding = rtne : tensor<128x128xf16, #mma> -> tensor<128x128xf8E4M3FN, #mma>
      %45 = ttg.convert_layout %44 {groups = [@nvws.mma]} : tensor<128x128xf8E4M3FN, #mma> -> tensor<128x128xf8E4M3FN, #blocked2>
      tt.store %36, %45, %43 {groups = [@nvws.mma]} : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>
    } {groups = [@nvws.mma, @nvws.tma_load]}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {nvws.epilogue = {num_warps = 4 : i32, start_warp = 8 : i32}, nvws.mma = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 4 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant {groups = [@nvws.mma]} true
    %c128_i32 = arith.constant {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} 128 : i32
    %c0_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 0 : i32
    %c1_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 1 : i32
    %c127_i32 = arith.constant {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} 127 : i32
    %cst = arith.constant {groups = [@nvws.epilogue, @nvws.mma]} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.get_program_id x {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
    %1 = arith.addi %arg3, %c127_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
    %2 = arith.divsi %1, %c128_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
    %3 = arith.remsi %0, %2 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
    %4 = arith.divsi %0, %2 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
    %5 = arith.muli %3, %c128_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
    %6 = arith.muli %4, %c128_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
    %7 = arith.addi %arg5, %c127_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %8 = arith.divsi %7, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    // CHECK: %[[TMEM:.*]] = ttng.tmem_alloc  {aref_buffer, groups = [@nvws.mma]} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: %[[AREF_TMEM:.*]] = ttng.aref_create %[[TMEM]] {groups = [@nvws.mma]} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %9 = ttng.tmem_alloc %cst {groups = [@nvws.epilogue, @nvws.mma]} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.warp_group start_warp(1) num_warps(1)
    // CHECK:  ttng.warp_group_return
    // CHECK: ttng.warp_group start_warp(0) num_warps(1)
    // CHECK:   %[[PUT_BUF_TMEM:.*]] = ttng.aref_put.enter %[[AREF_TMEM]][{{.*}}]
    // CHECK:   scf.for
    // CHECK:      ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[PUT_BUF_TMEM]]
    // CHECK:   ttng.aref_put.exit %[[AREF_TMEM]][{{.*}}]
    // CHECK:  ttng.warp_group_return
    // CHECK: ttng.warp_group start_warp(2) num_warps(4)
    // CHECK-NOT: scf.for
    // CHECK:  %[[GET_BUF_TMEM:.*]] = ttng.aref_get.enter %[[AREF_TMEM]][{{.*}}]
    // CHECK:  ttng.tmem_load %[[GET_BUF_TMEM]]
    // CHECK: ttng.aref_get.exit %[[AREF_TMEM]][{{.*}}]
    // CHECK:  ttng.warp_group_return
    %10:2 = scf.for %arg7 = %c0_i32 to %8 step %c1_i32 iter_args(%arg8 = %c0_i32, %arg9 = %false) -> (i32, i1)  : i32 {
      %29 = tt.reinterpret_tensor_descriptor %arg0 {groups = [@nvws.tma_load]} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
      %30 = tt.descriptor_load %29[%5, %arg8] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #blocked1>
      %31 = ttg.local_alloc %30 {groups = [@nvws.mma]} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
      %32 = tt.reinterpret_tensor_descriptor %arg1 {groups = [@nvws.tma_load]} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
      %33 = tt.descriptor_load %32[%6, %arg8] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #blocked1>
      %34 = ttg.local_alloc %33 {groups = [@nvws.mma]} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
      %35 = ttg.memdesc_trans %34 {groups = [@nvws.mma], order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
      ttng.tc_gen5_mma %31, %35, %9, %arg9, %true {groups = [@nvws.mma]} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %36 = arith.addi %arg8, %c128_i32 {groups = [@nvws.tma_load]} : i32
      scf.yield %36, %true : i32, i1
    } {groups = [@nvws.mma, @nvws.tma_load], groups.0 = [@nvws.tma_load], groups.1 = [@nvws.mma]}
    %11 = ttng.tmem_load %9 {groups = [@nvws.epilogue]} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %12 = tt.fp_to_fp %11 {groups = [@nvws.epilogue]}, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
    %13 = tt.make_range {end = 128 : i32, groups = [@nvws.epilogue], start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %14 = tt.make_range {end = 128 : i32, groups = [@nvws.epilogue], start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %15 = tt.splat %5 {groups = [@nvws.epilogue]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %16 = arith.addi %15, %13 {groups = [@nvws.epilogue]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %17 = tt.splat %6 {groups = [@nvws.epilogue]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %18 = arith.addi %17, %14 {groups = [@nvws.epilogue]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %19 = tt.expand_dims %16 {axis = 1 : i32, groups = [@nvws.epilogue]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %20 = tt.splat %arg6 {groups = [@nvws.epilogue]} : i32 -> tensor<128x1xi32, #blocked2>
    %21 = arith.muli %20, %19 {groups = [@nvws.epilogue]} : tensor<128x1xi32, #blocked2>
    %22 = tt.splat %arg2 {groups = [@nvws.epilogue]} : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2>
    %23 = tt.addptr %22, %21 {groups = [@nvws.epilogue]} : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<128x1xi32, #blocked2>
    %24 = tt.expand_dims %18 {axis = 0 : i32, groups = [@nvws.epilogue]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
    %25 = tt.broadcast %23 {groups = [@nvws.epilogue]} : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>
    %26 = tt.broadcast %24 {groups = [@nvws.epilogue]} : tensor<1x128xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
    %27 = tt.addptr %25, %26 {groups = [@nvws.epilogue]} : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<128x128xi32, #blocked2>
    %28 = ttg.convert_layout %12 {groups = [@nvws.epilogue]} : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked2>
    tt.store %27, %28 {groups = [@nvws.epilogue]} : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {nvws.epilogue = {num_warps = 4 : i32, start_warp = 8 : i32}, nvws.mma = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 4 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_persistent_tma_ws_cooperative_kernel(%arg0: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg1: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg2: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant {groups = [@nvws.mma]} true
    %c127_i32 = arith.constant {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} 127 : i32
    %c8_i32 = arith.constant {groups = [@nvws.epilogue, @nvws.tma_load]} 8 : i32
    %c128_i32 = arith.constant {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} 128 : i32
    %c0_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 0 : i32
    %c1_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 1 : i32
    %cst = arith.constant {groups = [@nvws.mma]} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = arith.addi %arg4, %c127_i32 {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %1 = arith.divsi %0, %c128_i32 {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %2 = arith.addi %arg5, %c127_i32 {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %3 = arith.divsi %2, %c128_i32 {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %4 = arith.muli %1, %3 {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %5 = tt.get_program_id x {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %6 = tt.get_num_programs x {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    // CHECK: %[[TMEM:.*]] = ttng.tmem_alloc  {aref_buffer, groups = [@nvws.mma]} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: %[[AREF_TMEM:.*]] = ttng.aref_create %[[TMEM]] {groups = [@nvws.mma]} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: ttng.warp_group start_warp(1) num_warps(1)
    // CHECK:  ttng.warp_group_return
    // CHECK: ttng.warp_group start_warp(0) num_warps(1)
    // CHECK:   scf.for %[[OUTER_IDX:.*]] = %[[LB:.*]] to {{.*}} step %[[STEP:.*]] iter_args(%[[OUTER_NORM_IDX:.*]] =
    // CHECK-NOT: tmem_alloc
    // CHECK:     %[[PUT_BUF_TMEM:.*]] = ttng.aref_put.enter %[[AREF_TMEM]][%[[OUTER_NORM_IDX]]]
    // CHECK:     scf.for
    // CHECK:       ttng.tc_gen5_mma {{.*}}, {{.*}}, %[[PUT_BUF_TMEM]]
    // CHECK:     ttng.aref_put.exit %[[AREF_TMEM]][%[[OUTER_NORM_IDX]]]
    // CHECK:  ttng.warp_group_return
    // CHECK: ttng.warp_group start_warp(2) num_warps(4)
    // CHECK: scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[NORM_IDX:.*]] = {{.*}})
    // CHECK-NOT: scf.for
    // CHECK:  %[[GET_BUF_TMEM:.*]] = ttng.aref_get.enter %[[AREF_TMEM]][%[[NORM_IDX]]]
    // CHECK:  ttng.tmem_load %[[GET_BUF_TMEM]]
    // CHECK:  ttng.aref_get.exit %[[AREF_TMEM]][%[[NORM_IDX]]]
    // CHECK: ttng.warp_group_return
    scf.for %arg7 = %5 to %4 step %6  : i32 {
      %7 = arith.muli %3, %c8_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %8 = arith.divsi %arg7, %7 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %9 = arith.muli %8, %c8_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %10 = arith.subi %1, %9 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %11 = arith.minsi %10, %c8_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %12 = arith.remsi %arg7, %7 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %13 = arith.remsi %12, %11 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %14 = arith.addi %9, %13 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %15 = arith.divsi %12, %11 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %16 = arith.muli %14, %c128_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %17 = arith.muli %15, %c128_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %18 = arith.addi %arg6, %c127_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %19 = arith.divsi %18, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      %20 = ttng.tmem_alloc %cst {groups = [@nvws.mma]} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %21:2 = scf.for %arg8 = %c0_i32 to %19 step %c1_i32 iter_args(%arg9 = %c0_i32, %arg10 = %false) -> (i32, i1)  : i32 {
        %47 = tt.reinterpret_tensor_descriptor %arg0 {groups = [@nvws.tma_load]} : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<128x128xf16>>
        %48 = tt.descriptor_load %47[%16, %arg9] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #blocked1>
        %49 = ttg.local_alloc %48 {groups = [@nvws.mma]} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %50 = tt.reinterpret_tensor_descriptor %arg1 {groups = [@nvws.tma_load]} : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<128x128xf16>>
        %51 = tt.descriptor_load %50[%arg9, %17] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #blocked1>
        %52 = ttg.local_alloc %51 {groups = [@nvws.mma]} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        ttng.tc_gen5_mma %49, %52, %20, %arg10, %true {groups = [@nvws.mma]} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %53 = arith.addi %arg9, %c128_i32 {groups = [@nvws.tma_load]} : i32
        scf.yield %53, %true : i32, i1
      } {groups = [@nvws.mma, @nvws.tma_load], groups.0 = [@nvws.tma_load], groups.1 = [@nvws.mma]}
      %22 = ttng.tmem_load %20 {groups = [@nvws.epilogue]} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %23 = arith.truncf %22 {groups = [@nvws.epilogue]} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
      %24 = tt.make_range {end = 128 : i32, groups = [@nvws.epilogue], start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %25 = tt.make_range {end = 128 : i32, groups = [@nvws.epilogue], start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %26 = tt.splat %16 {groups = [@nvws.epilogue]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %27 = arith.addi %26, %24 {groups = [@nvws.epilogue]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %28 = tt.splat %17 {groups = [@nvws.epilogue]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %29 = arith.addi %28, %25 {groups = [@nvws.epilogue]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %30 = tt.expand_dims %27 {axis = 1 : i32, groups = [@nvws.epilogue]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
      %31 = tt.splat %arg5 {groups = [@nvws.epilogue]} : i32 -> tensor<128x1xi32, #blocked2>
      %32 = arith.muli %30, %31 {groups = [@nvws.epilogue]} : tensor<128x1xi32, #blocked2>
      %33 = tt.splat %arg3 {groups = [@nvws.epilogue]} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
      %34 = tt.addptr %33, %32 {groups = [@nvws.epilogue]} : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
      %35 = tt.expand_dims %29 {axis = 0 : i32, groups = [@nvws.epilogue]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
      %36 = tt.broadcast %34 {groups = [@nvws.epilogue]} : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x128x!tt.ptr<f16>, #blocked2>
      %37 = tt.broadcast %35 {groups = [@nvws.epilogue]} : tensor<1x128xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
      %38 = tt.addptr %36, %37 {groups = [@nvws.epilogue]} : tensor<128x128x!tt.ptr<f16>, #blocked2>, tensor<128x128xi32, #blocked2>
      %39 = tt.splat %arg4 {groups = [@nvws.epilogue]} : i32 -> tensor<128x1xi32, #blocked2>
      %40 = arith.cmpi slt, %30, %39 {groups = [@nvws.epilogue]} : tensor<128x1xi32, #blocked2>
      %41 = tt.splat %arg5 {groups = [@nvws.epilogue]} : i32 -> tensor<1x128xi32, #blocked2>
      %42 = arith.cmpi slt, %35, %41 {groups = [@nvws.epilogue]} : tensor<1x128xi32, #blocked2>
      %43 = tt.broadcast %40 {groups = [@nvws.epilogue]} : tensor<128x1xi1, #blocked2> -> tensor<128x128xi1, #blocked2>
      %44 = tt.broadcast %42 {groups = [@nvws.epilogue]} : tensor<1x128xi1, #blocked2> -> tensor<128x128xi1, #blocked2>
      %45 = arith.andi %43, %44 {groups = [@nvws.epilogue]} : tensor<128x128xi1, #blocked2>
      %46 = ttg.convert_layout %23 {groups = [@nvws.epilogue]} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2>
      tt.store %38, %46, %45 {groups = [@nvws.epilogue]} : tensor<128x128x!tt.ptr<f16>, #blocked2>
    } {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, unpacked = true>
module attributes {nvws.epilogue = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.mma = {num_warps = 4 : i32, start_warp = 4 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 8 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @_matmul_og_ptma_NNT_fp8e4nvxfp8e4nvxfp8e4nv_64x256x128x1(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg20: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg21: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg22: !tt.ptr<i32>, %arg23: !tt.ptr<i32>, %arg24: i32, %arg25: i32) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant {groups = [@nvws.tma_load]} dense<4> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_0 = arith.constant {groups = [@nvws.tma_load]} dense<-4> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c0_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 0 : i32
    %c2147483647_i32 = arith.constant {groups = [@nvws.tma_load]} 2147483647 : i32
    %c128_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 128 : i32
    %cst_1 = arith.constant {groups = [@nvws.epilogue]} dense<0.000000e+00> : tensor<256xf32, #blocked1>
    %c1_i64 = arith.constant {groups = [@nvws.epilogue, @nvws.tma_load]} 1 : i64
    %c148_i32 = arith.constant {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} 148 : i32
    %c1_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 1 : i32
    %c127_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 127 : i32
    %c8_i32 = arith.constant {groups = [@nvws.epilogue, @nvws.tma_load]} 8 : i32
    %c256_i32 = arith.constant {groups = [@nvws.epilogue, @nvws.tma_load]} 256 : i32
    %c64_i32 = arith.constant {groups = [@nvws.epilogue, @nvws.tma_load]} 64 : i32
    %c16_i32 = arith.constant {groups = [@nvws.epilogue, @nvws.tma_load]} 16 : i32
    %c65535_i32 = arith.constant {groups = [@nvws.epilogue, @nvws.tma_load]} 65535 : i32
    %true = arith.constant {groups = [@nvws.mma]} true
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #blocked2>
    %0 = tt.load %arg22 {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : !tt.ptr<i32>
    %1 = arith.subi %arg24, %0 {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %2 = arith.extsi %arg7 {groups = [@nvws.tma_load]} : i32 to i64
    %3 = tt.make_tensor_descriptor %arg5, [%c2147483647_i32, %arg18], [%2, %c1_i64] {groups = [@nvws.tma_load]} : <f8E4M3FN>, <tensor<1x128xf8E4M3FN, #shared>>
    %4 = arith.extsi %arg9 {groups = [@nvws.tma_load]} : i32 to i64
    %5 = arith.extsi %arg10 {groups = [@nvws.tma_load]} : i32 to i64
    %6 = tt.make_tensor_descriptor %arg8, [%c128_i32, %arg17, %arg18], [%4, %5, %c1_i64] {groups = [@nvws.tma_load]} : <f8E4M3FN>, <tensor<1x256x128xf8E4M3FN, #shared1>>
    %7 = arith.addi %arg18, %c127_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %8 = arith.divsi %7, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %9 = arith.subi %arg24, %1 {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %10 = arith.muli %9, %arg25 {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %11 = tt.get_program_id x {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    scf.for %arg26 = %11 to %10 step %c148_i32  : i32 {
      %12 = arith.remsi %arg26, %10 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %13 = arith.muli %arg25, %c8_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %14 = arith.divsi %12, %13 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %15 = arith.muli %14, %c8_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %16 = arith.subi %9, %15 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %17 = arith.minsi %16, %c8_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %18 = arith.remsi %12, %17 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %19 = arith.addi %15, %18 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %20 = arith.remsi %12, %13 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %21 = arith.divsi %20, %17 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %22 = tt.addptr %arg23, %19 {groups = [@nvws.epilogue, @nvws.tma_load]} : !tt.ptr<i32>, i32
      %23 = tt.load %22 {groups = [@nvws.epilogue, @nvws.tma_load]} : !tt.ptr<i32>
      %24 = arith.andi %23, %c65535_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %25 = arith.shrsi %23, %c16_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %26 = tt.addptr %arg20, %24 {groups = [@nvws.epilogue, @nvws.tma_load]} : !tt.ptr<i32>, i32
      %27 = tt.load %26 {groups = [@nvws.epilogue, @nvws.tma_load]} : !tt.ptr<i32>
      %28 = tt.addptr %arg21, %24 {groups = [@nvws.epilogue, @nvws.tma_load]} : !tt.ptr<i32>, i32
      %29 = tt.load %28 {groups = [@nvws.epilogue, @nvws.tma_load]} : !tt.ptr<i32>
      %30 = arith.muli %25, %c64_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %31 = arith.muli %21, %c256_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
      %32 = tt.make_range {end = 64 : i32, groups = [@nvws.tma_load], start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %33 = tt.splat %30 {groups = [@nvws.tma_load]} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %34 = arith.addi %33, %32 {groups = [@nvws.tma_load]} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %35 = tt.splat %27 {groups = [@nvws.tma_load]} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %36 = arith.cmpi slt, %34, %35 {groups = [@nvws.tma_load]} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %37 = arith.extsi %29 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32 to i64
      %38 = tt.addptr %arg19, %37 {groups = [@nvws.tma_load]} : !tt.ptr<i32>, i64
      %39 = tt.splat %38 {groups = [@nvws.tma_load]} : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked}>>
      %40 = tt.addptr %39, %34 {groups = [@nvws.tma_load]} : tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %41 = tt.load %40, %36, %cst_0 {groups = [@nvws.tma_load]} : tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked}>>
      %42 = arith.divsi %41, %cst {groups = [@nvws.tma_load]} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %43 = ttng.tmem_alloc {groups = [@nvws.mma]} : () -> !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tmem_store %cst_2, %43, %true {groups = [@nvws.mma]} : tensor<64x256xf32, #blocked2> -> !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>
      // TMA gather involves a small tensor computation - the TMA group should have 4 warps
      // CHECK: ttng.warp_group start_warp(5) num_warps(4)
      // CHECK:  ttng.warp_group_return
      %44 = scf.for %arg27 = %c0_i32 to %8 step %c1_i32 iter_args(%arg28 = %false) -> (i1)  : i32 {
        %68 = arith.muli %arg27, %c128_i32 {groups = [@nvws.tma_load]} : i32
        %69 = tt.descriptor_gather %3[%42, %68] {groups = [@nvws.tma_load]} : (!tt.tensordesc<tensor<1x128xf8E4M3FN, #shared>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32) -> tensor<64x128xf8E4M3FN, #blocked3>
        %70 = ttg.local_alloc %69 {groups = [@nvws.mma]} : (tensor<64x128xf8E4M3FN, #blocked3>) -> !ttg.memdesc<64x128xf8E4M3FN, #shared, #smem>
        %71 = tt.descriptor_load %6[%24, %31, %68] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<1x256x128xf8E4M3FN, #shared1>> -> tensor<256x128xf8E4M3FN, #blocked3>
        %72 = ttg.local_alloc %71 {groups = [@nvws.mma]} : (tensor<256x128xf8E4M3FN, #blocked3>) -> !ttg.memdesc<256x128xf8E4M3FN, #shared, #smem>
        %73 = ttg.memdesc_trans %72 {groups = [@nvws.mma], order = array<i32: 1, 0>} : !ttg.memdesc<256x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x256xf8E4M3FN, #shared2, #smem>
        ttng.tc_gen5_mma %70, %73, %43, %arg28, %true {groups = [@nvws.mma]} : !ttg.memdesc<64x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x256xf8E4M3FN, #shared2, #smem>, !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true : i1
      } {groups = [@nvws.mma, @nvws.tma_load], groups.0 = [@nvws.mma]}
      %45 = arith.extsi %arg4 {groups = [@nvws.epilogue]} : i32 to i64
      %46 = arith.muli %37, %45 {groups = [@nvws.epilogue]} : i64
      %47 = tt.addptr %arg1, %46 {groups = [@nvws.epilogue]} : !tt.ptr<f8E4M3FN>, i64
      %48 = tt.make_tensor_descriptor %47, [%27, %arg17], [%45, %c1_i64] {groups = [@nvws.epilogue]} : <f8E4M3FN>, <tensor<64x256xf8E4M3FN, #shared>>
      %49 = tt.make_range {end = 256 : i32, groups = [@nvws.epilogue], start = 0 : i32} : tensor<256xi32, #blocked1>
      %50 = tt.splat %31 {groups = [@nvws.epilogue]} : i32 -> tensor<256xi32, #blocked1>
      %51 = arith.addi %50, %49 {groups = [@nvws.epilogue]} : tensor<256xi32, #blocked1>
      %52 = tt.splat %arg17 {groups = [@nvws.epilogue]} : i32 -> tensor<256xi32, #blocked1>
      %53 = arith.cmpi slt, %51, %52 {groups = [@nvws.epilogue]} : tensor<256xi32, #blocked1>
      %54 = tt.addptr %arg15, %24 {groups = [@nvws.epilogue]} : !tt.ptr<f32>, i32
      %55 = tt.splat %54 {groups = [@nvws.epilogue]} : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked1>
      %56 = tt.addptr %55, %51 {groups = [@nvws.epilogue]} : tensor<256x!tt.ptr<f32>, #blocked1>, tensor<256xi32, #blocked1>
      %57 = tt.load %56, %53, %cst_1 {groups = [@nvws.epilogue]} : tensor<256x!tt.ptr<f32>, #blocked1>
      %58 = tt.load %arg11 {groups = [@nvws.epilogue]} : !tt.ptr<f32>
      %59 = tt.splat %58 {groups = [@nvws.epilogue]} : f32 -> tensor<64x256xf32, #blocked2>
      %60 = ttng.tmem_load %43 {groups = [@nvws.epilogue]} : !ttg.memdesc<64x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x256xf32, #blocked2>
      %61 = arith.mulf %60, %59 {groups = [@nvws.epilogue]} : tensor<64x256xf32, #blocked2>
      %62 = ttg.convert_layout %57 {groups = [@nvws.epilogue]} : tensor<256xf32, #blocked1> -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %63 = tt.expand_dims %62 {axis = 0 : i32, groups = [@nvws.epilogue]} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xf32, #blocked2>
      %64 = tt.broadcast %63 {groups = [@nvws.epilogue]} : tensor<1x256xf32, #blocked2> -> tensor<64x256xf32, #blocked2>
      %65 = arith.addf %61, %64 {groups = [@nvws.epilogue]} : tensor<64x256xf32, #blocked2>
      %66 = tt.fp_to_fp %65 {groups = [@nvws.epilogue]}, rounding = rtne : tensor<64x256xf32, #blocked2> -> tensor<64x256xf8E4M3FN, #blocked2>
      %67 = ttg.convert_layout %66 {groups = [@nvws.epilogue]} : tensor<64x256xf8E4M3FN, #blocked2> -> tensor<64x256xf8E4M3FN, #blocked3>
      tt.descriptor_store %48[%30, %31], %67 {groups = [@nvws.epilogue]} : !tt.tensordesc<tensor<64x256xf8E4M3FN, #shared>>, tensor<64x256xf8E4M3FN, #blocked3>
    } {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]}
    tt.return
  }
}
