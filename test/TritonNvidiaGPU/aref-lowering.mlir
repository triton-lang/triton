// RUN: triton-opt %s -split-input-file --triton-nvidia-aref-lowering | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {nvws.mma = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 4 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_kernel_tma_persistent(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c148_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 148 : i32
    %c1_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 128 : i32
    %c127_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 127 : i32
    %cst = arith.constant {groups = [@nvws.mma]} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.get_program_id x {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %1 = arith.addi %arg3, %c127_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %2 = arith.divsi %1, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %3 = arith.addi %arg4, %c127_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %4 = arith.divsi %3, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %5 = arith.addi %arg5, %c127_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %6 = arith.divsi %5, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %7 = arith.muli %2, %4 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %8 = arith.divsi %7, %c148_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %9 = arith.remsi %7, %c148_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %10 = arith.cmpi slt, %0, %9 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    %11 = scf.if %10 -> (i32) {
      %25 = arith.addi %8, %c1_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
      scf.yield %25 : i32
    } else {
      scf.yield %8 : i32
    } {groups = [@nvws.mma, @nvws.tma_load]}
    %12 = arith.subi %0, %c148_i32 : i32
    %13 = arith.muli %4, %c8_i32 : i32
    %14 = arith.muli %6, %11 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    // CHECK-DAG: %[[ACC:.*]] = ttng.tmem_alloc
    %15 = ttng.tmem_alloc %cst {groups = [@nvws.mma]} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %16 = ttg.local_alloc  {aref_buffer} : () -> !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>
    %17 = ttg.local_alloc  {aref_buffer} : () -> !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>
    %18 = ttng.aref_create %16, %17 : <[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>
    // CHECK: scf.for %
    // CHECK: ttg.memdesc_subview
    // CHECK: ttg.memdesc_subview
    // CHECK: }
    // CHECK-DAG: %[[FINAL_BARRIER:.*]] = ttg.local_alloc
    // CHECK: ttng.init_barrier %[[FINAL_BARRIER]], 1
    %19 = nvvm.read.ptx.sreg.tid.x : i32
    %20 = arith.divsi %19, %c128_i32 : i32
    nvvm.barrier0 {init_barrier_sync}
    %21 = arith.cmpi eq, %20, %c1_i32 {groups = [@nvws.tma_load]} : i32
    ttng.warp_group start_warp(4) num_warps(4) :  {{
      // CHECK: scf.for
      %25:5 = scf.for %arg6 = %c0_i32 to %14 step %c1_i32 iter_args(%arg7 = %c-1_i32, %arg8 = %12, %arg9 = %c0_i32, %arg10 = %c0_i32, %arg11 = %c0_i32) -> (i32, i32, i32, i32, i32)  : i32 {
        %26 = arith.subi %6, %c1_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
        %27 = arith.cmpi eq, %arg7, %26 {groups = [@nvws.tma_load]} : i32
        %28 = arith.addi %arg7, %c1_i32 {groups = [@nvws.tma_load]} : i32
        %29 = arith.select %27, %c0_i32, %28 {groups = [@nvws.tma_load]} : i32
        %30 = arith.cmpi eq, %29, %c0_i32 {groups = [@nvws.tma_load]} : i32
        %31:3 = scf.if %30 -> (i32, i32, i32) {
          %36 = arith.addi %arg8, %c148_i32 {groups = [@nvws.tma_load]} : i32
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
        %32 = arith.muli %29, %c128_i32 {groups = [@nvws.tma_load]} : i32
        %c1_i64 = arith.constant {groups = [@nvws.epilogue, @nvws.tma_load]} 1 : i64
        %00 = arith.extsi %arg5 {groups = [@nvws.tma_load]} : i32 to i64
        %33 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%00, %c1_i64] {groups = [@nvws.tma_load]} : <f8E4M3FN>, <tensor<128x128xf8E4M3FN, #shared>>
        %34 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%00, %c1_i64] {groups = [@nvws.tma_load]} : <f8E4M3FN>, <tensor<128x128xf8E4M3FN, #shared>>
        ttng.aref_put %18[%arg11] {groups = [@nvws.tma_load]} : <[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>, i32 {
        ^bb0(%arg12: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg13: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>):
          %36 = tt.descriptor_load %33[%31#1, %32] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked>
          ttg.local_store %36, %arg12 {groups = [@nvws.tma_load]} : tensor<128x128xf8E4M3FN, #blocked> -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>
          %37 = tt.descriptor_load %34[%31#2, %32] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked>
          ttg.local_store %37, %arg13 {groups = [@nvws.tma_load]} : tensor<128x128xf8E4M3FN, #blocked> -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>
          ttng.aref_return
        }
        %35 = arith.addi %arg11, %c1_i32 {groups = [@nvws.tma_load]} : i32
        scf.yield {groups = [@nvws.tma_load]} %29, %31#0, %31#1, %31#2, %35 : i32, i32, i32, i32, i32
      } {groups = [@nvws.tma_load]}
      ttng.warp_group_return
    } {barId = 1 : i32, groups = [@nvws.tma_load]}}
    %22 = arith.cmpi sge, %20, %c0_i32 {groups = [@nvws.mma]} : i32
    %23 = arith.cmpi slt, %20, %c1_i32 {groups = [@nvws.mma]} : i32
    %24 = arith.andi %22, %23 {groups = [@nvws.mma]} : i1
    ttng.warp_group start_warp(4) num_warps(4) : {{
      // CHECK: scf.for {{.*}} iter_args(%arg{{[1-9]+}} = %{{.*}}, %arg{{[1-9]+}} = %{{.*}}, %arg{{[1-9]+}} = %{{.*}}, %[[TMA_PHASE:.*]] = %c0_i32, %[[MMA_COMPLETION_PHASE:.*]] = %c0_i32)
      %25:3 = scf.for %arg6 = %c0_i32 to %14 step %c1_i32 iter_args(%arg7 = %c-1_i32, %arg8 = %false, %arg9 = %c0_i32) -> (i32, i1, i32)  : i32 {
        %26:2 = ttng.aref_get.enter %18[%arg9] {groups = [@nvws.mma]} : <[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>
        %27 = ttg.memdesc_trans %26#1 {groups = [@nvws.mma], order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>
        %28 = arith.subi %6, %c1_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
        // CHECK: %[[TMA_BARRIER:.*]] = ttg.memdesc_subview
        // CHECK: ttng.wait_barrier %[[TMA_BARRIER]], %[[TMA_PHASE]]
        // CHECK-DAG: %[[A:.*]] = ttg.memdesc_subview
        // CHECK-DAG: %[[B:.*]] = ttg.memdesc_subview
        // CHECK-DAG: %[[B_TRANS:.*]] = ttg.memdesc_trans %[[B]]
        // CHECK-DAG: %[[UPDATED_TMA_PHASE:.*]] = arith.xori %[[TMA_PHASE]], %c1_i32 : i32
        // CHECK-DAG: %[[NEXT_TMA_PHASE:.*]] = arith.select %{{[1-9]+}}, %[[UPDATED_TMA_PHASE]], %[[TMA_PHASE]] : i32
        // CHECK-DAG: %[[BARRIER:.*]] = ttg.memdesc_subview
        // CHECK: tc_gen5_mma %[[A]], %[[B_TRANS]], %[[ACC]], {{.*}}, %true, %[[BARRIER]]
        ttng.tc_gen5_mma %26#0, %27, %15, %arg8, %true, %18[%arg9] {groups = [@nvws.mma]} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttng.aref<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>]>, i32
        %29 = arith.cmpi eq, %arg7, %28 {groups = [@nvws.mma]} : i32
        %30 = arith.addi %arg7, %c1_i32 {groups = [@nvws.mma]} : i32
        %31 = arith.select %29, %c0_i32, %30 {groups = [@nvws.mma]} : i32
        %32 = arith.cmpi eq, %31, %28 {groups = [@nvws.mma]} : i32
        %33 = arith.cmpi ne, %31, %28 {groups = [@nvws.mma]} : i32
	      // CHECK: %[[NEXT_PHASE:.*]] = scf.if
        scf.if %32 {
          %35 = arith.addi %arg6, %c1_i32 : i32
          %36 = arith.divsi %35, %6 : i32
          %37 = arith.muli %36, %c148_i32 : i32
          %38 = arith.addi %12, %37 : i32
          %39 = arith.divsi %38, %13 : i32
          %40 = arith.muli %39, %c8_i32 : i32
          %41 = arith.subi %2, %40 : i32
          %42 = arith.minsi %41, %c8_i32 : i32
          %43 = arith.remsi %38, %42 : i32
          %44 = arith.addi %40, %43 : i32
          %45 = arith.remsi %38, %13 : i32
          %46 = arith.divsi %45, %42 : i32
          %47 = arith.muli %44, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
          %48 = arith.muli %46, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
          // CHECK: nvws.arrive_barrier %[[FINAL_BARRIER]] true
          // CHECK: ttng.wait_barrier %[[FINAL_BARRIER]], %[[MMA_COMPLETION_PHASE]]
          // CHECK-DAG: %[[UPDATED_PHASE:.*]] = arith.xori %[[MMA_COMPLETION_PHASE]]
          // CHECK: ttng.tmem_load
          %49 = ttng.tmem_load %15 {groups = [@nvws.mma]} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          %50 = tt.fp_to_fp %49, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
          %51 = ttg.convert_layout %50 : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked1>
          %52 = tt.reinterpret_tensor_descriptor %arg2 : !tt.ptr<f32> to !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>
          tt.descriptor_store %52[%47, %48], %51 {groups = [@nvws.mma]} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, tensor<128x128xf8E4M3FN, #blocked1>
  	    // CHECK: scf.yield %[[UPDATED_PHASE]]
        } {groups = [@nvws.mma]}
        // CHECK: } else {
        // CHECK: scf.yield %[[MMA_COMPLETION_PHASE]]
        // CHECK: }
        // CHECK: scf.yield %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %[[NEXT_TMA_PHASE]], %[[NEXT_PHASE]]
        %34 = arith.addi %arg9, %c1_i32 {groups = [@nvws.mma]} : i32
        scf.yield {groups = [@nvws.mma]} %31, %33, %34 : i32, i1, i32
      } {groups = [@nvws.mma]}
      ttng.warp_group_return
    } {barId = 2 : i32, groups = [@nvws.mma]}}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {nvws.epilogue = {num_warps = 4 : i32, start_warp = 8 : i32}, nvws.mma = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 4 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_persistent_tma_ws_cooperative_kernel(%arg0: !tt.ptr<f16> {tt.nv_tma_desc = 1 : i32}, %arg1: !tt.ptr<f16> {tt.nv_tma_desc = 1 : i32}, %arg2: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant {groups = [@nvws.mma]} true
    %c127_i32 = arith.constant {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} 127 : i32
    %c8_i32 = arith.constant {groups = [@nvws.epilogue, @nvws.tma_load]} 8 : i32
    // CHECK-DAG: %[[C128:.*]] = arith.constant {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} 128
    // CHECK-DAG: %[[C1:.*]] = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 1 : i32
    // CHECK-DAG: %[[C2:.*]] = arith.constant 2
    %c128_i32 = arith.constant {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} 128 : i32
    %c0_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 0 : i32
    %c1_i32 = arith.constant {groups = [@nvws.mma, @nvws.tma_load]} 1 : i32
    %0 = arith.addi %arg4, %c127_i32 {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %1 = arith.divsi %0, %c128_i32 {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %2 = arith.addi %arg5, %c127_i32 {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %3 = arith.divsi %2, %c128_i32 {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %4 = arith.muli %1, %3 {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %5 = tt.get_program_id x {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %6 = tt.get_num_programs x {groups = [@nvws.epilogue, @nvws.mma, @nvws.tma_load]} : i32
    %7 = ttg.local_alloc  {aref_buffer, groups = [@nvws.tma_load]} : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    %8 = ttg.local_alloc  {aref_buffer, groups = [@nvws.tma_load]} : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    %9 = ttng.aref_create %7, %8 {groups = [@nvws.tma_load]} : <[!ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: %[[TMEM:.*]] = ttng.tmem_alloc
    %10 = ttng.tmem_alloc  {aref_buffer, groups = [@nvws.mma]} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: %[[TMEM_EMPTY:.*]] = ttg.local_alloc  {aref_empty_mbarriers} : () -> !ttg.memdesc<2xi64
    // CHECK: %[[TMEM_FULL:.*]] = ttg.local_alloc  {aref_full_mbarriers} : () -> !ttg.memdesc<2xi64
    // CHECK: %[[TMEM_EMPTY_SLICE:.*]] = ttg.memdesc_subview %[[TMEM_EMPTY]]
    // CHECK: init_barrier %[[TMEM_EMPTY_SLICE]], 128
    // CHECK: %[[TMEM_FULL_SLICE:.*]] = ttg.memdesc_subview %[[TMEM_FULL]]
    // CHECK: init_barrier %[[TMEM_FULL_SLICE]], 1
    %11 = ttng.aref_create %10 {groups = [@nvws.mma]} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    nvvm.barrier0 {init_barrier_sync}
    // CHECK: ttng.warp_group start_warp(4) num_warps(4)
    ttng.warp_group start_warp(4) num_warps(4) : {{
    // CHECK: scf.for
      %12 = scf.for %arg7 = %5 to %4 step %6 iter_args(%arg8 = %c0_i32) -> (i32)  : i32 {
        %13 = arith.muli %3, %c8_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %14 = arith.divsi %arg7, %13 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %15 = arith.muli %14, %c8_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %16 = arith.subi %1, %15 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %17 = arith.minsi %16, %c8_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %18 = arith.remsi %arg7, %13 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %19 = arith.remsi %18, %17 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %20 = arith.addi %15, %19 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %21 = arith.divsi %18, %17 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %22 = arith.muli %20, %c128_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %23 = arith.muli %21, %c128_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %24 = arith.addi %arg6, %c127_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
        %25 = arith.divsi %24, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
    	// CHECK: scf.for
	// CHECK: nvvm.barrier id =
        %26:2 = scf.for %arg9 = %c0_i32 to %25 step %c1_i32 iter_args(%arg10 = %c0_i32, %arg11 = %arg8) -> (i32, i32)  : i32 {
          %c1_i64 = arith.constant {groups = [@nvws.epilogue, @nvws.tma_load]} 1 : i64
          %00 = arith.extsi %arg5 {groups = [@nvws.tma_load]} : i32 to i64
          %27 = tt.make_tensor_descriptor %arg0, [%arg4, %arg5], [%00, %c1_i64] {groups = [@nvws.tma_load]} : <f16>, <tensor<128x128xf16, #shared>>
          %28 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%00, %c1_i64] {groups = [@nvws.tma_load]} : <f16>, <tensor<128x128xf16, #shared>>
          ttng.aref_put %9[%arg11] {groups = [@nvws.tma_load]} : <[!ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>]>, i32 {
          ^bb0(%arg12: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg13: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>):
            %31 = tt.descriptor_load %27[%22, %arg10] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked>
            ttg.local_store %31, %arg12 {groups = [@nvws.tma_load]} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
            %32 = tt.descriptor_load %28[%arg10, %23] {groups = [@nvws.tma_load]} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked>
            ttg.local_store %32, %arg13 {groups = [@nvws.tma_load]} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
            ttng.aref_return {groups = [@nvws.tma_load]}
          }
          %29 = arith.addi %arg10, %c128_i32 {groups = [@nvws.tma_load]} : i32
          %30 = arith.addi %arg11, %c1_i32 {groups = [@nvws.tma_load]} : i32
          scf.yield {groups = [@nvws.tma_load]} %29, %30 : i32, i32
        } {groups = [@nvws.tma_load]}
        scf.yield {groups = [@nvws.tma_load]} %26#1 : i32
      } {groups = [@nvws.tma_load]}
      ttng.warp_group_return
    } {barId = 1 : i32, groups = [@nvws.tma_load]}}
    // CHECK: ttng.warp_group start_warp(0) num_warps(4)
    ttng.warp_group start_warp(0) num_warps(4) : {{
      // CHECK:   scf.for %[[OUTER_IDX:.*]] = %[[LB:.*]] to {{.*}} step %[[STEP:.*]] iter_args(%[[NORM_IDX:.*]] = {{.*}}, %[[TMEM_EMPTY_PHASE:.*]] = %[[C1]]
      %12 = scf.for %arg7 = %5 to %4 step %6 iter_args(%arg8 = %c0_i32) -> (i32)  : i32 {
        // CHECK: %[[SUB:.*]] = arith.subi %[[OUTER_IDX]], %[[LB]]
        // CHECK: %[[OUTER_NORM_IDX:.*]] = arith.divsi %[[SUB]], %[[STEP]]
	// CHECK: %[[STAGE_IDX:.*]] = arith.remsi %[[OUTER_NORM_IDX]], %[[C2]]
	// CHECK: %[[TMEM_EMPTY_SLICE:.*]] = ttg.memdesc_subview %[[TMEM_EMPTY]][%[[STAGE_IDX]]]
	// CHECK: ttng.wait_barrier %[[TMEM_EMPTY_SLICE]], %[[TMEM_EMPTY_PHASE]]
        %13 = arith.subi %arg7, %5 {groups = [@nvws.mma]} : i32
        %14 = arith.divsi %13, %6 {groups = [@nvws.mma]} : i32
        %15 = arith.addi %arg6, %c127_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
        %16 = arith.divsi %15, %c128_i32 {groups = [@nvws.mma, @nvws.tma_load]} : i32
        %17 = ttng.aref_put.enter %11[%14] {groups = [@nvws.mma]} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, i32 -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
	// CHECK: scf.for
	// CHECK: nvvm.barrier id =
        %18:2 = scf.for %arg9 = %c0_i32 to %16 step %c1_i32 iter_args(%arg10 = %false, %arg11 = %arg8) -> (i1, i32)  : i32 {
          %19:2 = ttng.aref_get.enter %9[%arg11] {groups = [@nvws.mma]} : <[!ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>]>, i32 -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
          ttng.tc_gen5_mma %19#0, %19#1, %17, %arg10, %true, %9[%arg11] {groups = [@nvws.mma]} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttng.aref<[!ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>]>, i32
          %20 = arith.addi %arg11, %c1_i32 {groups = [@nvws.mma]} : i32
          scf.yield {groups = [@nvws.mma]} %true, %20 : i1, i32
        } {groups = [@nvws.mma]}
	// CHECK: }
	// CHECK: %[[STAGE_IDX2:.*]] = arith.remsi %[[OUTER_NORM_IDX]], %[[C2]]
	// CHECK: %[[TMEM_FULL_SLICE:.*]] = ttg.memdesc_subview %[[TMEM_FULL]][%[[STAGE_IDX2]]]
	// CHECK: nvws.arrive_barrier %[[TMEM_FULL_SLICE]] true
        ttng.aref_put.exit %11[%14] {groups = [@nvws.mma]} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, i32
        scf.yield {groups = [@nvws.mma]} %18#1 : i32
      } {groups = [@nvws.mma]}
      ttng.warp_group_return
    } {barId = 2 : i32, groups = [@nvws.mma]}}
    // CHECK: ttng.warp_group start_warp(8) num_warps(4)
    // CHECK: for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[OUTER_IDX:.*]] = {{.*}}, %[[PHASE:.*]] = {{.*}})
    ttng.warp_group start_warp(8) num_warps(4) : {{
      %12 = scf.for %arg7 = %5 to %4 step %6 iter_args(%arg8 = %c0_i32) -> (i32)  : i32 {
        %13 = arith.muli %3, %c8_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %14 = arith.divsi %arg7, %13 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %15 = arith.muli %14, %c8_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %16 = arith.subi %1, %15 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %17 = arith.minsi %16, %c8_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %18 = arith.remsi %arg7, %13 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %19 = arith.remsi %18, %17 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %20 = arith.addi %15, %19 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %21 = arith.divsi %18, %17 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %22 = arith.muli %20, %c128_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
        %23 = arith.muli %21, %c128_i32 {groups = [@nvws.epilogue, @nvws.tma_load]} : i32
	// CHECK-DAG: %[[TMEM_BUF_IDX:.*]] = arith.remsi %[[OUTER_IDX]], {{.*}}
	// CHECK-DAG: %[[TMEM_BUF_IDX2:.*]] = arith.remsi %[[OUTER_IDX]], {{.*}}
        // CHECK-DAG: %[[TMEM_FULL_SLICE:.*]] = ttg.memdesc_subview %[[TMEM_FULL]][%[[TMEM_BUF_IDX]]]
	// CHECK-DAG: %[[TMEM_SLICE:.*]] = ttg.memdesc_subview %[[TMEM]][%[[TMEM_BUF_IDX2]],
	// CHECK-DAG: wait_barrier %[[TMEM_FULL_SLICE]], %[[PHASE]]
        %24 = ttng.aref_get.enter %11[%arg8] {groups = [@nvws.epilogue]} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, i32 -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

        // CHECK: ttng.tmem_load %[[TMEM_SLICE]]
        %25 = ttng.tmem_load %24 {groups = [@nvws.epilogue]} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>

	// CHECK: %[[TMEM_BUF_IDX3:.*]] = arith.remsi %[[OUTER_IDX]], {{.*}}
        // CHECK: %[[TMEM_EMPTY_SLICE:.*]] = ttg.memdesc_subview %[[TMEM_EMPTY]][%[[TMEM_BUF_IDX3]]]
	// CHECK: nvws.arrive_barrier %[[TMEM_EMPTY_SLICE]] false
        ttng.aref_get.exit %11[%arg8] {groups = [@nvws.epilogue]} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, i32
        %26 = arith.truncf %25 {groups = [@nvws.epilogue]} : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
        %27 = tt.make_range {end = 128 : i32, groups = [@nvws.epilogue], start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %28 = tt.make_range {end = 128 : i32, groups = [@nvws.epilogue], start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %29 = tt.splat %22 {groups = [@nvws.epilogue]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %30 = arith.addi %29, %27 {groups = [@nvws.epilogue]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %31 = tt.splat %23 {groups = [@nvws.epilogue]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %32 = arith.addi %31, %28 {groups = [@nvws.epilogue]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %33 = tt.expand_dims %30 {axis = 1 : i32, groups = [@nvws.epilogue]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
        %34 = tt.splat %arg5 {groups = [@nvws.epilogue]} : i32 -> tensor<128x1xi32, #blocked2>
        %35 = arith.muli %33, %34 {groups = [@nvws.epilogue]} : tensor<128x1xi32, #blocked2>
        %36 = tt.splat %arg3 {groups = [@nvws.epilogue]} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
        %37 = tt.addptr %36, %35 {groups = [@nvws.epilogue]} : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
        %38 = tt.expand_dims %32 {axis = 0 : i32, groups = [@nvws.epilogue]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
        %39 = tt.broadcast %37 {groups = [@nvws.epilogue]} : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x128x!tt.ptr<f16>, #blocked2>
        %40 = tt.broadcast %38 {groups = [@nvws.epilogue]} : tensor<1x128xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
        %41 = tt.addptr %39, %40 {groups = [@nvws.epilogue]} : tensor<128x128x!tt.ptr<f16>, #blocked2>, tensor<128x128xi32, #blocked2>
        %42 = tt.splat %arg4 {groups = [@nvws.epilogue]} : i32 -> tensor<128x1xi32, #blocked2>
        %43 = arith.cmpi slt, %33, %42 {groups = [@nvws.epilogue]} : tensor<128x1xi32, #blocked2>
        %44 = tt.splat %arg5 {groups = [@nvws.epilogue]} : i32 -> tensor<1x128xi32, #blocked2>
        %45 = arith.cmpi slt, %38, %44 {groups = [@nvws.epilogue]} : tensor<1x128xi32, #blocked2>
        %46 = tt.broadcast %43 {groups = [@nvws.epilogue]} : tensor<128x1xi1, #blocked2> -> tensor<128x128xi1, #blocked2>
        %47 = tt.broadcast %45 {groups = [@nvws.epilogue]} : tensor<1x128xi1, #blocked2> -> tensor<128x128xi1, #blocked2>
        %48 = arith.andi %46, %47 {groups = [@nvws.epilogue]} : tensor<128x128xi1, #blocked2>
        %49 = ttg.convert_layout %26 : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #blocked2>
	// CHECK: tt.store
	// CHECK: nvvm.barrier id = {{.*}} number_of_threads = %[[C128]]
        tt.store %41, %49, %48 {groups = [@nvws.epilogue]} : tensor<128x128x!tt.ptr<f16>, #blocked2>
        %50 = arith.addi %arg8, %c1_i32 {groups = [@nvws.epilogue]} : i32
        scf.yield {groups = [@nvws.epilogue]} %50 : i32
      } {groups = [@nvws.epilogue]}
      ttng.warp_group_return
    } {barId = 3 : i32, groups = [@nvws.epilogue]}}
    tt.return
  }
}
