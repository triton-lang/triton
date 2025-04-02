// RUN: triton-opt %s -split-input-file --nvws-lower-aref | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {nvws.mma = {num_warps = 4 : i32, start_warp = 4 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 0 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_kernel_tma_persistent(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c148_i32 = arith.constant 148 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant {} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg5, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.divsi %7, %c148_i32 : i32
    %9 = arith.remsi %7, %c148_i32 : i32
    %10 = arith.cmpi slt, %0, %9 : i32
    %11 = scf.if %10 -> (i32) {
      %22 = arith.addi %8, %c1_i32 : i32
      scf.yield %22 : i32
    } else {
      scf.yield %8 : i32
    }
    %12 = arith.subi %0, %c148_i32 : i32
    %13 = arith.muli %4, %c8_i32 : i32
    %14 = arith.muli %6, %11 : i32
    %15 = ttng.tmem_alloc %cst {} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %16 = ttg.local_alloc  {aref_buffer} : () -> !ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>
    %17 = ttg.local_alloc  {aref_buffer} : () -> !ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>
    %18 = nvws.aref.create %16, %17 : !nvws.aref<[!ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>], 1>
    %19 = nvvm.read.ptx.sreg.tid.x : i32
    %20 = arith.divsi %19, %c128_i32 : i32
    %21 = arith.cmpi eq, %20, %c0_i32 : i32
    nvvm.barrier0
    nvws.warp_group
    partition0 num_warps(4) {
      nvvm.setmaxregister  decrease 40
      %22:5 = scf.for %arg6 = %c0_i32 to %14 step %c1_i32 iter_args(%arg7 = %c-1_i32, %arg8 = %12, %arg9 = %c0_i32, %arg10 = %c0_i32, %arg11 = %c0_i32) -> (i32, i32, i32, i32, i32) : i32 {
        %23 = arith.subi %6, %c1_i32 {} : i32
        %24 = arith.cmpi eq, %arg7, %23 {} : i32
        %25 = arith.addi %arg7, %c1_i32 {} : i32
        %26 = arith.select %24, %c0_i32, %25 {} : i32
        %27 = arith.cmpi eq, %26, %c0_i32 {} : i32
        %28:3 = scf.if %27 -> (i32, i32, i32) {
          %33 = arith.addi %arg8, %c148_i32 {} : i32
          %34 = arith.divsi %33, %13 {} : i32
          %35 = arith.muli %34, %c8_i32 {} : i32
          %36 = arith.subi %2, %35 {} : i32
          %37 = arith.minsi %36, %c8_i32 {} : i32
          %38 = arith.remsi %33, %37 {} : i32
          %39 = arith.addi %35, %38 {} : i32
          %40 = arith.remsi %33, %13 {} : i32
          %41 = arith.divsi %40, %37 {} : i32
          %42 = arith.muli %39, %c128_i32 {} : i32
          %43 = arith.muli %41, %c128_i32 {} : i32
          scf.yield %33, %42, %43 : i32, i32, i32
        } else {
          scf.yield %arg8, %arg9, %arg10 : i32, i32, i32
        }
        %29 = arith.muli %26, %c64_i32 {} : i32
        %30 = tt.reinterpret_tensor_descriptor %arg0 {} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x64xf8E4M3FN>>
        %31 = tt.reinterpret_tensor_descriptor %arg1 {} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x64xf8E4M3FN>>
        nvws.aref.put %18[%arg11] as (%arg12: !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>,
                                      %arg13: !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>) {
          %33 = tt.descriptor_load %30[%28#1, %29] {} : !tt.tensordesc<tensor<128x64xf8E4M3FN>> -> tensor<128x64xf8E4M3FN, #blocked1>
          ttg.local_store %33, %arg12 : tensor<128x64xf8E4M3FN, #blocked1> -> !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>
          %34 = tt.descriptor_load %31[%28#2, %29] {} : !tt.tensordesc<tensor<128x64xf8E4M3FN>> -> tensor<128x64xf8E4M3FN, #blocked1>
          ttg.local_store %34, %arg13 : tensor<128x64xf8E4M3FN, #blocked1> -> !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>
          nvws.aref.return
        } : (!nvws.aref<[!ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>], 1>, i32) -> ()
        %32 = arith.addi %arg11, %c1_i32 {} : i32
        scf.yield {} %26, %28#0, %28#1, %28#2, %32 : i32, i32, i32, i32, i32
      }
      nvws.warp_group.return
    } partition1 num_warps(4) {
      nvvm.setmaxregister  increase 232
      %22:3 = scf.for %arg6 = %c0_i32 to %14 step %c1_i32 iter_args(%arg7 = %c-1_i32, %arg8 = %false, %arg9 = %c0_i32) -> (i32, i1, i32)  : i32 {
        nvws.aref.get %18[%arg9] as (%arg10 : !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>,
                                     %arg11: !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>) {
          %24 = ttg.memdesc_trans %arg11 {order=array<i32: 1,0>} : !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf8E4M3FN, #shared1, #smem, mutable>
          ttng.tc_gen5_mma %arg10, %24, %15, %arg8, %true : (!ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<64x128xf8E4M3FN, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
          nvws.aref.return
        } : (!nvws.aref<[!ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>], 1>, i32) -> ()
        %25 = arith.subi %6, %c1_i32 {} : i32
        %26 = arith.cmpi eq, %arg7, %25 {} : i32
        %27 = arith.addi %arg7, %c1_i32 {} : i32
        %28 = arith.select %26, %c0_i32, %27 {} : i32
        %29 = arith.cmpi eq, %28, %25 {} : i32
        %30 = arith.cmpi ne, %28, %25 {} : i32
        scf.if %29 {
          %32 = arith.addi %arg6, %c1_i32 : i32
          %33 = arith.divsi %32, %6 : i32
          %34 = arith.muli %33, %c148_i32 : i32
          %35 = arith.addi %12, %34 : i32
          %36 = arith.divsi %35, %13 : i32
          %37 = arith.muli %36, %c8_i32 : i32
          %38 = arith.subi %2, %37 : i32
          %39 = arith.minsi %38, %c8_i32 : i32
          %40 = arith.remsi %35, %39 : i32
          %41 = arith.addi %37, %40 : i32
          %42 = arith.remsi %35, %13 : i32
          %43 = arith.divsi %42, %39 : i32
          %44 = arith.muli %41, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
          %45 = arith.muli %43, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
          %46 = ttng.tmem_load %15 {} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          %47 = tt.fp_to_fp %46, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
          %48 = ttg.convert_layout %47 : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked2>
          %49 = tt.reinterpret_tensor_descriptor %arg2 : !tt.ptr<f32> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
          tt.descriptor_store %49[%44, %45], %48 {} : !tt.tensordesc<tensor<128x128xf8E4M3FN>>, tensor<128x128xf8E4M3FN, #blocked2>
        }
        %31 = arith.addi %arg9, %c1_i32 {} : i32
        scf.yield {} %28, %30, %31 : i32, i1, i32
      }
      nvws.warp_group.return
    }
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {nvws.mma = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 4 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %c128_i32 = arith.constant {} 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant {} 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant {} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.get_program_id x {} : i32
    %1 = arith.addi %arg3, %c127_i32 {} : i32
    %2 = arith.divsi %1, %c128_i32 {} : i32
    %3 = arith.remsi %0, %2 {} : i32
    %4 = arith.divsi %0, %2 {} : i32
    %5 = arith.muli %3, %c128_i32 {} : i32
    %6 = arith.muli %4, %c128_i32 {} : i32
    %7 = arith.addi %arg5, %c63_i32 : i32
    %8 = arith.divsi %7, %c64_i32 : i32
    %9 = ttng.tmem_alloc %cst {} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %10 = ttg.local_alloc  {aref_buffer} : () -> !ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>
    %11 = ttg.local_alloc  {aref_buffer} : () -> !ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>
    %12 = nvws.aref.create %10, %11 : !nvws.aref<[!ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>], 1>
    %13 = nvvm.read.ptx.sreg.tid.x : i32
    %14 = arith.divsi %13, %c128_i32 : i32
    %15 = arith.cmpi eq, %14, %c1_i32 : i32
    nvvm.barrier0
    nvws.warp_group
    partition0 num_warps(4) {
      nvvm.setmaxregister  decrease 40
      %16:2 = scf.for %arg7 = %c0_i32 to %8 step %c1_i32 iter_args(%arg8 = %c0_i32, %arg9 = %c0_i32) -> (i32, i32)  : i32 {
        %17 = tt.reinterpret_tensor_descriptor %arg0 {} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x64xf8E4M3FN>>
        %18 = tt.reinterpret_tensor_descriptor %arg1 {} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x64xf8E4M3FN>>
        nvws.aref.put %12[%arg9] as (%arg10: !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>,
                                     %arg11: !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>) {
          %21 = tt.descriptor_load %17[%5, %arg8] {} : !tt.tensordesc<tensor<128x64xf8E4M3FN>> -> tensor<128x64xf8E4M3FN, #blocked1>
          ttg.local_store %21, %arg10 : tensor<128x64xf8E4M3FN, #blocked1> -> !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>
          %22 = tt.descriptor_load %18[%6, %arg8] {} : !tt.tensordesc<tensor<128x64xf8E4M3FN>> -> tensor<128x64xf8E4M3FN, #blocked1>
          ttg.local_store %22, %arg11 : tensor<128x64xf8E4M3FN, #blocked1> -> !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>
          nvws.aref.return
        } : (!nvws.aref<[!ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>], 1>, i32) -> ()
        %19 = arith.addi %arg8, %c64_i32 {} : i32
        %20 = arith.addi %arg9, %c1_i32 {} : i32
        scf.yield {} %19, %20 : i32, i32
      }
      nvws.warp_group.return
    }
    partition1 num_warps(4) {
      nvvm.setmaxregister  increase 232
      %16 = scf.for %arg7 = %c0_i32 to %8 step %c1_i32 iter_args(%arg8 = %c0_i32) -> (i32)  : i32 {
        nvws.aref.get %12[%arg8] as (%arg9 : !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>,
                                     %arg10 : !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>) {
          %36 = ttg.memdesc_trans %arg10 {order=array<i32: 1,0>} : !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf8E4M3FN, #shared1, #smem, mutable>
          ttng.tc_gen5_mma %arg9, %36, %9, %true, %true {} : (!ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<64x128xf8E4M3FN, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
          nvws.aref.return
        } : (!nvws.aref<[!ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>], 1>, i32) -> ()
        %37 = arith.addi %arg8, %c1_i32 {} : i32
        scf.yield {} %37 : i32
      }
      %17 = ttng.tmem_load %9 {} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %18 = tt.fp_to_fp %17 {}, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
      %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %21 = tt.splat %5 {} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %22 = arith.addi %21, %19 {} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %23 = tt.splat %6 {} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %24 = arith.addi %23, %20 {} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %25 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
      %26 = tt.splat %arg6 {} : i32 -> tensor<128x1xi32, #blocked2>
      %27 = arith.muli %26, %25 {} : tensor<128x1xi32, #blocked2>
      %28 = tt.splat %arg2 {} : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2>
      %29 = tt.addptr %28, %27 {} : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<128x1xi32, #blocked2>
      %30 = tt.expand_dims %24 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
      %31 = tt.broadcast %29 {} : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>
      %32 = tt.broadcast %30 {} : tensor<1x128xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
      %33 = tt.addptr %31, %32 {} : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<128x128xi32, #blocked2>
      %34 = ttg.convert_layout %18 : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked2>
      tt.store %33, %34 {} : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>
      nvws.warp_group.return
    }
    tt.return
  }
}
