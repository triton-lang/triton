// RUN: triton-opt %s -split-input-file --nvws-lower-aref | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {nvws.mma = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 4 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  // CHECK-LABEL: matmul_kernel
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %c128_i32 = arith.constant {} 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
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
    %tmem_aref = nvws.aref.create %9 : !nvws.aref<[!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %10 = ttg.local_alloc  {aref_buffer} : () -> !ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>
    %11 = ttg.local_alloc  {aref_buffer} : () -> !ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>
    %12 = nvws.aref.create %10, %11 : !nvws.aref<[!ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>], 1>
    %13 = nvvm.read.ptx.sreg.tid.x : i32
    %14 = arith.divsi %13, %c128_i32 : i32
    %15 = arith.cmpi eq, %14, %c1_i32 : i32
    nvvm.barrier0
    // CHECK: nvws.warp_group
    nvws.warp_group
    // CHECK-NEXT: partition0 num_warps(4) {
    // CHECK-NOT: nvws.aref.put
    // CHECK-NOT: nvws.aref.get
    partition0 num_warps(4) {
      nvvm.setmaxregister  decrease 40
      %16:2 = scf.for %arg7 = %c0_i32 to %8 step %c1_i32 iter_args(%arg8 = %c0_i32, %arg9 = %c0_i32) -> (i32, i32)  : i32 {
        %17 = tt.reinterpret_tensor_descriptor %arg0 {} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x64xf8E4M3FN>>
        %18 = tt.reinterpret_tensor_descriptor %arg1 {} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x64xf8E4M3FN>>
        nvws.aref.put %12[%arg9] as (%arg10: !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>,
                                     %arg11: !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>) {
          // CHECK: ttng.wait_barrier
          // CHECK-NEXT: ttg.memdesc_subview
          // CHECK-NEXT: ttg.memdesc_subview
          // CHECK-NEXT: arith.remsi
          // CHECK-NEXT: ttg.memdesc_subview
          // CHECK-NEXT: ttng.async_tma_copy_global_to_local
          // CHECK-NEXT: ttng.barrier_expect
          // CHECK-NEXT: ttng.async_tma_copy_global_to_local
          // CHECK-NEXT: ttng.barrier_expect
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
    // CHECK: partition1 num_warps(4) {
    partition1 num_warps(4) {
      nvvm.setmaxregister  increase 232
      nvws.aref.put %tmem_aref as (%arg11 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
        // CHECK: ttng.wait_barrier
        %16 = scf.for %arg7 = %c0_i32 to %8 step %c1_i32 iter_args(%arg8 = %c0_i32) -> (i32)  : i32 {
          nvws.aref.get %12[%arg8] as (%arg9 : !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>,
                                       %arg10 : !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>) {
            // CHECK: ttng.wait_barrier
            // CHECK-NEXT: ttg.memdesc_subview
            // CHECK-NEXT: ttg.memdesc_subview
            // CHECK-NEXT: ttg.memdesc_trans
            // CHECK-NEXT: arith.remsi
            // CHECK-NEXT: ttg.memdesc_subview
            // CHECK-NEXT: ttng.tc_gen5_mma
            %36 = ttg.memdesc_trans %arg10 {order=array<i32: 1,0>} : !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf8E4M3FN, #shared1, #smem, mutable>
            ttng.tc_gen5_mma %arg9, %36, %arg11, %true, %true {} : (!ttg.memdesc<128x64xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<64x128xf8E4M3FN, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
            nvws.aref.return
          } : (!nvws.aref<[!ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x64xf8E4M3FN, #shared, #smem, mutable>], 1>, i32) -> ()
          %37 = arith.addi %arg8, %c1_i32 {} : i32
          scf.yield %37 : i32
        }
        nvws.aref.return
      } : (!nvws.aref<[!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>) -> ()
      %17 = nvws.aref.get %tmem_aref as (%arg11 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
        %acc = ttng.tmem_load %arg11 {} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        nvws.aref.return %acc : tensor<128x128xf32, #blocked>
      } : (!nvws.aref<[!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>) -> (tensor<128x128xf32, #blocked>)
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
    // CHECK: }
    tt.return
  }
}

// -----


#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 1, 128], threadsPerWarp = [1, 32, 1], warpsPerCTA = [1, 4, 1], order = [0, 1, 2]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
// CHECK-LABEL: matmul_kernel_tma_persistent_nested
module attributes {nvws.mma = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 4 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_kernel_tma_persistent_nested(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant {} true
    %c8_i32 = arith.constant {} 8 : i32
    %c128_i32 = arith.constant {} 128 : i32
    %c0_i32 = arith.constant {} 0 : i32
    %c1_i32 = arith.constant {} 1 : i32
    %c2_i32 = arith.constant {} 2 : i32
    %c127_i32 = arith.constant {} 127 : i32
    %cst = arith.constant {} dense<0.000000e+00> : tensor<2x128x128xf32, #blocked2>
    %0 = tt.get_program_id x {} : i32
    %1 = arith.addi %arg3, %c127_i32 {} : i32
    %2 = arith.divsi %1, %c128_i32 {} : i32
    %3 = arith.addi %arg4, %c127_i32 {} : i32
    %4 = arith.divsi %3, %c128_i32 {} : i32
    %5 = arith.addi %arg5, %c127_i32 {} : i32
    %6 = arith.divsi %5, %c128_i32 {} : i32
    %7 = arith.muli %2, %4 {} : i32
    %8 = arith.muli %4, %c8_i32 {} : i32
    %9 = tt.get_num_programs x {} : i32
    %10 = nvvm.read.ptx.sreg.tid.x : i32
    %11 = arith.divsi %10, %c128_i32 : i32
    %12 = ttg.local_alloc  {aref_buffer} : () -> !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>
    %13 = ttg.local_alloc  {aref_buffer} : () -> !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>
    %14 = nvws.aref.create %12, %13 : !nvws.aref<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>], 1>
    %15 = ttng.tmem_alloc %cst : (tensor<2x128x128xf32, #blocked2>) -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %16 = nvws.aref.create %15 : !nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>
    nvvm.barrier0 {init_barrier_sync}
    // CHECK: nvws.warp_group
    nvws.warp_group
    // CHECK-NEXT: partition0 num_warps(4) {
    // CHECK-NOT: nvws.aref.put
    // CHECK-NOT: nvws.aref.get
    partition0 num_warps(4) {
      %20 = scf.for %arg6 = %0 to %7 step %9 iter_args(%arg7 = %c0_i32) -> (i32)  : i32 {
        %21 = arith.divsi %arg6, %8 {} : i32
        %22 = arith.muli %21, %c8_i32 {} : i32
        %23 = arith.subi %2, %22 {} : i32
        %24 = arith.minsi %23, %c8_i32 {} : i32
        %25 = arith.remsi %arg6, %24 {} : i32
        %26 = arith.addi %22, %25 {} : i32
        %27 = arith.remsi %arg6, %8 {} : i32
        %28 = arith.divsi %27, %24 {} : i32
        %29 = arith.muli %26, %c128_i32 {} : i32
        %30 = arith.muli %28, %c128_i32 {} : i32
        %31:2 = scf.for %arg8 = %c0_i32 to %6 step %c1_i32 iter_args(%arg9 = %c0_i32, %arg10 = %arg7) -> (i32, i32)  : i32 {
          %32 = tt.reinterpret_tensor_descriptor %arg0 {} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
          %33 = tt.reinterpret_tensor_descriptor %arg1 {} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
          nvws.aref.put %14[%arg10] as (%arg11: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg12: !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>) {
            // CHECK: ttng.wait_barrier
            // CHECK-NEXT: ttg.memdesc_subview
            // CHECK-NEXT: ttg.memdesc_subview
            // CHECK-NEXT: arith.remsi
            // CHECK-NEXT: ttg.memdesc_subview
            // CHECK-NEXT: ttng.async_tma_copy_global_to_local
            // CHECK-NEXT: ttng.barrier_expect
            // CHECK-NEXT: ttng.async_tma_copy_global_to_local
            // CHECK-NEXT: ttng.barrier_expect
            %36 = tt.descriptor_load %32[%29, %arg9] {} : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #blocked1>
            ttg.local_store %36, %arg11 : tensor<128x128xf8E4M3FN, #blocked1> -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>
            %37 = tt.descriptor_load %33[%30, %arg9] {} : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #blocked1>
            ttg.local_store %37, %arg12 : tensor<128x128xf8E4M3FN, #blocked1> -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>
            nvws.aref.return
          }: (!nvws.aref<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>], 1>, i32) -> ()
          %34 = arith.addi %arg9, %c128_i32 {} : i32
          %35 = arith.addi %arg10, %c1_i32 {} : i32
          scf.yield {} %34, %35 : i32, i32
        }
        scf.yield {} %31#1 : i32
      }
      nvws.warp_group.return
    // CHECK: }
    }
    // CHECK: partition1 num_warps(4) {
    partition1 num_warps(4) {
      //CHECK-NEXT: scf.for
      %20 = scf.for %arg6 = %0 to %7 step %9 iter_args(%arg7 = %c0_i32) -> (i32)  : i32 {
        %21 = arith.remsi %arg6, %c2_i32 : i32
        nvws.aref.put %16[%21] as (%arg11 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
          // CHECK: ttng.wait_barrier
          // CHECK-NEXT: ttg.memdesc_subview
          // CHECK-NEXT: scf.for
          %31:2 = scf.for %arg8 = %c0_i32 to %6 step %c1_i32 iter_args(%arg9 = %false, %arg10 = %arg7) -> (i1, i32)  : i32 {
            nvws.aref.get %14[%arg10] as (%arg12 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, %arg13 : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>) {
            // CHECK: ttng.wait_barrier
            // CHECK-NEXT: ttg.memdesc_subview
            // CHECK-NEXT: ttg.memdesc_subview
            // CHECK-NEXT: ttg.memdesc_trans
            // CHECK-NEXT: arith.remsi
            // CHECK-NEXT: ttg.memdesc_subview
            // CHECK-NEXT: ttng.tc_gen5_mma
              %37 = ttg.memdesc_trans %arg13 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>
              ttng.tc_gen5_mma %arg12, %37, %arg11, %arg9, %true : (!ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
              nvws.aref.return
            } : (!nvws.aref<[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>], 1>, i32) -> ()
            %38 = arith.addi %arg10, %c1_i32  : i32
            scf.yield {} %true, %38 : i1, i32
          }
          // CHECK: }
          // CHECK-NEXT: arith.remsi
          // CHECK-NEXT: ttg.memdesc_subview
          // CHECK-NEXT: ttng.arrive_barrier
          nvws.aref.return
        } : (!nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, i32) -> ()
        %31 = arith.addi %arg7, %6 : i32
        scf.yield {} %31 : i32
      }
      nvws.warp_group.return
    }
    // CHECK: partition2 num_warps(4) {
    partition2 num_warps(4) {
      %20 = scf.for %arg6 = %0 to %7 step %9 iter_args(%arg7 = %c0_i32) -> (i32)  : i32 {
        %21 = arith.divsi %arg6, %8 {} : i32
        %22 = arith.muli %21, %c8_i32 {} : i32
        %23 = arith.subi %2, %22 {} : i32
        %24 = arith.minsi %23, %c8_i32 {} : i32
        %25 = arith.remsi %arg6, %24 {} : i32
        %26 = arith.addi %22, %25 {} : i32
        %27 = arith.remsi %arg6, %8 {} : i32
        %28 = arith.divsi %27, %24 {} : i32
        %29 = arith.muli %26, %c128_i32 : i32
        %30 = arith.muli %28, %c128_i32 : i32
        %31 = arith.addi %arg7, %6 : i32
        %stage = arith.remsi %arg6, %c2_i32 : i32
        %32 = nvws.aref.get %16[%stage] as (%arg8 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
          // CHECK: ttng.wait_barrier
          // CHECK: ttng.wait_barrier
          // CHECK: ttng.tmem_load
          // CHECK: ttng.arrive_barrier
          %32 = ttng.tmem_load %arg8 {} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          nvws.aref.return %32 : tensor<128x128xf32, #blocked>
        } : (!nvws.aref<[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, i32) -> (tensor<128x128xf32, #blocked>)
        %33 = tt.fp_to_fp %32 {}, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
        %34 = ttg.convert_layout %33 : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked1>
        %35 = tt.reinterpret_tensor_descriptor %arg2 {} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
        tt.descriptor_store %35[%29, %30], %34 {} : !tt.tensordesc<tensor<128x128xf8E4M3FN>>, tensor<128x128xf8E4M3FN, #blocked1>
        scf.yield {} %31 : i32
      }
      nvws.warp_group.return
    // CHECK: }
    }
    tt.return
  }
}

// -----

module attributes {nvws.mma = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 4 : i32}, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  //CHECK-LABEL: matmul_kernel_hopper
  tt.func public @matmul_kernel_hopper(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant {} 127 : i32
    %c63_i32 = arith.constant {} 63 : i32
    %cst = arith.constant {} dense<0.000000e+00> : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>>
    %0 = tt.get_program_id x {} : i32
    %1 = arith.addi %arg3, %c127_i32 {} : i32
    %2 = arith.divsi %1, %c128_i32 {} : i32
    %3 = arith.remsi %0, %2 {} : i32
    %4 = arith.divsi %0, %2 {} : i32
    %5 = arith.muli %3, %c128_i32 {} : i32
    %6 = arith.muli %4, %c128_i32 {} : i32
    %7 = arith.addi %arg5, %c63_i32 {} : i32
    %8 = arith.divsi %7, %c64_i32 {} : i32
    %9 = nvvm.read.ptx.sreg.tid.x : i32
    %10 = arith.divsi %9, %c128_i32 : i32
    %11 = ttg.local_alloc  {aref_buffer} : () -> !ttg.memdesc<3x128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable>
    %12 = ttg.local_alloc  {aref_buffer} : () -> !ttg.memdesc<3x128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable>
    %13 = nvws.aref.create %11, %12 : !nvws.aref<[!ttg.memdesc<3x128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable>, !ttg.memdesc<3x128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable>], 1>
    %acc = nvws.aref.create %cst : !nvws.aref<[tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>>]>
    nvvm.barrier0 {init_barrier_sync}
    %14 = arith.cmpi eq, %10, %c1_i32 {} : i32
    // CHECK: nvws.warp_group
    nvws.warp_group
    // CHECK-NEXT: partition0 num_warps(4) {
    // CHECK-NOT: nvws.aref.put
    // CHECK-NOT: nvws.aref.get
    partition0 num_warps(4) {
      nvvm.setmaxregister  decrease 40 {}
      %18:2 = scf.for %arg7 = %c0_i32 to %8 step %c1_i32 iter_args(%arg8 = %c0_i32, %arg9 = %c0_i32) -> (i32, i32)  : i32 {
        %19 = tt.reinterpret_tensor_descriptor %arg0 {} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x64xf8E4M3FN>>
        %20 = tt.reinterpret_tensor_descriptor %arg1 {} : !tt.ptr<f32> to !tt.tensordesc<tensor<128x64xf8E4M3FN>>
        nvws.aref.put %13[%arg9] as (%arg10: !ttg.memdesc<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable>, %arg11: !ttg.memdesc<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable>) {
          // CHECK: ttng.wait_barrier
          // CHECK-NEXT: ttg.memdesc_subview
          // CHECK-NEXT: ttg.memdesc_subview
          // CHECK-NEXT: arith.remsi
          // CHECK-NEXT: ttg.memdesc_subview
          // CHECK-NEXT: ttng.async_tma_copy_global_to_local
          // CHECK-NEXT: ttng.barrier_expect
          // CHECK-NEXT: ttng.async_tma_copy_global_to_local
          // CHECK-NEXT: ttng.barrier_expect
          %23 = tt.descriptor_load %19[%5, %arg8] {} : !tt.tensordesc<tensor<128x64xf8E4M3FN>> -> tensor<128x64xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
          ttg.local_store %23, %arg10 : tensor<128x64xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> !ttg.memdesc<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable>
          %24 = tt.descriptor_load %20[%6, %arg8] {} : !tt.tensordesc<tensor<128x64xf8E4M3FN>> -> tensor<128x64xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
          ttg.local_store %24, %arg11 : tensor<128x64xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> !ttg.memdesc<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable>
          nvws.aref.return
        } : (!nvws.aref<[!ttg.memdesc<3x128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable>, !ttg.memdesc<3x128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable>], 1>, i32) -> ()
        %21 = arith.addi %arg8, %c64_i32 {} : i32
        %22 = arith.addi %arg9, %c1_i32 {} : i32
        scf.yield {} %21, %22 : i32, i32
      }
      nvws.warp_group.return
    } partition1 num_warps(4) {
      // CHECK: ttng.wait_barrier
      // CHECK-NEXT: ttg.memdesc_subview
      // CHECK-NEXT: ttg.memdesc_subview
      // CHECK-NEXT: ttg.memdesc_trans
      // CHECK-NEXT: ttng.fence_async_shared
      // CHECK-NEXT: ttng.warp_group_dot
      // CHECK-NEXT: ttng.warp_group_dot_wait
      // Signal empty
      // CHECK: scf.if
      // CHECK-NEXT: arith.remsi
      // CHECK-NEXT: ttg.memdesc_subview
      // CHECK-NEXT: ttng.arrive_barrier
      nvvm.setmaxregister  increase 232 {}
      %18:2 = scf.for %arg7 = %c0_i32 to %8 step %c1_i32 iter_args(%arg8 = %cst, %arg9 = %c0_i32) -> (tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>>, i32)  : i32 {
        %40 = nvws.aref.get %13[%arg9] as (%arg11 :!ttg.memdesc<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable>, %arg12 : !ttg.memdesc<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable>) {
          %38 = ttg.memdesc_trans %arg12 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>, #ttg.shared_memory, mutable>
          ttng.fence_async_shared {bCluster = false}
          %39 = ttng.warp_group_dot %arg11, %38, %arg8 : !ttg.memdesc<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable> * !ttg.memdesc<64x128xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>, #ttg.shared_memory, mutable> -> tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>>
          %40 = ttng.warp_group_dot_wait %39 {pendings = 0 : i32} : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>>
          nvws.aref.return %40 : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>>
        } : (!nvws.aref<[!ttg.memdesc<3x128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable>, !ttg.memdesc<3x128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>, #ttg.shared_memory, mutable>], 1>, i32) -> (tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>>)
        %41 = arith.subi %arg9, %c1_i32 : i32
        %42 = arith.addi %arg9, %c1_i32  : i32
        scf.yield %40, %42 : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>>, i32
      }
      // CHECK: }
      // CHECK: ttng.warp_group_dot_wait
      %20 = tt.fp_to_fp %18#0 {}, rounding = rtne : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>> -> tensor<128x128xf8E4M3FN, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>>
      %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
      %22 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
      %23 = tt.splat %5 {} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
      %24 = arith.addi %23, %21 {} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
      %25 = tt.splat %6 {} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
      %26 = arith.addi %25, %22 {} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
      %27 = tt.expand_dims %24 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %28 = tt.splat %arg6  : i32 -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %29 = arith.muli %28, %27  : tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %30 = tt.splat %arg2  : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %31 = tt.addptr %30, %29  : tensor<128x1x!tt.ptr<f8E4M3FN>, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %32 = tt.expand_dims %26 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %33 = tt.broadcast %31  : tensor<128x1x!tt.ptr<f8E4M3FN>, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %34 = tt.broadcast %32 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %35 = tt.addptr %33, %34 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<128x128xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %36 = ttg.convert_layout %20 : tensor<128x128xf8E4M3FN, #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>> -> tensor<128x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      tt.store %35, %36 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
      nvws.warp_group.return
    }
    tt.return
  }
}
