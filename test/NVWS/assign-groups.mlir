// RUN: triton-opt -split-input-file --nvws-assign-groups %s | FileCheck %s

// CHECK: module attributes {nvws.mma = {num_warps = 8 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 8 : i32},
// CHECK-LABEL: matmul_kernel_tma

// CHECK-NOT: groups
// CHECK: tt.descriptor_load {{.*}} {groups = [@nvws.tma_load]}
// CHECK-NEXT: ttg.local_alloc {{.*}} {groups = [@nvws.tma_load]}
// CHECK-NEXT: tt.descriptor_load {{.*}} {groups = [@nvws.tma_load]}
// CHECK-NEXT: ttg.local_alloc {{.*}} {groups = [@nvws.tma_load]}
// CHECK-NOT: groups
// CHECK-NEXT: ttg.memdesc_trans {{.*}} {groups = [@nvws.mma],
// CHECK-NEXT: ttng.tmem_alloc {{.*}} {groups = [@nvws.mma]}
// CHECK-NEXT: ttng.tc_gen5_mma {{.*}} {groups = [@nvws.mma]}
// CHECK-NEXT: ttng.tmem_load {{.*}} {groups = [@nvws.mma]}
// CHECK-NOT: groups
// CHECK: tt.store {{.*}} {groups = [@nvws.mma]}
// CHECK-NOT: groups

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_kernel_tma(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32) attributes {noinline = false} {
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.remsi %0, %2 : i32
    %4 = arith.divsi %0, %2 : i32
    %5 = arith.muli %3, %c128_i32 : i32
    %6 = arith.muli %4, %c128_i32 : i32
    %7 = arith.addi %arg5, %c63_i32 : i32
    %8 = arith.divsi %7, %c64_i32 : i32
    %9 = tt.reinterpret_tensor_descriptor %arg0 : !tt.ptr<f32> to !tt.tensordesc<tensor<128x64xf8E4M3FN>>
    %10 = tt.reinterpret_tensor_descriptor %arg1 : !tt.ptr<f32> to !tt.tensordesc<tensor<128x64xf8E4M3FN>>
    %11:2 = scf.for %arg8 = %c0_i32 to %8 step %c1_i32 iter_args(%arg9 = %cst, %arg10 = %c0_i32) -> (tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>, i32)  : i32 {

      %29 = tt.descriptor_load %9[%5, %arg10] : !tt.tensordesc<tensor<128x64xf8E4M3FN>> -> tensor<128x64xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>>
      %30 = ttg.local_alloc %29 : (tensor<128x64xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared, #ttg.shared_memory>
      %31 = tt.descriptor_load %10[%6, %arg10] : !tt.tensordesc<tensor<128x64xf8E4M3FN>> -> tensor<128x64xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>>
      %32 = ttg.local_alloc %31 : (tensor<128x64xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared1, #ttg.shared_memory>
      %33 = ttg.memdesc_trans %32 {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf8E4M3FN, #shared1, #ttg.shared_memory> -> !ttg.memdesc<64x128xf8E4M3FN, #shared, #ttg.shared_memory>
      %34 = ttng.tmem_alloc %arg9 : (tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>) -> !ttg.memdesc<128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %30, %33, %34, %true, %true : !ttg.memdesc<128x64xf8E4M3FN, #shared, #ttg.shared_memory>, !ttg.memdesc<64x128xf8E4M3FN, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>
      %35 = ttng.tmem_load %34 : !ttg.memdesc<128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>
      %36 = arith.addi %arg10, %c64_i32 : i32
      scf.yield %35, %36 : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>, i32
    }
    %12 = tt.fp_to_fp %11#0, rounding = rtne : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>> -> tensor<128x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %15 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %16 = arith.addi %15, %13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %17 = tt.splat %6 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %18 = arith.addi %17, %14 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>>
    %19 = tt.expand_dims %16 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>> -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %20 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %21 = arith.muli %20, %19 : tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %22 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %23 = tt.addptr %22, %21 : tensor<128x1x!tt.ptr<f8E4M3FN>, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>, tensor<128x1xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %24 = tt.expand_dims %18 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>}>> -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %25 = tt.broadcast %23 : tensor<128x1x!tt.ptr<f8E4M3FN>, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %26 = tt.broadcast %24 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>> -> tensor<128x128xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %27 = tt.addptr %25, %26 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>, tensor<128x128xi32, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    %28 = ttg.convert_layout %12 : tensor<128x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>> -> tensor<128x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    tt.store %27, %28 : tensor<128x128x!tt.ptr<f8E4M3FN>, #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>>
    tt.return
  }
}

// -----

// CHECK: module attributes {nvws.mma = {num_warps = 8 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 8 : i32},
// CHECK-LABEL: matmul_kernel_tma_persistent_blackwell

// CHECK-NOT: groups
// CHECK: tt.descriptor_load {{.*}} {groups = [@nvws.tma_load]}
// CHECK-NEXT: ttg.local_alloc {{.*}} {groups = [@nvws.tma_load]}
// CHECK-NEXT: tt.descriptor_load {{.*}} {groups = [@nvws.tma_load]}
// CHECK-NEXT: ttg.local_alloc {{.*}} {groups = [@nvws.tma_load]}
// CHECK-NEXT: ttg.memdesc_trans {{.*}} {groups = [@nvws.mma],
// CHECK-NEXT: ttng.tmem_alloc {{.*}} {groups = [@nvws.mma]}
// CHECK-NEXT: ttng.tc_gen5_mma {{.*}} {groups = [@nvws.mma]}
// CHECK-NEXT: ttng.tmem_load {{.*}} {groups = [@nvws.mma]}
// CHECK-NOT: groups
// CHECK: tt.descriptor_store {{.*}} {groups = [@nvws.mma]}
// CHECK-NOT: groups

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_kernel_tma_persistent_blackwell(%arg0: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg1: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg2: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %c148_i32 = arith.constant 148 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>
    %0 = tt.elementwise_inline_asm "prefetch.tensormap [$1]; // dummy $0" {constraints = "=r,l", packed_element = 1 : i32, pure = false} %arg0 : !tt.ptr<i8, 0> -> i32
    %1 = tt.elementwise_inline_asm "prefetch.tensormap [$1]; // dummy $0" {constraints = "=r,l", packed_element = 1 : i32, pure = false} %arg1 : !tt.ptr<i8, 0> -> i32
    %2 = tt.elementwise_inline_asm "prefetch.tensormap [$1]; // dummy $0" {constraints = "=r,l", packed_element = 1 : i32, pure = false} %arg2 : !tt.ptr<i8, 0> -> i32
    %3 = tt.get_program_id x : i32
    %4 = arith.addi %arg3, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.addi %arg4, %c255_i32 : i32
    %7 = arith.divsi %6, %c256_i32 : i32
    %8 = arith.addi %arg5, %c127_i32 : i32
    %9 = arith.divsi %8, %c128_i32 : i32
    %10 = arith.muli %5, %7 : i32
    %11 = arith.divsi %10, %c148_i32 : i32
    %12 = arith.remsi %10, %c148_i32 : i32
    %13 = arith.cmpi slt, %3, %12 : i32
    %14 = scf.if %13 -> (i32) {
      %22 = arith.addi %11, %c1_i32 : i32
      scf.yield %22 : i32
    } else {
      scf.yield %11 : i32
    }
    %15 = arith.subi %3, %c148_i32 : i32
    %16 = arith.muli %7, %c8_i32 : i32
    %17 = arith.muli %9, %14 : i32
    %18 = arith.subi %9, %c1_i32 : i32
    %19 = tt.reinterpret_tensor_descriptor %arg0 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
    %20 = tt.reinterpret_tensor_descriptor %arg1 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<256x128xf8E4M3FN>>
    %21:6 = scf.for %arg6 = %c0_i32 to %17 step %c1_i32 iter_args(%arg7 = %c-1_i32, %arg8 = %15, %arg9 = %c0_i32, %arg10 = %c0_i32, %arg11 = %cst, %arg12 = %15) -> (i32, i32, i32, i32, tensor<128x256xf32, #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>, i32)  : i32 {
      %22 = arith.cmpi eq, %arg7, %18 : i32
      %23 = arith.addi %arg7, %c1_i32 : i32
      %24 = arith.select %22, %c0_i32, %23 : i32
      %25 = arith.cmpi eq, %24, %c0_i32 : i32
      %26:3 = scf.if %25 -> (i32, i32, i32) {
        %38 = arith.addi %arg8, %c148_i32 : i32
        %39 = arith.divsi %38, %16 : i32
        %40 = arith.muli %39, %c8_i32 : i32
        %41 = arith.subi %5, %40 : i32
        %42 = arith.minsi %41, %c8_i32 : i32
        %43 = arith.remsi %38, %42 : i32
        %44 = arith.addi %40, %43 : i32
        %45 = arith.remsi %38, %16 : i32
        %46 = arith.divsi %45, %42 : i32
        %47 = arith.muli %44, %c128_i32 : i32
        %48 = arith.muli %46, %c256_i32 : i32
        scf.yield %38, %47, %48 : i32, i32, i32
      } else {
        scf.yield %arg8, %arg9, %arg10 : i32, i32, i32
      }
      %27 = arith.muli %24, %c128_i32 : i32
      %28 = tt.descriptor_load %19[%26#1, %27] : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>>
      %29 = ttg.local_alloc %28 : (tensor<128x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #ttg.shared_memory>
      %30 = tt.descriptor_load %20[%26#2, %27] : !tt.tensordesc<tensor<256x128xf8E4M3FN>> -> tensor<256x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>>
      %31 = ttg.local_alloc %30 : (tensor<256x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>>) -> !ttg.memdesc<256x128xf8E4M3FN, #shared1, #ttg.shared_memory>
      %32 = ttg.memdesc_trans %31 {order = array<i32: 1, 0>} : !ttg.memdesc<256x128xf8E4M3FN, #shared1, #ttg.shared_memory> -> !ttg.memdesc<128x256xf8E4M3FN, #shared, #ttg.shared_memory>
      %33 = ttng.tmem_alloc %arg11 : (tensor<128x256xf32, #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>) -> !ttg.memdesc<128x256xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_mma %29, %32, %33, %true, %true : !ttg.memdesc<128x128xf8E4M3FN, #shared, #ttg.shared_memory>, !ttg.memdesc<128x256xf8E4M3FN, #shared, #ttg.shared_memory>, !ttg.memdesc<128x256xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>, #ttng.tensor_memory, mutable>
      %34 = ttng.tmem_load %33 : !ttg.memdesc<128x256xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>
      %35 = arith.cmpi eq, %24, %18 : i32
      %36 = arith.select %35, %cst, %34 : tensor<128x256xf32, #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>
      %37 = scf.if %35 -> (i32) {
        %38 = arith.addi %arg12, %c148_i32 : i32
        %39 = arith.divsi %38, %16 : i32
        %40 = arith.muli %39, %c8_i32 : i32
        %41 = arith.subi %5, %40 : i32
        %42 = arith.minsi %41, %c8_i32 : i32
        %43 = arith.remsi %38, %42 : i32
        %44 = arith.addi %40, %43 : i32
        %45 = arith.remsi %38, %16 : i32
        %46 = arith.divsi %45, %42 : i32
        %47 = arith.muli %44, %c128_i32 : i32
        %48 = arith.muli %46, %c256_i32 : i32
        %49 = tt.fp_to_fp %34, rounding = rtne : tensor<128x256xf32, #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>> -> tensor<128x256xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>
        %50 = ttg.convert_layout %49 : tensor<128x256xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>> -> tensor<128x256xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>>
        %51 = tt.reinterpret_tensor_descriptor %arg2 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<128x256xf8E4M3FN>>
        tt.descriptor_store %51[%47, %48], %50 : !tt.tensordesc<tensor<128x256xf8E4M3FN>>, tensor<128x256xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>>
        scf.yield %38 : i32
      } else {
        scf.yield %arg12 : i32
      }
      scf.yield %24, %26#0, %26#1, %26#2, %36, %37 : i32, i32, i32, i32, tensor<128x256xf32, #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>, i32
    }
    tt.return
  }
}

// -----

// CHECK: module attributes {nvws.mma = {num_warps = 8 : i32, start_warp = 0 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 8 : i32}
// CHECK-LABEL: matmul_kernel_tma_persistent_nested

// CHECK-NOT: groups
// CHECK: tt.descriptor_load {{.*}} {groups = [@nvws.tma_load]}
// CHECK-NEXT: ttg.local_alloc {{.*}} {groups = [@nvws.tma_load]}
// CHECK-NEXT: tt.descriptor_load {{.*}} {groups = [@nvws.tma_load]}
// CHECK-NEXT: ttg.local_alloc {{.*}} {groups = [@nvws.tma_load]}
// CHECK-NEXT: ttg.memdesc_trans {{.*}} {groups = [@nvws.mma],
// CHECK-NEXT: ttng.tmem_alloc {{.*}} {groups = [@nvws.mma]}
// CHECK-NEXT: ttng.tc_gen5_mma {{.*}} {groups = [@nvws.mma]}
// CHECK-NEXT: ttng.tmem_load {{.*}} {groups = [@nvws.mma]}
// CHECK-NOT: groups
// CHECK: tt.descriptor_store {{.*}} {groups = [@nvws.mma]}
// CHECK-NOT: groups

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_tma_persistent_nested(%arg0: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg1: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg2: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %c148_i32 = arith.constant 148 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg5, %c127_i32 : i32
    %6 = arith.divsi %5, %c128_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.divsi %7, %c148_i32 : i32
    %9 = arith.remsi %7, %c148_i32 : i32
    %10 = arith.cmpi slt, %0, %9 : i32
    %11 = scf.if %10 -> (i32) {
      %18 = arith.addi %8, %c1_i32 : i32
      scf.yield %18 : i32
    } else {
      scf.yield %8 : i32
    }
    %12 = arith.subi %0, %c148_i32 : i32
    %13 = arith.muli %4, %c8_i32 : i32
    %14 = tt.reinterpret_tensor_descriptor %arg0 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
    %15 = tt.reinterpret_tensor_descriptor %arg1 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
    %16 = tt.reinterpret_tensor_descriptor %arg2 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
    %17 = scf.for %arg6 = %c0_i32 to %11 step %c1_i32 iter_args(%arg7 = %12) -> (i32)  : i32 {
      %18 = arith.addi %arg7, %c148_i32 : i32
      %19 = arith.divsi %18, %13 : i32
      %20 = arith.muli %19, %c8_i32 : i32
      %21 = arith.subi %2, %20 : i32
      %22 = arith.minsi %21, %c8_i32 : i32
      %23 = arith.remsi %18, %22 : i32
      %24 = arith.addi %20, %23 : i32
      %25 = arith.remsi %18, %13 : i32
      %26 = arith.divsi %25, %22 : i32
      %27 = arith.muli %24, %c128_i32 : i32
      %28 = arith.muli %26, %c128_i32 : i32
      %29:2 = scf.for %arg8 = %c0_i32 to %6 step %c1_i32 iter_args(%arg9 = %cst, %arg10 = %c0_i32) -> (tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>, i32)  : i32 {
        %32 = tt.descriptor_load %14[%27, %arg10] : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>>
        %33 = ttg.local_alloc %32 : (tensor<128x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #ttg.shared_memory>
        %34 = tt.descriptor_load %15[%28, %arg10] : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>>
        %35 = ttg.local_alloc %34 : (tensor<128x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #ttg.shared_memory>
        %36 = ttg.memdesc_trans %35 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #ttg.shared_memory> -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #ttg.shared_memory>
        %37 = ttng.tmem_alloc %arg9 : (tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>) -> !ttg.memdesc<128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>
        ttng.tc_gen5_mma %33, %36, %37, %true, %true : !ttg.memdesc<128x128xf8E4M3FN, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf8E4M3FN, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable>
        %38 = ttng.tmem_load %37 : !ttg.memdesc<128x128xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>
        %39 = arith.addi %arg10, %c128_i32 : i32
        scf.yield %38, %39 : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>, i32
      }
      %30 = tt.fp_to_fp %29#0, rounding = rtne : tensor<128x128xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>> -> tensor<128x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>>
      %31 = ttg.convert_layout %30 : tensor<128x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>> -> tensor<128x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>>
      tt.descriptor_store %16[%27, %28], %31 : !tt.tensordesc<tensor<128x128xf8E4M3FN>>, tensor<128x128xf8E4M3FN, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>>
      scf.yield %18 : i32
    }
    tt.return
  }
}

// -----

// CHECK: module attributes {nvws.epilogue = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.mma = {num_warps = 4 : i32, start_warp = 4 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 8 : i32}
// CHECK-LABEL: matmul_kernel

// CHECK-NOT: groups
// CHECK: ttng.tmem_alloc {{.*}} {groups = [@nvws.mma]}
// CHECK-NOT: groups
// CHECK: tt.descriptor_load {{.*}} {groups = [@nvws.tma_load]}
// CHECK: ttg.local_alloc {{.*}} {groups = [@nvws.tma_load]}
// CHECK-NOT: groups
// CHECK: tt.descriptor_load {{.*}} {groups = [@nvws.tma_load]}
// CHECK: ttg.local_alloc {{.*}} {groups = [@nvws.tma_load]}
// CHECK: ttg.memdesc_trans {{.*}} {groups = [@nvws.mma],
// CHECK: ttng.tc_gen5_mma {{.*}} {groups = [@nvws.mma]}
// CHECK-NOT: groups
// CHECK: ttng.tmem_load {{.*}} {groups = [@nvws.epilogue]}
// CHECK-NOT: groups
// CHECK: tt.store {{.*}} {groups = [@nvws.epilogue]}
// CHECK-NOT: groups

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.remsi %0, %2 : i32
    %4 = arith.divsi %0, %2 : i32
    %5 = arith.muli %3, %c128_i32 : i32
    %6 = arith.muli %4, %c128_i32 : i32
    %7 = arith.addi %arg5, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %10:2 = scf.for %arg7 = %c0_i32 to %8 step %c1_i32 iter_args(%arg8 = %c0_i32, %arg9 = %false) -> (i32, i1)  : i32 {
      %29 = tt.reinterpret_tensor_descriptor %arg0 : !tt.ptr<f32> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
      %30 = tt.descriptor_load %29[%5, %arg8] : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #blocked1>
      %31 = ttg.local_alloc %30 : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
      %32 = tt.reinterpret_tensor_descriptor %arg1 : !tt.ptr<f32> to !tt.tensordesc<tensor<128x128xf8E4M3FN>>
      %33 = tt.descriptor_load %32[%6, %arg8] : !tt.tensordesc<tensor<128x128xf8E4M3FN>> -> tensor<128x128xf8E4M3FN, #blocked1>
      %34 = ttg.local_alloc %33 : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
      %35 = ttg.memdesc_trans %34 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
      ttng.tc_gen5_mma %31, %35, %9, %arg9, %true : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %36 = arith.addi %arg8, %c128_i32 : i32
      scf.yield %36, %true : i32, i1
    }
    %11 = ttng.tmem_load %9 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %12 = tt.fp_to_fp %11, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %15 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %16 = arith.addi %15, %13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %17 = tt.splat %6 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %18 = arith.addi %17, %14 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %19 = tt.expand_dims %16 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %20 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked2>
    %21 = arith.muli %20, %19 : tensor<128x1xi32, #blocked2>
    %22 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2>
    %23 = tt.addptr %22, %21 : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<128x1xi32, #blocked2>
    %24 = tt.expand_dims %18 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
    %25 = tt.broadcast %23 : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>
    %26 = tt.broadcast %24 : tensor<1x128xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
    %27 = tt.addptr %25, %26 : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<128x128xi32, #blocked2>
    %28 = ttg.convert_layout %12 : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked2>
    tt.store %27, %28 : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>
    tt.return
  }
}

// -----

// CHECK: module attributes {nvws.epilogue = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.mma = {num_warps = 4 : i32, start_warp = 4 : i32}, nvws.tma_load = {num_warps = 4 : i32, start_warp = 8 : i32}
// CHECK-LABEL: matmul_persistent_tma_ws_cooperative_kernel

// CHECK-NOT: groups
// CHECK: ttng.tmem_alloc {{.*}} {groups = [@nvws.mma]}
// CHECK-NOT: groups
// CHECK: tt.descriptor_load {{.*}} {groups = [@nvws.tma_load]}
// CHECK: ttg.local_alloc {{.*}} {groups = [@nvws.tma_load]}
// CHECK-NOT: groups
// CHECK: tt.descriptor_load {{.*}} {groups = [@nvws.tma_load]}
// CHECK: ttg.local_alloc {{.*}} {groups = [@nvws.tma_load]}
// CHECK: ttng.tc_gen5_mma {{.*}} {groups = [@nvws.mma]}
// CHECK-NOT: groups
// CHECK: ttng.tmem_load {{.*}} {groups = [@nvws.epilogue]}
// CHECK-NOT: groups
// CHECK: tt.store {{.*}} {groups = [@nvws.epilogue]}
// CHECK-NOT: groups

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_persistent_tma_ws_cooperative_kernel(%arg0: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg1: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg2: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg3: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c127_i32 = arith.constant 127 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = arith.addi %arg4, %c127_i32 : i32
    %1 = arith.divsi %0, %c128_i32 : i32
    %2 = arith.addi %arg5, %c127_i32 : i32
    %3 = arith.divsi %2, %c128_i32 : i32
    %4 = arith.muli %1, %3 : i32
    %5 = tt.get_program_id x : i32
    %6 = tt.get_num_programs x : i32
    scf.for %arg7 = %5 to %4 step %6  : i32 {
      %7 = arith.muli %3, %c8_i32 : i32
      %8 = arith.divsi %arg7, %7 : i32
      %9 = arith.muli %8, %c8_i32 : i32
      %10 = arith.subi %1, %9 : i32
      %11 = arith.minsi %10, %c8_i32 : i32
      %12 = arith.remsi %arg7, %7 : i32
      %13 = arith.remsi %12, %11 : i32
      %14 = arith.addi %9, %13 : i32
      %15 = arith.divsi %12, %11 : i32
      %16 = arith.muli %14, %c128_i32 : i32
      %17 = arith.muli %15, %c128_i32 : i32
      %18 = arith.addi %arg6, %c127_i32 : i32
      %19 = arith.divsi %18, %c128_i32 : i32
      %20 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %21:2 = scf.for %arg8 = %c0_i32 to %19 step %c1_i32 iter_args(%arg9 = %c0_i32, %arg10 = %false) -> (i32, i1)  : i32 {
        %48 = tt.reinterpret_tensor_descriptor %arg0 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<128x128xf16>>
        %49 = tt.descriptor_load %48[%16, %arg9] : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #blocked1>
        %50 = ttg.local_alloc %49 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %51 = tt.reinterpret_tensor_descriptor %arg1 : !tt.ptr<i8, 0> to !tt.tensordesc<tensor<128x128xf16>>
        %52 = tt.descriptor_load %51[%arg9, %17] : !tt.tensordesc<tensor<128x128xf16>> -> tensor<128x128xf16, #blocked1>
        %53 = ttg.local_alloc %52 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        ttng.tc_gen5_mma %50, %53, %20, %arg10, %true : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %54 = arith.addi %arg9, %c128_i32 : i32
        scf.yield %54, %true : i32, i1
      }
      %22 = ttng.tmem_load %20 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %23 = arith.truncf %22 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
      %24 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %25 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %26 = tt.splat %16 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %27 = arith.addi %26, %24 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %28 = tt.splat %17 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %29 = arith.addi %28, %25 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %30 = tt.expand_dims %27 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
      %31 = tt.splat %arg5 : i32 -> tensor<128x1xi32, #blocked2>
      %32 = arith.muli %30, %31 : tensor<128x1xi32, #blocked2>
      %33 = tt.splat %arg3 : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2>
      %34 = tt.addptr %33, %32 : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<128x1xi32, #blocked2>
      %35 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
      %36 = tt.broadcast %34 : tensor<128x1x!tt.ptr<f8E4M3FN>, #blocked2> -> tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>
      %37 = tt.broadcast %35 : tensor<1x128xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
      %38 = tt.addptr %36, %37 : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>, tensor<128x128xi32, #blocked2>
      %39 = tt.splat %arg4 : i32 -> tensor<128x1xi32, #blocked2>
      %40 = arith.cmpi slt, %30, %39 : tensor<128x1xi32, #blocked2>
      %41 = tt.splat %arg5 : i32 -> tensor<1x128xi32, #blocked2>
      %42 = arith.cmpi slt, %35, %41 : tensor<1x128xi32, #blocked2>
      %43 = tt.broadcast %40 : tensor<128x1xi1, #blocked2> -> tensor<128x128xi1, #blocked2>
      %44 = tt.broadcast %42 : tensor<1x128xi1, #blocked2> -> tensor<128x128xi1, #blocked2>
      %45 = arith.andi %43, %44 : tensor<128x128xi1, #blocked2>
      %46 = tt.fp_to_fp %23, rounding = rtne : tensor<128x128xf16, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
      %47 = ttg.convert_layout %46 : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked2>
      tt.store %38, %47, %45 : tensor<128x128x!tt.ptr<f8E4M3FN>, #blocked2>
    }
    tt.return
  }
}
