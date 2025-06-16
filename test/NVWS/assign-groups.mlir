// RUN: triton-opt -split-input-file --nvws-assign-groups %s | FileCheck %s

// CHECK: module attributes {nvws.group.mma = {num_warps = 8 : i32, start_warp = 0 : i32}, nvws.group.tma_load = {num_warps = 4 : i32, start_warp = 8 : i32},
// CHECK-LABEL: matmul_kernel_tma

// CHECK-NOT: groups
// CHECK: tt.descriptor_load {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK-NEXT: ttg.local_alloc {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK-NEXT: tt.descriptor_load {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK-NEXT: ttg.local_alloc {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK-NOT: groups
// CHECK-NEXT: ttg.memdesc_trans {{.*}} {groups = [@nvws.group.mma],
// CHECK-NEXT: ttng.tmem_alloc {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NEXT: ttng.tc_gen5_mma {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NEXT: ttng.tmem_load {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NOT: groups
// CHECK: tt.store {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NOT: groups

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "nvws.warp-specialized" = true} {
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

// CHECK: module attributes {nvws.group.mma = {num_warps = 8 : i32, start_warp = 0 : i32}, nvws.group.tma_load = {num_warps = 4 : i32, start_warp = 8 : i32},
// CHECK-LABEL: matmul_kernel_tma_persistent_blackwell

// CHECK-NOT: groups
// CHECK: tt.descriptor_load {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK-NEXT: ttg.local_alloc {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK-NEXT: tt.descriptor_load {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK-NEXT: ttg.local_alloc {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK-NEXT: ttg.memdesc_trans {{.*}} {groups = [@nvws.group.mma],
// CHECK-NEXT: ttng.tmem_alloc {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NEXT: ttng.tc_gen5_mma {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NEXT: ttng.tmem_load {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NOT: groups
// CHECK: tt.descriptor_store {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NOT: groups

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "nvws.warp-specialized" = true} {
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

// CHECK: module attributes {nvws.group.mma = {num_warps = 8 : i32, start_warp = 0 : i32}, nvws.group.tma_load = {num_warps = 4 : i32, start_warp = 8 : i32}
// CHECK-LABEL: matmul_kernel_tma_persistent_nested

// CHECK-NOT: groups
// CHECK: tt.descriptor_load {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK-NEXT: ttg.local_alloc {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK-NEXT: tt.descriptor_load {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK-NEXT: ttg.local_alloc {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK-NEXT: ttg.memdesc_trans {{.*}} {groups = [@nvws.group.mma],
// CHECK-NEXT: ttng.tmem_alloc {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NEXT: ttng.tc_gen5_mma {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NEXT: ttng.tmem_load {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NOT: groups
// CHECK: tt.descriptor_store {{.*}} {groups = [@nvws.group.mma]}
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

// CHECK: module attributes {nvws.group.epilogue = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.group.mma = {num_warps = 4 : i32, start_warp = 4 : i32}, nvws.group.tma_load = {num_warps = 4 : i32, start_warp = 8 : i32}
// CHECK-LABEL: matmul_kernel

// CHECK-NOT: groups
// CHECK: ttng.tmem_alloc {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NOT: groups
// CHECK: tt.descriptor_load {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK: ttg.local_alloc {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK-NOT: groups
// CHECK: tt.descriptor_load {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK: ttg.local_alloc {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK: ttg.memdesc_trans {{.*}} {groups = [@nvws.group.mma],
// CHECK: ttng.tc_gen5_mma {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NOT: groups
// CHECK: ttng.tmem_load {{.*}} {groups = [@nvws.group.epilogue]}
// CHECK-NOT: groups
// CHECK: tt.store {{.*}} {groups = [@nvws.group.epilogue]}
// CHECK-NOT: groups

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "nvws.warp-specialized" = true} {
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

// CHECK: module attributes {nvws.group.epilogue = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.group.mma = {num_warps = 4 : i32, start_warp = 4 : i32}, nvws.group.tma_load = {num_warps = 4 : i32, start_warp = 8 : i32}
// CHECK-LABEL: matmul_persistent_tma_ws_cooperative_kernel

// CHECK-NOT: groups
// CHECK: ttng.tmem_alloc {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NOT: groups
// CHECK: tt.descriptor_load {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK: ttg.local_alloc {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK-NOT: groups
// CHECK: tt.descriptor_load {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK: ttg.local_alloc {{.*}} {groups = [@nvws.group.tma_load]}
// CHECK: ttng.tc_gen5_mma {{.*}} {groups = [@nvws.group.mma]}
// CHECK-NOT: groups
// CHECK: ttng.tmem_load {{.*}} {groups = [@nvws.group.epilogue]}
// CHECK-NOT: groups
// CHECK: tt.store {{.*}} {groups = [@nvws.group.epilogue]}
// CHECK-NOT: groups

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "nvws.warp-specialized" = true} {
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


// -----

// CHECK-LABEL: matmul_kernel_bias

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_kernel_bias(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<64> : tensor<128x64xi32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %true = arith.constant true
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg4, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.remsi %0, %2 : i32
    %4 = arith.divsi %0, %2 : i32
    %5 = arith.muli %3, %c128_i32 : i32
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %8 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %9 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %10 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %11 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %12 = arith.addi %10, %6 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = arith.addi %11, %7 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %14 = tt.splat %arg4 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %15 = arith.remsi %12, %14 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = arith.muli %4, %c128_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %18 = tt.splat %16 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %19 = arith.addi %17, %8 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %20 = arith.addi %18, %9 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %21 = tt.splat %arg5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %22 = tt.splat %arg5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %23 = arith.remsi %19, %21 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %24 = arith.remsi %20, %22 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %25 = tt.expand_dims %15 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %26 = tt.splat %arg7 : i32 -> tensor<128x1xi32, #blocked>
    %27 = arith.muli %25, %26 : tensor<128x1xi32, #blocked>
    %28 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %29 = tt.expand_dims %28 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %30 = tt.broadcast %27 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %31 = tt.broadcast %29 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %32 = arith.addi %30, %31 : tensor<128x64xi32, #blocked>
    %33 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %34 = tt.addptr %33, %32 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %35 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %36 = tt.expand_dims %35 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %37 = tt.splat %arg8 : i32 -> tensor<64x1xi32, #blocked2>
    %38 = arith.muli %36, %37 : tensor<64x1xi32, #blocked2>
    %39 = tt.expand_dims %23 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
    %40 = tt.expand_dims %24 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x128xi32, #blocked3>
    %41 = tt.broadcast %38 : tensor<64x1xi32, #blocked2> -> tensor<64x128xi32, #blocked2>
    %42 = tt.broadcast %39 : tensor<1x128xi32, #blocked2> -> tensor<64x128xi32, #blocked2>
    %43 = arith.addi %41, %42 : tensor<64x128xi32, #blocked2>
    %44 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked2>
    %45 = tt.addptr %44, %43 : tensor<64x128x!tt.ptr<f16>, #blocked2>, tensor<64x128xi32, #blocked2>
    %46 = arith.addi %arg6, %c63_i32 : i32
    %47 = arith.divsi %46, %c64_i32 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %48 = ttng.tmem_store %cst_0, %result[%token], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %49:4 = scf.for %arg10 = %c0_i32 to %47 step %c1_i32 iter_args(%arg11 = %34, %arg12 = %45, %arg13 = %false, %arg14 = %48) -> (tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked2>, i1, !ttg.async.token)  : i32 {
      // CHECK: tt.load {{.*}} {groups = [@nvws.group.tma_load]}
      %67 = tt.load %arg11 : tensor<128x64x!tt.ptr<f16>, #blocked>
      %68 = ttg.local_alloc %67 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // CHECK: tt.load {{.*}} {groups = [@nvws.group.tma_load]}
      %69 = tt.load %arg12 : tensor<64x128x!tt.ptr<f16>, #blocked2>
      %70 = ttg.local_alloc %69 : (tensor<64x128xf16, #blocked2>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %71 = ttng.tc_gen5_mma %68, %70, %result[%arg14], %arg13, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %72 = tt.addptr %arg11, %cst : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
      %73 = arith.muli %arg8, %c64_i32 : i32
      %74 = tt.splat %73 : i32 -> tensor<64x128xi32, #blocked2>
      %75 = tt.addptr %arg12, %74 : tensor<64x128x!tt.ptr<f16>, #blocked2>, tensor<64x128xi32, #blocked2>
      scf.yield %72, %75, %true, %71 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked2>, i1, !ttg.async.token
    }
    %result_1, %token_2 = ttng.tmem_load %result[%49#3] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %50 = arith.truncf %result_1 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    %51 = ttg.convert_layout %50 : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #blocked2>
    %52 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>, #blocked3>
    %53 = tt.addptr %52, %40 : tensor<1x128x!tt.ptr<f16>, #blocked3>, tensor<1x128xi32, #blocked3>
    // CHECK-NOT: tt.load {{.*}} {groups = [@nvws.group.tma_load]}
    %54 = tt.load %53 : tensor<1x128x!tt.ptr<f16>, #blocked3>
    %55 = tt.expand_dims %13 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %56 = tt.splat %arg9 : i32 -> tensor<128x1xi32, #blocked2>
    %57 = arith.muli %56, %55 : tensor<128x1xi32, #blocked2>
    %58 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
    %59 = tt.addptr %58, %57 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
    %60 = tt.expand_dims %19 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
    %61 = tt.broadcast %59 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x128x!tt.ptr<f16>, #blocked2>
    %62 = tt.broadcast %60 : tensor<1x128xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
    %63 = tt.addptr %61, %62 : tensor<128x128x!tt.ptr<f16>, #blocked2>, tensor<128x128xi32, #blocked2>
    %64 = ttg.convert_layout %54 : tensor<1x128xf16, #blocked3> -> tensor<1x128xf16, #blocked2>
    %65 = tt.broadcast %64 : tensor<1x128xf16, #blocked2> -> tensor<128x128xf16, #blocked2>
    %66 = arith.addf %51, %65 : tensor<128x128xf16, #blocked2>
    tt.store %63, %66 : tensor<128x128x!tt.ptr<f16>, #blocked2>
    tt.return
  }
}

// -----

// CHECK-LABEL: matmul_int4_rhs
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 16, 2], threadsPerWarp = [16, 2, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_int4_rhs(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0> : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant dense<0> : tensor<256xi32, #blocked2>
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_2 = arith.constant dense<4> : tensor<256x32xi8, #blocked>
    %true = arith.constant true
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked3>
    %0 = arith.divsi %arg6, %c2_i32 : i32
    %1 = arith.addi %arg5, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
    %3 = tt.get_program_id x : i32
    %4 = arith.divsi %3, %c16_i32 : i32
    %5 = arith.remsi %3, %c16_i32 : i32
    %6 = arith.divsi %4, %2 : i32
    %7 = arith.remsi %4, %2 : i32
    %8 = tt.addptr %arg7, %5 : !tt.ptr<i64>, i32
    %9 = tt.load %8 : !tt.ptr<i64>
    %10 = tt.addptr %8, %c1_i32 : !tt.ptr<i64>, i32
    %11 = tt.load %10 : !tt.ptr<i64>
    %12 = arith.muli %6, %c128_i32 : i32
    %13 = arith.extsi %12 : i32 to i64
    %14 = arith.addi %9, %13 : i64
    %15 = arith.cmpi sge, %14, %11 : i64
    %16 = arith.muli %7, %c256_i32 : i32
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %19 = arith.extsi %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %20 = arith.extsi %18 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %21 = tt.splat %14 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %22 = tt.splat %14 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %23 = arith.addi %21, %19 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %24 = arith.addi %22, %20 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %25 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %26 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %27 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked2>
    %28 = tt.splat %16 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %29 = tt.splat %16 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %30 = tt.splat %16 : i32 -> tensor<256xi32, #blocked2>
    %31 = arith.addi %28, %25 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %32 = arith.addi %29, %26 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %33 = arith.addi %30, %27 : tensor<256xi32, #blocked2>
    %34 = arith.extsi %arg4 : i32 to i64
    %35 = tt.splat %34 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %36 = arith.cmpi slt, %23, %35 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %37 = arith.select %36, %23, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %38 = tt.splat %arg5 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %39 = tt.splat %arg5 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %40 = tt.splat %arg5 : i32 -> tensor<256xi32, #blocked2>
    %41 = arith.cmpi slt, %31, %38 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %42 = arith.cmpi slt, %32, %39 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %43 = arith.cmpi slt, %33, %40 : tensor<256xi32, #blocked2>
    %44 = arith.select %41, %31, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %45 = arith.select %43, %33, %cst_1 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #blocked2>, tensor<256xi32, #blocked2>
    %46 = tt.expand_dims %37 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi64, #blocked1>
    %47 = arith.extsi %arg10 : i32 to i64
    %48 = tt.splat %47 : i64 -> tensor<128x1xi64, #blocked1>
    %49 = arith.muli %46, %48 : tensor<128x1xi64, #blocked1>
    %50 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %51 = tt.expand_dims %50 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %52 = arith.extsi %51 : tensor<1x64xi32, #blocked1> to tensor<1x64xi64, #blocked1>
    %53 = tt.broadcast %49 : tensor<128x1xi64, #blocked1> -> tensor<128x64xi64, #blocked1>
    %54 = tt.broadcast %52 : tensor<1x64xi64, #blocked1> -> tensor<128x64xi64, #blocked1>
    %55 = arith.addi %53, %54 : tensor<128x64xi64, #blocked1>
    %56 = tt.expand_dims %44 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %57 = tt.splat %arg11 : i32 -> tensor<256x1xi32, #blocked>
    %58 = arith.muli %56, %57 : tensor<256x1xi32, #blocked>
    %59 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %60 = tt.expand_dims %59 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %61 = tt.broadcast %58 : tensor<256x1xi32, #blocked> -> tensor<256x32xi32, #blocked>
    %62 = tt.broadcast %60 : tensor<1x32xi32, #blocked> -> tensor<256x32xi32, #blocked>
    %63 = arith.addi %61, %62 : tensor<256x32xi32, #blocked>
    %64 = arith.muli %5, %arg5 : i32
    %65 = arith.muli %64, %0 : i32
    %66 = tt.addptr %arg1, %65 : !tt.ptr<i8>, i32
    %67 = arith.divsi %arg6, %c256_i32 : i32
    %68 = arith.muli %64, %67 : i32
    %69 = tt.addptr %arg2, %68 : !tt.ptr<bf16>, i32
    %70 = arith.addi %arg6, %c63_i32 : i32
    %71 = arith.divsi %70, %c64_i32 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %72 = ttng.tmem_store %cst_3, %result[%token], %true : tensor<128x256xf32, #blocked3> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %73:2 = scf.for %arg17 = %c0_i32 to %71 step %c1_i32 iter_args(%arg18 = %false, %arg19 = %72) -> (i1, !ttg.async.token)  : i32 {
      %93 = arith.muli %arg17, %c64_i32 : i32
      %94 = arith.muli %arg17, %c32_i32 : i32
      %95 = tt.addptr %arg0, %93 : !tt.ptr<bf16>, i32
      %96 = tt.splat %95 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked1>
      %97 = tt.addptr %96, %55 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi64, #blocked1>
      %98 = tt.addptr %66, %94 : !tt.ptr<i8>, i32
      %99 = tt.splat %98 : !tt.ptr<i8> -> tensor<256x32x!tt.ptr<i8>, #blocked>
      %100 = tt.addptr %99, %63 : tensor<256x32x!tt.ptr<i8>, #blocked>, tensor<256x32xi32, #blocked>
      // CHECK: tt.load {{.*}} {groups = [@nvws.group.tma_load]}
      %101 = tt.load %97 : tensor<128x64x!tt.ptr<bf16>, #blocked1>
      %102 = ttg.local_alloc %101 : (tensor<128x64xbf16, #blocked1>) -> !ttg.memdesc<128x64xbf16, #shared, #smem>
      // CHECK: tt.load {{.*}} {groups = [@nvws.group.tma_load]}
      %103 = tt.load %100 : tensor<256x32x!tt.ptr<i8>, #blocked>
      %104 = arith.shli %103, %cst_2 : tensor<256x32xi8, #blocked>
      %105 = arith.shrsi %104, %cst_2 : tensor<256x32xi8, #blocked>
      %106 = arith.shrsi %103, %cst_2 : tensor<256x32xi8, #blocked>
      %107 = tt.join %105, %106 : tensor<256x32xi8, #blocked> -> tensor<256x32x2xi8, #blocked5>
      %108 = tt.reshape %107 : tensor<256x32x2xi8, #blocked5> -> tensor<256x64xi8, #blocked6>
      %109 = arith.sitofp %108 : tensor<256x64xi8, #blocked6> to tensor<256x64xbf16, #blocked6>
      %110 = arith.divsi %93, %c256_i32 : i32
      %111 = tt.addptr %69, %110 : !tt.ptr<bf16>, i32
      %112 = tt.splat %arg13 : i32 -> tensor<256xi32, #blocked2>
      %113 = arith.muli %45, %112 : tensor<256xi32, #blocked2>
      %114 = tt.splat %111 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked2>
      %115 = tt.addptr %114, %113 : tensor<256x!tt.ptr<bf16>, #blocked2>, tensor<256xi32, #blocked2>
      // CHECK-NOT: tt.load {{.*}} {groups = [@nvws.group.tma_load]}
      %116 = tt.load %115 : tensor<256x!tt.ptr<bf16>, #blocked2>
      %117 = ttg.convert_layout %116 : tensor<256xbf16, #blocked2> -> tensor<256xbf16, #ttg.slice<{dim = 1, parent = #blocked6}>>
      %118 = tt.expand_dims %117 {axis = 1 : i32} : tensor<256xbf16, #ttg.slice<{dim = 1, parent = #blocked6}>> -> tensor<256x1xbf16, #blocked6>
      %119 = tt.broadcast %118 : tensor<256x1xbf16, #blocked6> -> tensor<256x64xbf16, #blocked6>
      %120 = arith.mulf %109, %119 : tensor<256x64xbf16, #blocked6>
      %121 = ttg.local_alloc %120 : (tensor<256x64xbf16, #blocked6>) -> !ttg.memdesc<256x64xbf16, #shared, #smem>
      %122 = ttg.memdesc_trans %121 {order = array<i32: 1, 0>} : !ttg.memdesc<256x64xbf16, #shared, #smem> -> !ttg.memdesc<64x256xbf16, #shared1, #smem>
      %123 = ttng.tc_gen5_mma %102, %122, %result[%arg19], %arg18, %true : !ttg.memdesc<128x64xbf16, #shared, #smem>, !ttg.memdesc<64x256xbf16, #shared1, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %true, %123 : i1, !ttg.async.token
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: matmul_int4_lhs_tmem
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 16, 2], threadsPerWarp = [8, 4, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [128, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = false>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.warp-specialized" = true} {
  tt.func public @matmul_int4_lhs_tmem(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0> : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant dense<0> : tensor<128xi32, #blocked2>
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c127_i32 = arith.constant 127 : i32
    %cst_2 = arith.constant dense<4> : tensor<128x64xi8, #blocked>
    %true = arith.constant true
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked3>
    %0 = arith.divsi %arg6, %c2_i32 : i32
    %1 = arith.addi %arg5, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = tt.get_program_id x : i32
    %4 = arith.divsi %3, %c16_i32 : i32
    %5 = arith.remsi %3, %c16_i32 : i32
    %6 = arith.divsi %4, %2 : i32
    %7 = arith.remsi %4, %2 : i32
    %8 = tt.addptr %arg7, %5 : !tt.ptr<i64>, i32
    %9 = tt.load %8 : !tt.ptr<i64>
    %10 = tt.addptr %8, %c1_i32 : !tt.ptr<i64>, i32
    %11 = tt.load %10 : !tt.ptr<i64>
    %12 = arith.muli %6, %c128_i32 : i32
    %13 = arith.extsi %12 : i32 to i64
    %14 = arith.addi %9, %13 : i64
    %15 = arith.cmpi sge, %14, %11 : i64
    %16 = arith.muli %7, %c128_i32 : i32
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %21 = arith.extsi %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %22 = tt.splat %14 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %23 = arith.addi %22, %21 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %24 = tt.splat %16 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %25 = tt.splat %16 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %26 = tt.splat %16 : i32 -> tensor<128xi32, #blocked2>
    %27 = arith.addi %24, %18 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %28 = arith.addi %25, %19 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %29 = arith.addi %26, %20 : tensor<128xi32, #blocked2>
    %30 = arith.extsi %arg4 : i32 to i64
    %31 = tt.splat %30 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %32 = arith.cmpi slt, %23, %31 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %33 = arith.select %32, %23, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %34 = tt.splat %arg5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %35 = tt.splat %arg5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %36 = tt.splat %arg5 : i32 -> tensor<128xi32, #blocked2>
    %37 = arith.cmpi slt, %27, %34 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %38 = arith.cmpi slt, %28, %35 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %39 = arith.cmpi slt, %29, %36 : tensor<128xi32, #blocked2>
    %40 = arith.select %37, %27, %cst {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %41 = arith.select %39, %29, %cst_1 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #blocked2>, tensor<128xi32, #blocked2>
    %42 = tt.expand_dims %33 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi64, #blocked1>
    %43 = arith.extsi %arg10 : i32 to i64
    %44 = tt.splat %43 : i64 -> tensor<128x1xi64, #blocked1>
    %45 = arith.muli %42, %44 : tensor<128x1xi64, #blocked1>
    %46 = tt.expand_dims %19 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %47 = arith.extsi %46 : tensor<1x128xi32, #blocked1> to tensor<1x128xi64, #blocked1>
    %48 = tt.broadcast %45 : tensor<128x1xi64, #blocked1> -> tensor<128x128xi64, #blocked1>
    %49 = tt.broadcast %47 : tensor<1x128xi64, #blocked1> -> tensor<128x128xi64, #blocked1>
    %50 = arith.addi %48, %49 : tensor<128x128xi64, #blocked1>
    %51 = tt.expand_dims %40 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %52 = tt.splat %arg11 : i32 -> tensor<128x1xi32, #blocked>
    %53 = arith.muli %51, %52 : tensor<128x1xi32, #blocked>
    %54 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %55 = tt.expand_dims %54 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %56 = tt.broadcast %53 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %57 = tt.broadcast %55 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %58 = arith.addi %56, %57 : tensor<128x64xi32, #blocked>
    %59 = arith.muli %5, %arg5 : i32
    %60 = arith.muli %59, %0 : i32
    %61 = tt.addptr %arg1, %60 : !tt.ptr<i8>, i32
    %62 = arith.divsi %arg6, %c256_i32 : i32
    %63 = arith.muli %59, %62 : i32
    %64 = tt.addptr %arg2, %63 : !tt.ptr<bf16>, i32
    %65 = arith.addi %arg6, %c127_i32 : i32
    %66 = arith.divsi %65, %c128_i32 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %67 = ttng.tmem_store %cst_3, %result[%token], %true : tensor<128x128xf32, #blocked3> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %68:2 = scf.for %arg17 = %c0_i32 to %66 step %c1_i32 iter_args(%arg18 = %false, %arg19 = %67) -> (i1, !ttg.async.token)  : i32 {
      %89 = arith.muli %arg17, %c128_i32 : i32
      %90 = arith.muli %arg17, %c64_i32 : i32
      %91 = tt.addptr %arg0, %89 : !tt.ptr<bf16>, i32
      %92 = tt.splat %91 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked1>
      %93 = tt.addptr %92, %50 : tensor<128x128x!tt.ptr<bf16>, #blocked1>, tensor<128x128xi64, #blocked1>
      %94 = tt.addptr %61, %90 : !tt.ptr<i8>, i32
      %95 = tt.splat %94 : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #blocked>
      %96 = tt.addptr %95, %58 : tensor<128x64x!tt.ptr<i8>, #blocked>, tensor<128x64xi32, #blocked>
      // CHECK: tt.load {{.*}} {groups = [@nvws.group.tma_load]}
      %97 = tt.load %93 : tensor<128x128x!tt.ptr<bf16>, #blocked1>
      // CHECK: %[[INT4_TENSOR:.*]] = tt.load {{.*}} {groups = [@nvws.group.tma_load]}
      %98 = tt.load %96 : tensor<128x64x!tt.ptr<i8>, #blocked>
      %99 = arith.shli %98, %cst_2 : tensor<128x64xi8, #blocked>
      %100 = arith.shrsi %99, %cst_2 : tensor<128x64xi8, #blocked>
      // CHECK: arith.shrsi %[[INT4_TENSOR]], {{.*}} {groups = [@nvws.group.simt]}
      %101 = arith.shrsi %98, %cst_2 : tensor<128x64xi8, #blocked>
      %102 = tt.join %100, %101 : tensor<128x64xi8, #blocked> -> tensor<128x64x2xi8, #blocked4>
      %103 = tt.reshape %102 : tensor<128x64x2xi8, #blocked4> -> tensor<128x128xi8, #blocked5>
      %104 = arith.sitofp %103 : tensor<128x128xi8, #blocked5> to tensor<128x128xbf16, #blocked5>
      %105 = arith.divsi %89, %c256_i32 : i32
      %106 = tt.addptr %64, %105 : !tt.ptr<bf16>, i32
      %107 = tt.splat %arg13 : i32 -> tensor<128xi32, #blocked2>
      %108 = arith.muli %41, %107 : tensor<128xi32, #blocked2>
      %109 = tt.splat %106 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #blocked2>
      %110 = tt.addptr %109, %108 : tensor<128x!tt.ptr<bf16>, #blocked2>, tensor<128xi32, #blocked2>
      // CHECK-NOT: tt.load {{.*}} {groups = [@nvws.group.tma_load]}
      %111 = tt.load %110 : tensor<128x!tt.ptr<bf16>, #blocked2>
      %112 = ttg.convert_layout %111 : tensor<128xbf16, #blocked2> -> tensor<128xbf16, #ttg.slice<{dim = 1, parent = #blocked5}>>
      %113 = tt.expand_dims %112 {axis = 1 : i32} : tensor<128xbf16, #ttg.slice<{dim = 1, parent = #blocked5}>> -> tensor<128x1xbf16, #blocked5>
      %114 = tt.broadcast %113 : tensor<128x1xbf16, #blocked5> -> tensor<128x128xbf16, #blocked5>
      %115 = arith.mulf %104, %114 : tensor<128x128xbf16, #blocked5>
      %116 = ttg.convert_layout %115 : tensor<128x128xbf16, #blocked5> -> tensor<128x128xbf16, #blocked3>
      // CHECK: ttng.tmem_alloc {{.*}} {groups = [@nvws.group.simt]}
      %result_6 = ttng.tmem_alloc %116 : (tensor<128x128xbf16, #blocked3>) -> !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>
      %117 = ttg.local_alloc %97 : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
      %118 = ttg.memdesc_trans %117 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared1, #smem>
      %119 = ttng.tc_gen5_mma %result_6, %118, %result[%arg19], %arg18, %true : !ttg.memdesc<128x128xbf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xbf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %true, %119 : i1, !ttg.async.token
    }
    tt.return
  }
}
