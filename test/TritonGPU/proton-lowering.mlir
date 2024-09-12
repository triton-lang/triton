// RUN: triton-opt -split-input-file %s -tritongpu-proton-lowering -canonicalize -cse | FileCheck %s

// CHECK: #[[shared:.*]] = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
// CHECK: tt.func public @no_cse(%[[ARG:.*]]: !tt.ptr<i32>)
// CHECK: %[[IDX:.*]] = "triton_gpu.proton_init"() : () -> !tt.ptr<i32>
// CHECK: %[[BUF:.*]] = triton_gpu.local_alloc  : () -> !tt.memdesc<64xi32, #[[shared]], #triton_gpu.shared_memory, mutable>
// CHECK: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}>
// CHECK: "triton_gpu.proton_finalize"(%[[BUF]], %[[IDX]], %[[ARG]])
module attributes {"triton_gpu.num-warps" = 16 : i32, "triton_gpu.proton-slots" = 32 : i32} {
  tt.func public @no_cse(%arg0: !tt.ptr<i32>) attributes {noinline = false} {
    tt.proton_record <0, "start", "cycle", "warpgroup">
    tt.return
  }
}

// -----

// CHECK-LABEL: no_proton
// CHECK-NOT: triton_gpu.proton_init
// CHECK-NOT: triton_gpu.proton_finalize
module attributes {"triton_gpu.num-warps" = 16 : i32, "triton_gpu.proton-slots" = 32 : i32} {
  tt.func public @no_proton(%arg4: !tt.ptr<i32>) attributes {noinline = false} {
    tt.return
  }
}

// -----

// CHECK: #[[shared:.*]] = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
// CHECK: tt.func public @add_ttir_profile({{.*}}, %[[ARG:.*]]: !tt.ptr<i32>)
// CHECK-NEXT: %c4096_i32 = arith.constant 4096 : i32
// CHECK-NEXT: %[[IDX:.*]] = "triton_gpu.proton_init"() : () -> !tt.ptr<i32>
// CHECK-NEXT: %[[BUF:.*]] = triton_gpu.local_alloc  : () -> !tt.memdesc<64xi32, #[[shared]], #triton_gpu.shared_memory, mutable>
// CHECK-NEXT: tt.get_program_id
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = true{{.*}}regionId = 1
// CHECK-NEXT: arith.muli
// CHECK-NEXT: tt.make_range
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = true{{.*}}regionId = 4
// CHECK-NEXT: tt.splat
// CHECK-NEXT: arith.addi
// CHECK-NEXT: tt.splat
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = false{{.*}}regionId = 1
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = true{{.*}}regionId = 2
// CHECK-NEXT: arith.cmpi slt
// CHECK-NEXT: tt.splat
// CHECK-NEXT: tt.addptr
// CHECK-NEXT: tt.load
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = false{{.*}}regionId = 2
// CHECK-NEXT: tt.splat
// CHECK-NEXT: tt.addptr
// CHECK-NEXT: tt.load
// CHECK-NEXT: arith.addf
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = true{{.*}}regionId = 3
// CHECK-NEXT: tt.splat
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = false{{.*}}regionId = 4
// CHECK-NEXT: tt.addptr
// CHECK-NEXT: tt.store
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = false{{.*}}regionId = 3
// CHECK-NEXT: "triton_gpu.proton_finalize"(%[[BUF]], %[[IDX]], %[[ARG]])
// CHECK-NEXT: tt.return
#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [16], order = [0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-warps" = 16 : i32, "triton_gpu.proton-slots" = 32 : i32} {
  tt.func public @add_ttir_profile(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: !tt.ptr<i32>) attributes {noinline = false} {
    %c4096_i32 = arith.constant 4096 : i32
    %2 = tt.get_program_id x : i32
    tt.proton_record <1, "start", "cycle", "warpgroup">
    %3 = arith.muli %2, %c4096_i32 : i32
    %4 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    tt.proton_record <4, "start", "cycle", "warpgroup">
    %5 = tt.splat %3 : i32 -> tensor<4096xi32, #blocked>
    %6 = arith.addi %5, %4 : tensor<4096xi32, #blocked>
    %7 = tt.splat %arg3 : i32 -> tensor<4096xi32, #blocked>
    tt.proton_record <1, "end", "cycle", "warpgroup">
    tt.proton_record <2, "start", "cycle", "warpgroup">
    %8 = arith.cmpi slt, %6, %7 : tensor<4096xi32, #blocked>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked>
    %10 = tt.addptr %9, %6 : tensor<4096x!tt.ptr<f32>, #blocked>, tensor<4096xi32, #blocked>
    %11 = tt.load %10, %8 : tensor<4096x!tt.ptr<f32>, #blocked>
    tt.proton_record <2, "end", "cycle", "warpgroup">
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked>
    %13 = tt.addptr %12, %6 : tensor<4096x!tt.ptr<f32>, #blocked>, tensor<4096xi32, #blocked>
    %14 = tt.load %13, %8 : tensor<4096x!tt.ptr<f32>, #blocked>
    %15 = arith.addf %11, %14 : tensor<4096xf32, #blocked>
    tt.proton_record <3, "start", "cycle", "warpgroup">
    %16 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>, #blocked>
    tt.proton_record <4, "end", "cycle", "warpgroup">
    %17 = tt.addptr %16, %6 : tensor<4096x!tt.ptr<f32>, #blocked>, tensor<4096xi32, #blocked>
    tt.store %17, %15, %8 : tensor<4096x!tt.ptr<f32>, #blocked>
    tt.proton_record <3, "end", "cycle", "warpgroup">
    tt.return
  }
}


// -----

// CHECK: #[[SHARE:.*]] = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
// CHECK: tt.func public @matmul_ttgir_profile({{.*}}, %[[ARG:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:      %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT: %[[IDX:.*]] = "triton_gpu.proton_init"() : () -> !tt.ptr<i32>
// CHECK-NEXT: %[[BUF:.*]] = triton_gpu.local_alloc  : () -> !tt.memdesc<512xi32, #[[SHARE]], #triton_gpu.shared_memory, mutable>
// CHECK-NEXT: %2 = tt.get_program_id x : i32
// CHECK-NEXT: %3 = arith.addi %arg3, %c127_i32 : i32
// CHECK-NEXT: %4 = arith.divsi %3, %c128_i32 : i32
// CHECK:      %79 = triton_gpu.async_commit_group %78
// CHECK-NEXT: %80:14 = scf.for %arg10 = %c0_i32 to %21 step %c1_i32
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = true{{.*}}regionId = 0
// CHECK-NEXT: %83 = arith.subi %21, %c1_i32 : i32
// CHECK-NEXT: %84 = arith.cmpi slt, %arg10, %83 : i32
// CHECK:      %93 = triton_gpu.memdesc_subview
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = false{{.*}}regionId = 0
// CHECK-NEXT: %94 = triton_gpu.async_wait %arg21 {num = 0 : i32}
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = true{{.*}}regionId = 1
// CHECK:      %96 = triton_nvidia_gpu.warp_group_dot
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = false{{.*}}regionId = 1
// CHECK-NEXT: %97:3 = triton_nvidia_gpu.warp_group_dot_wait
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = true{{.*}}regionId = 2
// CHECK-NEXT: %98 = arith.addi %arg19, %c1_i32 : i32
// CHECK:      %127 = triton_gpu.async_copy_global_to_local
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = false{{.*}}regionId = 2
// CHECK-NEXT: %128 = triton_gpu.async_commit_group %127
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = true{{.*}}regionId = 3
// CHECK-NEXT: %129 = tt.splat %120 : i32 -> tensor<64x1xi32, #blocked>
// CHECK:      %132 = triton_gpu.memdesc_subview
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = true{{.*}}regionId = 5
// CHECK:      %135 = triton_gpu.async_copy_global_to_local
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = false{{.*}}regionId = 3
// CHECK-NEXT: %136 = triton_gpu.async_commit_group %135
// CHECK:      %138 = arith.cmpi ne, %arg22, %22 : i32
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = true{{.*}}regionId = 4
// CHECK-NEXT: scf.if %137 {
// CHECK-NEXT: %139:3 = triton_nvidia_gpu.warp_group_dot_wait
// CHECK:      %157 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked2>
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = false{{.*}}regionId = 5
// CHECK:      tt.store %154, %163, %161 : tensor<128x256x!tt.ptr<f16>, #blocked2>
// CHECK-NEXT: }
// CHECK-NEXT: "triton_gpu.local_record"(%[[BUF]], %[[IDX]]) <{{.*}}, isStart = false{{.*}}regionId = 4
// CHECK-NEXT: scf.yield
// CHECK: "triton_gpu.proton_finalize"(%[[BUF]], %[[IDX]], %[[ARG]])
// CHECK-NEXT: tt.return
// CHECK-NEXT: }
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#loc = loc(unknown)
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.proton-slots" = 256 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_ttgir_profile(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32 loc(#loc)
    %false = arith.constant false loc(#loc)
    %cst = arith.constant dense<0> : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %cst_0 = arith.constant dense<0> : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
    %c256_i32 = arith.constant 256 : i32 loc(#loc)
    %c128_i32 = arith.constant 128 : i32 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c8_i32 = arith.constant 8 : i32 loc(#loc)
    %c-1_i32 = arith.constant -1 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %c132_i32 = arith.constant 132 : i32 loc(#loc)
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1> loc(#loc)
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked> loc(#loc)
    %c64_i32 = arith.constant 64 : i32 loc(#loc)
    %c127_i32 = arith.constant 127 : i32 loc(#loc)
    %c255_i32 = arith.constant 255 : i32 loc(#loc)
    %c63_i32 = arith.constant 63 : i32 loc(#loc)
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma> loc(#loc)
    %0 = tt.get_program_id x : i32 loc(#loc)
    %1 = arith.addi %arg3, %c127_i32 : i32 loc(#loc)
    %2 = arith.divsi %1, %c128_i32 : i32 loc(#loc)
    %3 = arith.addi %arg4, %c255_i32 : i32 loc(#loc)
    %4 = arith.divsi %3, %c256_i32 : i32 loc(#loc)
    %5 = arith.addi %arg5, %c63_i32 : i32 loc(#loc)
    %6 = arith.divsi %5, %c64_i32 : i32 loc(#loc)
    %7 = arith.muli %2, %4 : i32 loc(#loc)
    %8 = arith.divsi %7, %c132_i32 : i32 loc(#loc)
    %9 = arith.remsi %7, %c132_i32 : i32 loc(#loc)
    %10 = arith.cmpi slt, %0, %9 : i32 loc(#loc)
    %11 = scf.if %10 -> (i32) {
      %81 = arith.addi %8, %c1_i32 : i32 loc(#loc)
      scf.yield %81 : i32 loc(#loc)
    } else {
      scf.yield %8 : i32 loc(#loc)
    } loc(#loc)
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> loc(#loc)
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc)
    %14 = arith.muli %4, %c8_i32 : i32 loc(#loc)
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc)
    %17 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %18 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc)
    %19 = arith.muli %6, %11 : i32 loc(#loc)
    %20 = arith.subi %6, %c1_i32 : i32 loc(#loc)
    %21 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1> loc(#loc)
    %22 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1> loc(#loc)
    %23 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked> loc(#loc)
    %24 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked> loc(#loc)
    %25 = tt.expand_dims %12 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1> loc(#loc)
    %26 = tt.expand_dims %13 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc)
    %27 = triton_gpu.local_alloc  : () -> !tt.memdesc<2x128x64xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc)
    %28 = triton_gpu.local_alloc  : () -> !tt.memdesc<2x64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc)
    %29 = arith.cmpi sgt, %19, %c0_i32 : i32 loc(#loc)
    %30 = arith.divsi %0, %14 : i32 loc(#loc)
    %31 = arith.muli %30, %c8_i32 : i32 loc(#loc)
    %32 = arith.subi %2, %31 : i32 loc(#loc)
    %33 = arith.minsi %32, %c8_i32 : i32 loc(#loc)
    %34 = arith.remsi %0, %33 : i32 loc(#loc)
    %35 = arith.addi %31, %34 : i32 loc(#loc)
    %36 = arith.remsi %0, %14 : i32 loc(#loc)
    %37 = arith.divsi %36, %33 : i32 loc(#loc)
    %38 = arith.muli %35, %c128_i32 : i32 loc(#loc)
    %39 = arith.muli %37, %c256_i32 : i32 loc(#loc)
    %40 = tt.splat %38 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
    %41 = arith.addi %40, %15 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
    %42 = tt.splat %39 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %43 = arith.addi %42, %17 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %44 = tt.splat %arg3 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
    %45 = arith.cmpi slt, %41, %44 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
    %46 = arith.select %45, %41, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
    %47 = tt.splat %arg4 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %48 = arith.cmpi slt, %43, %47 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %49 = arith.select %48, %43, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc)
    %50 = tt.expand_dims %46 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1> loc(#loc)
    %51 = arith.muli %50, %21 : tensor<128x1xi32, #blocked1> loc(#loc)
    %52 = tt.broadcast %51 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc)
    %53 = tt.broadcast %25 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc)
    %54 = arith.addi %52, %53 : tensor<128x64xi32, #blocked1> loc(#loc)
    %55 = tt.addptr %22, %54 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1> loc(#loc)
    %56 = tt.expand_dims %49 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked> loc(#loc)
    %57 = arith.muli %56, %23 : tensor<1x256xi32, #blocked> loc(#loc)
    %58 = tt.broadcast %26 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc)
    %59 = tt.broadcast %57 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc)
    %60 = arith.addi %58, %59 : tensor<64x256xi32, #blocked> loc(#loc)
    %61 = tt.addptr %24, %60 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked> loc(#loc)
    %62 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1> loc(#loc)
    %63 = arith.cmpi slt, %25, %62 : tensor<1x64xi32, #blocked1> loc(#loc)
    %64 = tt.broadcast %63 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1> loc(#loc)
    %65 = triton_gpu.memdesc_subview %27[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<2x128x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc)
    %66 = tt.splat %29 : i1 -> tensor<128x64xi1, #blocked1> loc(#loc)
    %67 = arith.andi %66, %64 : tensor<128x64xi1, #blocked1> loc(#loc)
    %68 = triton_gpu.async_copy_global_to_local %55, %65 mask %67 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc)
    %69 = triton_gpu.async_commit_group %68 loc(#loc)
    %70 = tt.splat %arg5 : i32 -> tensor<64x1xi32, #blocked> loc(#loc)
    %71 = arith.cmpi slt, %26, %70 : tensor<64x1xi32, #blocked> loc(#loc)
    %72 = tt.broadcast %71 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked> loc(#loc)
    %73 = triton_gpu.memdesc_subview %28[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<2x64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc)
    %74 = tt.splat %29 : i1 -> tensor<64x256xi1, #blocked> loc(#loc)
    %75 = arith.andi %74, %72 : tensor<64x256xi1, #blocked> loc(#loc)
    %76 = triton_gpu.async_copy_global_to_local %61, %73 mask %75 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc)
    %77 = triton_gpu.async_commit_group %76 loc(#loc)
    %78:14 = scf.for %arg10 = %c0_i32 to %19 step %c1_i32 iter_args(%arg11 = %c0_i32, %arg12 = %0, %arg13 = %35, %arg14 = %37, %arg15 = %cst_3, %arg16 = %46, %arg17 = %49, %arg18 = %false, %arg19 = %c0_i32, %arg20 = %c-1_i32, %arg21 = %77, %arg22 = %c0_i32, %arg23 = %35, %arg24 = %37) -> (i32, i32, i32, i32, tensor<128x256xf32, #mma>, tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, i1, i32, i32, !triton_gpu.async.token, i32, i32, i32)  : i32 {
      tt.proton_record <0, "start", "cycle", "warpgroup">
      %81 = arith.subi %19, %c1_i32 : i32 loc(#loc)
      %82 = arith.cmpi slt, %arg10, %81 : i32 loc(#loc)
      %83 = arith.cmpi eq, %arg11, %20 : i32 loc(#loc)
      %84 = arith.addi %arg11, %c1_i32 : i32 loc(#loc)
      %85 = arith.select %83, %c0_i32, %84 : i32 loc(#loc)
      %86 = arith.cmpi eq, %85, %c0_i32 : i32 loc(#loc)
      %87:5 = scf.if %86 -> (i32, i32, i32, tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) {
        %137 = arith.addi %arg12, %c132_i32 : i32 loc(#loc)
        %138 = arith.divsi %137, %14 : i32 loc(#loc)
        %139 = arith.muli %138, %c8_i32 : i32 loc(#loc)
        %140 = arith.subi %2, %139 : i32 loc(#loc)
        %141 = arith.minsi %140, %c8_i32 : i32 loc(#loc)
        %142 = arith.remsi %137, %141 : i32 loc(#loc)
        %143 = arith.addi %139, %142 : i32 loc(#loc)
        %144 = arith.remsi %137, %14 : i32 loc(#loc)
        %145 = arith.divsi %144, %141 : i32 loc(#loc)
        %146 = arith.muli %143, %c128_i32 : i32 loc(#loc)
        %147 = arith.muli %145, %c256_i32 : i32 loc(#loc)
        %148 = tt.splat %146 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
        %149 = arith.addi %148, %15 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
        %150 = tt.splat %147 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc)
        %151 = arith.addi %150, %17 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc)
        %152 = arith.cmpi slt, %149, %44 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
        %153 = arith.select %152, %149, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc)
        %154 = arith.cmpi slt, %151, %47 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc)
        %155 = arith.select %154, %151, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc)
        scf.yield %137, %143, %145, %153, %155 : i32, i32, i32, tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc)
      } else {
        scf.yield %arg12, %arg13, %arg14, %arg16, %arg17 : i32, i32, i32, tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc)
      } loc(#loc)
      %88 = arith.addi %arg20, %c1_i32 : i32 loc(#loc)
      %89 = arith.cmpi slt, %88, %c2_i32 : i32 loc(#loc)
      %90 = arith.select %89, %88, %c0_i32 : i32 loc(#loc)
      %91 = triton_gpu.memdesc_subview %27[%90, %c0_i32, %c0_i32] : !tt.memdesc<2x128x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc)
      tt.proton_record <0, "end", "cycle", "warpgroup">
      %92 = triton_gpu.async_wait %arg21 {num = 0 : i32} loc(#loc)
      tt.proton_record <1, "start", "cycle", "warpgroup">
      %93 = triton_gpu.memdesc_subview %28[%90, %c0_i32, %c0_i32] : !tt.memdesc<2x64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc)
      %94 = triton_nvidia_gpu.warp_group_dot %91, %93, %arg15, %arg18 {inputPrecision = 0 : i32, isAsync = true} : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable> * !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<128x256xf32, #mma> loc(#loc)
      tt.proton_record <1, "end", "cycle", "warpgroup">
      %95:3 = triton_nvidia_gpu.warp_group_dot_wait %94, %91, %93 {pendings = 1 : i32} : tensor<128x256xf32, #mma>, !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc)
      tt.proton_record <2, "start", "cycle", "warpgroup">
      %96 = arith.addi %arg19, %c1_i32 : i32 loc(#loc)
      %97 = arith.cmpi slt, %96, %c2_i32 : i32 loc(#loc)
      %98 = arith.select %97, %96, %c0_i32 : i32 loc(#loc)
      %99 = arith.muli %85, %c64_i32 : i32 loc(#loc)
      %100 = tt.splat %99 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> loc(#loc)
      %101 = tt.splat %99 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc)
      %102 = arith.addi %100, %12 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> loc(#loc)
      %103 = arith.addi %101, %13 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc)
      %104 = tt.expand_dims %87#3 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1> loc(#loc)
      %105 = arith.muli %104, %21 : tensor<128x1xi32, #blocked1> loc(#loc)
      %106 = tt.expand_dims %102 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1> loc(#loc)
      %107 = tt.broadcast %105 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc)
      %108 = tt.broadcast %106 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc)
      %109 = arith.addi %107, %108 : tensor<128x64xi32, #blocked1> loc(#loc)
      %110 = tt.addptr %22, %109 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1> loc(#loc)
      %111 = tt.expand_dims %103 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc)
      %112 = tt.expand_dims %87#4 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked> loc(#loc)
      %113 = arith.muli %112, %23 : tensor<1x256xi32, #blocked> loc(#loc)
      %114 = tt.broadcast %111 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc)
      %115 = tt.broadcast %113 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc)
      %116 = arith.addi %114, %115 : tensor<64x256xi32, #blocked> loc(#loc)
      %117 = tt.addptr %24, %116 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked> loc(#loc)
      %118 = arith.subi %arg5, %99 : i32 loc(#loc)
      %119 = tt.splat %118 : i32 -> tensor<1x64xi32, #blocked1> loc(#loc)
      %120 = arith.cmpi slt, %25, %119 : tensor<1x64xi32, #blocked1> loc(#loc)
      %121 = tt.broadcast %120 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1> loc(#loc)
      %122 = triton_gpu.memdesc_subview %27[%98, %c0_i32, %c0_i32] : !tt.memdesc<2x128x64xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc)
      %123 = tt.splat %82 : i1 -> tensor<128x64xi1, #blocked1> loc(#loc)
      %124 = arith.andi %123, %121 : tensor<128x64xi1, #blocked1> loc(#loc)
      %125 = triton_gpu.async_copy_global_to_local %110, %122 mask %124 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc)
      tt.proton_record <2, "end", "cycle", "warpgroup">
      %126 = triton_gpu.async_commit_group %125 loc(#loc)
      tt.proton_record <3, "start", "cycle", "warpgroup">
      %127 = tt.splat %118 : i32 -> tensor<64x1xi32, #blocked> loc(#loc)
      %128 = arith.cmpi slt, %26, %127 : tensor<64x1xi32, #blocked> loc(#loc)
      %129 = tt.broadcast %128 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked> loc(#loc)
      %130 = triton_gpu.memdesc_subview %28[%98, %c0_i32, %c0_i32] : !tt.memdesc<2x64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc)
      tt.proton_record <5, "start", "cycle", "warpgroup">
      %131 = tt.splat %82 : i1 -> tensor<64x256xi1, #blocked> loc(#loc)
      %132 = arith.andi %131, %129 : tensor<64x256xi1, #blocked> loc(#loc)
      %133 = triton_gpu.async_copy_global_to_local %117, %130 mask %132 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc)
      tt.proton_record <3, "end", "cycle", "warpgroup">
      %134 = triton_gpu.async_commit_group %133 loc(#loc)
      %135 = arith.cmpi eq, %arg22, %20 : i32 loc(#loc)
      %136 = arith.cmpi ne, %arg22, %20 : i32 loc(#loc)
      tt.proton_record <4, "start", "cycle", "warpgroup">
      scf.if %135 {
        %137:3 = triton_nvidia_gpu.warp_group_dot_wait %95#0, %91, %93 {pendings = 0 : i32} : tensor<128x256xf32, #mma>, !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc)
        %138 = arith.muli %arg23, %c128_i32 : i32 loc(#loc)
        %139 = tt.splat %138 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc)
        %140 = arith.addi %139, %16 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc)
        %141 = arith.muli %arg24, %c256_i32 : i32 loc(#loc)
        %142 = tt.splat %141 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc)
        %143 = arith.addi %142, %18 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc)
        %144 = tt.expand_dims %140 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2> loc(#loc)
        %145 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked2> loc(#loc)
        %146 = arith.muli %145, %144 : tensor<128x1xi32, #blocked2> loc(#loc)
        %147 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2> loc(#loc)
        %148 = tt.addptr %147, %146 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2> loc(#loc)
        %149 = tt.expand_dims %143 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2> loc(#loc)
        %150 = tt.broadcast %148 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x256x!tt.ptr<f16>, #blocked2> loc(#loc)
        %151 = tt.broadcast %149 : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2> loc(#loc)
        %152 = tt.addptr %150, %151 : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2> loc(#loc)
        %153 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked2> loc(#loc)
        %154 = arith.cmpi slt, %144, %153 : tensor<128x1xi32, #blocked2> loc(#loc)
        %155 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked2> loc(#loc)
        tt.proton_record <5, "end", "cycle", "warpgroup">
        %156 = arith.cmpi slt, %149, %155 : tensor<1x256xi32, #blocked2> loc(#loc)
        %157 = tt.broadcast %154 : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2> loc(#loc)
        %158 = tt.broadcast %156 : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2> loc(#loc)
        %159 = arith.andi %157, %158 : tensor<128x256xi1, #blocked2> loc(#loc)
        %160 = arith.truncf %137#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma> loc(#loc)
        %161 = triton_gpu.convert_layout %160 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked2> loc(#loc)
        tt.store %152, %161, %159 : tensor<128x256x!tt.ptr<f16>, #blocked2> loc(#loc)
      } loc(#loc)
      tt.proton_record <4, "end", "cycle", "warpgroup">
      scf.yield %85, %87#0, %87#1, %87#2, %95#0, %87#3, %87#4, %136, %98, %90, %134, %85, %87#1, %87#2 : i32, i32, i32, i32, tensor<128x256xf32, #mma>, tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, i1, i32, i32, !triton_gpu.async.token, i32, i32, i32 loc(#loc)
    } loc(#loc)
    %79 = triton_nvidia_gpu.warp_group_dot_wait %78#4 {pendings = 0 : i32} : tensor<128x256xf32, #mma> loc(#loc)
    %80 = triton_gpu.async_wait  {num = 0 : i32} loc(#loc)
    triton_gpu.local_dealloc %27 : !tt.memdesc<2x128x64xf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc)
    triton_gpu.local_dealloc %28 : !tt.memdesc<2x64x256xf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
