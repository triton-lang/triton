// RUN: triton-opt %s -split-input-file -tritonamdgpu-reorder-instructions | FileCheck %s

// Check that we place local_alloc, local_store (optional) and local_load right after definition of their operands
// in cases where local_alloc is in the loop but it's operand is not.
// This is useful for making sure that Q tensor in FA is hoisted out of the main loop and kept in registers
// throughout the computation.

// CHECK-LABEL: hoist_q_out_of_the_loop
//       CHECK: %[[TRUNCF:.+]] = arith.truncf
//       CHECK-NEXT:  %[[ALLOC:.+]] = triton_gpu.local_alloc %[[TRUNCF]]
//       CHECK-NEXT:  triton_gpu.local_load %[[ALLOC]]
//       CHECK: scf.for
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0], hasLeadingOffset = false}>
#mfma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @hoist_q_out_of_the_loop(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mfma>
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %arg7 : i32
    %3 = tt.addptr %arg0, %2 : !tt.ptr<f16>, i32
    %12 = tt.splat %3 : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #blocked1>
    %41 = tt.load %12 : tensor<256x128x!tt.ptr<f16>, #blocked1>
    %42 = arith.extf %41 : tensor<256x128xf16, #blocked1> to tensor<256x128xf32, #blocked1>
    %43 = tt.splat %cst : f32 -> tensor<256x128xf32, #blocked1>
    %44 = arith.mulf %42, %43 : tensor<256x128xf32, #blocked1>
    %45 = arith.truncf %44 : tensor<256x128xf32, #blocked1> to tensor<256x128xf16, #blocked1>
    %54:1 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg26 = %c0_i64) -> (i64)  : i32 {
      %73 = tt.splat %3 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked2>
      %74 = tt.load %73 : tensor<128x128x!tt.ptr<f16>, #blocked2>
      %75 = triton_gpu.local_alloc %45 : (tensor<256x128xf16, #blocked1>) -> !triton_gpu.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory>
      %76 = triton_gpu.local_load %75 : !triton_gpu.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory> -> tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %77 = triton_gpu.local_alloc %74 : (tensor<128x128xf16, #blocked2>) -> !triton_gpu.memdesc<128x128xf16, #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>, #triton_gpu.shared_memory>
      %78 = triton_gpu.local_load %77 : !triton_gpu.memdesc<128x128xf16, #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>, #triton_gpu.shared_memory> -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %79 = tt.dot %76, %78, %cst_2 : tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      %107 = arith.addi %arg26, %c128_i64 : i64
      scf.yield %107 : i64
    } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
    tt.return
  }
}


// -----
// Check that reordering described in hoist_q_out_of_the_loop is not done in the case where both
// local_alloc and it's src tensor defining op are in the loop.
// CHECK-LABEL: no_hoist_q_type_reordering
//       CHECK: scf.for
//       CHECK: %[[TRUNCF:.+]] = arith.truncf
//       CHECK-NEXT:  arith.constant
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0], hasLeadingOffset = false}>
#mfma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @no_hoist_q_type_reordering(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c0_i64 = arith.constant 0 : i64
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %arg7 : i32
    %3 = tt.addptr %arg0, %2 : !tt.ptr<f16>, i32
    %12 = tt.splat %3 : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #blocked1>
    %41 = tt.load %12 : tensor<256x128x!tt.ptr<f16>, #blocked1>
    %42 = arith.extf %41 : tensor<256x128xf16, #blocked1> to tensor<256x128xf32, #blocked1>
    %43 = tt.splat %cst : f32 -> tensor<256x128xf32, #blocked1>
    %44 = arith.mulf %42, %43 : tensor<256x128xf32, #blocked1>
    %54:1 = scf.for %arg21 = %c0_i32 to %arg20 step %c128_i32 iter_args(%arg26 = %c0_i64) -> (i64)  : i32 {
      %45 = arith.truncf %44 : tensor<256x128xf32, #blocked1> to tensor<256x128xf16, #blocked1>
      %cst_2 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mfma>
      %73 = tt.splat %3 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked2>
      %74 = tt.load %73 : tensor<128x128x!tt.ptr<f16>, #blocked2>
      %75 = triton_gpu.local_alloc %45 : (tensor<256x128xf16, #blocked1>) -> !triton_gpu.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory>
      %76 = triton_gpu.local_load %75 : !triton_gpu.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory> -> tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %77 = triton_gpu.local_alloc %74 : (tensor<128x128xf16, #blocked2>) -> !triton_gpu.memdesc<128x128xf16, #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>, #triton_gpu.shared_memory>
      %78 = triton_gpu.local_load %77 : !triton_gpu.memdesc<128x128xf16, #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>, #triton_gpu.shared_memory> -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %79 = tt.dot %76, %78, %cst_2 : tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      %107 = arith.addi %arg26, %c128_i64 : i64
      scf.yield %107 : i64
    } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>

// CHECK-LABEL: order_load_alloc_local_load_local_store
//       CHECK:   %[[LOAD:.+]] = tt.load
//       CHECK:   %[[ALLOC:.+]] = triton_gpu.local_alloc
//       CHECK:   triton_gpu.local_store %[[LOAD]], %[[ALLOC]]
//       CHECK:   triton_gpu.local_load %[[ALLOC]]
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @order_load_alloc_local_load_local_store(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %9 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %10 = triton_gpu.local_alloc : () -> !triton_gpu.memdesc<32x32xf32, #shared, mutable>
    triton_gpu.local_store %9, %10 : tensor<32x32xf32, #blocked> -> !triton_gpu.memdesc<32x32xf32, #shared, mutable>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %11 = triton_gpu.local_load %10 : !triton_gpu.memdesc<32x32xf32, #shared, mutable> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %12 = tt.dot %11, %cst_0, %cst : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %13 = triton_gpu.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----
// Move loads (and independent local_stores) as early as possible.
// For example in the matmul_loop below, the scf.for loop looks like this after pipeliner:
//   scf.for ... {
//     // stage 1
//     %a = tt.local_load %a_tile
//     %b = tt.local_load %b_tile
//     tt.dot %c, %a, %b
//     // stage 0
//     %aptr = tt.addptr %aptr, %k
//     %a_next = tt.load %aptr
//     %bptr = tt.addptr %bptr, %k
//     %b_next = tt.load %bptr
//     tt.local_store %a_next
//     tt.local_store %b_next
//     yield
//   }
//
//  Solution for num_stages=2 :
//   scf.for ... {
//     // stage 0.a
//     %aptr = tt.addptr %aptr, %k
//     %a_next = tt.load %aptr
//     %bptr = tt.addptr %bptr, %k
//     %b_next = tt.load %bptr
//     // stage 1
//     %a = tt.local_load %a_tile
//     %b = tt.local_load %b_tile
//     tt.dot %c, %a, %b
//     // stage 0.b
//     tt.local_store %a_next
//     tt.local_store %b_next
//     yield
//   }
//
//  Solution for num_stages=3 (double-buffered) :
//   scf.for ... {
//     // stage 1
//     tt.local_store %a_next_1
//     tt.local_store %b_next_1
//     // stage 0
//     %aptr = tt.addptr %aptr, %k
//     %a_next_2 = tt.load %aptr
//     %bptr = tt.addptr %bptr, %k
//     %b_next_2 = tt.load %bptr
//     // stage 2
//     %a = tt.local_load %a_tile
//     %b = tt.local_load %b_tile
//     tt.dot %c, %a, %b
//     yield
//   }

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = []}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared2 = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], hasLeadingOffset = false}>
#shared3 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared4 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32, triton_gpu.target = "hip:gfx942"} {

// CHECK-LABEL:  tt.func @matmul_loop
// CHECK:  %{{.*}}:6 = scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}})
// Stage 0.a
//        CHECK:       %[[ADDPTR_20:.*]] = tt.addptr %[[ARG6]], %{{.*}}
//        CHECK:       %[[SUBI_21:.*]] = arith.subi %{{.*}}, %{{.*}}
//        CHECK:       %[[CMPI_22:.*]] = arith.cmpi slt, %[[ARG5]], %[[SUBI_21]]
//        CHECK:       %[[SPLAT_23:.*]] = tt.splat %[[CMPI_22]]
//        CHECK:       %[[LOAD_24:.*]] = tt.load %[[ADDPTR_20]], %[[SPLAT_23]]
//        CHECK:       %[[ADDPTR_25:.*]] = tt.addptr %[[ARG7]], %{{.*}}
//        CHECK:       %[[SPLAT_26:.*]] = tt.splat %[[CMPI_22]]
//        CHECK:       %[[LOAD_27:.*]] = tt.load %[[ADDPTR_25]], %[[SPLAT_26]]
// Stage 1
//        CHECK:       %[[LOCAL_LOAD_28:.*]] = triton_gpu.local_load %[[ARG10]]
//        CHECK:       %[[LOCAL_LOAD_29:.*]] = triton_gpu.local_load %[[ARG11]]
//        CHECK:       %[[MULF_30:.*]] = arith.mulf %[[LOCAL_LOAD_29]], %{{.*}}
//        CHECK:       %[[DOT_31:.*]] = tt.dot %[[LOCAL_LOAD_28]], %[[MULF_30]], %[[ARG8]]
// Stage 0.b
//        CHECK:       %[[ADDI_32:.*]] = arith.addi %[[ARG9]], %{{.*}}
//        CHECK:       %[[CMPI_33:.*]] = arith.cmpi slt, %[[ADDI_32]], %{{.*}}
//        CHECK:       %[[SELECT_34:.*]] = arith.select %[[CMPI_33]], %[[ADDI_32]], %{{.*}}
//        CHECK:       %[[MEMDESC_SUBVIEW_35:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_34]], %{{.*}}, %{{.*}}]
//        CHECK:       triton_gpu.local_store %[[LOAD_24]], %[[MEMDESC_SUBVIEW_35]]
//        CHECK:       %[[MEMDESC_SUBVIEW_36:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_34]], %{{.*}}, %{{.*}}]
//        CHECK:       triton_gpu.local_store %[[LOAD_27]], %[[MEMDESC_SUBVIEW_36]]
//        CHECK:       scf.yield %[[ADDPTR_20]], %[[ADDPTR_25]], %[[DOT_31]], %[[SELECT_34]], %[[MEMDESC_SUBVIEW_35]], %[[MEMDESC_SUBVIEW_36]]
//        CHECK:   }

  tt.func @matmul_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<4.000000e+00> : tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_0 = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<128x32xi32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %0 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %3 = tt.broadcast %2 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %4 = tt.addptr %0, %3 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %5 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %9 = tt.addptr %5, %8 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %10 = triton_gpu.local_alloc  : () -> !triton_gpu.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %11 = triton_gpu.local_alloc  : () -> !triton_gpu.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %12 = arith.cmpi slt, %arg0, %arg1 : index
    %13 = tt.splat %12 : i1 -> tensor<128x32xi1, #blocked1>
    %14 = tt.load %4, %13 : tensor<128x32x!tt.ptr<f16>, #blocked1>
    %15 = tt.splat %12 : i1 -> tensor<32x128xi1, #blocked>
    %16 = tt.load %9, %15, %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked>
    %17 = triton_gpu.memdesc_subview %10[%c0_i32, %c0_i32, %c0_i32] : !triton_gpu.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !triton_gpu.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %14, %17 : tensor<128x32xf16, #blocked1> -> !triton_gpu.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %18 = triton_gpu.memdesc_subview %11[%c0_i32, %c0_i32, %c0_i32] : !triton_gpu.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !triton_gpu.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %16, %18 : tensor<32x128xf16, #blocked> -> !triton_gpu.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %19:6 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %4, %arg7 = %9, %arg8 = %cst_2, %arg9 = %c0_i32, %arg10 = %17, %arg11 = %18) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !triton_gpu.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !triton_gpu.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>) {
      %20 = arith.subi %arg1, %arg2 : index
      %21 = arith.cmpi slt, %arg5, %20 : index
      %22 = triton_gpu.local_load %arg10 : !triton_gpu.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %23 = triton_gpu.local_load %arg11 : !triton_gpu.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %24 = arith.mulf %23, %cst : tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %25 = tt.dot %22, %24, %arg8 : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %26 = tt.addptr %arg6, %cst_1 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %27 = tt.addptr %arg7, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %28 = tt.splat %21 : i1 -> tensor<128x32xi1, #blocked1>
      %29 = tt.load %26, %28 : tensor<128x32x!tt.ptr<f16>, #blocked1>
      %30 = tt.splat %21 : i1 -> tensor<32x128xi1, #blocked>
      %31 = tt.load %27, %30, %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked>
      %32 = arith.addi %arg9, %c1_i32 : i32
      %33 = arith.cmpi slt, %32, %c1_i32 : i32
      %34 = arith.select %33, %32, %c0_i32 : i32
      %35 = triton_gpu.memdesc_subview %10[%34, %c0_i32, %c0_i32] : !triton_gpu.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !triton_gpu.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %29, %35 : tensor<128x32xf16, #blocked1> -> !triton_gpu.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      %36 = triton_gpu.memdesc_subview %11[%34, %c0_i32, %c0_i32] : !triton_gpu.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !triton_gpu.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %31, %36 : tensor<32x128xf16, #blocked> -> !triton_gpu.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %26, %27, %25, %34, %35, %36 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !triton_gpu.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !triton_gpu.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    }
    triton_gpu.local_dealloc %10 : !triton_gpu.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %11 : !triton_gpu.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    tt.return %19#2 : tensor<128x128xf32, #mma>
  }


// This example tests that tt.load overlaps with independent ttg.local_store which
// overlaps with independent tt.dot.
// num_stages == 3, double buffered

// CHECK-LABEL:  tt.func @matmul_loop_mb
// CHECK:  %{{.*}}:8 = scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}})
// Stage 0
//        CHECK:       %[[ADDPTR_28:.*]] = tt.addptr %[[ARG6]], %{{.*}}
//        CHECK:       %[[MULI_29:.*]] = arith.muli %{{.*}}, %{{.*}}
//        CHECK:       %[[SUBI_30:.*]] = arith.subi %{{.*}}, %[[MULI_29]]
//        CHECK:       %[[CMPI_31:.*]] = arith.cmpi slt, %[[ARG5]], %[[SUBI_30]]
//        CHECK:       %[[SPLAT_32:.*]] = tt.splat %[[CMPI_31]]
//        CHECK:       %[[LOAD_33:.*]] = tt.load %[[ADDPTR_28]], %[[SPLAT_32]]
//        CHECK:       %[[ADDPTR_34:.*]] = tt.addptr %[[ARG7]], %{{.*}}
//        CHECK:       %[[SPLAT_35:.*]] = tt.splat %[[CMPI_31]]
//        CHECK:       %[[LOAD_36:.*]] = tt.load %[[ADDPTR_34]], %[[SPLAT_35]]
// Stage 1
//        CHECK:       %[[ADDI_37:.*]] = arith.addi %[[ARG9]], %{{.*}}
//        CHECK:       %[[CMPI_38:.*]] = arith.cmpi slt, %[[ADDI_37]], %{{.*}}
//        CHECK:       %[[SELECT_39:.*]] = arith.select %[[CMPI_38]], %[[ADDI_37]], %{{.*}}
//        CHECK:       %[[MEMDESC_SUBVIEW_40:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_39]], %{{.*}}, %{{.*}}]
//        CHECK:       triton_gpu.local_store %[[ARG12]], %[[MEMDESC_SUBVIEW_40]]
//        CHECK:       %[[MEMDESC_SUBVIEW_41:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_39]], %{{.*}}, %{{.*}}]
//        CHECK:       triton_gpu.local_store %[[ARG13]], %[[MEMDESC_SUBVIEW_41]]
// Stage 2
//        CHECK:       %[[LOCAL_LOAD_42:.*]] = triton_gpu.local_load %[[ARG10]]
//        CHECK:       %[[LOCAL_LOAD_43:.*]] = triton_gpu.local_load %[[ARG11]]
//        CHECK:       %[[MULF_44:.*]] = arith.mulf %[[LOCAL_LOAD_43]], %{{.*}}
//        CHECK:       %[[DOT_45:.*]] = tt.dot %[[LOCAL_LOAD_42]], %[[MULF_44]], %[[ARG8]]
//        CHECK:       scf.yield %[[ADDPTR_28]], %[[ADDPTR_34]], %[[DOT_45]], %[[SELECT_39]], %[[MEMDESC_SUBVIEW_40]], %[[MEMDESC_SUBVIEW_41]], %[[LOAD_33]], %[[LOAD_36]]
//        CHECK:   }

  tt.func @matmul_loop_mb(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<4.000000e+00> : tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_0 = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<128x32xi32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %0 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %3 = tt.broadcast %2 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %4 = tt.addptr %0, %3 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %5 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %9 = tt.addptr %5, %8 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %10 = triton_gpu.local_alloc  : () -> !triton_gpu.memdesc<2x128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %11 = triton_gpu.local_alloc  : () -> !triton_gpu.memdesc<2x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %12 = arith.cmpi slt, %arg0, %arg1 : index
    %13 = tt.splat %12 : i1 -> tensor<128x32xi1, #blocked1>
    %14 = tt.load %4, %13 : tensor<128x32x!tt.ptr<f16>, #blocked1>
    %15 = tt.splat %12 : i1 -> tensor<32x128xi1, #blocked>
    %16 = tt.load %9, %15, %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked>
    %17 = arith.addi %arg0, %arg2 : index
    %18 = arith.cmpi slt, %17, %arg1 : index
    %19 = tt.addptr %4, %cst_1 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %20 = tt.addptr %9, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %21 = tt.splat %18 : i1 -> tensor<128x32xi1, #blocked1>
    %22 = tt.load %19, %21 : tensor<128x32x!tt.ptr<f16>, #blocked1>
    %23 = tt.splat %18 : i1 -> tensor<32x128xi1, #blocked>
    %24 = tt.load %20, %23, %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked>
    %25 = triton_gpu.memdesc_subview %10[%c0_i32, %c0_i32, %c0_i32] : !triton_gpu.memdesc<2x128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !triton_gpu.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %14, %25 : tensor<128x32xf16, #blocked1> -> !triton_gpu.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %26 = triton_gpu.memdesc_subview %11[%c0_i32, %c0_i32, %c0_i32] : !triton_gpu.memdesc<2x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !triton_gpu.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %16, %26 : tensor<32x128xf16, #blocked> -> !triton_gpu.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %27:8 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %19, %arg7 = %20, %arg8 = %cst_2, %arg9 = %c0_i32, %arg10 = %25, %arg11 = %26, %arg12 = %22, %arg13 = %24) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !triton_gpu.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !triton_gpu.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>, tensor<128x32xf16, #blocked1>, tensor<32x128xf16, #blocked>) {
      %28 = arith.muli %arg2, %c2 : index
      %29 = arith.subi %arg1, %28 : index
      %30 = arith.cmpi slt, %arg5, %29 : index
      %31 = triton_gpu.local_load %arg10 : !triton_gpu.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %32 = triton_gpu.local_load %arg11 : !triton_gpu.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %33 = arith.mulf %32, %cst : tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %34 = tt.dot %31, %33, %arg8 : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %35 = tt.addptr %arg6, %cst_1 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %36 = tt.addptr %arg7, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %37 = tt.splat %30 : i1 -> tensor<128x32xi1, #blocked1>
      %38 = tt.load %35, %37 : tensor<128x32x!tt.ptr<f16>, #blocked1>
      %39 = tt.splat %30 : i1 -> tensor<32x128xi1, #blocked>
      %40 = tt.load %36, %39, %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked>
      %41 = arith.addi %arg9, %c1_i32 : i32
      %42 = arith.cmpi slt, %41, %c2_i32 : i32
      %43 = arith.select %42, %41, %c0_i32 : i32
      %44 = triton_gpu.memdesc_subview %10[%43, %c0_i32, %c0_i32] : !triton_gpu.memdesc<2x128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !triton_gpu.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %arg12, %44 : tensor<128x32xf16, #blocked1> -> !triton_gpu.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      %45 = triton_gpu.memdesc_subview %11[%43, %c0_i32, %c0_i32] : !triton_gpu.memdesc<2x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !triton_gpu.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %arg13, %45 : tensor<32x128xf16, #blocked> -> !triton_gpu.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %35, %36, %34, %43, %44, %45, %38, %40 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !triton_gpu.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !triton_gpu.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>, tensor<128x32xf16, #blocked1>, tensor<32x128xf16, #blocked>
    }
    triton_gpu.local_dealloc %10 : !triton_gpu.memdesc<2x128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %11 : !triton_gpu.memdesc<2x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    tt.return %27#2 : tensor<128x128xf32, #mma>
  }

// This example shows dependent loads and verifies all are moved early.
// CHECK-LABEL:  tt.func @indirect_bmm_vector
// CHECK:  %{{.*}}:7 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}})
// Stage 0
//        CHECK:       %[[ADDPTR_20:.*]] = tt.addptr %[[ARG8]], %{{.*}}
//        CHECK:       %[[SUBI_21:.*]] = arith.subi %{{.*}}, %{{.*}}
//        CHECK:       %[[CMPI_22:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_21]]
//        CHECK:       %[[SPLAT_23:.*]] = tt.splat %[[CMPI_22]]
//        CHECK:       %[[LOAD_24:.*]] = tt.load %[[ADDPTR_20]], %[[SPLAT_23]]
// Stage 1.a
//        CHECK:       %[[EXPAND_DIMS_25:.*]] = tt.expand_dims %[[ARG13]] {axis = 1 : i32}
//        CHECK:       %[[BROADCAST_26:.*]] = tt.broadcast %[[EXPAND_DIMS_25]]
//        CHECK:       %[[MULI_27:.*]] = arith.muli %{{.*}}, %[[BROADCAST_26]]
//        CHECK:       %[[ADDPTR_28:.*]] = tt.addptr %{{.*}}, %[[MULI_27]]
//        CHECK:       %[[SPLAT_29:.*]] = tt.splat %[[CMPI_22]]
//        CHECK:       %[[LOAD_30:.*]] = tt.load %[[ADDPTR_28]], %[[SPLAT_29]]
//        CHECK:       %[[ADDPTR_31:.*]] = tt.addptr %[[ARG9]], %{{.*}}
//        CHECK:       %[[SUBI_32:.*]] = arith.subi %{{.*}}, %{{.*}}
//        CHECK:       %[[CMPI_33:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_32]]
//        CHECK:       %[[SPLAT_34:.*]] = tt.splat %[[CMPI_33]]
//        CHECK:       %[[LOAD_35:.*]] = tt.load %[[ADDPTR_31]], %[[SPLAT_34]]
// Stage 2
//        CHECK:       %[[LOCAL_LOAD_36:.*]] = triton_gpu.local_load %[[ARG11]]
//        CHECK:       %[[LOCAL_LOAD_37:.*]] = triton_gpu.local_load %[[ARG12]]
//        CHECK:       %[[DOT_38:.*]] = tt.dot %[[LOCAL_LOAD_36]], %[[LOCAL_LOAD_37]], %[[ARG7]]
// Stage 1.b
//        CHECK:       %[[ADDI_39:.*]] = arith.addi %[[ARG10]], %{{.*}}
//        CHECK:       %[[CMPI_40:.*]] = arith.cmpi slt, %[[ADDI_39]], %{{.*}}
//        CHECK:       %[[SELECT_41:.*]] = arith.select %[[CMPI_40]], %[[ADDI_39]], %{{.*}}
//        CHECK:       %[[MEMDESC_SUBVIEW_42:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_41]], %{{.*}}, %{{.*}}]
//        CHECK:       triton_gpu.local_store %[[LOAD_24]], %[[MEMDESC_SUBVIEW_42]]
//        CHECK:       %[[MEMDESC_SUBVIEW_43:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_41]], %{{.*}}, %{{.*}}]
//        CHECK:       triton_gpu.local_store %[[LOAD_30]], %[[MEMDESC_SUBVIEW_43]]
//        CHECK:       scf.yield %[[DOT_38]], %[[ADDPTR_20]], %[[ADDPTR_31]], %[[SELECT_41]], %[[MEMDESC_SUBVIEW_42]], %[[MEMDESC_SUBVIEW_43]], %[[LOAD_35]]
//        CHECK:   }

  tt.func @indirect_bmm_vector(%arg0: tensor<16x16xi64, #blocked> {tt.constancy = 16 : i32, tt.divisibility = 16 : i32}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = 2 : i32, tt.divisibility = 16 : i32}, %arg3: tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = 16 : i32, tt.divisibility = 16 : i32}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}) -> tensor<16x16xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<1> : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %0 = triton_gpu.local_alloc  : () -> !triton_gpu.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %1 = triton_gpu.local_alloc  : () -> !triton_gpu.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %2 = arith.cmpi sgt, %arg1, %c0 : index
    %3 = tt.splat %2 : i1 -> tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.load %arg3, %3 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %5 = arith.cmpi sgt, %arg1, %c1 : index
    %6 = tt.addptr %arg3, %cst_0 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.splat %2 : i1 -> tensor<16x16xi1, #blocked1>
    %8 = tt.load %arg2, %7 : tensor<16x16x!tt.ptr<f16>, #blocked1>
    %9 = tt.expand_dims %4 {axis = 1 : i32} : tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
    %10 = tt.broadcast %9 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
    %11 = arith.muli %arg0, %10 : tensor<16x16xi64, #blocked>
    %12 = tt.addptr %arg5, %11 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
    %13 = tt.splat %2 : i1 -> tensor<16x16xi1, #blocked>
    %14 = tt.load %12, %13 : tensor<16x16x!tt.ptr<f16>, #blocked>
    %15 = tt.splat %5 : i1 -> tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.load %6, %15 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %17 = triton_gpu.memdesc_subview %0[%c0_i32, %c0_i32, %c0_i32] : !triton_gpu.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !triton_gpu.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %8, %17 : tensor<16x16xf16, #blocked1> -> !triton_gpu.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %18 = triton_gpu.memdesc_subview %1[%c0_i32, %c0_i32, %c0_i32] : !triton_gpu.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !triton_gpu.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %14, %18 : tensor<16x16xf16, #blocked> -> !triton_gpu.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %19:7 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst, %arg8 = %arg2, %arg9 = %6, %arg10 = %c0_i32, %arg11 = %17, %arg12 = %18, %arg13 = %16) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, i32, !triton_gpu.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, !triton_gpu.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) {
      %20 = arith.subi %arg1, %c2 : index
      %21 = arith.cmpi slt, %arg6, %20 : index
      %22 = arith.subi %arg1, %c1 : index
      %23 = arith.cmpi slt, %arg6, %22 : index
      %24 = triton_gpu.local_load %arg11 : !triton_gpu.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %25 = triton_gpu.local_load %arg12 : !triton_gpu.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %26 = tt.dot %24, %25, %arg7 : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
      %27 = tt.addptr %arg8, %arg4 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %28 = tt.addptr %arg9, %cst_0 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %29 = tt.splat %23 : i1 -> tensor<16x16xi1, #blocked1>
      %30 = tt.load %27, %29 : tensor<16x16x!tt.ptr<f16>, #blocked1>
      %31 = tt.expand_dims %arg13 {axis = 1 : i32} : tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi64, #blocked>
      %32 = tt.broadcast %31 : tensor<16x1xi64, #blocked> -> tensor<16x16xi64, #blocked>
      %33 = arith.muli %arg0, %32 : tensor<16x16xi64, #blocked>
      %34 = tt.addptr %arg5, %33 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi64, #blocked>
      %35 = tt.splat %23 : i1 -> tensor<16x16xi1, #blocked>
      %36 = tt.load %34, %35 : tensor<16x16x!tt.ptr<f16>, #blocked>
      %37 = tt.splat %21 : i1 -> tensor<16xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %38 = tt.load %28, %37 : tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %39 = arith.addi %arg10, %c1_i32 : i32
      %40 = arith.cmpi slt, %39, %c1_i32 : i32
      %41 = arith.select %40, %39, %c0_i32 : i32
      %42 = triton_gpu.memdesc_subview %0[%41, %c0_i32, %c0_i32] : !triton_gpu.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !triton_gpu.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %30, %42 : tensor<16x16xf16, #blocked1> -> !triton_gpu.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      %43 = triton_gpu.memdesc_subview %1[%41, %c0_i32, %c0_i32] : !triton_gpu.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !triton_gpu.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %36, %43 : tensor<16x16xf16, #blocked> -> !triton_gpu.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      scf.yield %26, %27, %28, %41, %42, %43, %38 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, i32, !triton_gpu.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, !triton_gpu.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    }
    triton_gpu.local_dealloc %0 : !triton_gpu.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %1 : !triton_gpu.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    tt.return %19#0 : tensor<16x16xf32, #mma>
  }
}

// -----

//   CHECK-LABEL: sink_convert_dealloc
// CHECK-COUNT-2: triton_gpu.local_dealloc %{{.+}} : !triton_gpu.memdesc<4x128x64xf16, #shared, mutable>
//         CHECK: triton_gpu.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_dealloc(%arg0: tensor<32x32xf32, #blocked>) attributes {noinline = false} {
    %0 = triton_gpu.local_alloc : () -> !triton_gpu.memdesc<4x128x64xf16, #shared, mutable>
    %1 = triton_gpu.local_alloc : () -> !triton_gpu.memdesc<4x128x64xf16, #shared, mutable>
    %2 = triton_gpu.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    triton_gpu.local_dealloc %0 : !triton_gpu.memdesc<4x128x64xf16, #shared, mutable>
    triton_gpu.local_dealloc %1 : !triton_gpu.memdesc<4x128x64xf16, #shared, mutable>
    %3 = arith.addf %2, %2 : tensor<32x32xf32, #blocked1>
    tt.return
  }
}

// -----

//   CHECK-LABEL: anchor_barrier
//         CHECK: gpu.barrier
//         CHECK: tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @anchor_barrier(%arg0: tensor<32x32x!tt.ptr<f16>, #blocked>) attributes {noinline = false} {
    %0 = triton_gpu.local_alloc : () -> !triton_gpu.memdesc<4x128x64xf16, #shared, mutable>
    gpu.barrier
    %2 = tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %1 = triton_gpu.local_alloc %2 : (tensor<32x32xf16, #blocked>) -> !triton_gpu.memdesc<4x128x64xf16, #shared, mutable>
    triton_gpu.local_dealloc %0 : !triton_gpu.memdesc<4x128x64xf16, #shared, mutable>
    triton_gpu.local_dealloc %1 : !triton_gpu.memdesc<4x128x64xf16, #shared, mutable>
    tt.return
  }
}


// -----

#mfma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: dont_hoist_scf_ops
  // Make sure we don't hoist scf ops above its dependencies.
  tt.func public @dont_hoist_scf_ops(%init: tensor<256x128xf32, #mfma>,
    %base: tensor<256x128x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>,
    %p1: tensor<128x128x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>, %i1: i1) -> (tensor<256x128xf32, #mfma>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    // CHECK: scf.for
    %54 = scf.for %arg21 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg = %init) -> (tensor<256x128xf32, #mfma>)  : i32 {
      // CHECK: arith.addi
      %f = arith.addi %arg21, %c128_i32 : i32
      // CHECK: scf.if
      // CHECK: tt.load
      %p0 = scf.if %i1 -> tensor<256x128x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>{
        %t = tt.splat %f : i32 -> tensor<256x128xi32>
        %padd = tt.addptr %base, %t : tensor<256x128x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>, tensor<256x128xi32>
        scf.yield %padd : tensor<256x128x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      } else {
        scf.yield %base : tensor<256x128x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      }
      %l = tt.load %p0 : tensor<256x128x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %r = tt.load %p1 : tensor<128x128x!tt.ptr<f16>, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %acc = tt.dot %l, %r, %arg : tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      scf.yield %acc : tensor<256x128xf32, #mfma>
    }
    tt.return %54 : tensor<256x128xf32, #mfma>
  }
}
