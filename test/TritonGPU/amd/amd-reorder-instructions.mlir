// RUN: triton-opt %s -split-input-file -tritonamdgpu-reorder-instructions | FileCheck %s

// Check that we order load, local_alloc and local_load one after another. This is useful
// for making sure that Q tensor in FA is hoisted out of the main loop and kept in registers
// throughout the computation.
// CHECK-LABEL: order_load_alloc_local_load
//       CHECK:   %[[LOAD:.+]] = tt.load
//       CHECK:   %[[ALLOC:.+]] = triton_gpu.local_alloc
//       CHECK:   triton_gpu.local_store %[[LOAD]], %[[ALLOC]]
//       CHECK:   triton_gpu.local_load %[[ALLOC]]
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @order_load_alloc_local_load(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %9 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %10 = triton_gpu.local_alloc : () -> !tt.memdesc<32x32xf32, #shared, mutable>
    triton_gpu.local_store %9, %10 : tensor<32x32xf32, #blocked> -> !tt.memdesc<32x32xf32, #shared, mutable>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %11 = triton_gpu.local_load %10 : !tt.memdesc<32x32xf32, #shared, mutable> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
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
//        CHECK:       %[[ADDPTR_20:.*]] = tt.addptr %[[ARG7]], %{{.*}}
//        CHECK:       %[[SUBI_21:.*]] = arith.subi %{{.*}}, %{{.*}}
//        CHECK:       %[[CMPI_22:.*]] = arith.cmpi slt, %[[ARG5]], %[[SUBI_21]]
//        CHECK:       %[[SPLAT_23:.*]] = tt.splat %[[CMPI_22]]
//        CHECK:       %[[LOAD_24:.*]] = tt.load %[[ADDPTR_20]], %[[SPLAT_23]], %{{.*}}
//        CHECK:       %[[ADDPTR_25:.*]] = tt.addptr %[[ARG6]], %{{.*}}
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
//        CHECK:       triton_gpu.local_store %[[LOAD_27]], %[[MEMDESC_SUBVIEW_35]]
//        CHECK:       %[[MEMDESC_SUBVIEW_36:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_34]], %{{.*}}, %{{.*}}]
//        CHECK:       triton_gpu.local_store %[[LOAD_24]], %[[MEMDESC_SUBVIEW_36]]
//        CHECK:       scf.yield %[[ADDPTR_25]], %[[ADDPTR_20]], %[[DOT_31]], %[[SELECT_34]], %[[MEMDESC_SUBVIEW_35]], %[[MEMDESC_SUBVIEW_36]]
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
    %10 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %11 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %12 = arith.cmpi slt, %arg0, %arg1 : index
    %13 = tt.splat %12 : i1 -> tensor<128x32xi1, #blocked1>
    %14 = tt.load %4, %13 : tensor<128x32x!tt.ptr<f16>, #blocked1>
    %15 = tt.splat %12 : i1 -> tensor<32x128xi1, #blocked>
    %16 = tt.load %9, %15, %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked>
    %17 = triton_gpu.memdesc_subview %10[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %14, %17 : tensor<128x32xf16, #blocked1> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %18 = triton_gpu.memdesc_subview %11[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %16, %18 : tensor<32x128xf16, #blocked> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %19:6 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %4, %arg7 = %9, %arg8 = %cst_2, %arg9 = %c0_i32, %arg10 = %17, %arg11 = %18) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>) {
      %20 = arith.subi %arg1, %arg2 : index
      %21 = arith.cmpi slt, %arg5, %20 : index
      %22 = triton_gpu.local_load %arg10 : !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %23 = triton_gpu.local_load %arg11 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
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
      %35 = triton_gpu.memdesc_subview %10[%34, %c0_i32, %c0_i32] : !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %29, %35 : tensor<128x32xf16, #blocked1> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      %36 = triton_gpu.memdesc_subview %11[%34, %c0_i32, %c0_i32] : !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %31, %36 : tensor<32x128xf16, #blocked> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %26, %27, %25, %34, %35, %36 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    }
    triton_gpu.local_dealloc %10 : !tt.memdesc<1x128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %11 : !tt.memdesc<1x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    tt.return %19#2 : tensor<128x128xf32, #mma>
  }

// This example tests that tt.load overlaps with independent ttg.local_store which
// overlaps with independent tt.dot.
// num_stages == 3, double buffered

// CHECK-LABEL:  tt.func @matmul_loop_mb
// CHECK:  %{{.*}}:8 = scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}})
// Stage 1
//        CHECK:       %[[ADDI_28:.*]] = arith.addi %[[ARG9]], %{{.*}}
//        CHECK:       %[[CMPI_29:.*]] = arith.cmpi slt, %[[ADDI_28]], %{{.*}}
//        CHECK:       %[[SELECT_30:.*]] = arith.select %[[CMPI_29]], %[[ADDI_28]], %{{.*}}
//        CHECK:       %[[MEMDESC_SUBVIEW_31:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_30]], %{{.*}}, %{{.*}}]
//        CHECK:       triton_gpu.local_store %[[ARG13]], %[[MEMDESC_SUBVIEW_31]]
//        CHECK:       %[[MEMDESC_SUBVIEW_32:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_30]], %{{.*}}, %{{.*}}]
//        CHECK:       triton_gpu.local_store %[[ARG12]], %[[MEMDESC_SUBVIEW_32]]
// Stage 0
//        CHECK:       %[[ADDPTR_33:.*]] = tt.addptr %[[ARG7]], %{{.*}}
//        CHECK:       %[[MULI_34:.*]] = arith.muli %{{.*}}, %{{.*}}
//        CHECK:       %[[SUBI_35:.*]] = arith.subi %{{.*}}, %[[MULI_34]]
//        CHECK:       %[[CMPI_36:.*]] = arith.cmpi slt, %[[ARG5]], %[[SUBI_35]]
//        CHECK:       %[[SPLAT_37:.*]] = tt.splat %[[CMPI_36]]
//        CHECK:       %[[LOAD_38:.*]] = tt.load %[[ADDPTR_33]], %[[SPLAT_37]], %{{.*}}
//        CHECK:       %[[ADDPTR_39:.*]] = tt.addptr %[[ARG6]], %{{.*}}
//        CHECK:       %[[SPLAT_40:.*]] = tt.splat %[[CMPI_36]]
//        CHECK:       %[[LOAD_41:.*]] = tt.load %[[ADDPTR_39]], %[[SPLAT_40]]
// Stage 2
//        CHECK:       %[[LOCAL_LOAD_42:.*]] = triton_gpu.local_load %[[ARG10]]
//        CHECK:       %[[LOCAL_LOAD_43:.*]] = triton_gpu.local_load %[[ARG11]]
//        CHECK:       %[[MULF_44:.*]] = arith.mulf %[[LOCAL_LOAD_43]], %{{.*}}
//        CHECK:       %[[DOT_45:.*]] = tt.dot %[[LOCAL_LOAD_42]], %[[MULF_44]], %[[ARG8]]
//        CHECK:       scf.yield %[[ADDPTR_39]], %[[ADDPTR_33]], %[[DOT_45]], %[[SELECT_30]], %[[MEMDESC_SUBVIEW_32]], %[[MEMDESC_SUBVIEW_31]], %[[LOAD_41]], %[[LOAD_38]]
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
    %10 = triton_gpu.local_alloc  : () -> !tt.memdesc<2x128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %11 = triton_gpu.local_alloc  : () -> !tt.memdesc<2x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
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
    %25 = triton_gpu.memdesc_subview %10[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<2x128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %14, %25 : tensor<128x32xf16, #blocked1> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    %26 = triton_gpu.memdesc_subview %11[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<2x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %16, %26 : tensor<32x128xf16, #blocked> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %27:8 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %19, %arg7 = %20, %arg8 = %cst_2, %arg9 = %c0_i32, %arg10 = %25, %arg11 = %26, %arg12 = %22, %arg13 = %24) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>, tensor<128x32xf16, #blocked1>, tensor<32x128xf16, #blocked>) {
      %28 = arith.muli %arg2, %c2 : index
      %29 = arith.subi %arg1, %28 : index
      %30 = arith.cmpi slt, %arg5, %29 : index
      %31 = triton_gpu.local_load %arg10 : !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %32 = triton_gpu.local_load %arg11 : !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
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
      %44 = triton_gpu.memdesc_subview %10[%43, %c0_i32, %c0_i32] : !tt.memdesc<2x128x32xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %arg12, %44 : tensor<128x32xf16, #blocked1> -> !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
      %45 = triton_gpu.memdesc_subview %11[%43, %c0_i32, %c0_i32] : !tt.memdesc<2x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %arg13, %45 : tensor<32x128xf16, #blocked> -> !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
      scf.yield %35, %36, %34, %43, %44, %45, %38, %40 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory, mutable>, !tt.memdesc<32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>, tensor<128x32xf16, #blocked1>, tensor<32x128xf16, #blocked>
    }
    triton_gpu.local_dealloc %10 : !tt.memdesc<2x128x32xf16, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %11 : !tt.memdesc<2x32x128xf16, #shared1, #triton_gpu.shared_memory, mutable>
    tt.return %27#2 : tensor<128x128xf32, #mma>
  }

// This example shows dependent loads and verifies all are moved early.
// CHECK-LABEL:  tt.func @indirect_bmm_vector
// CHECK:  %{{.*}}:7 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}})
// Stage 0
//        CHECK:       %[[ADDPTR_20:.*]] = tt.addptr %[[ARG9]], %{{.*}}
//        CHECK:       %[[SUBI_21:.*]] = arith.subi %{{.*}}, %{{.*}}
//        CHECK:       %[[CMPI_22:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_21]]
//        CHECK:       %[[SPLAT_23:.*]] = tt.splat %[[CMPI_22]]
//        CHECK:       %[[LOAD_24:.*]] = tt.load %[[ADDPTR_20]], %[[SPLAT_23]]
// Stage 1.a
//        CHECK:       %[[EXPAND_DIMS_25:.*]] = tt.expand_dims %[[ARG13]] {axis = 1 : i32}
//        CHECK:       %[[BROADCAST_26:.*]] = tt.broadcast %[[EXPAND_DIMS_25]]
//        CHECK:       %[[MULI_27:.*]] = arith.muli %{{.*}}, %[[BROADCAST_26]]
//        CHECK:       %[[ADDPTR_28:.*]] = tt.addptr %{{.*}}, %[[MULI_27]]
//        CHECK:       %[[SUBI_29:.*]] = arith.subi %{{.*}}, %{{.*}}
//        CHECK:       %[[CMPI_30:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_29]]
//        CHECK:       %[[SPLAT_31:.*]] = tt.splat %[[CMPI_30]]
//        CHECK:       %[[LOAD_32:.*]] = tt.load %[[ADDPTR_28]], %[[SPLAT_31]]
//        CHECK:       %[[ADDPTR_33:.*]] = tt.addptr %[[ARG8]], %{{.*}}
//        CHECK:       %[[SPLAT_34:.*]] = tt.splat %[[CMPI_30]]
//        CHECK:       %[[LOAD_35:.*]] = tt.load %[[ADDPTR_33]], %[[SPLAT_34]]
// Stage 2
//        CHECK:       %[[LOCAL_LOAD_36:.*]] = triton_gpu.local_load %[[ARG11]]
//        CHECK:       %[[LOCAL_LOAD_37:.*]] = triton_gpu.local_load %[[ARG12]]
//        CHECK:       %[[DOT_38:.*]] = tt.dot %[[LOCAL_LOAD_36]], %[[LOCAL_LOAD_37]], %[[ARG7]]
// Stage 1.b
//        CHECK:       %[[ADDI_39:.*]] = arith.addi %[[ARG10]], %{{.*}}
//        CHECK:       %[[CMPI_40:.*]] = arith.cmpi slt, %[[ADDI_39]], %{{.*}}
//        CHECK:       %[[SELECT_41:.*]] = arith.select %[[CMPI_40]], %[[ADDI_39]], %{{.*}}
//        CHECK:       %[[MEMDESC_SUBVIEW_42:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_41]], %{{.*}}, %{{.*}}]
//        CHECK:       triton_gpu.local_store %[[LOAD_35]], %[[MEMDESC_SUBVIEW_42]]
//        CHECK:       %[[MEMDESC_SUBVIEW_43:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_41]], %{{.*}}, %{{.*}}]
//        CHECK:       triton_gpu.local_store %[[LOAD_32]], %[[MEMDESC_SUBVIEW_43]]
//        CHECK:       scf.yield %[[DOT_38]], %[[ADDPTR_33]], %[[ADDPTR_20]], %[[SELECT_41]], %[[MEMDESC_SUBVIEW_42]], %[[MEMDESC_SUBVIEW_43]], %[[LOAD_24]]
//        CHECK:   }

  tt.func @indirect_bmm_vector(%arg0: tensor<16x16xi64, #blocked> {tt.constancy = 16 : i32, tt.divisibility = 16 : i32}, %arg1: index, %arg2: tensor<16x16x!tt.ptr<f16>, #blocked1> {tt.contiguity = 2 : i32, tt.divisibility = 16 : i32}, %arg3: tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, %arg4: tensor<16x16xi32, #blocked1> {tt.constancy = 16 : i32, tt.divisibility = 16 : i32}, %arg5: tensor<16x16x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}) -> tensor<16x16xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<1> : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %0 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %1 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
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
    %17 = triton_gpu.memdesc_subview %0[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %8, %17 : tensor<16x16xf16, #blocked1> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %18 = triton_gpu.memdesc_subview %1[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %14, %18 : tensor<16x16xf16, #blocked> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    %19:7 = scf.for %arg6 = %c0 to %arg1 step %c1 iter_args(%arg7 = %cst, %arg8 = %arg2, %arg9 = %6, %arg10 = %c0_i32, %arg11 = %17, %arg12 = %18, %arg13 = %16) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, i32, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) {
      %20 = arith.subi %arg1, %c2 : index
      %21 = arith.cmpi slt, %arg6, %20 : index
      %22 = arith.subi %arg1, %c1 : index
      %23 = arith.cmpi slt, %arg6, %22 : index
      %24 = triton_gpu.local_load %arg11 : !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %25 = triton_gpu.local_load %arg12 : !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
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
      %42 = triton_gpu.memdesc_subview %0[%41, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %30, %42 : tensor<16x16xf16, #blocked1> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      %43 = triton_gpu.memdesc_subview %1[%41, %c0_i32, %c0_i32] : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %36, %43 : tensor<16x16xf16, #blocked> -> !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
      scf.yield %26, %27, %28, %41, %42, %43, %38 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, i32, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, !tt.memdesc<16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>, tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    }
    triton_gpu.local_dealloc %0 : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_dealloc %1 : !tt.memdesc<1x16x16xf16, #shared2, #triton_gpu.shared_memory, mutable>
    tt.return %19#0 : tensor<16x16xf32, #mma>
  }
}

// -----
// This test ensures that loads will not be moved across `for` loops.

// CHECK-LABEL:  tt.func public @_attn_bwd
// CHECK:  tt.load
// CHECK:  tt.load
// CHECK:  scf.for
// CHECK:  }
// CHECK:  scf.for
// CHECK:  }
// Moved before the independent `tt.store` ops but not before the `for` ops.
// CHECK:  tt.load
// CHECK:  tt.load
// CHECK:  tt.load
// CHECK:  tt.load
// CHECK:  tt.load
// CHECK:  tt.load
// CHECK:  tt.store
// CHECK:  tt.store
// CHECK:  scf.for
// CHECK:  }
// CHECK:  scf.for
// CHECK:  }
// CHECK:  tt.store

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#mma1 = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
#shared2 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1], hasLeadingOffset = false}>
#shared3 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @_attn_bwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg9: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #mma>
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %cst_2 = arith.constant dense<0.693147182> : tensor<128x64xf32, #mma>
    %0 = tt.get_program_id z : i32
    %1 = arith.muli %0, %arg14 : i32
    %2 = arith.extsi %1 : i32 to i64
    %3 = arith.remsi %0, %arg13 : i32
    %4 = arith.muli %arg11, %3 : i32
    %5 = arith.divsi %0, %arg13 : i32
    %6 = arith.muli %arg10, %5 : i32
    %7 = arith.addi %4, %6 : i32
    %8 = arith.extsi %7 : i32 to i64
    %9 = tt.get_program_id x : i32
    %10 = tt.addptr %arg0, %8 : !tt.ptr<f16>, i64
    %11 = tt.addptr %arg1, %8 : !tt.ptr<f16>, i64
    %12 = tt.addptr %arg2, %8 : !tt.ptr<f16>, i64
    %13 = tt.addptr %arg4, %8 : !tt.ptr<f16>, i64
    %14 = tt.addptr %arg5, %8 : !tt.ptr<f16>, i64
    %15 = tt.addptr %arg6, %8 : !tt.ptr<f16>, i64
    %16 = tt.addptr %arg7, %8 : !tt.ptr<f16>, i64
    %17 = tt.addptr %arg8, %2 : !tt.ptr<f32>, i64
    %18 = tt.addptr %arg9, %2 : !tt.ptr<f32>, i64
    %19 = arith.muli %9, %c128_i32 : i32
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %22 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
    %23 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
    %24 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %25 = tt.splat %19 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %26 = tt.splat %19 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %27 = tt.splat %19 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
    %28 = tt.splat %19 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
    %29 = tt.splat %19 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %30 = arith.addi %25, %20 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %31 = arith.addi %26, %21 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %32 = arith.addi %27, %22 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
    %33 = arith.addi %28, %23 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
    %34 = arith.addi %29, %24 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %35 = tt.expand_dims %30 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
    %36 = tt.expand_dims %31 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %37 = tt.expand_dims %32 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>> -> tensor<128x1xi32, #mma1>
    %38 = tt.splat %arg12 : i32 -> tensor<128x1xi32, #mma>
    %39 = tt.splat %arg12 : i32 -> tensor<128x1xi32, #blocked>
    %40 = arith.muli %35, %38 : tensor<128x1xi32, #mma>
    %41 = arith.muli %36, %39 : tensor<128x1xi32, #blocked>
    %42 = tt.splat %11 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %43 = tt.addptr %42, %41 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %44 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %45 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %46 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %47 = tt.expand_dims %44 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
    %48 = tt.expand_dims %45 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %49 = tt.expand_dims %46 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %50 = tt.broadcast %43 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %51 = tt.broadcast %47 : tensor<1x64xi32, #mma> -> tensor<128x64xi32, #mma>
    %52 = tt.broadcast %48 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %53 = tt.addptr %50, %52 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %54 = tt.load %53 : tensor<128x64x!tt.ptr<f16>, #blocked>
    %55 = tt.splat %12 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %56 = tt.addptr %55, %41 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %57 = tt.broadcast %56 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %58 = tt.addptr %57, %52 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %59 = tt.load %58 : tensor<128x64x!tt.ptr<f16>, #blocked>
    %60 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %61 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
    %62 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %63 = tt.splat %19 : i32 -> tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %64 = tt.splat %19 : i32 -> tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %65 = arith.addi %63, %60 : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %66 = arith.addi %64, %62 : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %67 = tt.expand_dims %65 {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x16xi32, #blocked2>
    %68 = tt.splat %arg12 : i32 -> tensor<1x16xi32, #blocked2>
    %69 = arith.muli %67, %68 : tensor<1x16xi32, #blocked2>
    %70 = tt.splat %10 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked2>
    %71 = tt.addptr %70, %69 : tensor<1x16x!tt.ptr<f16>, #blocked2>, tensor<1x16xi32, #blocked2>
    %72 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %73 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %74 = tt.expand_dims %72 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %75 = tt.expand_dims %73 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xi32, #blocked3>
    %76 = tt.broadcast %71 : tensor<1x16x!tt.ptr<f16>, #blocked2> -> tensor<64x16x!tt.ptr<f16>, #blocked2>
    %77 = tt.broadcast %74 : tensor<64x1xi32, #blocked2> -> tensor<64x16xi32, #blocked2>
    %78 = tt.addptr %76, %77 : tensor<64x16x!tt.ptr<f16>, #blocked2>, tensor<64x16xi32, #blocked2>
    %79 = tt.expand_dims %66 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xi32, #blocked1>
    %80 = tt.splat %arg12 : i32 -> tensor<16x1xi32, #blocked1>
    %81 = arith.muli %79, %80 : tensor<16x1xi32, #blocked1>
    %82 = tt.splat %13 : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>, #blocked1>
    %83 = tt.addptr %82, %81 : tensor<16x1x!tt.ptr<f16>, #blocked1>, tensor<16x1xi32, #blocked1>
    %84 = tt.broadcast %83 : tensor<16x1x!tt.ptr<f16>, #blocked1> -> tensor<16x64x!tt.ptr<f16>, #blocked1>
    %85 = tt.broadcast %49 : tensor<1x64xi32, #blocked1> -> tensor<16x64xi32, #blocked1>
    %86 = tt.addptr %84, %85 : tensor<16x64x!tt.ptr<f16>, #blocked1>, tensor<16x64xi32, #blocked1>
    %87 = tt.splat %17 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
    %88 = tt.broadcast %37 : tensor<128x1xi32, #mma1> -> tensor<128x16xi32, #mma1>
    %89 = tt.splat %18 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
    %90 = arith.muli %arg12, %c16_i32 : i32
    %91 = tt.splat %90 : i32 -> tensor<64x16xi32, #blocked2>
    %92 = tt.splat %90 : i32 -> tensor<16x64xi32, #blocked1>
    %93:5 = scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg16 = %cst_1, %arg17 = %cst_1, %arg18 = %19, %arg19 = %78, %arg20 = %86) -> (tensor<128x64xf32, #mma>, tensor<128x64xf32, #mma>, i32, tensor<64x16x!tt.ptr<f16>, #blocked2>, tensor<16x64x!tt.ptr<f16>, #blocked1>)  : i32 {
      %206 = tt.load %arg19 : tensor<64x16x!tt.ptr<f16>, #blocked2>
      %207 = tt.splat %arg18 : i32 -> tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
      %208 = arith.addi %207, %61 : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
      %209 = tt.addptr %87, %208 : tensor<16x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma1}>>, tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
      %210 = tt.load %209 : tensor<16x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
      %211 = triton_gpu.local_alloc %54 : (tensor<128x64xf16, #blocked>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
      %212 = triton_gpu.local_load %211 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>>
      %213 = triton_gpu.local_alloc %206 : (tensor<64x16xf16, #blocked2>) -> !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>
      %214 = triton_gpu.local_load %213 : !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>>
      %215 = tt.dot %212, %214, %cst_0 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>> -> tensor<128x16xf32, #mma1>
      %216 = tt.expand_dims %210 {axis = 0 : i32} : tensor<16xf32, #triton_gpu.slice<{dim = 0, parent = #mma1}>> -> tensor<1x16xf32, #mma1>
      %217 = tt.broadcast %216 : tensor<1x16xf32, #mma1> -> tensor<128x16xf32, #mma1>
      %218 = arith.subf %215, %217 : tensor<128x16xf32, #mma1>
      %219 = math.exp2 %218 : tensor<128x16xf32, #mma1>
      %220 = tt.expand_dims %208 {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>> -> tensor<1x16xi32, #mma1>
      %221 = tt.broadcast %220 : tensor<1x16xi32, #mma1> -> tensor<128x16xi32, #mma1>
      %222 = arith.cmpi sge, %221, %88 : tensor<128x16xi32, #mma1>
      %223 = arith.select %222, %219, %cst_0 : tensor<128x16xi1, #mma1>, tensor<128x16xf32, #mma1>
      %224 = tt.load %arg20 : tensor<16x64x!tt.ptr<f16>, #blocked1>
      %225 = arith.truncf %223 : tensor<128x16xf32, #mma1> to tensor<128x16xf16, #mma1>
      %226 = triton_gpu.local_alloc %225 : (tensor<128x16xf16, #mma1>) -> !tt.memdesc<128x16xf16, #shared2, #triton_gpu.shared_memory>
      %227 = triton_gpu.local_load %226 : !tt.memdesc<128x16xf16, #shared2, #triton_gpu.shared_memory> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %228 = triton_gpu.local_alloc %224 : (tensor<16x64xf16, #blocked1>) -> !tt.memdesc<16x64xf16, #shared3, #triton_gpu.shared_memory>
      %229 = triton_gpu.local_load %228 : !tt.memdesc<16x64xf16, #shared3, #triton_gpu.shared_memory> -> tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %230 = tt.dot %227, %229, %arg16 : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma>
      %231 = tt.addptr %89, %208 : tensor<16x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma1}>>, tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
      %232 = tt.load %231 : tensor<16x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
      %233 = triton_gpu.local_alloc %224 : (tensor<16x64xf16, #blocked1>) -> !tt.memdesc<16x64xf16, #shared, #triton_gpu.shared_memory>
      %234 = tt.trans %233 {order = array<i32: 1, 0>} : !tt.memdesc<16x64xf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>
      %235 = triton_gpu.local_load %234 : !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>>
      %236 = triton_gpu.local_alloc %59 : (tensor<128x64xf16, #blocked>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
      %237 = triton_gpu.local_load %236 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>>
      %238 = tt.dot %237, %235, %cst_0 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>> -> tensor<128x16xf32, #mma1>
      %239 = tt.expand_dims %232 {axis = 0 : i32} : tensor<16xf32, #triton_gpu.slice<{dim = 0, parent = #mma1}>> -> tensor<1x16xf32, #mma1>
      %240 = tt.broadcast %239 : tensor<1x16xf32, #mma1> -> tensor<128x16xf32, #mma1>
      %241 = arith.subf %238, %240 : tensor<128x16xf32, #mma1>
      %242 = arith.mulf %223, %241 : tensor<128x16xf32, #mma1>
      %243 = arith.truncf %242 : tensor<128x16xf32, #mma1> to tensor<128x16xf16, #mma1>
      %244 = triton_gpu.local_alloc %206 : (tensor<64x16xf16, #blocked2>) -> !tt.memdesc<64x16xf16, #shared2, #triton_gpu.shared_memory>
      %245 = tt.trans %244 {order = array<i32: 1, 0>} : !tt.memdesc<64x16xf16, #shared2, #triton_gpu.shared_memory> -> !tt.memdesc<16x64xf16, #shared3, #triton_gpu.shared_memory>
      %246 = triton_gpu.local_load %245 : !tt.memdesc<16x64xf16, #shared3, #triton_gpu.shared_memory> -> tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %247 = triton_gpu.local_alloc %243 : (tensor<128x16xf16, #mma1>) -> !tt.memdesc<128x16xf16, #shared2, #triton_gpu.shared_memory>
      %248 = triton_gpu.local_load %247 : !tt.memdesc<128x16xf16, #shared2, #triton_gpu.shared_memory> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %249 = tt.dot %248, %246, %arg17 : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma>
      %250 = arith.addi %arg18, %c16_i32 : i32
      %251 = tt.addptr %arg19, %91 : tensor<64x16x!tt.ptr<f16>, #blocked2>, tensor<64x16xi32, #blocked2>
      %252 = tt.addptr %arg20, %92 : tensor<16x64x!tt.ptr<f16>, #blocked1>, tensor<16x64xi32, #blocked1>
      scf.yield %230, %249, %250, %251, %252 : tensor<128x64xf32, #mma>, tensor<128x64xf32, #mma>, i32, tensor<64x16x!tt.ptr<f16>, #blocked2>, tensor<16x64x!tt.ptr<f16>, #blocked1>
    }
    %94 = arith.addi %19, %c128_i32 : i32
    %95 = arith.subi %arg14, %94 : i32
    %96 = arith.divsi %95, %c32_i32 : i32
    %97 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %98 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %99 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %100 = tt.splat %94 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %101 = tt.splat %94 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %102 = arith.addi %100, %97 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %103 = arith.addi %101, %99 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %104 = tt.expand_dims %102 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x32xi32, #blocked3>
    %105 = tt.splat %arg12 : i32 -> tensor<1x32xi32, #blocked3>
    %106 = arith.muli %104, %105 : tensor<1x32xi32, #blocked3>
    %107 = tt.splat %10 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #blocked3>
    %108 = tt.addptr %107, %106 : tensor<1x32x!tt.ptr<f16>, #blocked3>, tensor<1x32xi32, #blocked3>
    %109 = tt.broadcast %108 : tensor<1x32x!tt.ptr<f16>, #blocked3> -> tensor<64x32x!tt.ptr<f16>, #blocked3>
    %110 = tt.broadcast %75 : tensor<64x1xi32, #blocked3> -> tensor<64x32xi32, #blocked3>
    %111 = tt.addptr %109, %110 : tensor<64x32x!tt.ptr<f16>, #blocked3>, tensor<64x32xi32, #blocked3>
    %112 = tt.expand_dims %103 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %113 = tt.splat %arg12 : i32 -> tensor<32x1xi32, #blocked>
    %114 = arith.muli %112, %113 : tensor<32x1xi32, #blocked>
    %115 = tt.splat %13 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %116 = tt.addptr %115, %114 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %117 = tt.broadcast %116 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %118 = tt.broadcast %48 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %119 = tt.addptr %117, %118 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
    %120 = tt.splat %17 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %121 = tt.splat %18 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %122 = arith.muli %arg12, %c32_i32 : i32
    %123 = tt.splat %122 : i32 -> tensor<64x32xi32, #blocked3>
    %124 = tt.splat %122 : i32 -> tensor<32x64xi32, #blocked>
    %125:5 = scf.for %arg15 = %c0_i32 to %96 step %c1_i32 iter_args(%arg16 = %93#0, %arg17 = %93#1, %arg18 = %94, %arg19 = %111, %arg20 = %119) -> (tensor<128x64xf32, #mma>, tensor<128x64xf32, #mma>, i32, tensor<64x32x!tt.ptr<f16>, #blocked3>, tensor<32x64x!tt.ptr<f16>, #blocked>)  : i32 {
      %206 = tt.load %arg19 : tensor<64x32x!tt.ptr<f16>, #blocked3>
      %207 = tt.splat %arg18 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
      %208 = arith.addi %207, %98 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
      %209 = tt.addptr %120, %208 : tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
      %210 = tt.load %209 : tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma}>>
      %211 = triton_gpu.local_alloc %54 : (tensor<128x64xf16, #blocked>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
      %212 = triton_gpu.local_load %211 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %213 = triton_gpu.local_alloc %206 : (tensor<64x32xf16, #blocked3>) -> !tt.memdesc<64x32xf16, #shared1, #triton_gpu.shared_memory>
      %214 = triton_gpu.local_load %213 : !tt.memdesc<64x32xf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %215 = tt.dot %212, %214, %cst : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
      %216 = tt.expand_dims %210 {axis = 0 : i32} : tensor<32xf32, #triton_gpu.slice<{dim = 0, parent = #mma}>> -> tensor<1x32xf32, #mma>
      %217 = tt.broadcast %216 : tensor<1x32xf32, #mma> -> tensor<128x32xf32, #mma>
      %218 = arith.subf %215, %217 : tensor<128x32xf32, #mma>
      %219 = math.exp2 %218 : tensor<128x32xf32, #mma>
      %220 = tt.load %arg20 : tensor<32x64x!tt.ptr<f16>, #blocked>
      %221 = arith.truncf %219 : tensor<128x32xf32, #mma> to tensor<128x32xf16, #mma>
      %222 = triton_gpu.convert_layout %221 : tensor<128x32xf16, #mma> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %223 = triton_gpu.local_alloc %220 : (tensor<32x64xf16, #blocked>) -> !tt.memdesc<32x64xf16, #shared3, #triton_gpu.shared_memory>
      %224 = triton_gpu.local_load %223 : !tt.memdesc<32x64xf16, #shared3, #triton_gpu.shared_memory> -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %225 = tt.dot %222, %224, %arg16 : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma>
      %226 = tt.addptr %121, %208 : tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
      %227 = tt.load %226 : tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma}>>
      %228 = triton_gpu.local_alloc %220 : (tensor<32x64xf16, #blocked>) -> !tt.memdesc<32x64xf16, #shared, #triton_gpu.shared_memory>
      %229 = tt.trans %228 {order = array<i32: 1, 0>} : !tt.memdesc<32x64xf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<64x32xf16, #shared1, #triton_gpu.shared_memory>
      %230 = triton_gpu.local_load %229 : !tt.memdesc<64x32xf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %231 = triton_gpu.local_alloc %59 : (tensor<128x64xf16, #blocked>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
      %232 = triton_gpu.local_load %231 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %233 = tt.dot %232, %230, %cst : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
      %234 = tt.expand_dims %227 {axis = 0 : i32} : tensor<32xf32, #triton_gpu.slice<{dim = 0, parent = #mma}>> -> tensor<1x32xf32, #mma>
      %235 = tt.broadcast %234 : tensor<1x32xf32, #mma> -> tensor<128x32xf32, #mma>
      %236 = arith.subf %233, %235 : tensor<128x32xf32, #mma>
      %237 = arith.mulf %219, %236 : tensor<128x32xf32, #mma>
      %238 = arith.truncf %237 : tensor<128x32xf32, #mma> to tensor<128x32xf16, #mma>
      %239 = triton_gpu.local_alloc %206 : (tensor<64x32xf16, #blocked3>) -> !tt.memdesc<64x32xf16, #shared2, #triton_gpu.shared_memory>
      %240 = tt.trans %239 {order = array<i32: 1, 0>} : !tt.memdesc<64x32xf16, #shared2, #triton_gpu.shared_memory> -> !tt.memdesc<32x64xf16, #shared3, #triton_gpu.shared_memory>
      %241 = triton_gpu.local_load %240 : !tt.memdesc<32x64xf16, #shared3, #triton_gpu.shared_memory> -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %242 = triton_gpu.convert_layout %238 : tensor<128x32xf16, #mma> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %243 = tt.dot %242, %241, %arg17 : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma>
      %244 = arith.addi %arg18, %c32_i32 : i32
      %245 = tt.addptr %arg19, %123 : tensor<64x32x!tt.ptr<f16>, #blocked3>, tensor<64x32xi32, #blocked3>
      %246 = tt.addptr %arg20, %124 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
      scf.yield %225, %243, %244, %245, %246 : tensor<128x64xf32, #mma>, tensor<128x64xf32, #mma>, i32, tensor<64x32x!tt.ptr<f16>, #blocked3>, tensor<32x64x!tt.ptr<f16>, #blocked>
    }
    %126 = tt.splat %16 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #mma>
    %127 = tt.addptr %126, %40 : tensor<128x1x!tt.ptr<f16>, #mma>, tensor<128x1xi32, #mma>
    %128 = tt.broadcast %127 : tensor<128x1x!tt.ptr<f16>, #mma> -> tensor<128x64x!tt.ptr<f16>, #mma>
    %129 = tt.addptr %128, %51 : tensor<128x64x!tt.ptr<f16>, #mma>, tensor<128x64xi32, #mma>
    %130 = arith.truncf %125#0 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
    tt.store %129, %130 : tensor<128x64x!tt.ptr<f16>, #mma>
    %131 = tt.splat %arg3 : f32 -> tensor<128x64xf32, #mma>
    %132 = arith.mulf %125#1, %131 : tensor<128x64xf32, #mma>
    %133 = tt.splat %15 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #mma>
    %134 = tt.addptr %133, %40 : tensor<128x1x!tt.ptr<f16>, #mma>, tensor<128x1xi32, #mma>
    %135 = tt.broadcast %134 : tensor<128x1x!tt.ptr<f16>, #mma> -> tensor<128x64x!tt.ptr<f16>, #mma>
    %136 = tt.addptr %135, %51 : tensor<128x64x!tt.ptr<f16>, #mma>, tensor<128x64xi32, #mma>
    %137 = arith.truncf %132 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
    tt.store %136, %137 : tensor<128x64x!tt.ptr<f16>, #mma>
    %138 = tt.splat %10 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %139 = tt.addptr %138, %41 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %140 = tt.broadcast %139 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %141 = tt.addptr %140, %52 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %142 = tt.load %141 : tensor<128x64x!tt.ptr<f16>, #blocked>
    %143 = tt.splat %13 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %144 = tt.addptr %143, %41 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %145 = tt.broadcast %144 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %146 = tt.addptr %145, %52 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %147 = tt.load %146 : tensor<128x64x!tt.ptr<f16>, #blocked>
    %148 = tt.splat %17 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
    %149 = tt.splat %17 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %150 = tt.addptr %148, %33 : tensor<128x!tt.ptr<f32>, #triton_gpu.slice<{dim = 1, parent = #mma1}>>, tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
    %151 = tt.addptr %149, %34 : tensor<128x!tt.ptr<f32>, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %152 = tt.load %150 : tensor<128x!tt.ptr<f32>, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
    %153 = tt.load %151 : tensor<128x!tt.ptr<f32>, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %154 = tt.expand_dims %152 {axis = 1 : i32} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma1}>> -> tensor<128x1xf32, #mma1>
    %155 = tt.expand_dims %153 {axis = 1 : i32} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %156 = tt.splat %11 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked2>
    %157 = tt.addptr %156, %69 : tensor<1x16x!tt.ptr<f16>, #blocked2>, tensor<1x16xi32, #blocked2>
    %158 = tt.broadcast %157 : tensor<1x16x!tt.ptr<f16>, #blocked2> -> tensor<64x16x!tt.ptr<f16>, #blocked2>
    %159 = tt.addptr %158, %77 : tensor<64x16x!tt.ptr<f16>, #blocked2>, tensor<64x16xi32, #blocked2>
    %160 = tt.splat %12 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked2>
    %161 = tt.addptr %160, %69 : tensor<1x16x!tt.ptr<f16>, #blocked2>, tensor<1x16xi32, #blocked2>
    %162 = tt.broadcast %161 : tensor<1x16x!tt.ptr<f16>, #blocked2> -> tensor<64x16x!tt.ptr<f16>, #blocked2>
    %163 = tt.addptr %162, %77 : tensor<64x16x!tt.ptr<f16>, #blocked2>, tensor<64x16xi32, #blocked2>
    %164 = tt.splat %18 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
    %165 = tt.splat %18 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %166 = tt.addptr %164, %33 : tensor<128x!tt.ptr<f32>, #triton_gpu.slice<{dim = 1, parent = #mma1}>>, tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
    %167 = tt.addptr %165, %34 : tensor<128x!tt.ptr<f32>, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %168 = tt.load %166 : tensor<128x!tt.ptr<f32>, #triton_gpu.slice<{dim = 1, parent = #mma1}>>
    %169 = tt.load %167 : tensor<128x!tt.ptr<f32>, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %170 = tt.broadcast %154 : tensor<128x1xf32, #mma1> -> tensor<128x16xf32, #mma1>
    %171 = tt.broadcast %37 : tensor<128x1xi32, #mma1> -> tensor<128x16xi32, #mma1>
    %172 = tt.expand_dims %168 {axis = 1 : i32} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma1}>> -> tensor<128x1xf32, #mma1>
    %173 = tt.broadcast %172 : tensor<128x1xf32, #mma1> -> tensor<128x16xf32, #mma1>
    %174 = arith.muli %arg12, %c16_i32 : i32
    %175 = tt.splat %174 : i32 -> tensor<64x16xi32, #blocked2>
    %176 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x64x16xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %177:5 = scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg16 = %cst_1, %arg17 = %19, %arg18 = %159, %arg19 = %163, %arg20 = %c-1_i32) -> (tensor<128x64xf32, #mma>, i32, tensor<64x16x!tt.ptr<f16>, #blocked2>, tensor<64x16x!tt.ptr<f16>, #blocked2>, i32)  : i32 {
      %206 = arith.addi %arg20, %c1_i32 : i32
      %207 = arith.cmpi slt, %206, %c1_i32 : i32
      %208 = arith.select %207, %206, %c0_i32 : i32
      %209 = tt.load %arg18 : tensor<64x16x!tt.ptr<f16>, #blocked2>
      %210 = tt.load %arg19 : tensor<64x16x!tt.ptr<f16>, #blocked2>
      %211 = triton_gpu.memdesc_subview %176[%208, %c0_i32, %c0_i32] : !tt.memdesc<1x64x16xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %210, %211 : tensor<64x16xf16, #blocked2> -> !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory, mutable>
      %212 = triton_gpu.local_load %211 : !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>>
      %213 = triton_gpu.local_alloc %142 : (tensor<128x64xf16, #blocked>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
      %214 = triton_gpu.local_load %213 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>>
      %215 = triton_gpu.local_alloc %209 : (tensor<64x16xf16, #blocked2>) -> !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>
      %216 = triton_gpu.local_load %215 : !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>>
      %217 = tt.dot %214, %216, %cst_0 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>> -> tensor<128x16xf32, #mma1>
      %218 = arith.subf %217, %170 : tensor<128x16xf32, #mma1>
      %219 = math.exp2 %218 : tensor<128x16xf32, #mma1>
      %220 = tt.splat %arg17 : i32 -> tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
      %221 = arith.addi %220, %61 : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>>
      %222 = tt.expand_dims %221 {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>> -> tensor<1x16xi32, #mma1>
      %223 = tt.broadcast %222 : tensor<1x16xi32, #mma1> -> tensor<128x16xi32, #mma1>
      %224 = arith.cmpi sge, %171, %223 : tensor<128x16xi32, #mma1>
      %225 = arith.select %224, %219, %cst_0 : tensor<128x16xi1, #mma1>, tensor<128x16xf32, #mma1>
      %226 = triton_gpu.local_alloc %147 : (tensor<128x64xf16, #blocked>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
      %227 = triton_gpu.local_load %226 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>>
      %228 = tt.dot %227, %212, %cst_0 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1, kWidth = 4}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma1, kWidth = 4}>> -> tensor<128x16xf32, #mma1>
      %229 = arith.subf %228, %173 : tensor<128x16xf32, #mma1>
      %230 = arith.mulf %225, %229 : tensor<128x16xf32, #mma1>
      %231 = arith.truncf %230 : tensor<128x16xf32, #mma1> to tensor<128x16xf16, #mma1>
      %232 = triton_gpu.local_alloc %209 : (tensor<64x16xf16, #blocked2>) -> !tt.memdesc<64x16xf16, #shared2, #triton_gpu.shared_memory>
      %233 = tt.trans %232 {order = array<i32: 1, 0>} : !tt.memdesc<64x16xf16, #shared2, #triton_gpu.shared_memory> -> !tt.memdesc<16x64xf16, #shared3, #triton_gpu.shared_memory>
      %234 = triton_gpu.local_load %233 : !tt.memdesc<16x64xf16, #shared3, #triton_gpu.shared_memory> -> tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %235 = triton_gpu.local_alloc %231 : (tensor<128x16xf16, #mma1>) -> !tt.memdesc<128x16xf16, #shared2, #triton_gpu.shared_memory>
      %236 = triton_gpu.local_load %235 : !tt.memdesc<128x16xf16, #shared2, #triton_gpu.shared_memory> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %237 = tt.dot %236, %234, %arg16 : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma>
      %238 = arith.addi %arg17, %c16_i32 : i32
      %239 = tt.addptr %arg18, %175 : tensor<64x16x!tt.ptr<f16>, #blocked2>, tensor<64x16xi32, #blocked2>
      %240 = tt.addptr %arg19, %175 : tensor<64x16x!tt.ptr<f16>, #blocked2>, tensor<64x16xi32, #blocked2>
      scf.yield %237, %238, %239, %240, %208 : tensor<128x64xf32, #mma>, i32, tensor<64x16x!tt.ptr<f16>, #blocked2>, tensor<64x16x!tt.ptr<f16>, #blocked2>, i32
    }
    triton_gpu.local_dealloc %176 : !tt.memdesc<1x64x16xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %178 = arith.divsi %19, %c32_i32 : i32
    %179 = arith.muli %178, %c32_i32 : i32
    %180 = arith.subi %19, %179 : i32
    %181 = tt.splat %180 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %182 = arith.addi %181, %97 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %183 = tt.expand_dims %182 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x32xi32, #blocked3>
    %184 = arith.muli %183, %105 : tensor<1x32xi32, #blocked3>
    %185 = tt.splat %11 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #blocked3>
    %186 = tt.addptr %185, %184 : tensor<1x32x!tt.ptr<f16>, #blocked3>, tensor<1x32xi32, #blocked3>
    %187 = tt.broadcast %186 : tensor<1x32x!tt.ptr<f16>, #blocked3> -> tensor<64x32x!tt.ptr<f16>, #blocked3>
    %188 = tt.addptr %187, %110 : tensor<64x32x!tt.ptr<f16>, #blocked3>, tensor<64x32xi32, #blocked3>
    %189 = tt.splat %12 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #blocked3>
    %190 = tt.addptr %189, %184 : tensor<1x32x!tt.ptr<f16>, #blocked3>, tensor<1x32xi32, #blocked3>
    %191 = tt.broadcast %190 : tensor<1x32x!tt.ptr<f16>, #blocked3> -> tensor<64x32x!tt.ptr<f16>, #blocked3>
    %192 = tt.addptr %191, %110 : tensor<64x32x!tt.ptr<f16>, #blocked3>, tensor<64x32xi32, #blocked3>
    %193 = tt.broadcast %155 : tensor<128x1xf32, #mma> -> tensor<128x32xf32, #mma>
    %194 = tt.expand_dims %169 {axis = 1 : i32} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %195 = tt.broadcast %194 : tensor<128x1xf32, #mma> -> tensor<128x32xf32, #mma>
    %196 = arith.muli %arg12, %c32_i32 : i32
    %197 = tt.splat %196 : i32 -> tensor<64x32xi32, #blocked3>
    %198 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x64x32xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %199:4 = scf.for %arg15 = %c0_i32 to %178 step %c1_i32 iter_args(%arg16 = %177#0, %arg17 = %188, %arg18 = %192, %arg19 = %c-1_i32) -> (tensor<128x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked3>, tensor<64x32x!tt.ptr<f16>, #blocked3>, i32)  : i32 {
      %206 = arith.addi %arg19, %c1_i32 : i32
      %207 = arith.cmpi slt, %206, %c1_i32 : i32
      %208 = arith.select %207, %206, %c0_i32 : i32
      %209 = tt.load %arg17 : tensor<64x32x!tt.ptr<f16>, #blocked3>
      %210 = tt.load %arg18 : tensor<64x32x!tt.ptr<f16>, #blocked3>
      %211 = triton_gpu.memdesc_subview %198[%208, %c0_i32, %c0_i32] : !tt.memdesc<1x64x32xf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x32xf16, #shared1, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %210, %211 : tensor<64x32xf16, #blocked3> -> !tt.memdesc<64x32xf16, #shared1, #triton_gpu.shared_memory, mutable>
      %212 = triton_gpu.local_load %211 : !tt.memdesc<64x32xf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %213 = triton_gpu.local_alloc %142 : (tensor<128x64xf16, #blocked>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
      %214 = triton_gpu.local_load %213 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %215 = triton_gpu.local_alloc %209 : (tensor<64x32xf16, #blocked3>) -> !tt.memdesc<64x32xf16, #shared1, #triton_gpu.shared_memory>
      %216 = triton_gpu.local_load %215 : !tt.memdesc<64x32xf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %217 = tt.dot %214, %216, %cst : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
      %218 = arith.subf %217, %193 : tensor<128x32xf32, #mma>
      %219 = math.exp2 %218 : tensor<128x32xf32, #mma>
      %220 = triton_gpu.local_alloc %147 : (tensor<128x64xf16, #blocked>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
      %221 = triton_gpu.local_load %220 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %222 = tt.dot %221, %212, %cst : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x32xf32, #mma>
      %223 = arith.subf %222, %195 : tensor<128x32xf32, #mma>
      %224 = arith.mulf %219, %223 : tensor<128x32xf32, #mma>
      %225 = arith.truncf %224 : tensor<128x32xf32, #mma> to tensor<128x32xf16, #mma>
      %226 = triton_gpu.local_alloc %209 : (tensor<64x32xf16, #blocked3>) -> !tt.memdesc<64x32xf16, #shared2, #triton_gpu.shared_memory>
      %227 = tt.trans %226 {order = array<i32: 1, 0>} : !tt.memdesc<64x32xf16, #shared2, #triton_gpu.shared_memory> -> !tt.memdesc<32x64xf16, #shared3, #triton_gpu.shared_memory>
      %228 = triton_gpu.local_load %227 : !tt.memdesc<32x64xf16, #shared3, #triton_gpu.shared_memory> -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %229 = triton_gpu.convert_layout %225 : tensor<128x32xf16, #mma> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %230 = tt.dot %229, %228, %arg16 : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x64xf32, #mma>
      %231 = tt.addptr %arg17, %197 : tensor<64x32x!tt.ptr<f16>, #blocked3>, tensor<64x32xi32, #blocked3>
      %232 = tt.addptr %arg18, %197 : tensor<64x32x!tt.ptr<f16>, #blocked3>, tensor<64x32xi32, #blocked3>
      scf.yield %230, %231, %232, %208 : tensor<128x64xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked3>, tensor<64x32x!tt.ptr<f16>, #blocked3>, i32
    }
    triton_gpu.local_dealloc %198 : !tt.memdesc<1x64x32xf16, #shared1, #triton_gpu.shared_memory, mutable>
    %200 = tt.splat %14 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #mma>
    %201 = tt.addptr %200, %40 : tensor<128x1x!tt.ptr<f16>, #mma>, tensor<128x1xi32, #mma>
    %202 = tt.broadcast %201 : tensor<128x1x!tt.ptr<f16>, #mma> -> tensor<128x64x!tt.ptr<f16>, #mma>
    %203 = tt.addptr %202, %51 : tensor<128x64x!tt.ptr<f16>, #mma>, tensor<128x64xi32, #mma>
    %204 = arith.mulf %199#0, %cst_2 : tensor<128x64xf32, #mma>
    %205 = arith.truncf %204 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
    tt.store %203, %205 : tensor<128x64x!tt.ptr<f16>, #mma>
    tt.return
  }
}
