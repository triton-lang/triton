// RUN: triton-opt %s -split-input-file -tritonamdgpu-reorder-instructions | FileCheck %s

// Check that we order load, local_alloc and local_load one after another. This is useful
// for making sure that Q tensor in FA is hoisted out of the main loop and kept in registers
// throughout the computation.
// CHECK-LABEL: order_load_alloc_local_load
//       CHECK:   %[[LOAD:.+]] = tt.load
//       CHECK-NEXT:   %[[ALLOC:.+]] = triton_gpu.local_alloc %[[LOAD]]
//       CHECK-NEXT:   triton_gpu.local_load %[[ALLOC]]
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @order_load_alloc_local_load(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %9 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %10 = triton_gpu.local_alloc %9 : (tensor<32x32xf32, #blocked>) -> !tt.memdesc<32x32xf32, #shared>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %11 = triton_gpu.local_load %10 : !tt.memdesc<32x32xf32, #shared> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
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
//     // stage 0
//     %aptr = tt.addptr %aptr, %k
//     %a_next_2 = tt.load %aptr
//     %bptr = tt.addptr %bptr, %k
//     %b_next_2 = tt.load %bptr
//     // stage 2
//     tt.local_store %a_next_1
//     tt.local_store %b_next_1
//     // stage 1
//     %a = tt.local_load %a_tile
//     %b = tt.local_load %b_tile
//     tt.dot %c, %a, %b
//     yield
//   }

//    - matmul
//    - num_stages=3
//    - dependent load
//    - atomic sync
//    - loop after loop
//    - fa?

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
// CHECK:  %[[SUBI_20:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[ADDPTR_21:.*]] = tt.addptr %[[ARG7]], %{{.*}}
// CHECK:  %[[CMPI_22:.*]] = arith.cmpi slt, %[[ARG5]], %[[SUBI_20]]
// CHECK:  %[[SPLAT_23:.*]] = tt.splat %[[CMPI_22]]
// CHECK:  %[[LOAD_24:.*]] = tt.load %[[ADDPTR_21]], %[[SPLAT_23]], %{{.*}}
// CHECK:  %[[ADDPTR_25:.*]] = tt.addptr %[[ARG6]], %{{.*}}
// CHECK:  %[[SPLAT_26:.*]] = tt.splat %[[CMPI_22]]
// CHECK:  %[[LOAD_27:.*]] = tt.load %[[ADDPTR_25]], %[[SPLAT_26]]
// CHECK:  %[[LOCAL_LOAD_28:.*]] = triton_gpu.local_load %[[ARG10]]
// CHECK:  %[[LOCAL_LOAD_29:.*]] = triton_gpu.local_load %[[ARG11]]
// CHECK:  %[[MULF_30:.*]] = arith.mulf %[[LOCAL_LOAD_29]], %{{.*}}
// CHECK:  %[[DOT_31:.*]] = tt.dot %[[LOCAL_LOAD_28]], %[[MULF_30]], %[[ARG8]]
// CHECK:  %[[ADDI_32:.*]] = arith.addi %[[ARG9]], %{{.*}}
// CHECK:  %[[CMPI_33:.*]] = arith.cmpi slt, %[[ADDI_32]], %{{.*}}
// CHECK:  %[[SELECT_34:.*]] = arith.select %[[CMPI_33]], %[[ADDI_32]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_35:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_34]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_27]], %[[MEMDESC_SUBVIEW_35]]
// CHECK:  %[[MEMDESC_SUBVIEW_36:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_34]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_24]], %[[MEMDESC_SUBVIEW_36]]
// CHECK:  scf.yield %[[ADDPTR_25]], %[[ADDPTR_21]], %[[DOT_31]], %[[SELECT_34]], %[[MEMDESC_SUBVIEW_35]], %[[MEMDESC_SUBVIEW_36]]
// CHECK:  }

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
// CHECK:  %[[MULI_28:.*]] = arith.muli %{{.*}}, %{{.*}}
// CHECK:  %[[ADDI_29:.*]] = arith.addi %[[ARG9]], %{{.*}}
// CHECK:  %[[CMPI_30:.*]] = arith.cmpi slt, %[[ADDI_29]], %{{.*}}
// CHECK:  %[[SELECT_31:.*]] = arith.select %[[CMPI_30]], %[[ADDI_29]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_32:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_31]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[ARG13]], %[[MEMDESC_SUBVIEW_32]]
// CHECK:  %[[MEMDESC_SUBVIEW_33:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_31]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[ARG12]], %[[MEMDESC_SUBVIEW_33]]
// CHECK:  %[[ADDPTR_34:.*]] = tt.addptr %[[ARG7]], %{{.*}}
// CHECK:  %[[SUBI_35:.*]] = arith.subi %{{.*}}, %[[MULI_28]]
// CHECK:  %[[CMPI_36:.*]] = arith.cmpi slt, %[[ARG5]], %[[SUBI_35]]
// CHECK:  %[[SPLAT_37:.*]] = tt.splat %[[CMPI_36]]
// CHECK:  %[[LOAD_38:.*]] = tt.load %[[ADDPTR_34]], %[[SPLAT_37]], %{{.*}}
// CHECK:  %[[ADDPTR_39:.*]] = tt.addptr %[[ARG6]], %{{.*}}
// CHECK:  %[[SPLAT_40:.*]] = tt.splat %[[CMPI_36]]
// CHECK:  %[[LOAD_41:.*]] = tt.load %[[ADDPTR_39]], %[[SPLAT_40]]
// CHECK:  %[[LOCAL_LOAD_42:.*]] = triton_gpu.local_load %[[ARG10]]
// CHECK:  %[[LOCAL_LOAD_43:.*]] = triton_gpu.local_load %[[ARG11]]
// CHECK:  %[[MULF_44:.*]] = arith.mulf %[[LOCAL_LOAD_43]], %{{.*}}
// CHECK:  %[[DOT_45:.*]] = tt.dot %[[LOCAL_LOAD_42]], %[[MULF_44]], %[[ARG8]]
// CHECK:  scf.yield %[[ADDPTR_39]], %[[ADDPTR_34]], %[[DOT_45]], %[[SELECT_31]], %[[MEMDESC_SUBVIEW_33]], %[[MEMDESC_SUBVIEW_32]], %[[LOAD_41]], %[[LOAD_38]]
// CHECK:  }

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
// CHECK:  %[[SUBI_20:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[ADDPTR_21:.*]] = tt.addptr %[[ARG9]], %{{.*}}
// CHECK:  %[[CMPI_22:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_20]]
// CHECK:  %[[SPLAT_23:.*]] = tt.splat %[[CMPI_22]]
// CHECK:  %[[LOAD_24:.*]] = tt.load %[[ADDPTR_21]], %[[SPLAT_23]]
// CHECK:  %[[EXPAND_DIMS_25:.*]] = tt.expand_dims %[[ARG13]] {axis = 1 : i32}
// CHECK:  %[[BROADCAST_26:.*]] = tt.broadcast %[[EXPAND_DIMS_25]]
// CHECK:  %[[MULI_27:.*]] = arith.muli %{{.*}}, %[[BROADCAST_26]]
// CHECK:  %[[ADDPTR_28:.*]] = tt.addptr %{{.*}}, %[[MULI_27]]
// CHECK:  %[[SUBI_29:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_30:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_29]]
// CHECK:  %[[SPLAT_31:.*]] = tt.splat %[[CMPI_30]]
// CHECK:  %[[LOAD_32:.*]] = tt.load %[[ADDPTR_28]], %[[SPLAT_31]]
// CHECK:  %[[ADDPTR_33:.*]] = tt.addptr %[[ARG8]], %{{.*}}
// CHECK:  %[[SPLAT_34:.*]] = tt.splat %[[CMPI_30]]
// CHECK:  %[[LOAD_35:.*]] = tt.load %[[ADDPTR_33]], %[[SPLAT_34]]
// CHECK:  %[[LOCAL_LOAD_36:.*]] = triton_gpu.local_load %[[ARG11]]
// CHECK:  %[[LOCAL_LOAD_37:.*]] = triton_gpu.local_load %[[ARG12]]
// CHECK:  %[[DOT_38:.*]] = tt.dot %[[LOCAL_LOAD_36]], %[[LOCAL_LOAD_37]], %[[ARG7]]
// CHECK:  %[[ADDI_39:.*]] = arith.addi %[[ARG10]], %{{.*}}
// CHECK:  %[[CMPI_40:.*]] = arith.cmpi slt, %[[ADDI_39]], %{{.*}}
// CHECK:  %[[SELECT_41:.*]] = arith.select %[[CMPI_40]], %[[ADDI_39]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_42:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_41]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_35]], %[[MEMDESC_SUBVIEW_42]]
// CHECK:  %[[MEMDESC_SUBVIEW_43:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_41]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_32]], %[[MEMDESC_SUBVIEW_43]]
// CHECK:  scf.yield %[[DOT_38]], %[[ADDPTR_33]], %[[ADDPTR_21]], %[[SELECT_41]], %[[MEMDESC_SUBVIEW_42]], %[[MEMDESC_SUBVIEW_43]], %[[LOAD_24]]
// CHECK:  }

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
