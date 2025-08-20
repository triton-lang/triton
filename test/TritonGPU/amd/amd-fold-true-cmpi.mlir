// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritonamdgpu-fold-true-cmpi -canonicalize | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @cmpsle(%arg0: !tt.ptr<f32>) -> i1 {
    %c0 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %cmpsle = arith.cmpi sle, %c0, %c1024_i32 : i32
    tt.return %cmpsle: i1
  }
}

// CHECK-LABEL:   tt.func @cmpsle(
// CHECK-SAME:                       %[[VAL_0:.*]]: !tt.ptr<f32>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = arith.constant true
// CHECK:           tt.return %[[VAL_1]] : i1
// CHECK:         }

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @assumepid(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    %c0 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %cmpsle = arith.cmpi sle, %pid, %c1024_i32 : i32
    llvm.intr.assume %cmpsle : i1
    %cmpsge = arith.cmpi sge, %pid, %c0 : i32
    llvm.intr.assume %cmpsge : i1
    %1 = arith.muli %pid, %c1024_i32 : i32
    %2 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.load %3 : tensor<1024x!tt.ptr<f32>>
    tt.return %4 : tensor<1024xf32>
  }
}

// CHECK-LABEL:   tt.func @assumepid(
// CHECK-SAME:                       %[[VAL_0:.*]]: !tt.ptr<f32>) -> tensor<1024xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant true
// CHECK:           %[[VAL_2:.*]] = arith.constant 1024 : i32
// CHECK:           %[[VAL_3:.*]] = tt.get_program_id x : i32
// CHECK:           llvm.intr.assume %[[VAL_1]] : i1
// CHECK:           llvm.intr.assume %[[VAL_1]] : i1
// CHECK:           %[[VAL_4:.*]] = arith.muli %[[VAL_3]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_5:.*]] = tt.addptr %[[VAL_0]], %[[VAL_4]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_6:.*]] = tt.splat %[[VAL_5]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_7:.*]] = tt.load %[[VAL_6]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.return %[[VAL_7]] : tensor<1024xf32>
// CHECK:         }

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @assume_matmul(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16>, %arg4: !tt.ptr<f16>) -> tensor<128x128xf32, #mma> {
    %c-1 = arith.constant -1 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %cst = arith.constant dense<4.000000e+00> : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_0 = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<128x32xi32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %0 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %3 = tt.broadcast %2 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %4 = tt.addptr %0, %3 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %5 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %9 = tt.addptr %5, %8 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #shared1, #smem, mutable>
    %12 = arith.cmpi slt, %arg0, %arg1 : index
    %13 = tt.splat %12 : i1 -> tensor<128x32xi1, #blocked1>
    %14 = tt.load %4, %13 : tensor<128x32x!tt.ptr<f16>, #blocked1>
    %15 = tt.splat %12 : i1 -> tensor<32x128xi1, #blocked>
    %16 = tt.load %9, %15, %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked>
    %17 = ttg.memdesc_index %10[%c0_i32] : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    ttg.local_store %14, %17 : tensor<128x32xf16, #blocked1> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %18 = ttg.memdesc_index %11[%c0_i32] : !ttg.memdesc<1x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    ttg.local_store %16, %18 : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    %19 = arith.subi %arg1, %arg2 : index
    %20:6 = scf.for %arg5 = %arg0 to %19 step %arg2 iter_args(%arg6 = %4, %arg7 = %9, %arg8 = %cst_2, %arg9 = %c0_i32, %arg10 = %17, %arg11 = %18) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !ttg.memdesc<128x32xf16, #shared, #smem, mutable>, !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>) {
      %33 = tt.addptr %arg6, %cst_1 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %34 = tt.addptr %arg7, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      llvm.intr.assume %true : i1
      %35 = tt.load %33 : tensor<128x32x!tt.ptr<f16>, #blocked1>
      %36 = ttg.local_load %arg10 : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %37 = tt.load %34 : tensor<32x128x!tt.ptr<f16>, #blocked>
      %38 = ttg.local_load %arg11 : !ttg.memdesc<32x128xf16, #shared1, #smem, mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %39 = arith.mulf %38, %cst : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %40 = tt.dot %36, %39, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %41 = arith.addi %arg9, %c1_i32 : i32
      %42 = arith.cmpi slt, %41, %c1_i32 : i32
      %43 = arith.select %42, %41, %c0_i32 : i32
      %44 = ttg.memdesc_index %10[%43] : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
      ttg.local_store %35, %44 : tensor<128x32xf16, #blocked1> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
      %45 = ttg.memdesc_index %11[%43] : !ttg.memdesc<1x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
      ttg.local_store %37, %45 : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
      scf.yield %33, %34, %40, %43, %44, %45 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !ttg.memdesc<128x32xf16, #shared, #smem, mutable>, !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    }
    %21 = arith.cmpi slt, %arg2, %c0 : index
    %22 = arith.select %21, %c1, %c-1 : index
    %23 = arith.subi %arg1, %arg0 : index
    %24 = arith.addi %23, %arg2 : index
    %25 = arith.addi %24, %22 : index
    %26 = arith.divsi %25, %arg2 : index
    %28 = ttg.local_load %20#4 : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %29 = ttg.local_load %20#5 : !ttg.memdesc<32x128xf16, #shared1, #smem, mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %30 = arith.mulf %29, %cst : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %27 = arith.cmpi sge, %26, %c1 : index
    llvm.intr.assume %27 : i1
    %31 = scf.if %27 -> (tensor<128x128xf32, #mma>) {
      %33 = tt.dot %28, %30, %20#2 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      scf.yield %33 : tensor<128x128xf32, #mma>
    } else {
      scf.yield %20#2 : tensor<128x128xf32, #mma>
    }
    %32 = arith.select %27, %31, %20#2 : tensor<128x128xf32, #mma>
    ttg.local_dealloc %10 : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable>
    ttg.local_dealloc %11 : !ttg.memdesc<1x32x128xf16, #shared1, #smem, mutable>
    tt.return %32 : tensor<128x128xf32, #mma>
  }
}

// CHECK: #[[$ATTR_2:.+]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
// CHECK: #[[$ATTR_3:.+]] = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
// CHECK: #[[$ATTR_4:.+]] = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
// CHECK: #[[$ATTR_5:.+]] = #ttg.shared_memory

// CHECK-LABEL:   tt.func @assume_matmul(
// CHECK:           %[[VAL_7:.*]] = arith.constant true
// CHECK:           %[[VAL_8:.*]] = arith.constant dense<4.000000e+00> : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$ATTR_2]], kWidth = 2}>>
// CHECK:           %[[VAL_23:.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xf16, #[[$ATTR_3]], #[[$ATTR_5]], mutable>
// CHECK:           %[[VAL_24:.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #[[$ATTR_4]], #[[$ATTR_5]], mutable>
// CHECK:           %[[VAL_33:.*]]:6 = scf.for
// CHECK:             scf.yield
// CHECK:           }
// CHECK-NEXT:      %[[VAL_54:.*]] = ttg.local_load %[[VAL_55:.*]]#4 : !ttg.memdesc<128x32xf16, #[[$ATTR_3]], #[[$ATTR_5]], mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$ATTR_2]], kWidth = 2}>>
// CHECK-NEXT:      %[[VAL_56:.*]] = ttg.local_load %[[VAL_55]]#5 : !ttg.memdesc<32x128xf16, #[[$ATTR_4]], #[[$ATTR_5]], mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$ATTR_2]], kWidth = 2}>>
// CHECK-NEXT:      %[[VAL_57:.*]] = arith.mulf %[[VAL_56]], %[[VAL_8]] : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$ATTR_2]], kWidth = 2}>>
// CHECK-NEXT:      llvm.intr.assume %[[VAL_7]] : i1
// CHECK-NEXT:      %[[VAL_58:.*]] = tt.dot %[[VAL_54]], %[[VAL_57]], %[[VAL_55]]#2 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$ATTR_2]], kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$ATTR_2]], kWidth = 2}>> -> tensor<128x128xf32, #[[$ATTR_2]]>
// CHECK-NEXT:      ttg.local_dealloc %[[VAL_23]] : !ttg.memdesc<1x128x32xf16, #[[$ATTR_3]], #[[$ATTR_5]], mutable>
// CHECK-NEXT:      ttg.local_dealloc %[[VAL_24]] : !ttg.memdesc<1x32x128xf16, #[[$ATTR_4]], #[[$ATTR_5]], mutable>
// CHECK-NEXT:      tt.return %[[VAL_58]] : tensor<128x128xf32, #[[$ATTR_2]]>
// CHECK-NEXT:      }

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @dontfoldtensor() -> tensor<128xi1> {
    %t0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %t1 = tt.make_range {end = 257 : i32, start = 129 : i32} : tensor<128xi32>
    %cmp = arith.cmpi sgt, %t1, %t0 : tensor<128xi32>
    tt.return %cmp: tensor<128xi1>
  }
}

// CHECK-LABEL:   tt.func @dontfoldtensor
// CHECK-NOT:       arith.constant dense<true>
// CHECK:           %[[VAL_0:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK:           %[[VAL_1:.*]] = tt.make_range {end = 257 : i32, start = 129 : i32} : tensor<128xi32>
// CHECK:           %[[VAL_2:.*]] = arith.cmpi sgt, %[[VAL_1]], %[[VAL_0]] : tensor<128xi32>
// CHECK:           tt.return %[[VAL_2]] : tensor<128xi1>
// CHECK:         }
