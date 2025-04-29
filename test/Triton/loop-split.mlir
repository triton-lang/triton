// RUN: triton-opt --split-input-file %s -triton-loop-split -canonicalize | FileCheck %s

tt.func @split_kernel_sgt(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32, %mid: i32) -> tensor<256xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // Check for 2 loops; first with subf, and second with addf
  // CHECK-LABEL: split_kernel_sgt
  // CHECK-NOT: arith.addi
  // CHECK: scf.for
  // CHECK:   arith.subf
  // CHECK: scf.for
  // CHECK:   arith.addf
  %2:2 = scf.for %arg3 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
    %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %cmp = arith.cmpi sgt, %arg3, %mid : i32
    %4 = scf.if %cmp -> (tensor<256xf32>) {
      %add = arith.addf %arg4, %3 : tensor<256xf32>
      scf.yield %add : tensor<256xf32>
    } else {
      %sub = arith.subf %arg4, %3 : tensor<256xf32>
      scf.yield %sub : tensor<256xf32>
    }
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  }
  tt.return %2#0 : tensor<256xf32>
}

// -----

tt.func @split_kernel_sge(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32, %mid: i32) -> tensor<256xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // Check the loop is split at mid-1
  // CHECK-LABEL: split_kernel_sge
  // CHECK: %[[MIN1:.*]] = arith.constant -1 : i32
  // CHECK: arith.addi {{.*}}, %[[MIN1]] : i32
  // CHECK: scf.for
  // CHECK:   arith.subf
  // CHECK: scf.for
  // CHECK:   arith.addf
  %2:2 = scf.for %arg3 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
    %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %cmp = arith.cmpi sge, %arg3, %mid : i32
    %4 = scf.if %cmp -> (tensor<256xf32>) {
      %add = arith.addf %arg4, %3 : tensor<256xf32>
      scf.yield %add : tensor<256xf32>
    } else {
      %sub = arith.subf %arg4, %3 : tensor<256xf32>
      scf.yield %sub : tensor<256xf32>
    }
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  }
  tt.return %2#0 : tensor<256xf32>
}

// -----

tt.func @split_kernel_slt(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32, %mid: i32) -> tensor<256xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // Check the 2 loops; first with addf, second with subf
  // CHECK: scf.for
  // CHECK:   arith.addf
  // CHECK: scf.for
  // CHECK:   arith.subf
  %2:2 = scf.for %arg3 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
    %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %cmp = arith.cmpi slt, %arg3, %mid : i32
    %4 = scf.if %cmp -> (tensor<256xf32>) {
      %add = arith.addf %arg4, %3 : tensor<256xf32>
      scf.yield %add : tensor<256xf32>
    } else {
      %sub = arith.subf %arg4, %3 : tensor<256xf32>
      scf.yield %sub : tensor<256xf32>
    }
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  }
  tt.return %2#0 : tensor<256xf32>
}

// -----

tt.func @split_kernel_sle(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32, %mid: i32) -> tensor<256xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // Check the loop is split at mid+1
  // CHECK-LABEL: split_kernel_sle
  // CHECK: %[[PLUS1:.*]] = arith.constant 1 : i32
  // CHECK: arith.addi {{.*}}, %[[PLUS1]] : i32
  // CHECK: scf.for
  // CHECK:   arith.addf
  // CHECK: scf.for
  // CHECK:   arith.subf
  %2:2 = scf.for %arg3 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
    %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %cmp = arith.cmpi sle, %arg3, %mid : i32
    %4 = scf.if %cmp -> (tensor<256xf32>) {
      %add = arith.addf %arg4, %3 : tensor<256xf32>
      scf.yield %add : tensor<256xf32>
    } else {
      %sub = arith.subf %arg4, %3 : tensor<256xf32>
      scf.yield %sub : tensor<256xf32>
    }
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  }
  tt.return %2#0 : tensor<256xf32>
}

// -----

tt.func @split_kernel_step10(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32, %arg2: i32) -> tensor<256xf32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // Check the mid-point is rounded up
  // CHECK-LABEL: split_kernel_step10
  // CHECK: %[[REM:.*]] = arith.remui %arg2, %c10_i32
  // CHECK: %[[CMP:.*]] = arith.cmpi ne, %[[REM]], %c0_i32
  // CHECK: %[[EXT:.*]] = arith.extui %[[CMP]] : i1 to i32
  // CHECK: %[[MIDP:.*]] = arith.addi %arg2, %[[EXT]] : i32

  // CHECK: scf.for {{.*}} = %c0_i32 to %[[MIDP]] step %c10_i32
  // CHECK:   arith.addf
  // CHECK: scf.for {{.*}} = %[[MIDP]] to %arg1 step %c10_i32
  // CHECK:   arith.subf
  %2:2 = scf.for %arg3 = %c0_i32 to %arg1 step %c10_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
    %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %cmp = arith.cmpi slt, %arg3, %arg2 : i32
    %4 = scf.if %cmp -> (tensor<256xf32>) {
      %add = arith.addf %arg4, %3 : tensor<256xf32>
      scf.yield %add : tensor<256xf32>
    } else {
      %sub = arith.subf %arg4, %3 : tensor<256xf32>
      scf.yield %sub : tensor<256xf32>
    }
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  }
  tt.return %2#0 : tensor<256xf32>
}

