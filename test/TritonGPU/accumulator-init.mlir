// RUN: triton-opt %s -split-input-file -tritongpu-optimize-accumulator-init | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {

// CHECK-LABEL: @constant_init
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: triton_nvidia_gpu.warp_group_dot {{.*}}, {{.*}}, {{.*}}, %[[FALSE]]
  tt.func @constant_init(%A: !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>, %B: !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = triton_nvidia_gpu.warp_group_dot %A, %B, %cst_2 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      scf.yield %acc: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @if_after_mma
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[FALSE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[ACC_NEXT:.+]] = triton_nvidia_gpu.warp_group_dot {{.*}}, {{.*}}, %[[ACC]], %[[USE_ACC]]
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[FALSE]], %[[TRUE]]
// CHECK: scf.if %[[CND]]
// CHECK: scf.yield %[[ACC_NEXT]]
// CHECK: else
// CHECK: scf.yield %[[ACC_NEXT]]
// CHECK: scf.yield {{.*}}, %[[USE_ACC_NEXT]]
  tt.func @if_after_mma(%A: !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>, %B: !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = triton_nvidia_gpu.warp_group_dot %A, %B, %arg4 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      %acc_ = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %acc : tensor<128x16xf32, #mma1>
      }
      scf.yield %acc_: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @if_after_mma_invert
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[FALSE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[ACC_NEXT:.+]] = triton_nvidia_gpu.warp_group_dot {{.*}}, {{.*}}, %[[ACC]], %[[USE_ACC]]
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[TRUE]], %[[FALSE]]
// CHECK: scf.if %[[CND]]
// CHECK: scf.yield %[[ACC_NEXT]]
// CHECK: else
// CHECK: scf.yield %[[ACC_NEXT]]
// CHECK: scf.yield {{.*}}, %[[USE_ACC_NEXT]]
  tt.func @if_after_mma_invert(%A: !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>, %B: !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = triton_nvidia_gpu.warp_group_dot %A, %B, %arg4 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      %acc_ = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %acc : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      }
      scf.yield %acc_: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @if_before_mma
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[TRUE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[FALSE]], %[[USE_ACC]]
// CHECK: %[[ACC_CND:.+]] = scf.if %[[CND]]
// CHECK: scf.yield %[[ACC]]
// CHECK: else
// CHECK: scf.yield %[[ACC]]
// CHECK: %[[ACC_NEXT:.+]] = triton_nvidia_gpu.warp_group_dot {{.*}}, {{.*}}, %[[ACC_CND]], %[[USE_ACC_NEXT]]
// CHECK: scf.yield {{.*}}, %[[TRUE]]
  tt.func @if_before_mma(%A: !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>, %B: !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc_ = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %arg4 : tensor<128x16xf32, #mma1>
      }
      %acc = triton_nvidia_gpu.warp_group_dot %A, %B, %acc_ : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      scf.yield %acc: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @if_before_mma_invert
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[TRUE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[USE_ACC]], %[[FALSE]]
// CHECK: %[[ACC_CND:.+]] = scf.if %[[CND]]
// CHECK: scf.yield %[[ACC]]
// CHECK: else
// CHECK: scf.yield %[[ACC]]
// CHECK: %[[ACC_NEXT:.+]] = triton_nvidia_gpu.warp_group_dot {{.*}}, {{.*}}, %[[ACC_CND]], %[[USE_ACC_NEXT]]
// CHECK: scf.yield {{.*}}, %[[TRUE]]
  tt.func @if_before_mma_invert(%A: !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>, %B: !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc_ = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %arg4 : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      }
      %acc = triton_nvidia_gpu.warp_group_dot %A, %B, %acc_ : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      scf.yield %acc: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @sel_after_mma
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[FALSE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[ACC_NEXT:.+]] = triton_nvidia_gpu.warp_group_dot {{.*}}, {{.*}}, %[[ACC]], %[[USE_ACC]]
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[FALSE]], %[[TRUE]]
// CHECK: scf.yield {{.*}}, %[[USE_ACC_NEXT]]
  tt.func @sel_after_mma(%A: !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>, %B: !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = triton_nvidia_gpu.warp_group_dot %A, %B, %arg4 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      %acc_ = arith.select %cnd, %cst_2, %acc : tensor<128x16xf32, #mma1>
      scf.yield %acc_: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @sel_before_mma
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[TRUE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[FALSE]], %[[USE_ACC]]
// CHECK: %[[ACC_NEXT:.+]] = triton_nvidia_gpu.warp_group_dot {{.*}}, {{.*}}, %[[ACC]], %[[USE_ACC_NEXT]]
// CHECK: scf.yield {{.*}}, %[[TRUE]]
  tt.func @sel_before_mma(%A: !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>, %B: !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc_ = arith.select %cnd, %cst_2, %arg4 : tensor<128x16xf32, #mma1>
      %acc = triton_nvidia_gpu.warp_group_dot %A, %B, %acc_ : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      scf.yield %acc: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }


// Check that we look only at the zeroing directly preceding the mma

// CHECK-LABEL: @if_before_and_after_mma
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[TRUE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[FALSE]], %[[USE_ACC]]
// CHECK: %[[ACC_CND:.+]] = scf.if %[[CND]]
// CHECK: scf.yield %[[ACC]]
// CHECK: else
// CHECK: scf.yield %[[ACC]]
// CHECK: %[[ACC_NEXT:.+]] = triton_nvidia_gpu.warp_group_dot {{.*}}, {{.*}}, %[[ACC_CND]], %[[USE_ACC_NEXT]]
// CHECK: scf.if %[[CND]]
// CHECK: scf.yield %[[C0_TENSOR]]
// CHECK: else
// CHECK: scf.yield %[[ACC_NEXT]]
// CHECK: scf.yield {{.*}}, %[[TRUE]]
  tt.func @if_before_and_after_mma(%A: !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>, %B: !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc_0 = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %arg4 : tensor<128x16xf32, #mma1>
      }
      %acc = triton_nvidia_gpu.warp_group_dot %A, %B, %acc_0 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      %acc_1 = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %acc : tensor<128x16xf32, #mma1>
      }
      scf.yield %acc_1: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @two_ifs_after_mma
// CHECK-DAG: %[[C0_TENSOR:.+]] = arith.constant dense<0.000000e+00>
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[FALSE:.+]] = arith.constant false
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[C0_TENSOR]], %[[USE_ACC:.+]] = %[[FALSE]])
// CHECK: %[[CND:.+]] = arith.cmpi
// CHECK: %[[ACC_NEXT:.+]] = triton_nvidia_gpu.warp_group_dot {{.*}}, {{.*}}, %[[ACC]], %[[USE_ACC]]
// CHECK: %[[ACC_CND:.+]] = scf.if %[[CND]]
// CHECK: scf.yield %[[C0_TENSOR]]
// CHECK: else
// CHECK: scf.yield %[[ACC_NEXT]]
// CHECK: %[[USE_ACC_NEXT:.*]] = arith.select %[[CND]], %[[FALSE]], %[[TRUE]]
// CHECK: scf.if %[[CND]]
// CHECK: scf.yield %[[ACC_CND]]
// CHECK: else
// CHECK: scf.yield %[[ACC_CND]]
// CHECK: scf.yield {{.*}}, %[[USE_ACC_NEXT]]
  tt.func @two_ifs_after_mma(%A: !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>, %B: !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = triton_nvidia_gpu.warp_group_dot %A, %B, %arg4 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      %acc_0 = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %acc : tensor<128x16xf32, #mma1>
      }
      %acc_1 = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %acc_0 : tensor<128x16xf32, #mma1>
      }
      scf.yield %acc_1: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// Check that we bail out in unsupported cases

// CHECK-LABEL: @non_zero_init
// CHECK-NOT: %[[ACC_NEXT:.+]] = triton_nvidia_gpu.warp_group_dot {{.*}}, {{.*}}, {{.*}}, {{.*}} : !tt.memdesc
  tt.func @non_zero_init(%A: !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>, %B: !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = triton_nvidia_gpu.warp_group_dot %A, %B, %arg4 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      %acc_ = arith.select %cnd, %cst_2, %acc : tensor<128x16xf32, #mma1>
      scf.yield %acc_: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @zero_init_dist_2
// CHECK-NOT: %[[ACC_NEXT:.+]] = triton_nvidia_gpu.warp_group_dot {{.*}}, {{.*}}, {{.*}}, {{.*}} : !tt.memdesc
  tt.func @zero_init_dist_2(%A: !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>, %B: !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17:2 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2, %arg5 = %cst_2) -> (tensor<128x16xf32, #mma1>, tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = triton_nvidia_gpu.warp_group_dot %A, %B, %arg5 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      %acc_ = arith.select %cnd, %cst_2, %acc : tensor<128x16xf32, #mma1>
      scf.yield %acc_, %arg4: tensor<128x16xf32, #mma1>, tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @if_defines_alternative
// CHECK-NOT: %[[ACC_NEXT:.+]] = triton_nvidia_gpu.warp_group_dot {{.*}}, {{.*}}, {{.*}}, {{.*}} : !tt.memdesc
  tt.func @if_defines_alternative(%A: !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>, %B: !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %acc = triton_nvidia_gpu.warp_group_dot %A, %B, %arg4 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      %acc_ = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        scf.yield %cst_2 : tensor<128x16xf32, #mma1>
      } else {
        %acc_alt = arith.addf %acc, %cst_3 : tensor<128x16xf32, #mma1>
        scf.yield %acc_alt : tensor<128x16xf32, #mma1>
      }
      scf.yield %acc_: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }

// CHECK-LABEL: @non_cond_override
// CHECK-NOT: %[[ACC_NEXT:.+]] = triton_nvidia_gpu.warp_group_dot {{.*}}, {{.*}}, {{.*}}, {{.*}} : !tt.memdesc
  tt.func @non_cond_override(%A: !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>, %B: !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>, %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %17 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<128x16xf32, #mma1>)  : i32 {
      %acc = triton_nvidia_gpu.warp_group_dot %A, %B, %arg4 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      %acc_ = arith.addf %acc, %cst_3 : tensor<128x16xf32, #mma1>
      scf.yield %acc_: tensor<128x16xf32, #mma1>
    }
    tt.return %17 : tensor<128x16xf32, #mma1>
  }
}
