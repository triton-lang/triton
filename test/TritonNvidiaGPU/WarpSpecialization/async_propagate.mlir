// RUN: triton-opt %s -split-input-file --triton-gpu-taskid-propagate=num-consumer-groups=1 | FileCheck %s

// CHECK-LABEL: @async_kernel
// CHECK: %0 = tt.get_program_id x {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
// CHECK: %5 = tt.splat %arg2 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<1024xi32>
// CHECK: %9 = tt.load %8, %6 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>
// CHECK: %10 = tt.splat %arg1 {async_task_id = dense<1> : vector<1xi32>} : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK: tt.store %11, %9 {async_task_id = dense<1> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>

module {
  tt.func public @async_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.load %8, %6 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 {async_task_id = dense<1> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %11, %9 {async_task_id = dense<1> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: @two_consumers
// CHECK: tt.get_program_id x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
// CHECK: tt.splat %arg0 {async_task_id = dense<0> : vector<1xi32>}
// CHECK: tt.load {{.*}} {async_task_id = dense<0> : vector<1xi32>}
// CHECK: tt.load {{.*}} {async_task_id = dense<0> : vector<1xi32>}
// CHECK: tt.splat %arg1 {async_task_id = dense<[1, 2]> : vector<2xi32>}
// CHECK: tt.store {{.*}} {async_task_id = dense<1> : vector<1xi32>}
// CHECK: tt.store {{.*}} {async_task_id = dense<2> : vector<1xi32>}

module {
  tt.func public @two_consumers(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.make_range {end = 2048 : i32, start = 1024 : i32} : tensor<1024xi32>
    %4 = tt.splat %1 : i32 -> tensor<1024xi32>
    %5 = arith.addi %4, %2 : tensor<1024xi32>
    %6 = arith.addi %4, %3 : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %5 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.addptr %7, %6 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %10 = tt.load %8 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>
    %11 = tt.load %9 {async_task_id = dense<0> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %13 = tt.addptr %12, %5 {async_task_id = dense<1> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %14 = tt.addptr %12, %6 {async_task_id = dense<2> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %13, %10 {async_task_id = dense<1> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>
    tt.store %14, %11 {async_task_id = dense<2> : vector<1xi32>} : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
