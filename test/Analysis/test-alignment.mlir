// RUN: triton-opt %s -test-print-alignment -split-input-file -o %t 2>&1 | FileCheck %s

// CHECK-LABEL: @cast
tt.func @cast() {
  // CHECK: contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1
  %cst = arith.constant 1 : i32
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1
  %0 = arith.extsi %cst : i32 to i64
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1
  %cst_tensor = arith.constant dense<1> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1
  %1 = tt.bitcast %cst_tensor : tensor<128xi32> -> tensor<128xi64>
  tt.return
}

// -----

// CHECK-LABEL: @add
tt.func @add() {
  // CHECK: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1
  %1 = arith.constant dense<1> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [128], divisibility = [1], constancy = [1], constant_value = <none>
  %2 = arith.addi %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = 127
  %3 = arith.constant dense<127> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128
  %4 = arith.addi %1, %3 : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @addptr
tt.func @addptr(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32}) {
  // CHECK: contiguity = [1], divisibility = [1], constancy = [1], constant_value = 1
  %cst1 = arith.constant 1 : i32
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %0 = tt.addptr %arg0, %cst1 : !tt.ptr<i1>, i32
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %1 = tt.addptr %arg1, %cst1 : !tt.ptr<i8>, i32
  // CHECK-NEXT: contiguity = [1], divisibility = [2], constancy = [1], constant_value = <none>
  %2 = tt.addptr %arg2, %cst1 : !tt.ptr<i16>, i32
  // CHECK-NEXT: contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>
  %3 = tt.addptr %arg3, %cst1 : !tt.ptr<i32>, i32
  // CHECK-NEXT: contiguity = [1], divisibility = [8], constancy = [1], constant_value = <none>
  %4 = tt.addptr %arg4, %cst1 : !tt.ptr<i64>, i32
  // CHECK-NEXT: contiguity = [1], divisibility = [4], constancy = [1], constant_value = 4
  %cst4 = arith.constant 4 : i32
  // CHECK-NEXT: contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>
  %5 = tt.addptr %arg0, %cst4 : !tt.ptr<i1>, i32
  // CHECK-NEXT: contiguity = [1], divisibility = [4], constancy = [1], constant_value = <none>
  %6 = tt.addptr %arg1, %cst4 : !tt.ptr<i8>, i32
  // CHECK-NEXT: contiguity = [1], divisibility = [8], constancy = [1], constant_value = <none>
  %7 = tt.addptr %arg2, %cst4 : !tt.ptr<i16>, i32
  // CHECK-NEXT: contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>
  %8 = tt.addptr %arg3, %cst4 : !tt.ptr<i32>, i32
  // CHECK-NEXT: contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>
  %9 = tt.addptr %arg4, %cst4 : !tt.ptr<i64>, i32
  // CHECK-NEXT: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [1, 1], constant_value = <none>
  %11 = tt.expand_dims %10 {axis = 0: i32} : (tensor<128xi32>) -> tensor<1x128xi32>
  // CHECK-NEXT: contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [128, 1], constant_value = <none>
  %12 = tt.broadcast %11 : (tensor<1x128xi32>) -> tensor<128x128xi32>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>
  %13 = tt.splat %arg0 : (!tt.ptr<i1>) -> tensor<128x128x!tt.ptr<i1>>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>
  %14 = tt.splat %arg1 : (!tt.ptr<i8>) -> tensor<128x128x!tt.ptr<i8>>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>
  %15 = tt.splat %arg2 : (!tt.ptr<i16>) -> tensor<128x128x!tt.ptr<i16>>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>
  %16 = tt.splat %arg3 : (!tt.ptr<i32>) -> tensor<128x128x!tt.ptr<i32>>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>
  %17 = tt.splat %arg4 : (!tt.ptr<i64>) -> tensor<128x128x!tt.ptr<i64>>
  // CHECK-NEXT: contiguity = [1, 128], divisibility = [1, 16], constancy = [128, 1], constant_value = <none>
  %18 = tt.addptr %13, %12 : tensor<128x128x!tt.ptr<i1>>, tensor<128x128xi32>
  // CHECK-NEXT: contiguity = [1, 128], divisibility = [1, 16], constancy = [128, 1], constant_value = <none>
  %19 = tt.addptr %14, %12 : tensor<128x128x!tt.ptr<i8>>, tensor<128x128xi32>
  // CHECK-NEXT: contiguity = [1, 128], divisibility = [2, 16], constancy = [128, 1], constant_value = <none>
  %20 = tt.addptr %15, %12 : tensor<128x128x!tt.ptr<i16>>, tensor<128x128xi32>
  // CHECK-NEXT: contiguity = [1, 128], divisibility = [4, 16], constancy = [128, 1], constant_value = <none>
  %21 = tt.addptr %16, %12 : tensor<128x128x!tt.ptr<i32>>, tensor<128x128xi32>
  // CHECK-NEXT: contiguity = [1, 128], divisibility = [8, 16], constancy = [128, 1], constant_value = <none>
  %22 = tt.addptr %17, %12 : tensor<128x128x!tt.ptr<i64>>, tensor<128x128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @sub
tt.func @sub() {
  // CHECK: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1
  %1 = arith.constant dense<1> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [128], divisibility = [1], constancy = [1], constant_value = <none>
  %2 = arith.subi %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = 129
  %3 = arith.constant dense<129> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128
  %4 = arith.subi %3, %1 : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @mul
tt.func @mul() {
  // CHECK: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1
  %1 = arith.constant dense<1> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %2 = arith.muli %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128
  %3 = arith.constant dense<128> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128
  %4 = arith.muli %3, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [2], constancy = [128], constant_value = 2
  %5 = arith.constant dense<2> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [256], constancy = [128], constant_value = 256
  %6 = arith.muli %4, %5 : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @div
tt.func @div() {
  // CHECK: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1
  %1 = arith.constant dense<1> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %2 = arith.divsi %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %3 = arith.divui %1, %0 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64
  %4 = arith.constant dense<64> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [64], constant_value = <none>
  %5 = arith.divsi %0, %4 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %6 = arith.divsi %4, %0 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64
  %7 = arith.divsi %4, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [2], constancy = [128], constant_value = 66
  %8 = arith.constant dense<66> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [2], constant_value = <none>
  %9 = arith.divui %0, %8 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [128], divisibility = [8192], constancy = [1], constant_value = <none>
  %10 = tt.make_range {end = 8320 : i32, start = 8192 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [64], constant_value = <none>
  %11 = arith.divsi %10, %4 : tensor<128xi32>
  tt.return
}


// -----

// CHECK-LABEL: @rem
tt.func @rem() {
  // CHECK: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = 1
  %1 = arith.constant dense<1> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0
  %2 = arith.remsi %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %3 = arith.remui %1, %0 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64
  %4 = arith.constant dense<64> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [64], divisibility = [64], constancy = [1], constant_value = <none>
  %5 = arith.remsi %0, %4 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [64], constancy = [1], constant_value = <none>
  %6 = arith.remsi %4, %0 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [2], constancy = [128], constant_value = 66
  %7 = arith.constant dense<66> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [2], divisibility = [2], constancy = [1], constant_value = <none>
  %8 = arith.remui %0, %7 : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @broadcast
tt.func @broadcast() {
  // CHECK: contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64
  %0 = arith.constant dense<64> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [64, 1], constancy = [128, 1], constant_value = 64
  %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<128xi32>) -> tensor<128x1xi32>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [64, 1], constancy = [128, 128], constant_value = 64
  %2 = tt.broadcast %1 : (tensor<128x1xi32>) -> tensor<128x128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @splat
tt.func @splat(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // CHECK: contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 128], constant_value = <none>
  %0 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<128x128x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL: @cmp
tt.func @cmp() {
  // CHECK: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0
  %1 = arith.constant dense<0> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>
  %2 = arith.cmpi eq, %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>
  %3 = arith.cmpi slt, %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %4 = arith.cmpi sle, %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>
  %5 = arith.cmpi sge, %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8
  %6 = arith.constant dense<8> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>
  %7 = arith.cmpi sgt, %0, %6 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = 0
  %8 = arith.cmpi sgt, %1, %6 : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @logic
tt.func @logic() {
  // CHECK: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [64], constancy = [128], constant_value = 64
  %1 = arith.constant dense<64> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [64], constant_value = <none>
  %2 = arith.divsi %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8
  %3 = arith.constant dense<8> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>
  %4 = arith.divsi %0, %3 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %5 = arith.andi %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %6 = arith.ori %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %7 = arith.xori %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>
  %8 = arith.andi %2, %4 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>
  %9 = arith.ori %2, %4 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [8], constant_value = <none>
  %10 = arith.xori %2, %4 : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @select
tt.func @select() {
  // CHECK: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0
  %1 = arith.constant dense<0> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>
  %2 = arith.cmpi eq, %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>
  %3 = arith.cmpi slt, %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0
  %4 = arith.constant 0 : i1
  // CHECK-NEXT: contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0
  %7 = tt.splat %4 : (i1) -> tensor<128xi1>
  // CHECK-NEXT: contiguity = [1], divisibility = [4611686018427387904], constancy = [128], constant_value = 0
  %5 = arith.select %4, %3, %7 : tensor<128xi1>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [128], constant_value = <none>
  %8 = "triton_gpu.select"(%7, %3, %2) : (tensor<128xi1>, tensor<128xi1>, tensor<128xi1>) -> tensor<128xi1>
  tt.return
}

// -----

tt.func @shift() {
  // CHECK: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8
  %1 = arith.constant dense<8> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [4], constancy = [128], constant_value = 4
  %2 = arith.constant dense<4> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [274877906944], constancy = [1], constant_value = <none>
  %3 = arith.shli %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [67108864], constancy = [1], constant_value = <none>
  %4 = arith.shrsi %0, %2 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [128], constancy = [128], constant_value = 128
  %5 = arith.shli %1, %2 : tensor<128xi32>
  tt.return
}

// -----

tt.func @max_min() {
  // CHECK: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [128], divisibility = [64], constancy = [1], constant_value = <none>
  %1 = tt.make_range {end = 192 : i32, start = 64 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %2 = arith.maxsi %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %3 = arith.minsi %0, %1 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [8], constancy = [128], constant_value = 8
  %4 = arith.constant dense<8> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [4], constancy = [128], constant_value = 4
  %5 = arith.constant dense<4> : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [1], constant_value = 8
  %6 = arith.maxsi %4, %5 : tensor<128xi32>
  tt.return
}

// -----

// CHECK-LABEL: @for
tt.func @for() {
  // CHECK: contiguity = [1, 1], divisibility = [4611686018427387904, 4611686018427387904], constancy = [128, 32], constant_value = 0
  %a_init = arith.constant dense<0> : tensor<128x32xi32>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 32], constant_value = 1
  %b_init = arith.constant dense<1> : tensor<128x32xi32>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [4, 4], constancy = [128, 32], constant_value = 4
  %c_init = arith.constant dense<4> : tensor<128x32xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [128], constancy = [1], constant_value = 128
  %ub = arith.constant 128 : index
  // CHECK-NEXT: contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0
  %lb = arith.constant 0 : index
  // CHECK-NEXT: contiguity = [1], divisibility = [16], constancy = [1], constant_value = 16
  %step = arith.constant 16 : index
  %a, %b, %c = scf.for %iv = %lb to %ub step %step iter_args(%a = %a_init, %b = %b_init, %c = %c_init) -> (tensor<128x32xi32>, tensor<128x32xi32>, tensor<128x32xi32>) {
    // CHECK-NEXT: contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>
    %t = arith.index_cast %iv : index to i32
    // CHECK: contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 32], constant_value = <none>
    // CHECK: contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 32], constant_value = <none>
    // CHECK: contiguity = [1, 1], divisibility = [4, 4], constancy = [128, 32], constant_value = 4
    scf.yield %b, %a, %c : tensor<128x32xi32>, tensor<128x32xi32>, tensor<128x32xi32>
  }
  tt.return
}

// -----

// CHECK-LABEL: @permute_2d
tt.func @permute_2d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
  // CHECK: contiguity = [1, 1], divisibility = [1, 1], constancy = [128, 128], constant_value = 1
  %cst = arith.constant dense<true> : tensor<128x128xi1>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
  // CHECK-NEXT: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: contiguity = [128, 1], divisibility = [1073741824, 1], constancy = [1, 1], constant_value = <none>
  %2 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<128xi32>) -> tensor<128x1xi32>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 1], constant_value = <none>
  %3 = tt.splat %arg1 : (i32) -> tensor<128x1xi32>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [17179869184, 16], constancy = [1, 1], constant_value = <none>
  %4 = arith.muli %2, %3 : tensor<128x1xi32>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 1], constant_value = <none>
  %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<128x1x!tt.ptr<f32>>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>
  %6 = tt.addptr %5, %4 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
  // CHECK-NEXT: contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [1, 1], constant_value = <none>
  %7 = tt.expand_dims %1 {axis = 0 : i32}: (tensor<128xi32>) -> tensor<1x128xi32>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 128], constant_value = <none>
  %8 = tt.broadcast %6 : (tensor<128x1x!tt.ptr<f32>>) -> tensor<128x128x!tt.ptr<f32>>
  // CHECK-NEXT: contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [128, 1], constant_value = <none>
  %9 = tt.broadcast %7 : (tensor<1x128xi32>) -> tensor<128x128xi32>
  // CHECK-NEXT: contiguity = [1, 128], divisibility = [4, 16], constancy = [1, 1], constant_value = <none>
  %10 = tt.addptr %8, %9 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
  // CHECK-NEXT: contiguity = [128, 1], divisibility = [1073741824, 1], constancy = [1, 1], constant_value = <none>
  %11 = tt.expand_dims %0 {axis = 1 : i32}: (tensor<128xi32>) -> tensor<128x1xi32>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [16, 16], constancy = [128, 1], constant_value = <none>
  %12 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<128x1x!tt.ptr<f32>>
  // CHECK-NEXT: contiguity = [128, 1], divisibility = [16, 4], constancy = [1, 1], constant_value = <none>
  %13 = tt.addptr %12, %11 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
  // CHECK-NEXT: contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [1, 1], constant_value = <none>
  %14 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<128xi32>) -> tensor<1x128xi32>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 128], constant_value = <none>
  %15 = tt.splat %arg3 : (i32) -> tensor<1x128xi32>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [16, 17179869184], constancy = [1, 1], constant_value = <none>
  %16 = arith.muli %14, %15 : tensor<1x128xi32>
  // CHECK-NEXT: contiguity = [128, 1], divisibility = [16, 4], constancy = [1, 128], constant_value = <none>
  %17 = tt.broadcast %13 : (tensor<128x1x!tt.ptr<f32>>) -> tensor<128x128x!tt.ptr<f32>>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [16, 17179869184], constancy = [128, 1], constant_value = <none>
  %18 = tt.broadcast %16 : (tensor<1x128xi32>) -> tensor<128x128xi32>
  // CHECK-NEXT: contiguity = [128, 1], divisibility = [16, 4], constancy = [1, 1], constant_value = <none>
  %19 = tt.addptr %17, %18 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
  // CHECK-NEXT: contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>
  %20 = tt.load %10, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf32>
  tt.store %19, %20, %cst : tensor<128x128xf32>
  tt.return
}

// -----

module {

// This is a tiny test for verifying StoreOp-related alignment, It simply store a constant to a buffer.
// CHECK-LABEL: @store_constant_align
tt.func @store_constant_align(%addr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32 {tt.divisibility = 16 : i32}) {
  // CHECK: contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %pid = tt.get_program_id {axis = 0 : i32} : i32
  // CHECK-NEXT: contiguity = [1], divisibility = [128], constancy = [1], constant_value = 128
  %c128_i32 = arith.constant 128 : i32
  // CHECK-NEXT: contiguity = [1], divisibility = [128], constancy = [1], constant_value = <none>
  %1 = arith.muli %pid, %c128_i32 : i32
  // CHECK-NEXT: contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
 // CHECK-NEXT: contiguity = [1], divisibility = [128], constancy = [128], constant_value = <none>
  %3 = tt.splat %1 : (i32) -> tensor<128xi32>
 // CHECK-NEXT: contiguity = [128], divisibility = [128], constancy = [1], constant_value = <none>
  %4 = arith.addi %3, %2 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [16], constancy = [128], constant_value = <none>
  %5 = tt.splat %addr : (!tt.ptr<f32>) -> tensor<128x!tt.ptr<f32>>
  // CHECK-NEXT: contiguity = [128], divisibility = [16], constancy = [1], constant_value = <none>
  %6 = tt.addptr %5, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [16], constancy = [128], constant_value = <none>
  %9 = tt.splat %n : (i32) -> tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>
  %mask = arith.cmpi slt, %4, %9 : tensor<128xi32>
  // CHECK-NEXT: contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %cst = arith.constant dense<0.0> : tensor<128xf32>
  tt.store %5, %cst, %mask : tensor<128xf32>
  tt.return
}

}

// -----

// This IR is dumped from vecadd test.
// Note, the hint {tt.divisibility = 16 : i32} for %n_elements affects the alignment of mask.
// CHECK-LABEL: @vecadd_mask_align_16
tt.func @vecadd_mask_align_16(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32 {tt.divisibility = 16 : i32}) {
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %3 = tt.splat %1 : (i32) -> tensor<64xi32>
  %4 = arith.addi %3, %2 : tensor<64xi32>
  %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %7 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %9 = tt.splat %n_elements : (i32) -> tensor<64xi32>
  // CHECK: arith.cmpi slt, %{{.*}} => contiguity = [1], divisibility = [1], constancy = [16], constant_value = <none>
  %mask = arith.cmpi slt, %4, %9 : tensor<64xi32>
  %11 = tt.load %6, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
  %12 = tt.load %8, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
  %13 = arith.addf %11, %12 : tensor<64xf32>
  %14 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  // CHECK: tt.addptr %{{.*}} => contiguity = [64], divisibility = [16], constancy = [1], constant_value = <none>
  %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  tt.store %15, %13, %mask : tensor<64xf32>
  tt.return
}

// -----

// This IR is dumped from vecadd test.
// Note, there is no divisibility hint for %n_elements, Triton should assume its divisibility to be 1 by default.
// CHECK-LABEL: @vecadd_mask_align_1
tt.func @vecadd_mask_align_1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32) {
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %3 = tt.splat %1 : (i32) -> tensor<64xi32>
  %4 = arith.addi %3, %2 : tensor<64xi32>
  %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %7 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %9 = tt.splat %n_elements : (i32) -> tensor<64xi32>
  // CHECK: arith.cmpi slt, %{{.*}} => contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %10 = arith.cmpi slt, %4, %9 : tensor<64xi32>
  %11 = tt.load %6, %10 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
  %12 = tt.load %8, %10 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
  %13 = arith.addf %11, %12 : tensor<64xf32>
  %14 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  tt.store %15, %13, %10 : tensor<64xf32>
  tt.return
}
