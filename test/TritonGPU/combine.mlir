// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions 2>&1 | FileCheck %s

#layout0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#layout1 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

#layout2 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#layout3 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>


module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {

// CHECK: [[$target_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
// CHECK-LABEL: cst
tt.func @cst() -> tensor<1024xi32, #layout1> {
  %cst = arith.constant dense<0> : tensor<1024xi32, #layout0>
  %1 = triton_gpu.convert_layout %cst : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: tt.return %cst : tensor<1024xi32, [[$target_layout]]>
  tt.return %1: tensor<1024xi32, #layout1>
}

// CHECK-LABEL: range
tt.func @range() -> tensor<1024xi32, #layout1> {
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %1 = triton_gpu.convert_layout %0 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: tt.return %0 : tensor<1024xi32, [[$target_layout]]>
  tt.return %1: tensor<1024xi32, #layout1>
}

// CHECK-LABEL: splat
tt.func @splat(%arg0: i32) -> tensor<1024xi32, #layout1> {
  %0 = tt.splat %arg0 : (i32) -> tensor<1024xi32, #layout0>
  %1 = triton_gpu.convert_layout %0 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  // CHECK: tt.return %0 : tensor<1024xi32, [[$target_layout]]>
  tt.return %1: tensor<1024xi32, #layout1>
}

// CHECK-LABEL: remat
tt.func @remat(%arg0: i32) -> tensor<1024xi32, #layout1> {
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %2 = arith.muli %0, %1 : tensor<1024xi32, #layout0>
  %3 = triton_gpu.convert_layout %2 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  %4 = tt.splat %arg0 : (i32) -> tensor<1024xi32, #layout0>
  %5 = triton_gpu.convert_layout %2 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  %6 = arith.addi %3, %5 : tensor<1024xi32, #layout1>
  tt.return %6: tensor<1024xi32, #layout1>
  // CHECK: %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[$target_layout]]>
  // CHECK: %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[$target_layout]]>
  // CHECK: %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[$target_layout]]>
  // CHECK: %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[$target_layout]]>
  // CHECK: %4 = arith.muli %0, %2 : tensor<1024xi32, [[$target_layout]]>
  // CHECK: %5 = arith.muli %1, %3 : tensor<1024xi32, [[$target_layout]]>
  // CHECK: %6 = arith.addi %4, %5 : tensor<1024xi32, [[$target_layout]]>
  // CHECK: tt.return %6 : tensor<1024xi32, [[$target_layout]]>
}

// Always rematerialize single value loads
// CHECK-LABEL: remat_single_value
tt.func @remat_single_value(%arg: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %0 = tt.splat %arg : (!tt.ptr<i32>) -> tensor<1x!tt.ptr<i32>, #layout1>
  %1 = tt.load %0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  %2 = triton_gpu.convert_layout %1 : (tensor<1xi32, #layout1>) -> tensor<1xi32, #layout0>
  %3 = triton_gpu.convert_layout %0 : (tensor<1x!tt.ptr<i32>, #layout1>) -> tensor<1x!tt.ptr<i32>, #layout0>
  tt.store %3, %2 : tensor<1xi32, #layout0>
  tt.return
}

tt.func @remat_fast_load(%arg: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %0 = tt.splat %arg : (!tt.ptr<i32>) -> tensor<16x!tt.ptr<i32>, #layout1>
  %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #layout1>
  %2 = tt.addptr %0, %1 : tensor<16x!tt.ptr<i32>, #layout1>, tensor<16xi32, #layout1>
  %3 = tt.load %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16xi32, #layout1>
  // CHECK-NOT: triton_gpu.convert_layout
  %4 = triton_gpu.convert_layout %3 : (tensor<16xi32, #layout1>) -> tensor<16xi32, #layout0>
  %5 = triton_gpu.convert_layout %2 : (tensor<16x!tt.ptr<i32>, #layout1>) -> tensor<16x!tt.ptr<i32>, #layout0>
  tt.store %5, %4 : tensor<16xi32, #layout0>
  tt.return
}

// Hoist the convert on top of ext to make it cheaper.
// CHECK-LABEL: hoist_above_ext
tt.func @hoist_above_ext(%arg0: tensor<1024xf16, #layout0>, %arg1: f32) -> tensor<1024xf32, #layout1> {
// CHECK: %[[CVT:.+]] = triton_gpu.convert_layout
// CHECK: arith.extf %[[CVT]]
// CHECK-NOT: triton_gpu.convert_layout
// CHECK: tt.return
  %0 = arith.extf %arg0 : tensor<1024xf16, #layout0> to tensor<1024xf32, #layout0>
  %1 = tt.splat %arg1 : (f32) -> tensor<1024xf32, #layout0>
  %2 = arith.addf %0, %1 : tensor<1024xf32, #layout0>
  %3 = triton_gpu.convert_layout %2 : (tensor<1024xf32, #layout0>) -> tensor<1024xf32, #layout1>
  tt.return %3 : tensor<1024xf32, #layout1>
}

// CHECK-LABEL: hoist_above_ext2
tt.func @hoist_above_ext2(%arg0: tensor<1024xf16, #layout0>, %arg1: f16) -> tensor<1024xf32, #layout1> {
// CHECK: %[[CVT:.+]] = triton_gpu.convert_layout
// CHECK: arith.extf %[[CVT]]
// CHECK-NOT: triton_gpu.convert_layout
// CHECK: tt.return
  %0 = arith.extf %arg0 : tensor<1024xf16, #layout0> to tensor<1024xf32, #layout0>
  %1 = tt.splat %arg1 : (f16) -> tensor<1024xf16, #layout0>
  %2 = arith.extf %1 : tensor<1024xf16, #layout0> to tensor<1024xf32, #layout0>
  %3 = arith.addf %0, %2 : tensor<1024xf32, #layout0>
  %4 = triton_gpu.convert_layout %3 : (tensor<1024xf32, #layout0>) -> tensor<1024xf32, #layout1>
  tt.return %4 : tensor<1024xf32, #layout1>
}

// Hoist the convert on top of broadcast to make it cheaper.
// CHECK-LABEL: hoist_above_broadcast
tt.func @hoist_above_broadcast(%arg0: tensor<1024x1xf32, #layout2>, %arg1: f32) -> tensor<1024x128xf32, #layout3> {
// CHECK: %[[CVT:.+]] = triton_gpu.convert_layout
// CHECK: tt.broadcast %[[CVT]]
// CHECK-NOT: triton_gpu.convert_layout
// CHECK: tt.return
  %0 = tt.broadcast %arg0 : (tensor<1024x1xf32, #layout2>) -> tensor<1024x128xf32, #layout2>
  %1 = tt.splat %arg1 : (f32) -> tensor<1024x128xf32, #layout2>
  %2 = arith.addf %0, %1 : tensor<1024x128xf32, #layout2>
  %3 = triton_gpu.convert_layout %2 : (tensor<1024x128xf32, #layout2>) -> tensor<1024x128xf32, #layout3>
  tt.return %3 : tensor<1024x128xf32, #layout3>
}


// CHECK-LABEL: if
tt.func @if(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  // CHECK-NOT: triton_gpu.convert_layout
  %c32_i32 = arith.constant dense<32> : tensor<1024xi32, #layout1>
  %0 = tt.get_program_id x : i32
  %1 = tt.splat %0 : (i32) -> tensor<1024xi32, #layout1>
  %2 = arith.muli %1, %c32_i32 : tensor<1024xi32, #layout1>
  %3 = arith.addi %2, %c32_i32 : tensor<1024xi32, #layout1>
  %4 = arith.cmpi sgt, %0, %arg0 : i32
  %5 = tt.splat %arg1 : (!tt.ptr<i32>) -> tensor<1024x!tt.ptr<i32>, #layout0>
  scf.if %4 {
    %6 = triton_gpu.convert_layout %2 : (tensor<1024xi32, #layout1>) -> tensor<1024xi32, #layout0>
    tt.store %5, %6 : tensor<1024xi32, #layout0>
  }
  tt.return
}

// CHECK-LABEL: if_convert_else_not
tt.func @if_convert_else_not(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %c32_i32 = arith.constant dense<32> : tensor<1024xi32, #layout0>
  %0 = tt.get_program_id x : i32
  %1 = tt.splat %0 : (i32) -> tensor<1024xi32, #layout0>
  %9 = tt.splat %0 : (i32) -> tensor<1024xi32, #layout1>
  %2 = arith.muli %1, %c32_i32 : tensor<1024xi32, #layout0>
  %3 = arith.addi %2, %c32_i32 : tensor<1024xi32, #layout0>
  %4 = arith.cmpi sgt, %0, %arg0 : i32
  %5 = tt.splat %arg1 : (!tt.ptr<i32>) -> tensor<1024x!tt.ptr<i32>, #layout1>
  %8 = scf.if %4 -> tensor<1024xi32, #layout1> {
    %6 = triton_gpu.convert_layout %2 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
    scf.yield %6 : tensor<1024xi32, #layout1>
  } else {
    scf.yield %9 : tensor<1024xi32, #layout1>
  }
  // CHECK-NOT: triton_gpu.convert_layout
  tt.store %5, %8 : tensor<1024xi32, #layout1>
  tt.return
}

// CHECK-LABEL: if_not_else_convert
tt.func @if_not_else_convert(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %c32_i32 = arith.constant dense<32> : tensor<1024xi32, #layout0>
  %0 = tt.get_program_id x : i32
  %1 = tt.splat %0 : (i32) -> tensor<1024xi32, #layout0>
  %9 = tt.splat %0 : (i32) -> tensor<1024xi32, #layout1>
  %2 = arith.muli %1, %c32_i32 : tensor<1024xi32, #layout0>
  %3 = arith.addi %2, %c32_i32 : tensor<1024xi32, #layout0>
  %4 = arith.cmpi sgt, %0, %arg0 : i32
  %5 = tt.splat %arg1 : (!tt.ptr<i32>) -> tensor<1024x!tt.ptr<i32>, #layout1>
  %8 = scf.if %4 -> tensor<1024xi32, #layout1> {
    scf.yield %9 : tensor<1024xi32, #layout1>
  } else {
    %7 = triton_gpu.convert_layout %3 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
    scf.yield %7 : tensor<1024xi32, #layout1>
  }
  // CHECK-NOT: triton_gpu.convert_layout
  tt.store %5, %8 : tensor<1024xi32, #layout1>
  tt.return
}

// CHECK-LABEL: if_else_both_convert
tt.func @if_else_both_convert(%arg0: i32, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %c32_i32 = arith.constant dense<32> : tensor<1024xi32, #layout0>
  %0 = tt.get_program_id x : i32
  %1 = tt.splat %0 : (i32) -> tensor<1024xi32, #layout0>
  %2 = arith.muli %1, %c32_i32 : tensor<1024xi32, #layout0>
  %3 = arith.addi %2, %c32_i32 : tensor<1024xi32, #layout0>
  %4 = arith.cmpi sgt, %0, %arg0 : i32
  %5 = tt.splat %arg1 : (!tt.ptr<i32>) -> tensor<1024x!tt.ptr<i32>, #layout1>
  %8 = scf.if %4 -> tensor<1024xi32, #layout1> {
    %6 = triton_gpu.convert_layout %2 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
    scf.yield %6 : tensor<1024xi32, #layout1>
  } else {
    %7 = triton_gpu.convert_layout %3 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
    scf.yield %7 : tensor<1024xi32, #layout1>
  }
  // TODO(csigg): seems like the whole function is converted to layout1.
  // disabledCHECK: triton_gpu.convert_layout
  // CHECK-NOT: triton_gpu.convert_layout
  tt.store %5, %8 : tensor<1024xi32, #layout1>
  tt.return
}

}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#slice1dim1 = #triton_gpu.slice<{dim = 1, parent = #blocked1}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2a = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#slice2dim0 = #triton_gpu.slice<{dim = 0, parent = #blocked2}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>

// A mnist model from torch inductor.
// Check if topological sort is working correct and there's no unnecessary convert
// CHECK-LABEL: mnist
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {
tt.func public @mnist(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32) {
  // CHECK-NOT: triton_gpu.convert_layout
  %cst = arith.constant dense<10> : tensor<16x1xi32, #blocked2>
  %cst_0 = arith.constant dense<10> : tensor<1x16xi32, #blocked3>
  %c16_i32 = arith.constant 16 : i32
  %cst_1 = arith.constant dense<64> : tensor<16x1xi32, #blocked2>
  %cst_2 = arith.constant dense<0xFF800000> : tensor<16x16xf32, #blocked2>
  %cst_3 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked2>
  %cst_4 = arith.constant dense<0> : tensor<16x16xi32, #blocked2>
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c16_i32 : i32
  %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked0>
  %3 = triton_gpu.convert_layout %2 : (tensor<16xi32, #blocked0>) -> tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %4 = tt.expand_dims %3 {axis = 1 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<16x1xi32, #blocked1>
  %5 = triton_gpu.convert_layout %4 : (tensor<16x1xi32, #blocked1>) -> tensor<16x1xi32, #blocked2>
  %6 = tt.splat %1 : (i32) -> tensor<16x1xi32, #blocked2>
  %7 = arith.addi %6, %5 : tensor<16x1xi32, #blocked2>
  %8 = arith.cmpi "slt", %7, %cst_1 : tensor<16x1xi32, #blocked2>
  %9 = triton_gpu.convert_layout %2 : (tensor<16xi32, #blocked0>) -> tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
  %10 = tt.expand_dims %9 {axis = 0 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x16xi32, #blocked3>
  %11 = arith.cmpi "slt", %10, %cst_0 : tensor<1x16xi32, #blocked3>
  %12 = arith.muli %7, %cst : tensor<16x1xi32, #blocked2>
  %13 = tt.broadcast %10 : (tensor<1x16xi32, #blocked3>) -> tensor<16x16xi32, #blocked3>
  %14 = triton_gpu.convert_layout %13 : (tensor<16x16xi32, #blocked3>) -> tensor<16x16xi32, #blocked2>
  %15 = tt.broadcast %12 : (tensor<16x1xi32, #blocked2>) -> tensor<16x16xi32, #blocked2>
  %16 = arith.addi %14, %15 : tensor<16x16xi32, #blocked2>
  %17 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<16x16x!tt.ptr<f32>, #blocked2>
  %18 = tt.addptr %17, %16 : tensor<16x16x!tt.ptr<f32>, #blocked2>, tensor<16x16xi32, #blocked2>
  %19 = tt.broadcast %11 : (tensor<1x16xi1, #blocked3>) -> tensor<16x16xi1, #blocked3>
  %20 = triton_gpu.convert_layout %19 : (tensor<16x16xi1, #blocked3>) -> tensor<16x16xi1, #blocked2>
  %21 = tt.broadcast %8 : (tensor<16x1xi1, #blocked2>) -> tensor<16x16xi1, #blocked2>
  %22 = arith.andi %20, %21 : tensor<16x16xi1, #blocked2>
  %23 = triton_gpu.convert_layout %18 : (tensor<16x16x!tt.ptr<f32>, #blocked2>) -> tensor<16x16x!tt.ptr<f32>, #blocked4>
  %24 = triton_gpu.convert_layout %22 : (tensor<16x16xi1, #blocked2>) -> tensor<16x16xi1, #blocked4>
  %25 = tt.load %23, %24 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<16x16xf32, #blocked4>
  %26 = triton_gpu.convert_layout %25 : (tensor<16x16xf32, #blocked4>) -> tensor<16x16xf32, #blocked2>
  %27 = arith.cmpf "olt", %cst_2, %26 : tensor<16x16xf32, #blocked2>
  %28 = arith.andi %22, %27 : tensor<16x16xi1, #blocked2>
  %29 = arith.select %28, %26, %cst_2 : tensor<16x16xi1, #blocked2>, tensor<16x16xf32, #blocked2>
  %30 = "tt.reduce" (%29) ({
  ^bb0(%arg4: f32, %arg5: f32):
    %max = arith.maximumf %arg4, %arg5 : f32
    tt.reduce.return %max : f32
  }) {axis = 1 : i32} : (tensor<16x16xf32, #blocked2>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
  %31 = triton_gpu.convert_layout %30 : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<16xf32, #blocked0>
  %32 = triton_gpu.convert_layout %31 : (tensor<16xf32, #blocked0>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %33 = tt.expand_dims %32 {axis = 1 : i32} : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<16x1xf32, #blocked1>
  %34 = triton_gpu.convert_layout %33 : (tensor<16x1xf32, #blocked1>) -> tensor<16x1xf32, #blocked2>
  %35 = arith.sitofp %cst_4 : tensor<16x16xi32, #blocked2> to tensor<16x16xf32, #blocked2>
  %36 = arith.addf %35, %cst_3 : tensor<16x16xf32, #blocked2>
  %37 = triton_gpu.convert_layout %18 : (tensor<16x16x!tt.ptr<f32>, #blocked2>) -> tensor<16x16x!tt.ptr<f32>, #blocked4>
  %38 = triton_gpu.convert_layout %22 : (tensor<16x16xi1, #blocked2>) -> tensor<16x16xi1, #blocked4>
  %39 = tt.load %37, %38 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<16x16xf32, #blocked4>
  %40 = triton_gpu.convert_layout %39 : (tensor<16x16xf32, #blocked4>) -> tensor<16x16xf32, #blocked2>
  %41 = tt.broadcast %34 : (tensor<16x1xf32, #blocked2>) -> tensor<16x16xf32, #blocked2>
  %42 = arith.subf %40, %41 : tensor<16x16xf32, #blocked2>
  %43 = math.exp %42 : tensor<16x16xf32, #blocked2>
  %44 = arith.addf %36, %43 : tensor<16x16xf32, #blocked2>
  %45 = arith.select %22, %44, %36 : tensor<16x16xi1, #blocked2>, tensor<16x16xf32, #blocked2>
  %46 = "tt.reduce" (%45) ({
  ^bb0(%arg4: f32, %arg5: f32):
    %add = arith.addf %arg4, %arg5 : f32
    tt.reduce.return %add : f32
  }) {axis = 1 : i32} : (tensor<16x16xf32, #blocked2>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
  %47 = triton_gpu.convert_layout %46 : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<16xf32, #blocked0>
  %48 = triton_gpu.convert_layout %47 : (tensor<16xf32, #blocked0>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %49 = tt.expand_dims %48 {axis = 1 : i32} : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<16x1xf32, #blocked1>
  %50 = triton_gpu.convert_layout %49 : (tensor<16x1xf32, #blocked1>) -> tensor<16x1xf32, #blocked2>
  %51 = triton_gpu.convert_layout %18 : (tensor<16x16x!tt.ptr<f32>, #blocked2>) -> tensor<16x16x!tt.ptr<f32>, #blocked4>
  %52 = triton_gpu.convert_layout %22 : (tensor<16x16xi1, #blocked2>) -> tensor<16x16xi1, #blocked4>
  %53 = tt.load %51, %52 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<16x16xf32, #blocked4>
  %54 = triton_gpu.convert_layout %53 : (tensor<16x16xf32, #blocked4>) -> tensor<16x16xf32, #blocked2>
  %55 = arith.subf %54, %41 : tensor<16x16xf32, #blocked2>
  %56 = math.log %50 : tensor<16x1xf32, #blocked2>
  %57 = tt.broadcast %56 : (tensor<16x1xf32, #blocked2>) -> tensor<16x16xf32, #blocked2>
  %58 = arith.subf %55, %57 : tensor<16x16xf32, #blocked2>
  %59 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<16x16x!tt.ptr<f32>, #blocked2>
  %60 = tt.addptr %59, %16 : tensor<16x16x!tt.ptr<f32>, #blocked2>, tensor<16x16xi32, #blocked2>
  %61 = triton_gpu.convert_layout %60 : (tensor<16x16x!tt.ptr<f32>, #blocked2>) -> tensor<16x16x!tt.ptr<f32>, #blocked4>
  %62 = triton_gpu.convert_layout %58 : (tensor<16x16xf32, #blocked2>) -> tensor<16x16xf32, #blocked4>
  %63 = triton_gpu.convert_layout %22 : (tensor<16x16xi1, #blocked2>) -> tensor<16x16xi1, #blocked4>
  tt.store %61, %62, %63 : tensor<16x16xf32, #blocked4>
  tt.return
}
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
// cmpf and cmpi have different operands and result types
// CHECK-LABEL: cmp
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.num-ctas" = 1 : i32} {
tt.func public @cmp(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) {
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %c0 = arith.constant 0 : index
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant dense<-3.40282347E+38> : tensor<64x64xf32, #blocked2>
  %cst_0 = arith.constant dense<4194304> : tensor<64x1xi32, #blocked2>
  %cst_1 = arith.constant dense<12> : tensor<64x1xi32, #blocked2>
  %cst_2 = arith.constant dense<2048> : tensor<1x64xi32, #blocked3>
  %cst_3 = arith.constant dense<0> : tensor<64x64xi32, #blocked2>
  %cst_4 = arith.constant dense<2048> : tensor<64x1xi32, #blocked2>
  %cst_5 = arith.constant dense<49152> : tensor<64x1xi32, #blocked2>
  %cst_6 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked2>
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked0>
  %3 = triton_gpu.convert_layout %2 : (tensor<64xi32, #blocked0>) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %4 = tt.expand_dims %3 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<64x1xi32, #blocked1>
  %5 = triton_gpu.convert_layout %4 : (tensor<64x1xi32, #blocked1>) -> tensor<64x1xi32, #blocked2>
  %6 = tt.splat %1 : (i32) -> tensor<64x1xi32, #blocked2>
  %7 = arith.addi %6, %5 : tensor<64x1xi32, #blocked2>
  %8 = arith.cmpi "slt", %7, %cst_5 : tensor<64x1xi32, #blocked2>
  %9 = triton_gpu.convert_layout %2 : (tensor<64xi32, #blocked0>) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
  %10 = tt.expand_dims %9 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x64xi32, #blocked3>
  %11 = arith.remsi %7, %cst_4 : tensor<64x1xi32, #blocked2>
  %12 = arith.divsi %7, %cst_4 : tensor<64x1xi32, #blocked2>
  %13 = arith.sitofp %cst_3 : tensor<64x64xi32, #blocked2> to tensor<64x64xf32, #blocked2>
  %14 = arith.addf %13, %cst_6 : tensor<64x64xf32, #blocked2>
  %15 = arith.muli %7, %cst_4 : tensor<64x1xi32, #blocked2>
  %16 = tt.broadcast %15 : (tensor<64x1xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %17 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<64x64x!tt.ptr<f16>, #blocked2>
  %18 = tt.broadcast %8 : (tensor<64x1xi1, #blocked2>) -> tensor<64x64xi1, #blocked2>
  %19 = arith.muli %11, %cst_4 : tensor<64x1xi32, #blocked2>
  %20 = tt.broadcast %19 : (tensor<64x1xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %21 = arith.divsi %12, %cst_1 : tensor<64x1xi32, #blocked2>
  %22 = arith.muli %21, %cst_0 : tensor<64x1xi32, #blocked2>
  %23 = tt.broadcast %22 : (tensor<64x1xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %24 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<64x64x!tt.ptr<f32>, #blocked2>
  %25 = scf.for %arg6 = %c0 to %c2048 step %c64 iter_args(%arg7 = %14) -> (tensor<64x64xf32, #blocked2>) {
    %44 = arith.index_cast %arg6 : index to i32
    %45 = tt.splat %44 : (i32) -> tensor<1x64xi32, #blocked3>
    %46 = arith.addi %45, %10 : tensor<1x64xi32, #blocked3>
    %47 = arith.cmpi "slt", %46, %cst_2 : tensor<1x64xi32, #blocked3>
    %48 = tt.broadcast %46 : (tensor<1x64xi32, #blocked3>) -> tensor<64x64xi32, #blocked3>
    %49 = triton_gpu.convert_layout %48 : (tensor<64x64xi32, #blocked3>) -> tensor<64x64xi32, #blocked2>
    %50 = arith.addi %49, %16 : tensor<64x64xi32, #blocked2>
    %51 = tt.addptr %17, %50 : tensor<64x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xi32, #blocked2>
    %52 = tt.broadcast %47 : (tensor<1x64xi1, #blocked3>) -> tensor<64x64xi1, #blocked3>
    %53 = triton_gpu.convert_layout %52 : (tensor<64x64xi1, #blocked3>) -> tensor<64x64xi1, #blocked2>
    %54 = arith.andi %53, %18 : tensor<64x64xi1, #blocked2>
    %55 = triton_gpu.convert_layout %51 : (tensor<64x64x!tt.ptr<f16>, #blocked2>) -> tensor<64x64x!tt.ptr<f16>, #blocked4>
    %56 = triton_gpu.convert_layout %54 : (tensor<64x64xi1, #blocked2>) -> tensor<64x64xi1, #blocked4>
    %57 = tt.load %55, %56 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<64x64xf16, #blocked4>
    %58 = triton_gpu.convert_layout %57 : (tensor<64x64xf16, #blocked4>) -> tensor<64x64xf16, #blocked2>
    %59 = arith.extf %58 : tensor<64x64xf16, #blocked2> to tensor<64x64xf32, #blocked2>
    %60 = arith.addi %49, %20 : tensor<64x64xi32, #blocked2>
    %61 = arith.addi %60, %23 : tensor<64x64xi32, #blocked2>
    %62 = tt.addptr %24, %61 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %63 = triton_gpu.convert_layout %62 : (tensor<64x64x!tt.ptr<f32>, #blocked2>) -> tensor<64x64x!tt.ptr<f32>, #blocked5>
    %64 = triton_gpu.convert_layout %54 : (tensor<64x64xi1, #blocked2>) -> tensor<64x64xi1, #blocked5>
    %65 = tt.load %63, %64 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<64x64xf32, #blocked5>
    %66 = triton_gpu.convert_layout %65 : (tensor<64x64xf32, #blocked5>) -> tensor<64x64xf32, #blocked2>
    %67 = arith.addf %59, %66 : tensor<64x64xf32, #blocked2>
    %68 = arith.cmpf "une", %67, %67 : tensor<64x64xf32, #blocked2>
    %69 = arith.cmpf "ogt", %67, %cst : tensor<64x64xf32, #blocked2>
    %70 = arith.select %69, %67, %cst : tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>
    %71 = arith.select %68, %67, %70 : tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>
    %72 = math.exp %71 : tensor<64x64xf32, #blocked2>
    %73 = arith.addf %arg7, %72 : tensor<64x64xf32, #blocked2>
    %74 = arith.select %54, %73, %arg7 : tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>
    scf.yield %74 : tensor<64x64xf32, #blocked2>
  }
  %26 = "tt.reduce" (%25) ({
  ^bb0(%arg8: f32, %arg9: f32):
    %add = arith.addf %arg8, %arg9 : f32
    tt.reduce.return %add : f32
  }) {axis = 1 : i32} : (tensor<64x64xf32, #blocked2>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
  %27 = triton_gpu.convert_layout %26 : (tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64xf32, #blocked0>
  %28 = triton_gpu.convert_layout %27 : (tensor<64xf32, #blocked0>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %29 = tt.expand_dims %28 {axis = 1 : i32} : (tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<64x1xf32, #blocked1>
  %30 = triton_gpu.convert_layout %29 : (tensor<64x1xf32, #blocked1>) -> tensor<64x1xf32, #blocked2>
  %31 = arith.muli %7, %cst_4 : tensor<64x1xi32, #blocked2>
  %32 = tt.broadcast %31 : (tensor<64x1xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %33 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<64x64x!tt.ptr<f16>, #blocked2>
  %34 = tt.broadcast %8 : (tensor<64x1xi1, #blocked2>) -> tensor<64x64xi1, #blocked2>
  %35 = arith.muli %11, %cst_4 : tensor<64x1xi32, #blocked2>
  %36 = tt.broadcast %35 : (tensor<64x1xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %37 = arith.divsi %12, %cst_1 : tensor<64x1xi32, #blocked2>
  %38 = arith.muli %37, %cst_0 : tensor<64x1xi32, #blocked2>
  %39 = tt.broadcast %38 : (tensor<64x1xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
  %40 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<64x64x!tt.ptr<f32>, #blocked2>
  %41 = tt.broadcast %30 : (tensor<64x1xf32, #blocked2>) -> tensor<64x64xf32, #blocked2>
  %42 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x64x!tt.ptr<f32>, #blocked2>
  %43 = tt.splat %arg3 : (!tt.ptr<f16>) -> tensor<64x64x!tt.ptr<f16>, #blocked2>
  scf.for %arg6 = %c0 to %c2048 step %c64 {
    %44 = arith.index_cast %arg6 : index to i32
    %45 = tt.splat %44 : (i32) -> tensor<1x64xi32, #blocked3>
    %46 = arith.addi %45, %10 : tensor<1x64xi32, #blocked3>
    %47 = arith.cmpi "slt", %46, %cst_2 : tensor<1x64xi32, #blocked3>
    %48 = tt.broadcast %46 : (tensor<1x64xi32, #blocked3>) -> tensor<64x64xi32, #blocked3>
    %49 = triton_gpu.convert_layout %48 : (tensor<64x64xi32, #blocked3>) -> tensor<64x64xi32, #blocked2>
    %50 = arith.addi %49, %32 : tensor<64x64xi32, #blocked2>
    %51 = tt.addptr %33, %50 : tensor<64x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xi32, #blocked2>
    %52 = tt.broadcast %47 : (tensor<1x64xi1, #blocked3>) -> tensor<64x64xi1, #blocked3>
    %53 = triton_gpu.convert_layout %52 : (tensor<64x64xi1, #blocked3>) -> tensor<64x64xi1, #blocked2>
    %54 = arith.andi %53, %34 : tensor<64x64xi1, #blocked2>
    %55 = triton_gpu.convert_layout %51 : (tensor<64x64x!tt.ptr<f16>, #blocked2>) -> tensor<64x64x!tt.ptr<f16>, #blocked4>
    %56 = triton_gpu.convert_layout %54 : (tensor<64x64xi1, #blocked2>) -> tensor<64x64xi1, #blocked4>
    %57 = tt.load %55, %56 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<64x64xf16, #blocked4>
    %58 = triton_gpu.convert_layout %57 : (tensor<64x64xf16, #blocked4>) -> tensor<64x64xf16, #blocked2>
    %59 = arith.extf %58 : tensor<64x64xf16, #blocked2> to tensor<64x64xf32, #blocked2>
    %60 = arith.addi %49, %36 : tensor<64x64xi32, #blocked2>
    %61 = arith.addi %60, %39 : tensor<64x64xi32, #blocked2>
    %62 = tt.addptr %40, %61 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %63 = triton_gpu.convert_layout %62 : (tensor<64x64x!tt.ptr<f32>, #blocked2>) -> tensor<64x64x!tt.ptr<f32>, #blocked5>
    %64 = triton_gpu.convert_layout %54 : (tensor<64x64xi1, #blocked2>) -> tensor<64x64xi1, #blocked5>
    %65 = tt.load %63, %64 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<64x64xf32, #blocked5>
    %66 = triton_gpu.convert_layout %65 : (tensor<64x64xf32, #blocked5>) -> tensor<64x64xf32, #blocked2>
    %67 = arith.addf %59, %66 : tensor<64x64xf32, #blocked2>
    %68 = arith.cmpf "une", %67, %67 : tensor<64x64xf32, #blocked2>
    %69 = arith.cmpf "ogt", %67, %cst : tensor<64x64xf32, #blocked2>
    %70 = arith.select %69, %67, %cst : tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>
    %71 = arith.select %68, %67, %70 : tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2>
    %72 = math.exp %71 : tensor<64x64xf32, #blocked2>
    %73 = arith.divf %72, %41 : tensor<64x64xf32, #blocked2>
    %74 = tt.addptr %42, %50 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %75 = triton_gpu.convert_layout %74 : (tensor<64x64x!tt.ptr<f32>, #blocked2>) -> tensor<64x64x!tt.ptr<f32>, #blocked5>
    %76 = triton_gpu.convert_layout %73 : (tensor<64x64xf32, #blocked2>) -> tensor<64x64xf32, #blocked5>
    %77 = triton_gpu.convert_layout %54 : (tensor<64x64xi1, #blocked2>) -> tensor<64x64xi1, #blocked5>
    tt.store %75, %76, %77 : tensor<64x64xf32, #blocked5>
    %78 = tt.addptr %43, %50 : tensor<64x64x!tt.ptr<f16>, #blocked2>, tensor<64x64xi32, #blocked2>
    %79 = arith.truncf %73 : tensor<64x64xf32, #blocked2> to tensor<64x64xf16, #blocked2>
    %80 = triton_gpu.convert_layout %78 : (tensor<64x64x!tt.ptr<f16>, #blocked2>) -> tensor<64x64x!tt.ptr<f16>, #blocked4>
    %81 = triton_gpu.convert_layout %79 : (tensor<64x64xf16, #blocked2>) -> tensor<64x64xf16, #blocked4>
    %82 = triton_gpu.convert_layout %54 : (tensor<64x64xi1, #blocked2>) -> tensor<64x64xi1, #blocked4>
    tt.store %80, %81, %82 : tensor<64x64xf16, #blocked4>
  }
  tt.return
}
}

// -----

// Just make sure it doesn't crash on non-tensor types.
// CHECK-LABEL: if_no_tensor
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {
tt.func public @if_no_tensor(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}) {
  // CHECK-NOT: triton_gpu.convert_layout
  %c-1_i64 = arith.constant -1 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c-1_i32 = arith.constant -1 : i32
  %0 = tt.get_program_id x : i32
  %1 = tt.addptr %arg3, %0 : !tt.ptr<i64>, i32
  %2 = tt.load %1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i64
  %3 = arith.cmpi eq, %2, %c-1_i64 : i64
  %4 = arith.select %3, %c-1_i32, %arg2 : i32
  %5 = scf.if %3 -> (!tt.ptr<f32>) {
    scf.yield %arg0 : !tt.ptr<f32>
  } else {
    %10 = tt.addptr %arg0, %2 : !tt.ptr<f32>, i64
    scf.yield %10 : !tt.ptr<f32>
  }
  %6 = arith.extsi %4 : i32 to i64
  %7 = arith.cmpi slt, %2, %6 : i64
  %8 = tt.load %5, %7, %cst {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
  %9 = tt.addptr %arg1, %0 : !tt.ptr<f32>, i32
  tt.store %9, %8 {cache = 1 : i32, evict = 1 : i32} : f32
  tt.return
}
}

// -----

// Check if the SimplifyReduceCvt rewriter pattern doesn't hang.
// CHECK-LABEL: reduce_cvt
// CHECK-NOT: triton_gpu.convert_layout
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [2, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 2 : i32, "triton_gpu.num-ctas" = 1 : i32} {
  tt.func public @reduce_cvt1(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32) {
    %cst = arith.constant dense<0> : tensor<1x2xi32, #blocked>
    %cst_0 = arith.constant dense<2> : tensor<1x2xi32, #blocked>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #blocked1>
    %1 = triton_gpu.convert_layout %0 : (tensor<2xi32, #blocked1>) -> tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<2xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x2xi32, #blocked>
    %3 = arith.cmpi "slt", %2, %cst_0 : tensor<1x2xi32, #blocked>
    %4 = "tt.reduce" (%cst) ({
    ^bb0(%arg3: i32, %arg4: i32):
      %add = arith.addi %arg3, %arg4 : i32
      tt.reduce.return %add : i32
    }) {axis = 1 : i32} : (tensor<1x2xi32, #blocked>) -> tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %5 = triton_gpu.convert_layout %4 : (tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<1xi32, #blocked1>
    %6 = triton_gpu.convert_layout %5 : (tensor<1xi32, #blocked1>) -> tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : (tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<1x1xi32, #blocked2>
    %8 = triton_gpu.convert_layout %7 : (tensor<1x1xi32, #blocked2>) -> tensor<1x1xi32, #blocked>
    %9 = tt.splat %arg0 : (!tt.ptr<i64>) -> tensor<1x2x!tt.ptr<i64>, #blocked>
    %10 = tt.addptr %9, %2 : tensor<1x2x!tt.ptr<i64>, #blocked>, tensor<1x2xi32, #blocked>
    %11 = tt.broadcast %8 : (tensor<1x1xi32, #blocked>) -> tensor<1x2xi32, #blocked>
    %12 = arith.extsi %11 : tensor<1x2xi32, #blocked> to tensor<1x2xi64, #blocked>
    %13 = triton_gpu.convert_layout %10 : (tensor<1x2x!tt.ptr<i64>, #blocked>) -> tensor<1x2x!tt.ptr<i64>, #blocked3>
    %14 = triton_gpu.convert_layout %12 : (tensor<1x2xi64, #blocked>) -> tensor<1x2xi64, #blocked3>
    %15 = triton_gpu.convert_layout %3 : (tensor<1x2xi1, #blocked>) -> tensor<1x2xi1, #blocked3>
    tt.store %13, %14, %15 {cache = 1 : i32, evict = 1 : i32} : tensor<1x2xi64, #blocked3>
    tt.return
  }
}

// -----

// CHECK-LABEL: reduce_cvt2
// Match the reduction
// CHECK-NOT: triton_gpu.convert_layout
// CHECK: tt.reduce
// CHECK-SAME: axis = 1
// CHECK: (tensor<1x256xf32, #{{.*}}>) -> tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #{{.*}}}>>
// CHECK: triton_gpu.convert_layout
// CHECK: tt.expand_dims
// CHECK-NOT: triton_gpu.convert_layout
// CHECK: tt.return
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {
  tt.func public @reduce_cvt2(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x256xf32, #blocked>
    %c3136_i32 = arith.constant 3136 : index
    %c256_i32 = arith.constant 256 : index
    %c0_i32 = arith.constant 0 : index
    %cst_0 = arith.constant dense<3.136000e+03> : tensor<1x1xf32, #blocked>
    %cst_1 = arith.constant dense<50176> : tensor<1x256xi32, #blocked>
    %cst_2 = arith.constant dense<196> : tensor<1x1xi32, #blocked>
    %cst_3 = arith.constant dense<196> : tensor<1x256xi32, #blocked>
    %cst_4 = arith.constant dense<3136> : tensor<1x256xi32, #blocked>
    %cst_5 = arith.constant dense<256> : tensor<1x1xi32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #blocked1>
    %2 = triton_gpu.convert_layout %1 : (tensor<1xi32, #blocked1>) -> tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<1x1xi32, #blocked2>
    %4 = triton_gpu.convert_layout %3 : (tensor<1x1xi32, #blocked2>) -> tensor<1x1xi32, #blocked>
    %5 = tt.splat %0 : (i32) -> tensor<1x1xi32, #blocked>
    %6 = arith.addi %5, %4 : tensor<1x1xi32, #blocked>
    %7 = arith.cmpi "slt", %6, %cst_5 : tensor<1x1xi32, #blocked>
    %8 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %9 = triton_gpu.convert_layout %8 : (tensor<256xi32, #blocked1>) -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %10 = tt.expand_dims %9 {axis = 0 : i32} : (tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x256xi32, #blocked>
    %11 = arith.muli %6, %cst_2 : tensor<1x1xi32, #blocked>
    %12 = tt.broadcast %11 : (tensor<1x1xi32, #blocked>) -> tensor<1x256xi32, #blocked>
    %13 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<1x256x!tt.ptr<f32>, #blocked>
    %14 = tt.broadcast %7 : (tensor<1x1xi1, #blocked>) -> tensor<1x256xi1, #blocked>
    %15 = scf.for %arg5 = %c0_i32 to %c3136_i32 step %c256_i32 iter_args(%arg6 = %cst) -> (tensor<1x256xf32, #blocked>) {
      %42 = arith.index_cast %arg5 : index to i32
      %43 = tt.splat %42 : (i32) -> tensor<1x256xi32, #blocked>
      %44 = arith.addi %43, %10 : tensor<1x256xi32, #blocked>
      %45 = arith.cmpi "slt", %44, %cst_4 : tensor<1x256xi32, #blocked>
      %46 = arith.remsi %44, %cst_3 : tensor<1x256xi32, #blocked>
      %47 = arith.divsi %44, %cst_3 : tensor<1x256xi32, #blocked>
      %48 = arith.addi %46, %12 : tensor<1x256xi32, #blocked>
      %49 = arith.muli %47, %cst_1 : tensor<1x256xi32, #blocked>
      %50 = arith.addi %48, %49 : tensor<1x256xi32, #blocked>
      %51 = tt.addptr %13, %50 : tensor<1x256x!tt.ptr<f32>, #blocked>, tensor<1x256xi32, #blocked>
      %52 = arith.andi %45, %14 : tensor<1x256xi1, #blocked>
      %53 = triton_gpu.convert_layout %51 : (tensor<1x256x!tt.ptr<f32>, #blocked>) -> tensor<1x256x!tt.ptr<f32>, #blocked3>
      %54 = triton_gpu.convert_layout %52 : (tensor<1x256xi1, #blocked>) -> tensor<1x256xi1, #blocked3>
      %55 = triton_gpu.convert_layout %cst : (tensor<1x256xf32, #blocked>) -> tensor<1x256xf32, #blocked3>
      %56 = tt.load %53, %54, %55 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<1x256xf32, #blocked3>
      %57 = triton_gpu.convert_layout %56 : (tensor<1x256xf32, #blocked3>) -> tensor<1x256xf32, #blocked>
      %58 = arith.addf %arg6, %57 : tensor<1x256xf32, #blocked>
      %59 = arith.select %52, %58, %arg6 : tensor<1x256xi1, #blocked>, tensor<1x256xf32, #blocked>
      scf.yield %59 : tensor<1x256xf32, #blocked>
    }
    %16 = "tt.reduce" (%15) ({
    ^bb0(%arg7: f32, %arg8: f32):
      %add = arith.addf %arg7, %arg8 : f32
      tt.reduce.return %add : f32

    }) {axis = 1 : i32} : (tensor<1x256xf32, #blocked>) -> tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %17 = triton_gpu.convert_layout %16 : (tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<1xf32, #blocked1>
    %18 = triton_gpu.convert_layout %17 : (tensor<1xf32, #blocked1>) -> tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %19 = tt.expand_dims %18 {axis = 1 : i32} : (tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<1x1xf32, #blocked2>
    %20 = triton_gpu.convert_layout %19 : (tensor<1x1xf32, #blocked2>) -> tensor<1x1xf32, #blocked>
    %21 = arith.divf %20, %cst_0 : tensor<1x1xf32, #blocked>
    %22 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<1x1x!tt.ptr<f32>, #blocked>
    %23 = tt.addptr %22, %6 : tensor<1x1x!tt.ptr<f32>, #blocked>, tensor<1x1xi32, #blocked>
    %24 = triton_gpu.convert_layout %23 : (tensor<1x1x!tt.ptr<f32>, #blocked>) -> tensor<1x1x!tt.ptr<f32>, #blocked>
    %25 = triton_gpu.convert_layout %21 : (tensor<1x1xf32, #blocked>) -> tensor<1x1xf32, #blocked>
    %26 = triton_gpu.convert_layout %7 : (tensor<1x1xi1, #blocked>) -> tensor<1x1xi1, #blocked>
    tt.store %24, %25, %26 {cache = 1 : i32, evict = 1 : i32} : tensor<1x1xf32, #blocked>
    tt.return
  }
}

// -----

// Ensure that RematerializeForward doesn't apply when a convert has multiple uses
// CHECK-LABEL: loop_convert_multi_uses
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32, "triton_gpu.num-ctas" = 1 : i32} {
  tt.func public @loop_convert_multi_uses(%arg0: i32 {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<16xf32, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16xf32, #blocked>
    %cst_1 = arith.constant dense<1> : tensor<16xi32, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked1>
    %cst_3 = arith.constant dense<1> : tensor<16x1xi32, #blocked1>
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %arg0 : i32
    %3 = arith.remsi %1, %arg0 : i32
    %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %5 = arith.muli %0, %c16_i32 : i32
    %6 = tt.splat %5 : (i32) -> tensor<16xi32, #blocked>
    %7 = arith.addi %6, %4 : tensor<16xi32, #blocked>
    %8 = arith.muli %2, %arg3 : i32
    %9 = arith.muli %3, %arg4 : i32
    %10 = arith.addi %8, %9 : i32
    %11 = triton_gpu.convert_layout %7 : (tensor<16xi32, #blocked>) -> tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %12 = tt.expand_dims %11 {axis = 1 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<16x1xi32, #blocked2>
    %13 = triton_gpu.convert_layout %12 : (tensor<16x1xi32, #blocked2>) -> tensor<16x1xi32, #blocked1>
    %14 = tt.splat %arg6 : (i32) -> tensor<16x1xi32, #blocked1>
    %15 = arith.muli %13, %14 : tensor<16x1xi32, #blocked1>
    %16 = triton_gpu.convert_layout %4 : (tensor<16xi32, #blocked>) -> tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %17 = tt.expand_dims %16 {axis = 0 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x16xi32, #blocked3>
    %18 = tt.broadcast %15 : (tensor<16x1xi32, #blocked1>) -> tensor<16x16xi32, #blocked1>
    %19 = tt.broadcast %17 : (tensor<1x16xi32, #blocked3>) -> tensor<16x16xi32, #blocked3>
    %20 = triton_gpu.convert_layout %19 : (tensor<16x16xi32, #blocked3>) -> tensor<16x16xi32, #blocked1>
    %21 = arith.addi %18, %20 : tensor<16x16xi32, #blocked1>
    %22 = tt.splat %arg2 : (!tt.ptr<f16>) -> tensor<16x16x!tt.ptr<f16>, #blocked1>
    %23 = arith.cmpi "slt", %13, %cst_3 : tensor<16x1xi32, #blocked1>
    %24 = tt.broadcast %23 : (tensor<16x1xi1, #blocked1>) -> tensor<16x16xi1, #blocked1>
    %25 = arith.truncf %cst_2 : tensor<16x16xf32, #blocked1> to tensor<16x16xf16, #blocked1>
    %26 = arith.muli %2, %arg11 : i32
    %27 = arith.muli %3, %arg12 : i32
    %28 = arith.addi %26, %27 : i32
    %29 = tt.splat %arg10 : (!tt.ptr<f32>) -> tensor<16x!tt.ptr<f32>, #blocked>
    %30 = arith.cmpi "slt", %7, %cst_1 : tensor<16xi32, #blocked>
    %31 = arith.muli %2, %arg8 : i32
    %32 = arith.muli %3, %arg9 : i32
    %33 = arith.addi %31, %32 : i32
    %34 = tt.splat %arg7 : (!tt.ptr<f32>) -> tensor<16x!tt.ptr<f32>, #blocked>
    %35:3 = scf.for %arg17 = %c0_i32 to %arg1 step %c1_i32 iter_args(%arg18 = %cst_2, %arg19 = %cst_0, %arg20 = %cst) -> (tensor<16x16xf32, #blocked1>, tensor<16xf32, #blocked>, tensor<16xf32, #blocked>)  : i32 {
      %60 = arith.muli %arg17, %arg5 : i32
      %61 = arith.addi %10, %60 : i32
      %62 = tt.splat %61 : (i32) -> tensor<16x16xi32, #blocked1>
      %63 = arith.addi %62, %21 : tensor<16x16xi32, #blocked1>
      %64 = tt.addptr %22, %63 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
      %65 = triton_gpu.convert_layout %64 : (tensor<16x16x!tt.ptr<f16>, #blocked1>) -> tensor<16x16x!tt.ptr<f16>, #blocked4>
      %66 = triton_gpu.convert_layout %24 : (tensor<16x16xi1, #blocked1>) -> tensor<16x16xi1, #blocked4>
      %67 = triton_gpu.convert_layout %25 : (tensor<16x16xf16, #blocked1>) -> tensor<16x16xf16, #blocked4>
      %68 = tt.load %65, %66, %67 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #blocked4>
      %69 = triton_gpu.convert_layout %68 : (tensor<16x16xf16, #blocked4>) -> tensor<16x16xf16, #blocked1>
      %70 = arith.addi %28, %arg17 : i32
      %71 = tt.splat %70 : (i32) -> tensor<16xi32, #blocked>
      %72 = arith.addi %71, %7 : tensor<16xi32, #blocked>
      %73 = tt.addptr %29, %72 : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
      %74 = triton_gpu.convert_layout %73 : (tensor<16x!tt.ptr<f32>, #blocked>) -> tensor<16x!tt.ptr<f32>, #blocked>
      %75 = triton_gpu.convert_layout %30 : (tensor<16xi1, #blocked>) -> tensor<16xi1, #blocked>
      %76 = triton_gpu.convert_layout %cst_0 : (tensor<16xf32, #blocked>) -> tensor<16xf32, #blocked>
      %77 = tt.load %74, %75, %76 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16xf32, #blocked>
      %78 = arith.addi %33, %arg17 : i32
      %79 = tt.splat %78 : (i32) -> tensor<16xi32, #blocked>
      %80 = arith.addi %79, %7 : tensor<16xi32, #blocked>
      %81 = tt.addptr %34, %80 : tensor<16x!tt.ptr<f32>, #blocked>, tensor<16xi32, #blocked>
      %82 = triton_gpu.convert_layout %81 : (tensor<16x!tt.ptr<f32>, #blocked>) -> tensor<16x!tt.ptr<f32>, #blocked>
      %83 = triton_gpu.convert_layout %30 : (tensor<16xi1, #blocked>) -> tensor<16xi1, #blocked>
      %84 = triton_gpu.convert_layout %cst_0 : (tensor<16xf32, #blocked>) -> tensor<16xf32, #blocked>
      %85 = tt.load %82, %83, %84 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16xf32, #blocked>
      %86 = arith.cmpf "ogt", %arg20, %85 : tensor<16xf32, #blocked>
      %87 = arith.select %86, %arg20, %85 : tensor<16xi1, #blocked>, tensor<16xf32, #blocked>
      %88 = arith.subf %arg20, %87 : tensor<16xf32, #blocked>
      %89 = math.exp %88 : tensor<16xf32, #blocked>
      %90 = arith.subf %85, %87 : tensor<16xf32, #blocked>
      %91 = math.exp %90 : tensor<16xf32, #blocked>
      %92 = arith.mulf %89, %arg19 : tensor<16xf32, #blocked>
      %93 = arith.mulf %91, %77 : tensor<16xf32, #blocked>
      %94 = arith.addf %92, %93 : tensor<16xf32, #blocked>
      %95 = arith.divf %91, %94 : tensor<16xf32, #blocked>
      %96 = arith.divf %arg19, %94 : tensor<16xf32, #blocked>
      %97 = arith.mulf %96, %89 : tensor<16xf32, #blocked>
      %98 = triton_gpu.convert_layout %97 : (tensor<16xf32, #blocked>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %99 = tt.expand_dims %98 {axis = 1 : i32} : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<16x1xf32, #blocked2>
      %100 = triton_gpu.convert_layout %99 : (tensor<16x1xf32, #blocked2>) -> tensor<16x1xf32, #blocked1>
      %101 = tt.broadcast %100 : (tensor<16x1xf32, #blocked1>) -> tensor<16x16xf32, #blocked1>
      %102 = arith.mulf %arg18, %101 : tensor<16x16xf32, #blocked1>
      %103 = triton_gpu.convert_layout %95 : (tensor<16xf32, #blocked>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %104 = tt.expand_dims %103 {axis = 1 : i32} : (tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<16x1xf32, #blocked2>
      %105 = triton_gpu.convert_layout %104 : (tensor<16x1xf32, #blocked2>) -> tensor<16x1xf32, #blocked1>
      %106 = tt.broadcast %105 : (tensor<16x1xf32, #blocked1>) -> tensor<16x16xf32, #blocked1>
      %107 = arith.extf %69 : tensor<16x16xf16, #blocked1> to tensor<16x16xf32, #blocked1>
      %108 = arith.mulf %107, %106 : tensor<16x16xf32, #blocked1>
      %109 = arith.addf %102, %108 : tensor<16x16xf32, #blocked1>
      scf.yield %109, %94, %87 : tensor<16x16xf32, #blocked1>, tensor<16xf32, #blocked>, tensor<16xf32, #blocked>
    }
    %36 = arith.muli %2, %arg14 : i32
    %37 = arith.muli %3, %arg15 : i32
    %38 = arith.addi %36, %37 : i32
    %39 = triton_gpu.convert_layout %7 : (tensor<16xi32, #blocked>) -> tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %40 = tt.expand_dims %39 {axis = 1 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<16x1xi32, #blocked2>
    %41 = triton_gpu.convert_layout %40 : (tensor<16x1xi32, #blocked2>) -> tensor<16x1xi32, #blocked1>
    %42 = tt.splat %arg16 : (i32) -> tensor<16x1xi32, #blocked1>
    %43 = arith.muli %41, %42 : tensor<16x1xi32, #blocked1>
    %44 = triton_gpu.convert_layout %4 : (tensor<16xi32, #blocked>) -> tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %45 = tt.expand_dims %44 {axis = 0 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x16xi32, #blocked3>
    %46 = tt.broadcast %43 : (tensor<16x1xi32, #blocked1>) -> tensor<16x16xi32, #blocked1>
    %47 = tt.broadcast %45 : (tensor<1x16xi32, #blocked3>) -> tensor<16x16xi32, #blocked3>
    %48 = triton_gpu.convert_layout %47 : (tensor<16x16xi32, #blocked3>) -> tensor<16x16xi32, #blocked1>
    %49 = arith.addi %46, %48 : tensor<16x16xi32, #blocked1>
    %50 = tt.splat %38 : (i32) -> tensor<16x16xi32, #blocked1>
    %51 = arith.addi %50, %49 : tensor<16x16xi32, #blocked1>
    %52 = tt.splat %arg13 : (!tt.ptr<f16>) -> tensor<16x16x!tt.ptr<f16>, #blocked1>
    %53 = tt.addptr %52, %51 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
    %54 = arith.cmpi "slt", %41, %cst_3 : tensor<16x1xi32, #blocked1>
    %55 = tt.broadcast %54 : (tensor<16x1xi1, #blocked1>) -> tensor<16x16xi1, #blocked1>
    %56 = arith.truncf %35#0 : tensor<16x16xf32, #blocked1> to tensor<16x16xf16, #blocked1>
    %57 = triton_gpu.convert_layout %53 : (tensor<16x16x!tt.ptr<f16>, #blocked1>) -> tensor<16x16x!tt.ptr<f16>, #blocked4>
    %58 = triton_gpu.convert_layout %56 : (tensor<16x16xf16, #blocked1>) -> tensor<16x16xf16, #blocked4>
    %59 = triton_gpu.convert_layout %55 : (tensor<16x16xi1, #blocked1>) -> tensor<16x16xi1, #blocked4>
    tt.store %57, %58, %59 {cache = 1 : i32, evict = 1 : i32} : tensor<16x16xf16, #blocked4>
    tt.return
  }
}

// -----

// Check if MoveConvertOutOfLoop hangs because of adding additional conversions
// CHECK-LABEL: @loop_print
// CHECK-NOT: triton_gpu.convert_layout
//     CHECK: tt.return
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32, "triton_gpu.num-ctas" = 1 : i32} {
  tt.func public @loop_print(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<32> : tensor<32x128xi32, #blocked>
    %cst_0 = arith.constant dense<32> : tensor<128x32xi32, #blocked1>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %1 = triton_gpu.convert_layout %0 : (tensor<128xi32, #blocked2>) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi32, #blocked1>
    %3 = tt.splat %arg6 : (i32) -> tensor<128x1xi32, #blocked1>
    %4 = arith.muli %2, %3 : tensor<128x1xi32, #blocked1>
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked2>
    %6 = triton_gpu.convert_layout %5 : (tensor<32xi32, #blocked2>) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x32xi32, #blocked3>
    %8 = tt.broadcast %4 : (tensor<128x1xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
    %9 = tt.broadcast %7 : (tensor<1x32xi32, #blocked3>) -> tensor<128x32xi32, #blocked3>
    %10 = triton_gpu.convert_layout %9 : (tensor<128x32xi32, #blocked3>) -> tensor<128x32xi32, #blocked1>
    %11 = arith.addi %8, %10 : tensor<128x32xi32, #blocked1>
    %12 = triton_gpu.convert_layout %5 : (tensor<32xi32, #blocked2>) -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<32x1xi32, #blocked1>
    %14 = triton_gpu.convert_layout %13 : (tensor<32x1xi32, #blocked1>) -> tensor<32x1xi32, #blocked>
    %15 = triton_gpu.convert_layout %0 : (tensor<128xi32, #blocked2>) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %16 = tt.expand_dims %15 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x128xi32, #blocked3>
    %17 = tt.broadcast %14 : (tensor<32x1xi32, #blocked>) -> tensor<32x128xi32, #blocked>
    %18 = tt.broadcast %16 : (tensor<1x128xi32, #blocked3>) -> tensor<32x128xi32, #blocked3>
    %19 = triton_gpu.convert_layout %18 : (tensor<32x128xi32, #blocked3>) -> tensor<32x128xi32, #blocked>
    %20 = arith.addi %17, %19 : tensor<32x128xi32, #blocked>
    %21 = arith.addi %arg5, %c31_i32 : i32
    %22 = arith.divsi %21, %c32_i32 : i32
    %23 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %24 = tt.splat %arg1 : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %25:3 = scf.for %arg7 = %c0_i32 to %22 step %c1_i32 iter_args(%arg8 = %cst_1, %arg9 = %11, %arg10 = %20) -> (f32, tensor<128x32xi32, #blocked1>, tensor<32x128xi32, #blocked>)  : i32 {
      tt.print "a_offsets: " : %arg9 : tensor<128x32xi32, #blocked1>
      %27 = tt.addptr %23, %arg9 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %28 = triton_gpu.convert_layout %27 : (tensor<128x32x!tt.ptr<f16>, #blocked1>) -> tensor<128x32x!tt.ptr<f16>, #blocked4>
      %29 = tt.load %28 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked4>
      %30 = triton_gpu.convert_layout %29 : (tensor<128x32xf16, #blocked4>) -> tensor<128x32xf16, #blocked1>
      %31 = tt.addptr %24, %arg10 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %32 = triton_gpu.convert_layout %31 : (tensor<32x128x!tt.ptr<f16>, #blocked>) -> tensor<32x128x!tt.ptr<f16>, #blocked5>
      %33 = tt.load %32 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked5>
      %34 = triton_gpu.convert_layout %33 : (tensor<32x128xf16, #blocked5>) -> tensor<32x128xf16, #blocked>
      %35 = "tt.reduce"(%30) <{axis = 0 : i32}> ({
      ^bb0(%arg11: f16, %arg12: f16):
        %46 = arith.addf %arg11, %arg12 : f16
        tt.reduce.return %46 : f16
      }) : (tensor<128x32xf16, #blocked1>) -> tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %36 = triton_gpu.convert_layout %35 : (tensor<32xf16, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<32xf16, #blocked2>
      %37 = "tt.reduce"(%36) <{axis = 0 : i32}> ({
      ^bb0(%arg11: f16, %arg12: f16):
        %46 = arith.addf %arg11, %arg12 : f16
        tt.reduce.return %46 : f16
      }) : (tensor<32xf16, #blocked2>) -> f16
      %38 = "tt.reduce"(%34) <{axis = 0 : i32}> ({
      ^bb0(%arg11: f16, %arg12: f16):
        %46 = arith.addf %arg11, %arg12 : f16
        tt.reduce.return %46 : f16
      }) : (tensor<32x128xf16, #blocked>) -> tensor<128xf16, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %39 = triton_gpu.convert_layout %38 : (tensor<128xf16, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<128xf16, #blocked2>
      %40 = "tt.reduce"(%39) <{axis = 0 : i32}> ({
      ^bb0(%arg11: f16, %arg12: f16):
        %46 = arith.addf %arg11, %arg12 : f16
        tt.reduce.return %46 : f16
      }) : (tensor<128xf16, #blocked2>) -> f16
      %41 = arith.addf %37, %40 : f16
      %42 = arith.extf %41 : f16 to f32
      %43 = arith.addf %arg8, %42 : f32
      %44 = arith.addi %arg9, %cst_0 : tensor<128x32xi32, #blocked1>
      %45 = arith.addi %arg10, %cst : tensor<32x128xi32, #blocked>
      scf.yield %43, %44, %45 : f32, tensor<128x32xi32, #blocked1>, tensor<32x128xi32, #blocked>
    }
    %26 = arith.truncf %25#0 : f32 to f16
    tt.store %arg2, %26 {cache = 1 : i32, evict = 1 : i32} : f16
    tt.return
  }
}

// -----

// Check if SimplifyReduceCvt handles the cvt,reduce->reduce,cvt conversion but not the general push forward conversion
// CHECK-LABEL: reduce_cvt3
// CHECK: tt.dot
// CHECK-NEXT: tt.reduce
// CHECK: triton_gpu.convert_layout
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32, "triton_gpu.num-ctas" = 1 : i32} {
  tt.func public @reduce_cvt3(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %cst_0 = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>
    %1 = triton_gpu.convert_layout %0 : (tensor<32xi32, #blocked1>) -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<32x1xi32, #blocked2>
    %3 = triton_gpu.convert_layout %2 : (tensor<32x1xi32, #blocked2>) -> tensor<32x1xi32, #blocked>
    %4 = arith.muli %3, %cst_0 : tensor<32x1xi32, #blocked>
    %5 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %7 = triton_gpu.convert_layout %0 : (tensor<32xi32, #blocked1>) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %8 = tt.expand_dims %7 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>) -> tensor<1x32xi32, #blocked3>
    %9 = tt.broadcast %6 : (tensor<32x1x!tt.ptr<f16>, #blocked>) -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %10 = tt.broadcast %8 : (tensor<1x32xi32, #blocked3>) -> tensor<32x32xi32, #blocked3>
    %11 = triton_gpu.convert_layout %10 : (tensor<32x32xi32, #blocked3>) -> tensor<32x32xi32, #blocked>
    %12 = tt.addptr %9, %11 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %13 = tt.splat %arg1 : (!tt.ptr<f16>) -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %14 = tt.addptr %13, %4 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %15 = tt.broadcast %14 : (tensor<32x1x!tt.ptr<f16>, #blocked>) -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %16 = tt.addptr %15, %11 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %17 = triton_gpu.convert_layout %12 : (tensor<32x32x!tt.ptr<f16>, #blocked>) -> tensor<32x32x!tt.ptr<f16>, #blocked4>
    %18 = tt.load %17 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf16, #blocked4>
    %19 = triton_gpu.convert_layout %18 : (tensor<32x32xf16, #blocked4>) -> tensor<32x32xf16, #blocked>
    %20 = triton_gpu.convert_layout %16 : (tensor<32x32x!tt.ptr<f16>, #blocked>) -> tensor<32x32x!tt.ptr<f16>, #blocked4>
    %21 = tt.load %20 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf16, #blocked4>
    %22 = triton_gpu.convert_layout %21 : (tensor<32x32xf16, #blocked4>) -> tensor<32x32xf16, #blocked>
    %23 = triton_gpu.convert_layout %22 : (tensor<32x32xf16, #blocked>) -> tensor<32x32xf16, #shared>
    %24 = tt.trans %23 : (tensor<32x32xf16, #shared>) -> tensor<32x32xf16, #shared1>
    %25 = triton_gpu.convert_layout %24 : (tensor<32x32xf16, #shared1>) -> tensor<32x32xf16, #blocked>
    %26 = triton_gpu.convert_layout %19 : (tensor<32x32xf16, #blocked>) -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked5}>>
    %27 = triton_gpu.convert_layout %25 : (tensor<32x32xf16, #blocked>) -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked5}>>
    %28 = triton_gpu.convert_layout %cst : (tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked5>
    %29 = tt.dot %26, %27, %28 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked5}>> * tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked5}>> -> tensor<32x32xf32, #blocked5>
    %30 = triton_gpu.convert_layout %29 : (tensor<32x32xf32, #blocked5>) -> tensor<32x32xf32, #blocked>
    %31:2 = "tt.reduce"(%30, %11) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: i32, %arg5: f32, %arg6: i32):
      %37 = arith.cmpf "oeq", %arg3, %arg5 : f32
      %38 = arith.cmpi "slt", %arg4, %arg6 : i32
      %39 = arith.andi %37, %38 : i1
      %40 = arith.cmpf "ogt", %arg3, %arg5 : f32
      %41 = arith.ori %40, %39 : i1
      %42 = arith.select %41, %arg3, %arg5 : f32
      %43 = arith.select %41, %arg4, %arg6 : i32
      tt.reduce.return %42, %43 : f32, i32
    }) : (tensor<32x32xf32, #blocked>, tensor<32x32xi32, #blocked>) -> (tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>)
    %32 = triton_gpu.convert_layout %31#1 : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32xi32, #blocked1>
    %33 = tt.splat %arg2 : (!tt.ptr<i32>) -> tensor<32x!tt.ptr<i32>, #blocked1>
    %34 = tt.addptr %33, %0 : tensor<32x!tt.ptr<i32>, #blocked1>, tensor<32xi32, #blocked1>
    %35 = triton_gpu.convert_layout %34 : (tensor<32x!tt.ptr<i32>, #blocked1>) -> tensor<32x!tt.ptr<i32>, #blocked1>
    %36 = triton_gpu.convert_layout %32 : (tensor<32xi32, #blocked1>) -> tensor<32xi32, #blocked1>
    tt.store %35, %36 {cache = 1 : i32, evict = 1 : i32} : tensor<32xi32, #blocked1>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
// CHECK-LABEL: axis_mismatch
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {
tt.func @axis_mismatch(%arg0: f32) -> tensor<1xf32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> {
// CHECK: %[[R:.+]] = "tt.reduce"(%0) <{axis = 1 : i32}>
// CHECK: %[[C:.+]] = triton_gpu.convert_layout %[[R]]
// CHECK: tt.return %[[C]]
  %0 = tt.splat %arg0 : (f32) -> tensor<1x16xf32, #blocked>
  %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%arg9: f32, %arg10: f32):
    %60 = arith.addf %arg9, %arg10 : f32
    tt.reduce.return %60 : f32
  }) : (tensor<1x16xf32, #blocked>) -> tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
  %2 = triton_gpu.convert_layout %1 : (tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<1xf32, #blocked1>
  %3 = triton_gpu.convert_layout %2 : (tensor<1xf32, #blocked1>) -> tensor<1xf32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
  tt.return %3: tensor<1xf32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
}
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {
// CHECK-LABEL: reduce_to_scalar
//   CHECK-NOT:   triton_gpu.convert_layout
//       CHECK:   tt.return
tt.func @reduce_to_scalar(%ptr: tensor<1024x!tt.ptr<f32>, #blocked>) -> (f32, i32) {
  %0 = tt.load %ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked>
  %1 = triton_gpu.convert_layout %0 : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked1>
  %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked1>
  %3:2 = "tt.reduce"(%1, %2) <{axis = 0 : i32}> ({
    ^bb0(%arg7: f32, %arg8: i32, %arg9: f32, %arg10: i32):
    %51 = arith.cmpf "oeq", %arg7, %arg9 : f32
    %52 = arith.cmpi "slt", %arg8, %arg10 : i32
    %53 = arith.andi %51, %52 : i1
    %54 = arith.cmpf "ogt", %arg7, %arg9 : f32
    %55 = arith.ori %54, %53 : i1
    %56 = arith.select %55, %arg7, %arg9 : f32
    %57 = arith.select %55, %arg8, %arg10 : i32
    tt.reduce.return %56, %57 : f32, i32
  }) : (tensor<1024xf32, #blocked1>, tensor<1024xi32, #blocked1>) -> (f32, i32)
  tt.return %3#0, %3#1: f32, i32
}
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {
// CHECK-LABEL: whileop
//       CHECK: %[[L:.+]] = tt.load %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked>
//       CHECK: %[[W:.+]] = scf.while (%[[I:.+]] = %[[L]], %{{.*}} = %{{.*}}) : (tensor<1024xf32, #blocked>, i1) -> tensor<1024xf32, #blocked> {
//       CHECK:   scf.condition(%{{.*}}) %[[I]] : tensor<1024xf32, #blocked>
//       CHECK: } do {
//       CHECK: ^bb0(%[[ARG1:.+]]: tensor<1024xf32, #blocked>):
//       CHECK:    %[[ADD:.+]] = arith.addf %[[ARG1]], %[[ARG1]] : tensor<1024xf32, #blocked>
//       CHECK:    scf.yield %[[ADD]], %{{.*}} : tensor<1024xf32, #blocked>, i1
//       CHECK:  }
//       CHECK:  tt.store %{{.*}}, %[[W]] {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32, #blocked>
tt.func @whileop(%ptr: tensor<1024x!tt.ptr<f32>, #blocked>, %cond: i1) {
  %0 = tt.load %ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked>
  %1 = triton_gpu.convert_layout %0 : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked1>
  %2 = scf.while (%arg0 = %1, %arg1 = %cond) : (tensor<1024xf32, #blocked1>, i1) -> (tensor<1024xf32, #blocked1>) {
      scf.condition(%arg1) %arg0 : tensor<1024xf32, #blocked1>
    } do {
    ^bb0(%arg0: tensor<1024xf32, #blocked1>):
      %4 = triton_gpu.convert_layout %arg0 : (tensor<1024xf32, #blocked1>) -> tensor<1024xf32, #blocked>
      %5 = arith.addf %4, %4 : tensor<1024xf32, #blocked>
      %6 = triton_gpu.convert_layout %5 : (tensor<1024xf32, #blocked>) -> tensor<1024xf32, #blocked1>
      scf.yield %6, %cond : tensor<1024xf32, #blocked1>, i1
    }
  %3 = triton_gpu.convert_layout %2 : (tensor<1024xf32, #blocked1>) -> tensor<1024xf32, #blocked>
  tt.store %ptr, %3 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32, #blocked>
  tt.return
}
}

// -----


#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {

// Check that we don't insert convert in between the loop and async_wait as this is a required barrier.
// CHECK-LABEL: workaround
//       CHECK: scf.for
//       CHECK:   scf.yield
//  CHECK-NEXT: }
//  CHECK-NEXT: triton_gpu.async_wait
//  CHECK-NEXT: triton_gpu.convert_layout
  tt.func @workaround(%arg0: !tt.ptr<f32, 1>, %arg1: i32, %arg2: !tt.ptr<f32, 1>, %arg3: i32, %arg4: i32) {
    %cst = arith.constant dense<64> : tensor<64x64xi32, #blocked>
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<64x1xi32, #blocked>
    %3 = tt.splat %arg1 : (i32) -> tensor<64x1xi32, #blocked>
    %4 = arith.muli %2, %3 : tensor<64x1xi32, #blocked>
    %5 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<64x1x!tt.ptr<f32, 1>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<64x1x!tt.ptr<f32, 1>, #blocked>, tensor<64x1xi32, #blocked>
    %7 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x64xi32, #blocked1>
    %8 = tt.broadcast %6 : (tensor<64x1x!tt.ptr<f32, 1>, #blocked>) -> tensor<64x64x!tt.ptr<f32, 1>, #blocked>
    %9 = tt.broadcast %7 : (tensor<1x64xi32, #blocked1>) -> tensor<64x64xi32, #blocked1>
    %10 = triton_gpu.convert_layout %9 : (tensor<64x64xi32, #blocked1>) -> tensor<64x64xi32, #blocked>
    %11 = tt.addptr %8, %10 : tensor<64x64x!tt.ptr<f32, 1>, #blocked>, tensor<64x64xi32, #blocked>
    %12 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<64x1x!tt.ptr<f32, 1>, #blocked>
    %13 = tt.addptr %12, %2 : tensor<64x1x!tt.ptr<f32, 1>, #blocked>, tensor<64x1xi32, #blocked>
    %14 = tt.splat %arg3 : (i32) -> tensor<1x64xi32, #blocked1>
    %15 = arith.muli %7, %14 : tensor<1x64xi32, #blocked1>
    %16 = tt.broadcast %13 : (tensor<64x1x!tt.ptr<f32, 1>, #blocked>) -> tensor<64x64x!tt.ptr<f32, 1>, #blocked>
    %17 = tt.broadcast %15 : (tensor<1x64xi32, #blocked1>) -> tensor<64x64xi32, #blocked1>
    %18 = triton_gpu.convert_layout %17 : (tensor<64x64xi32, #blocked1>) -> tensor<64x64xi32, #blocked>
    %19 = tt.addptr %16, %18 : tensor<64x64x!tt.ptr<f32, 1>, #blocked>, tensor<64x64xi32, #blocked>
    %20:2 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %cst_0, %arg7 = %11) -> (tensor<64x64xf32, #blocked>, tensor<64x64x!tt.ptr<f32, 1>, #blocked>) {
      %21 = triton_gpu.convert_layout %arg7 : (tensor<64x64x!tt.ptr<f32, 1>, #blocked>) -> tensor<64x64x!tt.ptr<f32, 1>, #blocked2>
      %22 = tt.load %21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf32, #blocked2>
      %23 = triton_gpu.convert_layout %22 : (tensor<64x64xf32, #blocked2>) -> tensor<64x64xf32, #blocked>
      %24 = arith.addf %arg6, %23 : tensor<64x64xf32, #blocked>
      %25 = tt.addptr %arg7, %cst : tensor<64x64x!tt.ptr<f32, 1>, #blocked>, tensor<64x64xi32, #blocked>
      scf.yield %24, %25 : tensor<64x64xf32, #blocked>, tensor<64x64x!tt.ptr<f32, 1>, #blocked>
    }
    triton_gpu.async_wait {num = 0 : i32}
    tt.store %19, %20#0 {cache = 1 : i32, evict = 1 : i32} : tensor<64x64xf32, #blocked>
    tt.return
  }
}
