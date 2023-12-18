// RUN: triton-opt %s -split-input-file -tritongpu-rewrite-tensor-pointer | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func public @matmul_kernel_0d1d2d3d456d7d8c9c10d11c121314c(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32) {
    %c127_i32 = arith.constant 127 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg5, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.muli %2, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %4, %7 : i32
    %9 = arith.cmpi "slt", %8, %c8_i32 : i32
    %10 = arith.select %9, %8, %c8_i32 : i32
    %11 = arith.remsi %0, %10 : i32
    %12 = arith.addi %7, %11 : i32
    %13 = arith.remsi %0, %5 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c128_i32 : i32
    %16 = arith.muli %14, %c128_i32 : i32
    %17 = arith.extsi %arg4 : i32 to i64
    %18 = arith.extsi %arg6 : i32 to i64
    %19 = arith.extsi %arg7 : i32 to i64
    // CHECK-NOT: tt.make_tensor_ptr
    %20 = tt.make_tensor_ptr %arg0, [%17, %18], [%19, %c1_i64], [%15, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #blocked>, 1>
    %21 = arith.extsi %arg5 : i32 to i64
    %22 = arith.extsi %arg8 : i32 to i64
    // CHECK-NOT: tt.make_tensor_ptr
    %23 = tt.make_tensor_ptr %arg1, [%18, %21], [%c1_i64, %22], [%c0_i32, %16] {order = array<i32: 0, 1>} : <tensor<64x128xf16, #blocked1>, 1>
    %24:3 = scf.for %arg11 = %c0_i32 to %arg6 step %c64_i32 iter_args(%arg12 = %cst, %arg13 = %20, %arg14 = %23) -> (tensor<128x128xf32, #blocked>, !tt.ptr<tensor<128x64xf16, #blocked>, 1>, !tt.ptr<tensor<64x128xf16, #blocked1>, 1>)  : i32 {
      // CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 2 : i32} : tensor<128x64xf16,
      %28 = tt.load %arg13 {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 2 : i32} : !tt.ptr<tensor<128x64xf16, #blocked>, 1> -> tensor<128x64xf16, #blocked>
      // CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 2 : i32} : tensor<64x128xf16,
      %29 = tt.load %arg14 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 2 : i32} : !tt.ptr<tensor<64x128xf16, #blocked1>, 1> -> tensor<64x128xf16, #blocked1>
      %30 = triton_gpu.convert_layout %28 : (tensor<128x64xf16, #blocked>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>>
      %31 = triton_gpu.convert_layout %29 : (tensor<64x128xf16, #blocked1>) -> tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked2}>>
      %32 = triton_gpu.convert_layout %arg12 : (tensor<128x128xf32, #blocked>) -> tensor<128x128xf32, #blocked2>
      %33 = tt.dot %30, %31, %32 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x128xf32, #blocked2>
      %34 = triton_gpu.convert_layout %33 : (tensor<128x128xf32, #blocked2>) -> tensor<128x128xf32, #blocked>
      // CHECK-NOT: tt.advance
      %35 = tt.advance %arg13, [%c0_i32, %c64_i32] : <tensor<128x64xf16, #blocked>, 1>
      // CHECK-NOT: tt.advance
      %36 = tt.advance %arg14, [%c64_i32, %c0_i32] : <tensor<64x128xf16, #blocked1>, 1>
      scf.yield %34, %35, %36 : tensor<128x128xf32, #blocked>, !tt.ptr<tensor<128x64xf16, #blocked>, 1>, !tt.ptr<tensor<64x128xf16, #blocked1>, 1>
    }
    %25 = arith.truncf %24#0 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %26 = arith.extsi %arg10 : i32 to i64
    %27 = tt.make_tensor_ptr %arg3, [%17, %21], [%26, %c1_i64], [%15, %16] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #blocked>, 1>
    // CHECK: tt.store %{{.*}}, %{{.*}}, %{{.*}} {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #blocked>
    tt.store %27, %25 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<128x128xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @if_for_if(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x1xf32, #blocked>
    %c63_i32 = arith.constant 63 : i32
    %c-16_i32 = arith.constant -16 : i32
    %c132_i32 = arith.constant 132 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.muli %0, %c64_i32 : i32
    %4 = arith.extsi %arg3 : i32 to i64
    %5 = arith.extsi %arg4 : i32 to i64
    %6 = arith.extsi %arg5 : i32 to i64
    // CHECK-NOT: tt.make_tensor_ptr
    %7 = tt.make_tensor_ptr %arg0, [%4, %5], [%6, %c1_i64], [%3, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #blocked>, 1>
    %8 = arith.cmpi "sge", %2, %c132_i32 : i32
    scf.if %8 {
      %9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
      %10 = tt.splat %arg7 : (i32) -> tensor<64x1xi32, #blocked>
      %11 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<64x1x!tt.ptr<f32, 1>, #blocked>
      %12 = scf.for %arg8 = %0 to %2 step %c132_i32 iter_args(%arg9 = %7) -> (!tt.ptr<tensor<64x16xf16, #blocked>, 1>)  : i32 {
        %13 = arith.cmpi "sge", %arg8, %c132_i32 : i32
        %14 = scf.if %13 -> (!tt.ptr<tensor<64x16xf16, #blocked>, 1>) {
          %25 = arith.subi %arg8, %0 : i32
          %26 = arith.muli %25, %c64_i32 : i32
          // CHECK-NOT: tt.advance
          %27 = tt.advance %arg9, [%26, %c-16_i32] : <tensor<64x16xf16, #blocked>, 1>
          scf.yield %27 : !tt.ptr<tensor<64x16xf16, #blocked>, 1>
        } else {
          scf.yield %arg9 : !tt.ptr<tensor<64x16xf16, #blocked>, 1>
        }
        %15 = arith.muli %arg8, %c64_i32 : i32
        %16 = tt.splat %15 : (i32) -> tensor<64xi32, #blocked1>
        %17 = arith.addi %9, %16 : tensor<64xi32, #blocked1>
        %18 = triton_gpu.convert_layout %17 : (tensor<64xi32, #blocked1>) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
        %19 = tt.expand_dims %18 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64x1xi32, #blocked2>
        %20 = triton_gpu.convert_layout %19 : (tensor<64x1xi32, #blocked2>) -> tensor<64x1xi32, #blocked>
        %21 = arith.muli %20, %10 : tensor<64x1xi32, #blocked>
        %22 = tt.addptr %11, %21 : tensor<64x1x!tt.ptr<f32, 1>, #blocked>, tensor<64x1xi32, #blocked>
        %23 = triton_gpu.convert_layout %22 : (tensor<64x1x!tt.ptr<f32, 1>, #blocked>) -> tensor<64x1x!tt.ptr<f32, 1>, #blocked>
        %24 = triton_gpu.convert_layout %cst : (tensor<64x1xf32, #blocked>) -> tensor<64x1xf32, #blocked>
        tt.store %23, %24 {cache = 1 : i32, evict = 1 : i32} : tensor<64x1xf32, #blocked>
        scf.yield %14 : !tt.ptr<tensor<64x16xf16, #blocked>, 1>
      }
    }
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // Make sure the load generated has the right number of operands when there is nomask.
  // CHECK-LABEL: @no_mask
  tt.func public @no_mask(%arg0: !tt.ptr<i8, 1> {tt.divisibility = 16 : i32}) -> tensor<1024x1024xi8, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %0 = tt.make_tensor_ptr %arg0, [%c1024_i64, %c1024_i64], [%c1024_i64, %c1024_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<1024x1024xi8, #blocked1>, 1>
    // CHECK: tt.load %{{.+}} {boundaryCheck = array<i32>, cache = 1 : i32, evict = 2 : i32, isVolatile = false, padding = 1 : i32} : tensor<1024x1024xi8, #{{.*}}>
    %1 = tt.load %0 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 2 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<1024x1024xi8, #blocked1>, 1> -> tensor<1024x1024xi8, #blocked>
    tt.return %1 : tensor<1024x1024xi8, #blocked>
  }
}
