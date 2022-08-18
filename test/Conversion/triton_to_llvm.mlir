// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu=num-warps=2 -convert-triton-gpu-to-llvm | FileCheck %s

func @test_splat(%ptr: !tt.ptr<f32>) {
  // Here, 128 elements, 64(2*32) threads, so each need to process 2 elements
  //
  // CHECK: %0 = llvm.bitcast %arg0 : !llvm.ptr<f32, 1> to !llvm.ptr<f32, 1>
  // CHECK: %1 = llvm.mlir.undef : !llvm.struct<(ptr<f32, 1>, ptr<f32, 1>)>
  // CHECK: %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<(ptr<f32, 1>, ptr<f32, 1>)>
  // CHECK: %3 = llvm.insertvalue %0, %2[1] : !llvm.struct<(ptr<f32, 1>, ptr<f32, 1>)>
  %ptrs = tt.splat %ptr : (!tt.ptr<f32>) -> tensor<128x!tt.ptr<f32>>
  %a = arith.constant 1.0 : f32
  %true = arith.constant 1 : i1
  %b = tt.splat %a : (f32) -> tensor<128xf32>

  // Here, each thread process only 1 element
  // CHECK: %{{.*}} = llvm.mlir.undef : !llvm.struct<(i1)>
  %mask = tt.splat %true : (i1) -> tensor<64xi1>

  return
}

// -----

func @test_store_splat(%ptr: !tt.ptr<f32>) {
  %ptrs = tt.splat %ptr : (!tt.ptr<f32>) -> tensor<128x!tt.ptr<f32>>
  %a = arith.constant 1.0 : f32
  %true = arith.constant 1 : i1

  %vs = tt.splat %a : (f32) -> tensor<128xf32>
  %mask = tt.splat %true : (i1) -> tensor<128xi1>

  // CHECK: %{{.*}} = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 st.global.v32.b1 [ $1 + 0 ], { $2 };",
  // CHECK-SAME: "b,l,r" %{{.*}}, %{{.*}}, %{{.*}} : (i1, !llvm.ptr<f32, 1>, i32) -> !llvm.struct<()>
  tt.store %ptrs, %vs, %mask, {} : tensor<128xf32>

  return
}
