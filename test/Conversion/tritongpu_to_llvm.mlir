// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm | FileCheck %s

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.func @test_empty_kernel(%arg0: i64, %arg1: !llvm.ptr<f16, 1>)
  // Here the 128 comes from the 4 in module attribute multiples 32
  // CHECK:  attributes {nvvm.kernel = 1 : ui1, nvvm.maxntid = [128 : i32]} {{.*}}
  func.func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {
    // CHECK:  llvm.return
    return
  }
} // end module

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_load
  func.func @basic_load(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK: llvm.inline_asm
    %1 = tt.load %a_ptr_init, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: vectorized_load
  func.func @vectorized_load(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.b32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.b32
    %1 = tt.load %a_ptr_init, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: vectorized_load_f16
  func.func @vectorized_load_f16(%a_ptr_init: tensor<256x!tt.ptr<f16>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf16, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.b16
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.b16
    %1 = tt.load %a_ptr_init, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf16, #blocked0>
    return
  }
}

// -----

// TODO: masked load with vectorization is pending on TODO
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: masked_load_const_other
  func.func @masked_load_const_other(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked0>
    %1 = tt.load %a_ptr_init, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    return
  }
}

// -----

// TODO: masked load with vectorization is pending on TODO
#blocked0 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: masked_load_const_other_vec
  func.func @masked_load_const_other_vec(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked0>
    %1 = tt.load %a_ptr_init, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"triton_gpu.num-warps" = 2 : i32} {
  // CHECK-LABEL: global_load_store_no_vec
  func.func @global_load_store_no_vec(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : (i32) -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 4 elements from vector0
    // CHECK: "@${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: "@${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: "@${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: "@${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 4 elements from vector1
    // CHECK: "@${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: "@${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: "@${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: "@${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    %9 = tt.load %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    %10 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 4 elements to global
    // CHECK: @${{.*}} st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    // CHECK: @${{.*}} st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    // CHECK: @${{.*}} st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    // CHECK: @${{.*}} st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    tt.store %13, %11 : tensor<256xf32, #blocked0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"triton_gpu.num-warps" = 2 : i32} {
  // CHECK-LABEL: global_load_store_vec4
  func.func @global_load_store_vec4(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : (i32) -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 4 elements from A with single one vectorized load instruction
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 4 elements from B with single one vectorized load instruction
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    %9 = tt.load %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    %10 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 4 elements to global with single one vectorized store instruction
    // CHECK: @$5 st.global.v4.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} };
    tt.store %13, %11 : tensor<256xf32, #blocked0>
    return
  }
}

// -----

// This test verifies the vectorization of Load and Store Ops.
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
// Note, the %n_elements doesn't have a "tt.divisibility" hint, so Triton assumes it's divisibility is 1, this should effect the mask's alignment and further restrict the load/store ops' vector width to be 1.
module attributes {"triton_gpu.num-warps" = 2 : i32} {
  func.func @vecadd_masked_vec1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32) {
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %3 = tt.splat %1 : (i32) -> tensor<64xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<64xi32, #blocked>
    %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %7 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %9 = tt.splat %n_elements : (i32) -> tensor<64xi32, #blocked>
    %10 = "triton_gpu.cmpi"(%4, %9) {predicate = 2 : i64} : (tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64xi1, #blocked>
    // load op has a vector width = 1 due to the %mask's alignment
    // CHECK: ld.global.b32
    %11 = tt.load %6, %10 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32, #blocked>
    %12 = tt.load %8, %10 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32, #blocked>
    %13 = arith.addf %11, %12 : tensor<64xf32, #blocked>
    %14 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    tt.store %15, %13, %10 : tensor<64xf32, #blocked>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_store_vec2
    func.func @global_load_store_vec2(%arg0: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : (i32) -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 8 elements from A with four vectorized load instruction
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 8 elements from B with four vectorized load instruction
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    %9 = tt.load %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    %10 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 8 elements to global with four vectorized store instruction
    // CHECK: @${{.*}} st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: @${{.*}} st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: @${{.*}} st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: @${{.*}} st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    tt.store %13, %11 : tensor<256xf32, #blocked0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_store_vec8
    func.func @global_load_store_vec8(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : (i32) -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 8 elements from A with two vectorized load instruction
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 8 elements from B with two vectorized load instruction
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    %9 = tt.load %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    %10 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 8 elements to global with two vectorized store instruction
    // CHECK: @$5 st.global.v4.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} };
    // CHECK: @$5 st.global.v4.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} };
    tt.store %13, %11 : tensor<256xf32, #blocked0>
    return
  }
}

// TODO: Add a testcase to verify the optimization when ptr of the LoadOp
//       is from an addptr with const idx

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_view_broadcast
  func.func @basic_view_broadcast(%arg : tensor<256xf32,#blocked0>) {
    // CHECK: llvm.mlir.undef
    // CHECK: %[[T0:.*]] = llvm.extractvalue
    // CHECK: %[[T1:.*]] = llvm.extractvalue
    %0 = tt.view %arg : (tensor<256xf32, #blocked0>) -> tensor<256x1xf32,#blocked2>
    // CHECK: llvm.mlir.undef
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T1]]
    %1 = tt.broadcast %0 : (tensor<256x1xf32,#blocked2>) -> tensor<256x4xf32, #blocked2>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_make_range
  func.func @basic_make_range() {
    // CHECK: nvvm.read.ptx.sreg.tid.x
    // CHECK: llvm.mlir.undef
    // CHECK: llvm.insertvalue
    // CHECK: llvm.insertvalue
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_addf
  func.func @basic_addf(%arg0 : tensor<256xf32,#blocked0>, %arg1 : tensor<256xf32,#blocked0>) {
    // CHECK: llvm.fadd
    // CHECK: llvm.fadd
    %1 = arith.addf %arg0, %arg1 : tensor<256xf32,#blocked0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_addi
  func.func @basic_addi(%arg0 : tensor<256xi32,#blocked0>, %arg1 : tensor<256xi32,#blocked0>) {
    // CHECK: llvm.add
    // CHECK: llvm.add
    %1 = arith.addi %arg0, %arg1 : tensor<256xi32,#blocked0>
    return
  }
}

// -----

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_program_id
  func.func @basic_program_id() {
    // CHECK: nvvm.read.ptx.sreg.ctaid.x : i32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_addptr
  func.func @basic_addptr(%arg0 : tensor<256x!tt.ptr<f32>,#blocked0>, %arg1 : tensor<256xi32,#blocked0>) {
    // CHECK: llvm.getelementptr
    // CHECK: llvm.getelementptr
    %0 = tt.addptr %arg0, %arg1 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    return
  }
}

// -----

#shared0 = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: basic_alloc_tensor
  func.func @basic_alloc_tensor() {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK-NEXT: llvm.bitcast
    // CHECK-NEXT: llvm.mlir.constant
    // CHECK-NEXT: llvm.getelementptr
    // CHECK-NEXT: llvm.bitcast
    %0 = triton_gpu.alloc_tensor : tensor<16x16xf16, #shared0>
    return
  }
}

// -----

#shared0 = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: basic_extract_slice
  func.func @basic_extract_slice() {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.getelementptr
    %index = arith.constant 1 : i32
    %0 = triton_gpu.alloc_tensor : tensor<128x16x32xf32, #shared0>
    %1 = triton_gpu.extract_slice %0[%index, 0, 0][1, 16, 32][1, 1, 1] : tensor<128x16x32xf32, #shared0> to tensor<16x32xf32, #shared0>
    return
  }
}

// -----

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_async_wait
  func.func @basic_async_wait() {
    // CHECK: cp.async.wait_group 0x4
    triton_gpu.async_wait {num = 4: i32}
    return
  }
}

// -----

#block0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [4], warpsPerCTA = [4], order = [0]}>
#block1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [8], warpsPerCTA = [4], order = [0]}>
#block2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#block3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice2d1 = #triton_gpu.slice<{dim = 1, parent=#block2}>
#slice3d0 = #triton_gpu.slice<{dim = 0, parent=#block3}>
#AL = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#A = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_fallback
  func.func @basic_insert_slice_async_fallback(%arg0: !tt.ptr<f16> {tt.divisibility = 1 : i32}) {
    %off0_ = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #slice2d1>
    %off1_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<64xi32, #slice3d0>
    %off0 = tt.expand_dims %off0_ {axis = 1 : i32} : (tensor<16xi32, #slice2d1>) -> tensor<16x1xi32, #block2>
    %off1 = tt.expand_dims %off1_ {axis = 0 : i32} : (tensor<64xi32, #slice3d0>) -> tensor<1x64xi32, #block3>
    %broadcast_off0_scalar = tt.broadcast %off0 : (tensor<16x1xi32, #block2>) -> tensor<16x64xi32, #block2>
    %cst_scalar = arith.constant 64 : i32
    %cst = tt.splat %cst_scalar : (i32) -> tensor<16x64xi32, #block2>
    %broadcast_off0_ = arith.muli %broadcast_off0_scalar, %cst : tensor<16x64xi32, #block2>
    %broadcast_off1_ = tt.broadcast %off1 : (tensor<1x64xi32, #block3>) -> tensor<16x64xi32, #block3>
    %broadcast_off0 = triton_gpu.convert_layout %broadcast_off0_ : (tensor<16x64xi32, #block2>) -> tensor<16x64xi32, #AL>
    %broadcast_off1 = triton_gpu.convert_layout %broadcast_off1_ : (tensor<16x64xi32, #block3>) -> tensor<16x64xi32, #AL>
    %off = arith.addi %broadcast_off0, %broadcast_off1 : tensor<16x64xi32, #AL>
    %a_init = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<16x64x!tt.ptr<f16>, #AL>
    %a_ptr = tt.addptr %a_init, %off : tensor<16x64x!tt.ptr<f16>, #AL>, tensor<16x64xi32, #AL>
    %tensor = triton_gpu.alloc_tensor : tensor<2x16x64xf16, #A>
    %index = arith.constant 1 : i32

    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<8xi32>, 3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<8xi32>, 3>
    %a = triton_gpu.insert_slice_async %a_ptr, %tensor, %index {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x64x!tt.ptr<f16>, #AL> -> tensor<2x16x64xf16, #A>
    return
  }
}

// -----

#block0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [4], warpsPerCTA = [4], order = [0]}>
#block1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [8], warpsPerCTA = [4], order = [0]}>
#block2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#block3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice2d1 = #triton_gpu.slice<{dim = 1, parent=#block2}>
#slice3d0 = #triton_gpu.slice<{dim = 0, parent=#block3}>
#AL = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#A = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_v4
  func.func @basic_insert_slice_async_v4(%arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) {
    %off0_ = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #slice2d1>
    %off1_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<64xi32, #slice3d0>
    %off0 = tt.expand_dims %off0_ {axis = 1 : i32} : (tensor<16xi32, #slice2d1>) -> tensor<16x1xi32, #block2>
    %off1 = tt.expand_dims %off1_ {axis = 0 : i32} : (tensor<64xi32, #slice3d0>) -> tensor<1x64xi32, #block3>
    %broadcast_off0_scalar = tt.broadcast %off0 : (tensor<16x1xi32, #block2>) -> tensor<16x64xi32, #block2>
    %cst_scalar = arith.constant 64 : i32
    %cst = tt.splat %cst_scalar : (i32) -> tensor<16x64xi32, #block2>
    %broadcast_off0_ = arith.muli %broadcast_off0_scalar, %cst : tensor<16x64xi32, #block2>
    %broadcast_off1_ = tt.broadcast %off1 : (tensor<1x64xi32, #block3>) -> tensor<16x64xi32, #block3>
    %broadcast_off0 = triton_gpu.convert_layout %broadcast_off0_ : (tensor<16x64xi32, #block2>) -> tensor<16x64xi32, #AL>
    %broadcast_off1 = triton_gpu.convert_layout %broadcast_off1_ : (tensor<16x64xi32, #block3>) -> tensor<16x64xi32, #AL>
    %off = arith.addi %broadcast_off0, %broadcast_off1 : tensor<16x64xi32, #AL>
    %a_init = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<16x64x!tt.ptr<f32>, #AL>
    %a_ptr = tt.addptr %a_init, %off : tensor<16x64x!tt.ptr<f32>, #AL>, tensor<16x64xi32, #AL>
    %tensor = triton_gpu.alloc_tensor : tensor<2x16x64xf32, #A>
    %index = arith.constant 1 : i32

    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-SAME: cp.async.cg.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x10, 0x10
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-SAME: cp.async.cg.shared.global [ ${{.*}} + 16 ], [ ${{.*}} + 0 ], 0x10, 0x10
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-SAME: cp.async.commit_group
    %a = triton_gpu.insert_slice_async %a_ptr, %tensor, %index {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x64x!tt.ptr<f32>, #AL> -> tensor<2x16x64xf32, #A>
    triton_gpu.async_commit_group
    return
  }
}

// -----

#block0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [4], warpsPerCTA = [4], order = [0]}>
#block1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [8], warpsPerCTA = [4], order = [0]}>
#block2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#block3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice2d1 = #triton_gpu.slice<{dim = 1, parent=#block2}>
#slice3d0 = #triton_gpu.slice<{dim = 0, parent=#block3}>
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#A = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_v1
  func.func @basic_insert_slice_async_v1(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}) {
    %off0_ = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #slice2d1>
    %off1_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice3d0>
    %off0 = tt.expand_dims %off0_ {axis = 1 : i32} : (tensor<16xi32, #slice2d1>) -> tensor<16x1xi32, #block2>
    %off1 = tt.expand_dims %off1_ {axis = 0 : i32} : (tensor<32xi32, #slice3d0>) -> tensor<1x32xi32, #block3>
    %broadcast_off0_scalar = tt.broadcast %off0 : (tensor<16x1xi32, #block2>) -> tensor<16x32xi32, #block2>
    %cst_scalar = arith.constant 32 : i32
    %cst = tt.splat %cst_scalar : (i32) -> tensor<16x32xi32, #block2>
    %broadcast_off0_ = arith.muli %broadcast_off0_scalar, %cst : tensor<16x32xi32, #block2>
    %broadcast_off1_ = tt.broadcast %off1 : (tensor<1x32xi32, #block3>) -> tensor<16x32xi32, #block3>
    %broadcast_off0 = triton_gpu.convert_layout %broadcast_off0_ : (tensor<16x32xi32, #block2>) -> tensor<16x32xi32, #AL>
    %broadcast_off1 = triton_gpu.convert_layout %broadcast_off1_ : (tensor<16x32xi32, #block3>) -> tensor<16x32xi32, #AL>
    %off = arith.addi %broadcast_off0, %broadcast_off1 : tensor<16x32xi32, #AL>
    %a_init = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<16x32x!tt.ptr<f32>, #AL>
    %a_ptr = tt.addptr %a_init, %off : tensor<16x32x!tt.ptr<f32>, #AL>, tensor<16x32xi32, #AL>
    %tensor = triton_gpu.alloc_tensor : tensor<2x16x32xf32, #A>
    %index = arith.constant 1 : i32

    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.commit_group
    %a = triton_gpu.insert_slice_async %a_ptr, %tensor, %index {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x32x!tt.ptr<f32>, #AL> -> tensor<2x16x32xf32, #A>
    triton_gpu.async_commit_group
    return
  }
}

// -----

#block0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [8], warpsPerCTA = [4], order = [0]}>
#block2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#block3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice2d1 = #triton_gpu.slice<{dim = 1, parent=#block2}>
#slice3d0 = #triton_gpu.slice<{dim = 0, parent=#block3}>
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#A = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_v1_multictas
  func.func @basic_insert_slice_async_v1_multictas(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}) {
    %off0_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice2d1>
    %off1_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice3d0>
    %off0 = tt.expand_dims %off0_ {axis = 1 : i32} : (tensor<32xi32, #slice2d1>) -> tensor<32x1xi32, #block2>
    %off1 = tt.expand_dims %off1_ {axis = 0 : i32} : (tensor<32xi32, #slice3d0>) -> tensor<1x32xi32, #block3>
    %broadcast_off0_scalar = tt.broadcast %off0 : (tensor<32x1xi32, #block2>) -> tensor<32x32xi32, #block2>
    %cst_scalar = arith.constant 32 : i32
    %cst = tt.splat %cst_scalar : (i32) -> tensor<32x32xi32, #block2>
    %broadcast_off0_ = arith.muli %broadcast_off0_scalar, %cst : tensor<32x32xi32, #block2>
    %broadcast_off1_ = tt.broadcast %off1 : (tensor<1x32xi32, #block3>) -> tensor<32x32xi32, #block3>
    %broadcast_off0 = triton_gpu.convert_layout %broadcast_off0_ : (tensor<32x32xi32, #block2>) -> tensor<32x32xi32, #AL>
    %broadcast_off1 = triton_gpu.convert_layout %broadcast_off1_ : (tensor<32x32xi32, #block3>) -> tensor<32x32xi32, #AL>
    %off = arith.addi %broadcast_off0, %broadcast_off1 : tensor<32x32xi32, #AL>
    %a_init = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>, #AL>
    %a_ptr = tt.addptr %a_init, %off : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    %tensor = triton_gpu.alloc_tensor : tensor<2x32x32xf32, #A>
    %index = arith.constant 1 : i32

    // CHECK: llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.mul
    // CHECK: llvm.add
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.commit_group
    %a = triton_gpu.insert_slice_async %a_ptr, %tensor, %index {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32x!tt.ptr<f32>, #AL> -> tensor<2x32x32xf32, #A>
    triton_gpu.async_commit_group
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK: basic_splat
  func.func @basic_splat(%ptr: !tt.ptr<f32>) {
    // CHECK: llvm.mlir.undef
    // CHECK: llvm.insertvalue
    // CHECK: llvm.insertvalue
    %0 = tt.splat %ptr : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>,#blocked0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_store
  func.func @basic_store(%ptrs: tensor<256x!tt.ptr<f32>, #blocked0>, %vals: tensor<256xf32, #blocked0>, %mask: tensor<256xi1, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    // CHECK: llvm.inline_asm
    // CHECK-SAME: st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    tt.store %ptrs, %vals, %mask : tensor<256xf32, #blocked0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global external @global_smem() {addr_space = 3 : i32} : !llvm.array<0 x i8>
  // CHECK-LABEL: convert_layout_blocked_blocked
  func.func @convert_layout_blocked_blocked(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<1xf32>, 3>
    %0 = triton_gpu.convert_layout %arg0 : (tensor<16x16xf32, #blocked0>) -> tensor<16x16xf32, #blocked1>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global external @global_smem() {addr_space = 3 : i32} : !llvm.array<0 x i8>
  // CHECK-LABEL: convert_layout_blocked_blocked_vec
  func.func @convert_layout_blocked_blocked_vec(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<4xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<4xf32>, 3>
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<4xf32>, 3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<4xf32>, 3>
    %0 = triton_gpu.convert_layout %arg0 : (tensor<16x16xf32, #blocked0>) -> tensor<16x16xf32, #blocked1>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global external @global_smem() {addr_space = 3 : i32} : !llvm.array<0 x i8>
  // CHECK-LABEL: convert_layout_blocked_blocked_multi_rep
  func.func @convert_layout_blocked_blocked_multi_rep(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<4xf32>, 3>
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<4xf32>, 3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<4xf32>, 3>
    // CHECK: nvvm.barrier0
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<4xf32>, 3>
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<4xf32>, 3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<4xf32>, 3>
    %0 = triton_gpu.convert_layout %arg0 : (tensor<16x16xf32, #blocked0>) -> tensor<16x16xf32, #blocked1>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 1, perPhase=2, maxPhase=8 ,order = [1, 0]}>
#mma0 = #triton_gpu.mma<{versionMajor=2, warpsPerCTA=[1,1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot
  func.func @convert_dot(%A: tensor<16x16xf16, #blocked0>, %B: tensor<16x16xf16, #blocked0>) {
    %AA = triton_gpu.convert_layout %A : (tensor<16x16xf16, #blocked0>) -> tensor<16x16xf16, #shared0>
    %BB = triton_gpu.convert_layout %B : (tensor<16x16xf16, #blocked0>) -> tensor<16x16xf16, #shared0>
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ldmatrix.sync.aligned.m8n8.x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ldmatrix.sync.aligned.m8n8.x4
    %AA_DOT = triton_gpu.convert_layout %AA : (tensor<16x16xf16, #shared0>) -> tensor<16x16xf16, #dot_operand_a>
    %BB_DOT = triton_gpu.convert_layout %BB : (tensor<16x16xf16, #shared0>) -> tensor<16x16xf16, #dot_operand_b>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma0>

    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    %D = tt.dot %AA_DOT, %BB_DOT, %cst0 {allowTF32 = true, transA = false, transB = false} : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>

    return
  }
}

// TODO: problems in MLIR's parser on slice layout
// #blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
// module attributes {"triton_gpu.num-warps" = 1 : i32} {
//   func.func @make_range_sliced_layout() {
//     %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked0}>>
//     return
//   }
// }

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #triton_gpu.mma<{versionMajor = 2, warpsPerCTA = [2, 2]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global external @global_smem() {addr_space = 3 : i32} : !llvm.array<0 x i8>
  // CHECK-LABEL: convert_layout_mmav2_block
  func.func @convert_layout_mmav2_blocked(%arg0: tensor<32x16xf32, #mma>) {
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<2xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<2xf32>, 3>
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<4xf32>, 3>
    %0 = triton_gpu.convert_layout %arg0 : (tensor<32x16xf32, #mma>) -> tensor<32x16xf32, #blocked0>
    return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #triton_gpu.mma<{versionMajor = 1, versionMinor = 3, warpsPerCTA = [2, 2]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global external @global_smem() {addr_space = 3 : i32} : !llvm.array<0 x i8>
  // CHECK-LABEL: convert_layout_mmav1_block
  func.func @convert_layout_mmav1_blocked(%arg0: tensor<32x64xf32, #mma>) {
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<2xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<2xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<2xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<2xf32>, 3>
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<vector<4xf32>, 3>
    %0 = triton_gpu.convert_layout %arg0 : (tensor<32x64xf32, #mma>) -> tensor<32x64xf32, #blocked>
    return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global external @global_smem() {addr_space = 3 : i32} : !llvm.array<0 x i8>
  // CHECK-LABEL: convert_layout_blocked_shared
  func.func @convert_layout_blocked_shared(%arg0: tensor<128x32xf32, #blocked0>) {
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<8xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<8xf32>, 3>
    %0 = triton_gpu.convert_layout %arg0 : (tensor<128x32xf32, #blocked0>) -> tensor<128x32xf32, #shared0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_blocked1d_to_slice0
  func.func @convert_blocked1d_to_slice0(%src:tensor<32xi32, #blocked0>) {
    // CHECK-COUNT-4: llvm.load {{.*}} : !llvm.ptr<vector<1xi32>, 3>
    %cvt = triton_gpu.convert_layout %src : (tensor<32xi32, #blocked0>) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_blocked1d_to_slice1
  func.func @convert_blocked1d_to_slice1(%src:tensor<32xi32, #blocked0>) {
    // CHECK-COUNT-32: llvm.load {{.*}} : !llvm.ptr<vector<1xi32>, 3>
    %cvt = triton_gpu.convert_layout %src : (tensor<32xi32, #blocked0>) -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_blocked_to_blocked_ptr
  func.func @convert_blocked_to_blocked_ptr(%src:tensor<32x!tt.ptr<f32>, #blocked0>) {
    // CHECK: llvm.ptrtoint
    // CHECK: llvm.store
    // CHECK: nvvm.barrier0
    // CHECK: llvm.inttoptr
    // CHECK-COUNT-4: llvm.insertvalue
    %cvt = triton_gpu.convert_layout %src : (tensor<32x!tt.ptr<f32>, #blocked0>) -> tensor<32x!tt.ptr<f32>, #blocked1>
    return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#mma = #triton_gpu.mma<{versionMajor = 2, warpsPerCTA = [2, 2]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  func.func @matmul_kernel_dot_operand_layout(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:tensor<128x32xf16, #shared>, %b:tensor<32x256xf16, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    // CHECK: ldmatrix.sync.aligned.m8n8.x4.shared.b16
    %a_mat = triton_gpu.convert_layout %a : (tensor<128x32xf16, #shared>) -> tensor<128x32xf16, #dot_operand_a>
    %b_mat = triton_gpu.convert_layout %b : (tensor<32x256xf16, #shared>) -> tensor<32x256xf16, #dot_operand_b>

    %28 = tt.dot %a_mat, %b_mat, %cst {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #dot_operand_a> * tensor<32x256xf16, #dot_operand_b> -> tensor<128x256xf32, #mma>
    %38 = triton_gpu.convert_layout %28 : (tensor<128x256xf32, #mma>) -> tensor<128x256xf32, #blocked>

    %30 = tt.splat %ptr : (!tt.ptr<f32>) -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : (tensor<128x1x!tt.ptr<f32>, #blocked>) -> tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<128x256xf32, #blocked>
    return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#mma = #triton_gpu.mma<{versionMajor = 1, versionMinor = 3, warpsPerCTA = [2, 2]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma, isMMAv1Row=true}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma, isMMAv1Row=true}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  func.func @matmul884_kernel_dot_operand_layout(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:tensor<32x64xf16, #shared0>, %b:tensor<64x64xf16, #shared1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
    // CHECK: ldmatrix.sync.aligned.m8n8.x4.shared.b16
    %a_mat = triton_gpu.convert_layout %a : (tensor<32x64xf16, #shared0>) -> tensor<32x64xf16, #dot_operand_a>
    %b_mat = triton_gpu.convert_layout %b : (tensor<64x64xf16, #shared1>) -> tensor<64x64xf16, #dot_operand_b>

    %28 = tt.dot %a_mat, %b_mat, %cst {allowTF32 = true, transA = false, transB = false} : tensor<32x64xf16, #dot_operand_a> * tensor<64x64xf16, #dot_operand_b> -> tensor<32x64xf32, #mma>
    %38 = triton_gpu.convert_layout %28 : (tensor<32x64xf32, #mma>) -> tensor<32x64xf32, #blocked>
    %30 = tt.splat %ptr : (!tt.ptr<f32>) -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : (tensor<32x1x!tt.ptr<f32>, #blocked>) -> tensor<32x64x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<32x64xf32, #blocked>
    return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  func.func @matmul_fmadot(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:tensor<32x16xf32, #shared>, %b:tensor<16x32xf32, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    // CHECK: llvm.intr.fmuladd
    %a_mat = triton_gpu.convert_layout %a : (tensor<32x16xf32, #shared>) -> tensor<32x16xf32, #dot_operand_a>
    %b_mat = triton_gpu.convert_layout %b : (tensor<16x32xf32, #shared>) -> tensor<16x32xf32, #dot_operand_b>

    %28 = tt.dot %a_mat, %b_mat, %cst {allowTF32 = false, transA = false, transB = false} : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #blocked>
    %30 = tt.splat %ptr : (!tt.ptr<f32>) -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : (tensor<32x1x!tt.ptr<f32>, #blocked>) -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %36, %28 : tensor<32x32xf32, #blocked>
    return
  }
}

// -----

#mma = #triton_gpu.mma<{versionMajor=2, warpsPerCTA=[2, 2]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: matmul_tf32dot
  func.func @matmul_tf32dot(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:tensor<32x16xf32, #shared>, %b:tensor<16x32xf32, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ldmatrix.sync.aligned.m8n8.x4.shared.b16
    // CHECK-SAME: (vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>)
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ldmatrix.sync.aligned.m8n8.x4.shared.b16
    // CHECK-SAME: (vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>)
    %a_mat = triton_gpu.convert_layout %a : (tensor<32x16xf32, #shared>) -> tensor<32x16xf32, #dot_operand_a>
    %b_mat = triton_gpu.convert_layout %b : (tensor<16x32xf32, #shared>) -> tensor<16x32xf32, #dot_operand_b>

    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    %28 = tt.dot %a_mat, %b_mat, %cst {allowTF32 = true, transA = false, transB = false} : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #mma>
    %38 = triton_gpu.convert_layout %28 : (tensor<32x32xf32, #mma>) -> tensor<32x32xf32, #blocked>

    %30 = tt.splat %ptr : (!tt.ptr<f32>) -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : (tensor<32x1x!tt.ptr<f32>, #blocked>) -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<32x32xf32, #blocked>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32
  func.func @atomic_add_f32(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xi1, #blocked0>, %arg2 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.icmp "slt"
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$3 atom.global.gpu.add.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$3 atom.global.gpu.add.f32
    %0 = "tt.atomic_rmw" (%arg0, %arg2, %arg1) {atomic_rmw_op = 5 : i32} : (tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xf32, #blocked0>, tensor<256xi1, #blocked0>) -> tensor<256xf32, #blocked0>
    return
  }
}

// -----

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32_scalar
  func.func @atomic_add_f32_scalar(%arg0 : !tt.ptr<f32>, %arg1 : i1, %arg2 : f32) {
    // CHECK: llvm.icmp "eq"
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$3 atom.global.gpu.add.f32
    %0 = "tt.atomic_rmw" (%arg0, %arg2, %arg1) {atomic_rmw_op = 5 : i32} : (!tt.ptr<f32>, f32, i1) -> f32
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: store_f32
  func.func @store_f32(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.icmp "slt"
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$2 st.global.b32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$2 st.global.b32
    tt.store %arg0, %arg1 : tensor<256xf32, #blocked0>
    return
  }
}

// -----

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: store_f32_scalar
  func.func @store_f32_scalar(%arg0 : !tt.ptr<f32>, %arg1 : f32) {
    // CHECK: llvm.icmp "slt"
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$2 st.global.b32
    tt.store %arg0, %arg1 : f32
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {

func.func @test_get_program_id(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
  %blockidx = tt.get_program_id {axis=0:i32} : i32
  %blockidy = tt.get_program_id {axis=1:i32} : i32
  %blockidz = tt.get_program_id {axis=2:i32} : i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.x
  // CHECK: nvvm.read.ptx.sreg.ctaid.y
  // CHECK: nvvm.read.ptx.sreg.ctaid.z
  %v0 = arith.addi %blockidx, %blockidy : i32
  %v1 = arith.addi %v0, %blockidz : i32
  %0 = tt.splat %v1 : (i32) -> tensor<32xi32, #blocked0>
  tt.store %a, %0 : tensor<32xi32, #blocked0>

  return
}

}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  func.func @test_get_num_program(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
    // CHECK: nvvm.read.ptx.sreg.nctaid.x
    // CHECK: nvvm.read.ptx.sreg.nctaid.y
    // CHECK: nvvm.read.ptx.sreg.nctaid.z
    %blockdimx = tt.get_num_programs {axis=0:i32} : i32
    %blockdimy = tt.get_num_programs {axis=1:i32} : i32
    %blockdimz = tt.get_num_programs {axis=2:i32} : i32
    %v0 = arith.addi %blockdimx, %blockdimy : i32
    %v1 = arith.addi %v0, %blockdimz : i32
    %0 = tt.splat %v1 : (i32) -> tensor<32xi32, #blocked0>
    tt.store %a, %0 : tensor<32xi32, #blocked0>

    return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: test_index_cache
  func.func @test_index_cache() {
    // CHECK: nvvm.read.ptx.sreg.tid.x
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    // CHECK-NOT: nvvm.read.ptx.sreg.tid.x
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: test_base_index_cache
  func.func @test_base_index_cache(%arg0: tensor<128x32xf32, #blocked0>) {
    // CHECK: nvvm.read.ptx.sreg.tid.x
    %0 = triton_gpu.convert_layout %arg0 : (tensor<128x32xf32, #blocked0>) -> tensor<128x32xf32, #shared0>
    // CHECK-NOT: nvvm.read.ptx.sreg.tid.x
    %1 = triton_gpu.convert_layout %arg0 : (tensor<128x32xf32, #blocked0>) -> tensor<128x32xf32, #shared0>
    return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: test_index_cache_different_block
  func.func @test_index_cache_different_block(%arg0: tensor<128x32xf32, #blocked0>, %arg1: i1) {
    // CHECK: nvvm.read.ptx.sreg.tid.x
    %0 = triton_gpu.convert_layout %arg0 : (tensor<128x32xf32, #blocked0>) -> tensor<128x32xf32, #shared0>
    scf.if %arg1 {
      // CHECK-NOT: nvvm.read.ptx.sreg.tid.x
      %1 = triton_gpu.convert_layout %arg0 : (tensor<128x32xf32, #blocked0>) -> tensor<128x32xf32, #shared0>
    }
    return
  }
}
