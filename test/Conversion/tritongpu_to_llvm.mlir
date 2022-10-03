// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm | FileCheck %s

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.func @test_empty_kernel(%arg0: i32, %arg1: !llvm.ptr<f16, 1>)
  // Here the 128 comes from the 4 in module attribute multiples 32
  // CHECK:  attributes {nvvm.kernel = 1 : ui1, nvvm.maxntid = 128 : i32} {{.*}}
  func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {
    // CHECK:  llvm.return
    return
  }
} // end module

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_load
  func @basic_load(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
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
  func @vectorized_load(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
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
  func @vectorized_load_f16(%a_ptr_init: tensor<256x!tt.ptr<f16>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf16, #blocked0>) {
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
  func @masked_load_const_other(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>) {
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
  func @masked_load_const_other_vec(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked0>
    %1 = tt.load %a_ptr_init, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"triton_gpu.num-warps" = 2 : i32} {
  // CHECK-LABEL: global_load_store_no_vec
  func @global_load_store_no_vec(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : (i32) -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>
    %7 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>

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
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>

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
  func @global_load_store_vec4(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : (i32) -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>
    %7 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>

    // Load 4 elements from A with single one vectorized load instruction
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 4 elements from B with single one vectorized load instruction
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    %9 = tt.load %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    %10 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>

    // Store 4 elements to global with single one vectorized store instruction
    // CHECK: @$5 st.global.b32.v4 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} };
    tt.store %13, %11 : tensor<256xf32, #blocked0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_store_vec8
    func @global_load_store_vec8(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : (i32) -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>
    %7 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>

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
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>

    // Store 8 elements to global with two vectorized store instruction
    // CHECK: @$5 st.global.b32.v4 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} };
    // CHECK: @$5 st.global.b32.v4 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} };
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
  func @basic_view_broadcast(%arg : tensor<256xf32,#blocked0>) {
    // CHECK: llvm.mlir.undef
    // CHECK: %[[T0:.*]] = llvm.extractvalue
    // CHECK: %[[T1:.*]] = llvm.extractvalue
    %0 = tt.view %arg : (tensor<256xf32, #blocked0>) -> tensor<256x1xf32,#blocked2>
    // CHECK: llvm.mlir.undef
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T1]]
    %1 = tt.broadcast %0 : (tensor<256x1xf32,#blocked2>) -> tensor<256x4xf32, #blocked2>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_make_range
  func @basic_make_range() {
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
  func @basic_addf(%arg0 : tensor<256xf32,#blocked0>, %arg1 : tensor<256xf32,#blocked0>) {
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
  func @basic_addi(%arg0 : tensor<256xi32,#blocked0>, %arg1 : tensor<256xi32,#blocked0>) {
    // CHECK: llvm.add
    // CHECK: llvm.add
    %1 = arith.addi %arg0, %arg1 : tensor<256xi32,#blocked0>
    return
  }
}

// -----

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_program_id
  func @basic_program_id() {
    // CHECK: nvvm.read.ptx.sreg.ctaid.x : i32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_addptr
  func @basic_addptr(%arg0 : tensor<256x!tt.ptr<f32>,#blocked0>, %arg1 : tensor<256xi32,#blocked0>) {
    // CHECK: llvm.getelementptr
    // CHECK: llvm.getelementptr
    %0 = tt.addptr %arg0, %arg1 : tensor<256x!tt.ptr<f32>, #blocked0>
    return
  }
}

// -----

#shared0 = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.mlir.global internal @global_smem
  // CHECK-LABEL: basic_alloc_tensor
  func @basic_alloc_tensor() {
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
  // CHECK: llvm.mlir.global internal @global_smem
  // CHECK-LABEL: basic_extract_slice
  func @basic_extract_slice() {
    // CHECK: %[[BASE0:.*]] = llvm.mlir.addressof @global_smem
    // CHECK-NEXT: %[[BASE1:.*]] = llvm.bitcast %[[BASE0]]
    // CHECK-NEXT: %[[OFFSET0:.*]] = llvm.mlir.constant
    // CHECK-NEXT: %[[OFFSET1:.*]] = llvm.mlir.constant
    // CHECK-NEXT: llvm.getelementptr %[[BASE1]][%[[OFFSET1]]]
    // CHECK-NEXT: %[[BASE2:.*]] = llvm.bitcast
    // CHECK-NEXT: %[[OFFSET2:.*]] = llvm.mlir.constant
    // CHECK-NEXT: %[[OFFSET3:.*]] = llvm.mul %[[OFFSET0]], %[[OFFSET2]]
    // CHECK-NEXT: llvm.getelementptr %[[BASE2]][%[[OFFSET3]]]
    %index = arith.constant 1 : i32
    %0 = triton_gpu.alloc_tensor : tensor<128x16x32xf32, #shared0>
    %1 = triton_gpu.extract_slice %0, %index {axis = 0: i32} : tensor<128x16x32xf32, #shared0> -> tensor<16x32xf32, #shared0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK: basic_splat
  func @basic_splat(%ptr: !tt.ptr<f32>) {
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
  func @basic_store(%ptrs: tensor<256x!tt.ptr<f32>, #blocked0>, %vals: tensor<256xf32, #blocked0>, %mask: tensor<256xi1, #blocked0>) {
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-SAME: st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-SAME: st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    tt.store %ptrs, %vals, %mask : tensor<256xf32, #blocked0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global internal @global_smem() {addr_space = 3 : i32} : !llvm.array<1088 x i8>
  // CHECK-LABEL: convert_layout_blocked_blocked
  func @convert_layout_blocked_blocked(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: nvvm.barrier0
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
  // CHECK: llvm.mlir.global internal @global_smem() {addr_space = 3 : i32} : !llvm.array<1280 x i8>
  // CHECK-LABEL: convert_layout_blocked_blocked_vec
  func @convert_layout_blocked_blocked_vec(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: nvvm.barrier0
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
  // CHECK: llvm.mlir.global internal @global_smem() {addr_space = 3 : i32} : !llvm.array<640 x i8>
  // CHECK-LABEL: convert_layout_blocked_blocked_multi_rep
  func @convert_layout_blocked_blocked_multi_rep(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: nvvm.barrier0
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
#mma0 = #triton_gpu.mma<{version=2, warpsPerCTA=[1,1]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot
  func @convert_dot(%A: tensor<16x16xf16, #blocked0>, %B: tensor<16x16xf16, #blocked0>) {
    %AA = triton_gpu.convert_layout %A : (tensor<16x16xf16, #blocked0>) -> tensor<16x16xf16, #shared0>
    %BB = triton_gpu.convert_layout %B : (tensor<16x16xf16, #blocked0>) -> tensor<16x16xf16, #shared0>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma0>
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ldmatrix.sync.aligned.m8n8.x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ldmatrix.sync.aligned.m8n8.x4

    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    %D = tt.dot %AA, %BB, %cst0 {allowTF32 = true} : tensor<16x16xf16, #shared0> * tensor<16x16xf16, #shared0> -> tensor<16x16xf32, #mma0>

    return
  }
}

// TODO: problems in MLIR's parser on slice layout
// #blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
// module attributes {"triton_gpu.num-warps" = 1 : i32} {
//   func @make_range_sliced_layout() {
//     %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked0}>>
//     return
//   }
// }

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #triton_gpu.mma<{version = 2, warpsPerCTA = [2, 2]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global internal @global_smem() {addr_space = 3 : i32} : !llvm.array<2560 x i8>
  // CHECK-LABEL: convert_layout_mma_block
  func @convert_layout_mma_blocked(%arg0: tensor<32x16xf32, #mma>) {
    // CHECK: nvvm.barrier0
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

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global internal @global_smem() {addr_space = 3 : i32} : !llvm.array<16384 x i8>
  // CHECK-LABEL: convert_layout_blocked_shared
  func @convert_layout_blocked_shared(%arg0: tensor<128x32xf32, #blocked0>) {
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<8xf32>, 3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<vector<8xf32>, 3>
    %0 = triton_gpu.convert_layout %arg0 : (tensor<128x32xf32, #blocked0>) -> tensor<128x32xf32, #shared0>
    return
  }
}