// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck %s

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_load
    tt.func @buffer_load(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0>{tt.divisibility=16:i32}) {
        // CHECK: %[[c_mask:.*]] = llvm.mlir.constant(true) : i1
        // CHECK: %[[offset:.*]] = llvm.select %[[c_mask]]
        // CHECK: %[[aux:.*]] = llvm.mlir.constant(3 : i32) : i32
        // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, %[[offset]], {{.*}}, %[[aux]]
        %c0 = arith.constant 0 : i32
        %ret = amdgpu.buffer_load %arg0[%offset] cacheModifier = cs stride = %c0 : tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_load_mask
    tt.func @buffer_load_mask(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0> {tt.divisibility=16:i32}, %N : i32 {tt.divisibility = 16 : i32}) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<128xi32, #blocked0>
        %5 = tt.splat %N: i32 -> tensor<128xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<128xi32, #blocked0>
        // CHECK: %[[mask:.*]] = llvm.extractvalue %{{.*}} : !llvm.struct<(i1, i1, i1, i1)>
        // CHECK: %[[offset:.*]] = llvm.select %[[mask]]
        // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, %[[offset]]
        %ret = amdgpu.buffer_load %arg0[%offset], %7 stride = %c256_i32 : tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_load_mask_other
    tt.func @buffer_load_mask_other(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0> {tt.divisibility=16:i32}, %N : i32 {tt.divisibility = 16 : i32}) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<128xi32, #blocked0>
        %5 = tt.splat %N: i32 -> tensor<128xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<128xi32, #blocked0>
        %other = arith.constant dense<0.00e+00> : tensor<128xf32, #blocked0>
        // CHECK: %[[mask:.*]] = llvm.extractvalue %{{.*}} : !llvm.struct<(i1, i1, i1, i1)>
        // CHECK: %[[offset:.*]] = llvm.select %[[mask]]
        // CHECK: rocdl.raw.ptr.buffer.load {{.*}}, %[[offset]]
        // CHECK: llvm.select
        %ret = amdgpu.buffer_load %arg0[%offset], %7, %other stride = %c256_i32: tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_store
    tt.func @buffer_store(%value : tensor<128xf32, #blocked0>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0>{tt.divisibility=16:i32}) {
        // CHECK: %[[c_mask:.*]] = llvm.mlir.constant(true) : i1
        // CHECK: %[[offset:.*]] = llvm.select %[[c_mask]]
        // CHECK: %[[aux:.*]] = llvm.mlir.constant(3 : i32) : i32
        // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, %[[offset]], {{.*}}, %[[aux]]
        %c256_i32 = arith.constant 256 : i32
        amdgpu.buffer_store %value, %arg0[%offset] cacheModifier = cs stride = %c256_i32 : tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_store_mask
    tt.func @buffer_store_mask(%value : tensor<128xf32, #blocked0>, %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0> {tt.divisibility=16:i32}, %N : i32 {tt.divisibility = 16 : i32}) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<128xi32, #blocked0>
        %5 = tt.splat %N: i32 -> tensor<128xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<128xi32, #blocked0>
        // CHECK: %[[mask0:.*]] = llvm.extractvalue %{{.*}} : !llvm.struct<(i1, i1, i1, i1)>
        // CHECK: %[[mask1:.*]] = llvm.and %[[mask0]], {{.*}}
        // CHECK: %[[offset:.*]] = llvm.select %[[mask1]]
        // CHECK: rocdl.raw.ptr.buffer.store {{.*}}, {{.*}}, %[[offset]]
        amdgpu.buffer_store %value, %arg0[%offset], %7 stride = %N : tensor<128xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: buffer_load_store_vec4
    tt.func @buffer_load_store_vec4(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
        // Load 8 elements from A with two vectorized load instructions
        // CHECK-COUNT-2: rocdl.raw.ptr.buffer.load {{.*}} : vector<4xf32>
        %9 = amdgpu.buffer_load %arg0[%4] stride = %arg3 : tensor<256xf32, #blocked0>
        // Load 8 elements from B with two vectorized load instructions
        // CHECK-COUNT-2: rocdl.raw.ptr.buffer.load {{.*}} : vector<4xf32>
        %10 = amdgpu.buffer_load %arg1[%4] stride = %arg3 : tensor<256xf32, #blocked0>
        %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
        // Store 8 elements into C with two vectorized store instructions
        // CHECK-COUNT-2: rocdl.raw.ptr.buffer.store {{.*}} : vector<4xf32>
        amdgpu.buffer_store %11, %arg2[%4] stride = %arg3 : tensor<256xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: buffer_load_store_vec1
    tt.func @buffer_load_store_vec1(%arg0: !tt.ptr<f32> , %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
        %5 = tt.splat %arg3 : i32 -> tensor<256xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<256xi32, #blocked0>
        // Load 8 elements from A with eight scalar load instructions
        // CHECK-COUNT-8: rocdl.raw.ptr.buffer.load {{.*}} : f32
        %9 = amdgpu.buffer_load %arg0[%4], %7 stride = %arg3 : tensor<256xf32, #blocked0>
        // Load 8 elements from B with two scalar load instructions
        // CHECK-COUNT-8: rocdl.raw.ptr.buffer.load {{.*}} : f32
        %10 = amdgpu.buffer_load %arg1[%4], %7 stride = %arg3 : tensor<256xf32, #blocked0>
        %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
        // Store 8 elements into C with two scalar store instructions
        // CHECK-COUNT-8: rocdl.raw.ptr.buffer.store {{.*}} : f32
        amdgpu.buffer_store %11, %arg2[%4], %7 stride = %arg3 : tensor<256xf32, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_load_store_vec2
    tt.func @buffer_load_store_vec2(%arg0: !tt.ptr<f16> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f16>{tt.divisibility = 4 : i32}, %arg2: !tt.ptr<f16>{tt.divisibility = 4: i32}, %arg3: i32{tt.divisibility = 4: i32}) {
        %c256_i32 = arith.constant 256 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c256_i32 : i32
        %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
        %5 = tt.splat %arg3 : i32 -> tensor<256xi32, #blocked0>
        %7 = arith.cmpi slt, %4, %5: tensor<256xi32, #blocked0>
        // Load 8 fp16 elements from A with four i32 scalar load instructions
        // CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}} : i32
        %9 = amdgpu.buffer_load %arg0[%4], %7 stride = %arg3 : tensor<256xf16, #blocked0>
        // Load 8 fp16 elements from B with four i32 scalar load instructions
        // CHECK-COUNT-4: rocdl.raw.ptr.buffer.load {{.*}} : i32
        %10 = amdgpu.buffer_load %arg1[%4], %7 stride = %arg3 : tensor<256xf16, #blocked0>
        %11 = arith.addf %9, %10 : tensor<256xf16, #blocked0>
        // Store 8 fp16 elements into C with four i32 scalar store instructionss
        // CHECK-COUNT-4: rocdl.raw.ptr.buffer.store {{.*}} : i32
        amdgpu.buffer_store %11, %arg2[%4], %7 stride = %arg3 : tensor<256xf16, #blocked0>
        tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
    // CHECK-LABEL: buffer_atomic
    tt.func @buffer_atomic_rmw_fadd(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked0>{tt.divisibility=16:i32}, %N: i32, %values : tensor<128xf32, #blocked0>) {
        %c128_i32 = arith.constant 128 : i32
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c128_i32 : i32
        %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %3 = tt.splat %1 : i32 -> tensor<128xi32, #blocked0>
        %4 = arith.addi %3, %2 : tensor<128xi32, #blocked0>
        %5 = tt.splat %N: i32 -> tensor<128xi32, #blocked0>
        %mask = arith.cmpi slt, %4, %5: tensor<128xi32, #blocked0>
        // CHECK: %[[mask0:.*]] = llvm.extractvalue %{{.*}} : !llvm.struct<(i1, i1, i1, i1)>
        // There should be a single release fence before any atomics
        // CHECK: llvm.fence syncscope("agent") release
        // CHECK: %[[mask1:.*]] = llvm.and %[[mask0]], {{.*}}
        // CHECK: %[[offset:.*]] = llvm.select %[[mask1]]

        // We will have 4 calls to fadd, since the sizePerThread is 4. We should have a vmcnt between each call.
        %ret = amdgpu.buffer_atomic_rmw fadd, acq_rel, gpu, %values, %arg0[%offset], %mask : tensor<128xf32, #blocked0>

        // CHECK: %[[result:.*]] = llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"({{.*}}, {{.*}}, %[[mask1:.*]], {{.*}}, {{.*}}) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32
        // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "s_waitcnt vmcnt(0) ", ""  : () -> !llvm.void
        // CHECK: %[[result:.*]] = llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"({{.*}}, {{.*}}, %[[mask1:.*]], {{.*}}, {{.*}}) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32
        // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "s_waitcnt vmcnt(0) ", ""  : () -> !llvm.void
        // CHECK: %[[result:.*]] = llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"({{.*}}, {{.*}}, %[[mask1:.*]], {{.*}}, {{.*}}) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32
        // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "s_waitcnt vmcnt(0) ", ""  : () -> !llvm.void
        // CHECK: %[[result:.*]] = llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"({{.*}}, {{.*}}, %[[mask1:.*]], {{.*}}, {{.*}}) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32

        // There should be a single acquire fence after all of the atomics
        // CHECK: llvm.fence syncscope("agent") acquire
        tt.return
  }
}
