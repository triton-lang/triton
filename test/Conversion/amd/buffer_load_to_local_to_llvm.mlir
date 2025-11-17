// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx950 | FileCheck %s --check-prefixes=COMMON,GFX950
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --verify-diagnostics | FileCheck %s --check-prefixes=COMMON,GFX942

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_to_local_simple
  tt.func public @buffer_load_to_local_simple(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: !tt.ptr<f32>,
                                %arg2: tensor<32x64xi32, #blocked>,
                                %arg3: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>) {
    // Each thread needs to load 8 elements and we load 1 (sizePerThread) per buffer load instruction
    // COMMON: rocdl.make.buffer.rsrc
    // COMMON-NOT: rocdl.make.buffer.rsrc
    // COMMON-COUNT-8: rocdl.raw.ptr.buffer.load.lds
    // COMMON-NOT: rocdl.raw.ptr.buffer.load.lds
    %65 = amdg.buffer_load_to_local %arg1[%arg2] into %arg3 : <f32>[tensor<32x64xi32, #blocked>] -> <32x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 32], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 0 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_to_local_vectorized_2xf16
  tt.func public @buffer_load_to_local_vectorized_2xf16(%arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>) {
    %cst = arith.constant dense<64> : tensor<1x64xi32, #blocked>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %4 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %5 = arith.muli %4, %cst : tensor<1x64xi32, #blocked>
    %6 = tt.broadcast %5 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %7 = arith.addi %3, %6 : tensor<64x64xi32, #blocked>

    // Each thread needs to load 2 elements and we load 2 (sizePerThread) per buffer load instruction
    // COMMON: rocdl.make.buffer.rsrc
    // COMMON-NOT: rocdl.make.buffer.rsrc
    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON-NOT: rocdl.raw.ptr.buffer.load.lds
    %8 = amdg.buffer_load_to_local %arg1[%7] into %arg2 : <f16>[tensor<64x64xi32, #blocked>]  -> <64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 32], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 0 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_to_local_vectorized_8xf16
  tt.func public @buffer_load_to_local_vectorized_8xf16(%arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>) {
    %cst = arith.constant dense<64> : tensor<1x64xi32, #blocked>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %4 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %5 = arith.muli %4, %cst : tensor<1x64xi32, #blocked>
    %6 = tt.broadcast %5 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %7 = arith.addi %3, %6 : tensor<64x64xi32, #blocked>

    // Each thread needs to load 8 elements and we load 8 (sizePerThread) per buffer load instruction
    // GFX950: rocdl.make.buffer.rsrc
    // GFX950-NOT: rocdl.make.buffer.rsrc
    // GFX950: rocdl.raw.ptr.buffer.load.lds
    // GFX950-NOT: rocdl.raw.ptr.buffer.load.lds

    // GFX942 does not support vectorization > 4bytes so we cannot lower it
    // GFX942-NOT: rocdl.raw.ptr.buffer.load.lds
    // GFX942: amdg.buffer_load_to_local
    %8 = amdg.buffer_load_to_local %arg1[%7] into %arg2 : <f16>[tensor<64x64xi32, #blocked>]  -> <64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_to_local_vectorized_8xf16
  tt.func public @buffer_load_to_local_vectorized_8xf16(%arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !ttg.memdesc<256x8xf16, #shared, #smem, mutable>) {
    %cst = arith.constant dense<8> : tensor<256x1xi32, #blocked>
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %3 = arith.muli %2, %cst : tensor<256x1xi32, #blocked>
    %4 = tt.broadcast %3 : tensor<256x1xi32, #blocked> -> tensor<256x8xi32, #blocked>
    %5 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x8xi32, #blocked>
    %6 = tt.broadcast %5 : tensor<1x8xi32, #blocked> -> tensor<256x8xi32, #blocked>
    %7 = arith.addi %4, %6 : tensor<256x8xi32, #blocked>

    // Each thread needs to load 8 elements and we load 8 (sizePerThread) per buffer load instruction
    // GFX950: rocdl.make.buffer.rsrc
    // GFX950-NOT: rocdl.make.buffer.rsrc
    // GFX950: rocdl.raw.ptr.buffer.load.lds
    // GFX950-NOT: rocdl.raw.ptr.buffer.load.lds

    // GFX942 does not support vectorization > 4bytes so we cannot lower it
    // GFX942-NOT: rocdl.raw.ptr.buffer.load.lds
    // GFX942: amdg.buffer_load_to_local
    %8 = amdg.buffer_load_to_local %arg1[%7] into %arg2 : <f16>[tensor<256x8xi32, #blocked>]  -> <256x8xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_to_local_mask_other
  tt.func public @buffer_load_to_local_mask_other(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: !tt.ptr<f32>,
                                %arg2: tensor<32x32xi32, #blocked>,
                                %arg3: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
                                %arg4: i32) {
    // We need the splat to allow the AxisAnalysis to work during lowering
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %29 = arith.addi %arg4, %c31_i32 : i32
    %30 = arith.divsi %29, %c32_i32 : i32
    %31 = arith.cmpi sgt, %30, %c0_i32 : i32

    %51 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %52 = tt.expand_dims %51 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %65 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #blocked>
    %66 = arith.cmpi slt, %52, %65 : tensor<32x1xi32, #blocked>
    %67 = tt.broadcast %66 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>

    %70 = tt.splat %31 : i1 -> tensor<32x32xi1, #blocked>
    %71 = arith.andi %70, %67 : tensor<32x32xi1, #blocked>

    // Each thread needs to load 4 elements and we load 1 (sizePerThread) per buffer load instruction
    // Note that mask/other alignment is 1 so we need 4 conditionals

    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON: llvm.cond_br
    // COMMON: llvm.store

    // Make sure branch condition is set properly when there is other value.
    // COMMON: [[AND:%.*]] = llvm.and
    // COMMON: llvm.cond_br [[AND]]

    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON: llvm.cond_br
    // COMMON: llvm.store

    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON: llvm.cond_br
    // COMMON: llvm.store

    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON: llvm.cond_br
    // COMMON: llvm.store

    // COMMON-NOT: rocdl.raw.ptr.buffer.load.lds
    // COMMON-NOT: _predicated_store
    // COMMON-NOT: llvm.cond_br
    // COMMON-NOT: llvm.store

    amdg.buffer_load_to_local %arg1[%arg2] mask=%67 other=%cst_0 into %arg3 : <f32>[tensor<32x32xi32, #blocked>] tensor<32x32xf32, #blocked>  -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_to_local_cache_mods
  tt.func public @buffer_load_to_local_cache_mods(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg2: !ttg.memdesc<64xf32, #shared, #smem, mutable>) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    // The first constant 0 skips the LDS offset which is also 0
    // COMMON: %[[VOFFSET:.*]] = llvm.select
    // COMMON-NEXT: %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // COMMON-NEXT: %[[aux_ca:.*]] = llvm.mlir.constant(0 : i32) : i32
    // COMMON-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // COMMON-NEXT: rocdl.raw.ptr.buffer.load.lds {{.*}}, {{.*}}, {{.*}}, %[[VOFFSET]], %[[IMM1]], %[[IMM0]], %[[aux_ca]]
    %1 = amdg.buffer_load_to_local %arg0[%0] cacheModifier = ca into %arg2: <f32>[tensor<64xi32, #blocked>] -> <64xf32, #shared, #smem, mutable>
    // COMMON: llvm.getelementptr
    // COMMON: %[[aux_cg:.*]] = llvm.mlir.constant(3 : i32) : i32
    // COMMON: rocdl.raw.ptr.buffer.load.lds {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[aux_cg]]
    %2 = amdg.buffer_load_to_local %arg0[%0] cacheModifier = cg into %arg2: <f32>[tensor<64xi32, #blocked>] -> <64xf32, #shared, #smem, mutable>
    // COMMON: llvm.getelementptr
    // COMMON: %[[aux_cv:.*]] = llvm.mlir.constant(17 : i32) : i32
    // COMMON: rocdl.raw.ptr.buffer.load.lds {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[aux_cv]]
    %3 = amdg.buffer_load_to_local %arg0[%0] cacheModifier = cv into %arg2: <f32>[tensor<64xi32, #blocked>] -> <64xf32, #shared, #smem, mutable>

    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_swizzled_simple
  tt.func public @buffer_load_swizzled_simple(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: !tt.ptr<f32>,
                                %arg2: tensor<16x64xi32, #blocked>,
                                %arg3: !ttg.memdesc<16x64xf32, #shared, #smem, mutable>) {
    // Each thread needs to load 2 elements and we load 1 (sizePerThread) per buffer load instruction
    // COMMON: rocdl.make.buffer.rsrc
    // COMMON-NOT: rocdl.make.buffer.rsrc
    // COMMON: rocdl.ds_bpermute
    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON: rocdl.ds_bpermute
    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON-NOT: rocdl.raw.ptr.buffer.load.lds
    %65 = amdg.buffer_load_to_local %arg1[%arg2] into %arg3 : <f32>[tensor<16x64xi32, #blocked>] -> <16x64xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_to_local_swizzled_mask_other
  tt.func public @buffer_load_to_local_swizzled_mask_other(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg1: !tt.ptr<f32>,
                                %arg2: tensor<32x32xi32, #blocked>,
                                %arg3: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
                                %arg4: i32) {
    // We need the splat to allow the AxisAnalysis to work during lowering
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %29 = arith.addi %arg4, %c31_i32 : i32
    %30 = arith.divsi %29, %c32_i32 : i32
    %31 = arith.cmpi sgt, %30, %c0_i32 : i32

    %51 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %52 = tt.expand_dims %51 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %65 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #blocked>
    %66 = arith.cmpi slt, %52, %65 : tensor<32x1xi32, #blocked>
    %67 = tt.broadcast %66 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>

    %70 = tt.splat %31 : i1 -> tensor<32x32xi1, #blocked>
    %71 = arith.andi %70, %67 : tensor<32x32xi1, #blocked>

    // Each thread needs to load 4 elements and we load 1 (sizePerThread) per buffer load instruction
    // Note that mask/other alignment is 1 so we need 4 conditionals

    // COMMON: rocdl.ds_bpermute
    // COMMON: rocdl.ballot
    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON: llvm.cond_br
    // COMMON: llvm.store

    // COMMON: rocdl.ds_bpermute
    // COMMON: rocdl.ballot
    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON: llvm.cond_br
    // COMMON: llvm.store

    // COMMON: rocdl.ds_bpermute
    // COMMON: rocdl.ballot
    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON: llvm.cond_br
    // COMMON: llvm.store

    // COMMON: rocdl.ds_bpermute
    // COMMON: rocdl.ballot
    // COMMON: rocdl.raw.ptr.buffer.load.lds
    // COMMON: llvm.cond_br
    // COMMON: llvm.store

    // COMMON-NOT: rocdl.ds_bpermute
    // COMMON-NOT: rocdl.ballot
    // COMMON-NOT: rocdl.raw.ptr.buffer.load.lds
    // COMMON-NOT: _predicated_store

    amdg.buffer_load_to_local %arg1[%arg2] mask=%67 other=%cst_0 into %arg3 : <f32>[tensor<32x32xi32, #blocked>] tensor<32x32xf32, #blocked>  -> <32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 32], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 0 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_to_local_swizzled_vectorized_8xf16
  tt.func public @buffer_load_to_local_swizzled_vectorized_8xf16(%arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>) {
    %cst = arith.constant dense<64> : tensor<1x64xi32, #blocked>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %3 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %4 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %5 = arith.muli %4, %cst : tensor<1x64xi32, #blocked>
    %6 = tt.broadcast %5 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %7 = arith.addi %3, %6 : tensor<64x64xi32, #blocked>

    // Each thread needs to load 8 elements and we load 8 (sizePerThread) per buffer load instruction
    // GFX950: rocdl.make.buffer.rsrc
    // GFX950: rocdl.raw.ptr.buffer.load.lds
    // GFX950-NOT: rocdl.raw.ptr.buffer.load.lds

    // GFX942 does not support vectorization > 4bytes so we cannot lower it
    // GFX942-NOT: rocdl.raw.ptr.buffer.load.lds
    // GFX942: amdg.buffer_load_to_local
    %8 = amdg.buffer_load_to_local %arg1[%7] into %arg2 : <f16>[tensor<64x64xi32, #blocked>]  -> <64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // COMMON-LABEL: buffer_load_to_local_wave_id
  tt.func public @buffer_load_to_local_wave_id(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg2: !ttg.memdesc<64xf32, #shared, #smem, mutable>, %arg3: i32) {
    // COMMON: %0 = rocdl.workitem.id.x : i32
    // COMMON-NEXT: %1 = llvm.mlir.constant(63 : i32) : i32
    // COMMON-NEXT: %2 = llvm.and %0, %1 : i32
    // COMMON-NEXT: %3 = llvm.mlir.constant(64 : i32) : i32
    // COMMON-NEXT: %4 = llvm.mlir.constant(64 : i32) : i32
    // COMMON-NEXT: %5 = rocdl.workitem.id.x : i32
    // COMMON-NEXT: %6 = llvm.mlir.constant(63 : i32) : i32
    // COMMON-NEXT: %7 = llvm.and %5, %6 : i32
    // COMMON-NEXT: %8 = llvm.udiv %7, %4 : i32
    // COMMON-NEXT: %9 = rocdl.readfirstlane %8 : i32
    // COMMON-NEXT: %10 = llvm.mlir.constant(0 : i32) : i32
    // COMMON-NEXT: %11 = rocdl.workitem.id.x : i32
    // COMMON-NEXT: %12 = llvm.mlir.constant(63 : i32) : i32
    // COMMON-NEXT: %13 = llvm.and %11, %12 : i32
    // COMMON-NEXT: %14 = llvm.mlir.constant(64 : i32) : i32
    // COMMON-NEXT: %15 = llvm.mlir.constant(64 : i32) : i32
    // COMMON-NEXT: %16 = rocdl.workitem.id.x : i32
    // COMMON-NEXT: %17 = llvm.mlir.constant(63 : i32) : i32
    // COMMON-NEXT: %18 = llvm.and %16, %17 : i32
    // COMMON-NEXT: %19 = llvm.udiv %18, %15 : i32
    // COMMON-NEXT: %20 = rocdl.readfirstlane %19 : i32

    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %1 = amdg.buffer_load_to_local %arg0[%0] into %arg2: <f32>[tensor<64xi32, #blocked>] -> <64xf32, #shared, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    %cond = llvm.icmp "eq" %arg3, %c0_i32 : i32
    cf.cond_br %cond, ^bb1, ^bb2
    ^bb1:
      amdg.buffer_load_to_local %arg0[%0] into %arg2: <f32>[tensor<64xi32, #blocked>] -> <64xf32, #shared, #smem, mutable>
      cf.br ^bb1
    ^bb2:
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared1D = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // COMMON-LABEL: buffer_load_to_local_contiguity_hint
  tt.func @buffer_load_to_local_contiguity_hint(%ptr: !tt.ptr<f16>, %off: tensor<256xi32, #blocked>, %lds: !ttg.memdesc<256xf16, #shared1D, #smem, mutable>) {
    // Check we load 4 bytes
    // COMMON: %[[LOAD_BYTES:.*]] = llvm.mlir.constant(4 : i32) : i32
    // COMMON: rocdl.raw.ptr.buffer.load.lds %{{.*}}, %{{.*}}, %[[LOAD_BYTES]]
    %0 = amdg.buffer_load_to_local %ptr[%off] into %lds {contiguity = 2 : i32} : <f16>[tensor<256xi32, #blocked>] -> <256xf16, #shared1D, #smem, mutable>
    tt.return
  }
}
