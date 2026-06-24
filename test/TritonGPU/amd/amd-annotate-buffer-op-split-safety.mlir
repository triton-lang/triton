// RUN: triton-opt %s -split-input-file --tritonamdgpu-annotate-buffer-op-split-safety | FileCheck %s

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: @no_split_negative_summand
    // CHECK: amdg.buffer_load
    // CHECK-NOT: amdgpu.split_soffset_safe
    tt.func @no_split_negative_summand(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
        %cn64 = arith.constant -64 : i32
        %neg = tt.splat %cn64 : i32 -> tensor<128xi32, #blocked0>
        %range = tt.make_range {end = 192 : i32, start = 64 : i32} : tensor<128xi32, #blocked0>
        %offset = arith.addi %neg, %range : tensor<128xi32, #blocked0>
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: @split_all_nonneg
    // CHECK: amdg.buffer_load
    // CHECK-SAME: amdgpu.split_soffset_safe
    tt.func @split_all_nonneg(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
        %c5 = arith.constant 5 : i32
        %base = tt.splat %c5 : i32 -> tensor<128xi32, #blocked0>
        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
        %offset = arith.addi %base, %range : tensor<128xi32, #blocked0>
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: @no_split_negative_summand_store
    // CHECK: amdg.buffer_store
    // CHECK-NOT: amdgpu.split_soffset_safe
    tt.func @no_split_negative_summand_store(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %value : tensor<128xf32, #blocked0>) {
        %cn64 = arith.constant -64 : i32
        %neg = tt.splat %cn64 : i32 -> tensor<128xi32, #blocked0>
        %range = tt.make_range {end = 192 : i32, start = 64 : i32} : tensor<128xi32, #blocked0>
        %offset = arith.addi %neg, %range : tensor<128xi32, #blocked0>
        amdg.buffer_store %value, %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: @no_split_convert_layout_negative_summand
    // CHECK: amdg.buffer_load
    // CHECK-NOT: amdgpu.split_soffset_safe
    tt.func @no_split_convert_layout_negative_summand(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
        %cn64 = arith.constant -64 : i32
        %neg = tt.splat %cn64 : i32 -> tensor<128xi32, #blocked1>
        %range = tt.make_range {end = 192 : i32, start = 64 : i32} : tensor<128xi32, #blocked1>
        %add = arith.addi %neg, %range : tensor<128xi32, #blocked1>
        %offset = ttg.convert_layout %add : tensor<128xi32, #blocked1> -> tensor<128xi32, #blocked0>
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: @split_convert_layout_all_nonneg
    // CHECK: amdg.buffer_load
    // CHECK-SAME: amdgpu.split_soffset_safe
    tt.func @split_convert_layout_all_nonneg(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
        %c5 = arith.constant 5 : i32
        %base = tt.splat %c5 : i32 -> tensor<128xi32, #blocked1>
        %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
        %add = arith.addi %base, %range : tensor<128xi32, #blocked1>
        %offset = ttg.convert_layout %add : tensor<128xi32, #blocked1> -> tensor<128xi32, #blocked0>
        %ret = amdg.buffer_load %arg0[%offset] : tensor<128xf32, #blocked0>
        tt.return
    }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked2d = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: @no_split_reshape_negative_summand
    // CHECK: amdg.buffer_load
    // CHECK-NOT: amdgpu.split_soffset_safe
    tt.func @no_split_reshape_negative_summand(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
        %cn64 = arith.constant -64 : i32
        %neg = tt.splat %cn64 : i32 -> tensor<128xi32, #blocked0>
        %range = tt.make_range {end = 192 : i32, start = 64 : i32} : tensor<128xi32, #blocked0>
        %add = arith.addi %neg, %range : tensor<128xi32, #blocked0>
        %offset = tt.reshape %add allow_reorder : tensor<128xi32, #blocked0> -> tensor<1x128xi32, #blocked2d>
        %ret = amdg.buffer_load %arg0[%offset] : tensor<1x128xf32, #blocked2d>
        tt.return
    }
}

// -----

#blocked2d = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice2d0 = #ttg.slice<{dim = 0, parent = #blocked2d}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: @no_split_expand_dims_broadcast_negative_summand
    // CHECK: amdg.buffer_load
    // CHECK-NOT: amdgpu.split_soffset_safe
    tt.func @no_split_expand_dims_broadcast_negative_summand(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
        %cn64 = arith.constant -64 : i32
        %neg = tt.splat %cn64 : i32 -> tensor<128xi32, #slice2d0>
        %range = tt.make_range {end = 192 : i32, start = 64 : i32} : tensor<128xi32, #slice2d0>
        %add = arith.addi %neg, %range : tensor<128xi32, #slice2d0>
        %expanded = tt.expand_dims %add {axis = 0 : i32} : tensor<128xi32, #slice2d0> -> tensor<1x128xi32, #blocked2d>
        %offset = tt.broadcast %expanded : tensor<1x128xi32, #blocked2d> -> tensor<2x128xi32, #blocked2d>
        %ret = amdg.buffer_load %arg0[%offset] : tensor<2x128xf32, #blocked2d>
        tt.return
    }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: @no_split_overflowing_add
    // CHECK: amdg.buffer_load
    // CHECK-NOT: amdgpu.split_soffset_safe
    tt.func @no_split_overflowing_add(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
        %range = tt.make_range {end = 1073741826 : i32, start = 1073741824 : i32} : tensor<2xi32, #blocked0>
        %offset = arith.addi %range, %range : tensor<2xi32, #blocked0>
        %ret = amdg.buffer_load %arg0[%offset] : tensor<2xf32, #blocked0>
        tt.return
    }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: @no_split_overflowing_mul
    // CHECK: amdg.buffer_load
    // CHECK-NOT: amdgpu.split_soffset_safe
    tt.func @no_split_overflowing_mul(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
        %big = arith.constant dense<1073741824> : tensor<1xi32, #blocked0>
        %range = tt.make_range {end = 4 : i32, start = 3 : i32} : tensor<1xi32, #blocked0>
        %offset = arith.muli %range, %big : tensor<1xi32, #blocked0>
        %ret = amdg.buffer_load %arg0[%offset] : tensor<1xf32, #blocked0>
        tt.return
    }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: @no_split_overflowing_shl
    // CHECK: amdg.buffer_load
    // CHECK-NOT: amdgpu.split_soffset_safe
    tt.func @no_split_overflowing_shl(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
        %sh = arith.constant dense<30> : tensor<1xi32, #blocked0>
        %range = tt.make_range {end = 4 : i32, start = 3 : i32} : tensor<1xi32, #blocked0>
        %offset = arith.shli %range, %sh : tensor<1xi32, #blocked0>
        %ret = amdg.buffer_load %arg0[%offset] : tensor<1xf32, #blocked0>
        tt.return
    }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: @no_split_shrui_by_zero
    // CHECK: amdg.buffer_load
    // CHECK-NOT: amdgpu.split_soffset_safe
    tt.func @no_split_shrui_by_zero(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
        %neg = arith.constant dense<-1> : tensor<1xi32, #blocked0>
        %zero = arith.constant dense<0> : tensor<1xi32, #blocked0>
        %offset = arith.shrui %neg, %zero : tensor<1xi32, #blocked0>
        %ret = amdg.buffer_load %arg0[%offset] : tensor<1xf32, #blocked0>
        tt.return
    }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: @no_split_cat_hides_negative_leaf
    // CHECK: amdg.buffer_load
    // CHECK-NOT: amdgpu.split_soffset_safe
    tt.func @no_split_cat_hides_negative_leaf(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
        %two = arith.constant dense<2> : tensor<2xi32, #blocked0>
        %range = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #blocked0>
        %leaf = arith.subi %range, %two : tensor<2xi32, #blocked0>
        %offset = arith.addi %leaf, %two : tensor<2xi32, #blocked0>
        %cat = tt.cat %offset, %offset : tensor<2xi32, #blocked0> -> tensor<4xi32, #blocked0>
        %ret = amdg.buffer_load %arg0[%cat] : tensor<4xf32, #blocked0>
        tt.return
    }
}
