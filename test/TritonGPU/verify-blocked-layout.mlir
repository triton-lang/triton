// RUN: triton-opt --split-input-file %s --verify-diagnostics

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[16, 1],
    warpsPerCTA=[4, 1],
    order=[0, 1], CGALayout = [[0, 0]]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        // expected-error @+1 {{threads per warp}}
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<8x1x!tt.ptr<i32,1>, #blocked>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[32, 1],
    warpsPerCTA=[4, 2],
    order=[0, 1], CGALayout = [[0, 0]]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        // expected-error @+1 {{warps per CTA}}
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<8x1x!tt.ptr<i32,1>, #blocked>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[32, 1],
    warpsPerCTA=[4, 1],
    order=[0, 1]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        // expected-error @+1 {{CTAs per CGA}}
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<8x1x!tt.ptr<i32,1>, #blocked>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[32, 1],
    warpsPerCTA=[4, 1],
    order=[0, 1], CGALayout = [[0, 0]]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        // Note it's a 3d tensor here, but #blocked is 2D.
        // expected-error @+1 {{rank}}
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<8x1x1x!tt.ptr<i32,1>, #blocked>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[32, 1],
    warpsPerCTA=[4, 1],
    order=[0, 1], CGALayout = [[0, 0]]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: tensor<8xf32, #blocked>) {
        // expected-error @+1 {{rank}}
        %t = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<8xf32, #blocked> -> tensor<8x1xf32, #blocked>
        tt.return
    }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn() {
        // expected-error @+1 {{CTAs per CGA}}
        %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x16xf32, #shared, #smem, mutable>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#linear = #ttg.linear<{register = [[1], [16]], lane = [[0], [0], [2], [4], [8]], warp = [[0], [0]], block = []}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @invalid_cat_layout() {
    %lhs = arith.constant dense<0> : tensor<16xi32, #blocked>
    %rhs = arith.constant dense<1> : tensor<16xi32, #blocked>
    // expected-error @+1 {{tt.cat result encoding requires 4 non-broadcast register values, but operands provide 2}}
    %cat = tt.cat %lhs, %rhs : tensor<16xi32, #blocked> -> tensor<32xi32, #linear>
    tt.return
  }
}
