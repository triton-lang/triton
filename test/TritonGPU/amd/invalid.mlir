// RUN: triton-opt --split-input-file %s --verify-diagnostics

// expected-error @+1 {{Transposed WMMA is supported only for version 2}}
#wmma = #ttg.amd_wmma<{version = 1, isTranspose = true, warpsPerCTA = [2, 2]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<32x32x!tt.ptr<i32,1>, #wmma>
        tt.return
    }
}

// -----

// expected-error @+1 {{WMMA version must be in the [1, 2] range}}
#wmma = #ttg.amd_wmma<{version = 0, isTranspose = false, warpsPerCTA = [2, 2]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<32x32x!tt.ptr<i32,1>, #wmma>
        tt.return
    }
}
