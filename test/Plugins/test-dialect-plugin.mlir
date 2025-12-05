// RUN: TRITON_PASS_PLUGIN_PATH=%shlibdir/../plugins/libMLIRLoweringDialectPlugin.so triton-opt -split-input-file --convert-plugin-gpu-to-llvm --convert-triton-gpu-to-llvm %s | \
// RUN: FileCheck %s

// REQUIRES: shared-libs

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-warps" = 8 : i32} {
  tt.func @convert_plugin() {
    %0 = arith.constant 0 : i32
    %1 = plugin.magic %0 : i32
    %2 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    tt.return
  }
}
