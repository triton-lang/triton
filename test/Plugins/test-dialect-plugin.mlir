// RUN: TRITON_PASS_PLUGIN_PATH=%shlibdir/../plugins/libTritonPluginsTestLib.so triton-opt -split-input-file -tritongpu-plugin %s | FileCheck %s --check-prefix=CHECK-PLUGIN
// RUN: TRITON_PASS_PLUGIN_PATH=%shlibdir/../plugins/libTritonPluginsTestLib.so triton-opt -split-input-file %s | FileCheck %s -allow-unused-prefixes --check-prefix=CHECK-NOFLAG
// RUN: triton-opt -split-input-file %s | FileCheck %s -allow-unused-prefixes --check-prefix=CHECK-BASE

// REQUIRES: shared-libs

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  tt.func @bar() -> i32 {
    %0 = tt.get_program_id x : i32
    %1 = pluginlowering.foo %0 : i32
    tt.return %1 : i32
  }
}  // module

