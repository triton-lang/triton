// RUN: TRITON_PASS_PLUGIN_PATH=%shlibdir/../plugins/libTritonPluginsTestLib.so triton-opt -split-input-file -tritongpu-plugin %s | FileCheck %s --check-prefix=CHECK-PLUGIN
// RUN: TRITON_PASS_PLUGIN_PATH=%shlibdir/../plugins/libTritonPluginsTestLib.so triton-opt -split-input-file %s | FileCheck %s -allow-unused-prefixes --check-prefix=CHECK-NOFLAG
// RUN: triton-opt -split-input-file %s | FileCheck %s -allow-unused-prefixes --check-prefix=CHECK-BASE

// REQUIRES: shared-libs

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-PLUGIN: func @foo()
  tt.func @bar() {
    tt.return
  }
}  // module

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-NOFLAG: func @bar()
  tt.func @bar() {
    tt.return
  }
}  // module

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  // CHECK-BASE: func @bar()
  tt.func @bar() {
    tt.return
  }
}  // module
