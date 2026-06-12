// RUN: LD_PRELOAD=%shlibdir/../plugins/libtriton.so \
// RUN: TRITON_PLUGIN_PATHS=%shlibdir/../plugins/libTritonPluginsTestLib.so \
// RUN: triton-opt \
// RUN: -split-input-file -tritongpu-plugin %s | FileCheck %s --check-prefix=CHECK-PLUGIN

// RUN: LD_PRELOAD=%shlibdir/../plugins/libtriton.so \
// RUN: TRITON_PLUGIN_PATHS=%shlibdir/../plugins/libTritonPluginsTestLib.so \
// RUN: triton-opt \
// RUN: -split-input-file %s | FileCheck %s -allow-unused-prefixes --check-prefix=CHECK-NOFLAG

// RUN: triton-opt -split-input-file %s | FileCheck %s -allow-unused-prefixes --check-prefix=CHECK-BASE

// REQUIRES: triton-ext-enabled
// XFAIL: *

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
