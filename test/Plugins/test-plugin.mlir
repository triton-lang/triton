// RUN: TRITON_PASS_PLUGIN_PATH=%shlibdir/../plugins/libGPUExtensionTestLib.so triton-opt %s -tritongpu-plugin

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {

  // CHECK: func @foo()
  tt.func @bar() {
    tt.return
  }
}  // module
