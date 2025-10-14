// RUN: env TRITON_PASS_PLUGIN_PATH=%shlibdir/../plugins/libTritonPluginsTestLib.so
// RUN: TRITON_PASS_PLUGIN_PATH=%shlibdir/../plugins/libTritonPluginsTestLib.so triton-opt -tritongpu-plugin %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {

  // CHECK: func @foo()
  tt.func @bar() {
    tt.return
  }
}  // module
