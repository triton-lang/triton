// RUN: TRITON_PASS_PLUGIN_PATH=libGPUExtensionTestLib.so triton-opt -tritongpu-plugin

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {


// CHECK-LABEL: func @foo()
tt.func @bar() {
    tt.return
  }


}  // module
