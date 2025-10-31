# Triton TTIR and TTGIR Out of Tree Plugin Passes

## Overview

## Example
export LLVM_BUILD_SHARED_LIBS=1;  make dev-install-llvm
TRITON_PASS_PLUGIN_PATH=libTritonPluginsTestLib.so triton-opt -tritongpu-plugin test/Plugins/test-plugin.mlir

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:80"} {
  tt.func @foo() {
    tt.return
  }
}

After the out of tree pass runs, becomes:

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:80"} {
  tt.func @bar() {
    tt.return
  }
}

Function "foo" is renamed to "bar" by the out of tree pass.

## Known Issues
