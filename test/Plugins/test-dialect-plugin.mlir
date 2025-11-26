// RUN: TRITON_PASS_PLUGIN_PATH=%shlibdir/../plugins/libMLIRLoweringDialectPlugin.so triton-opt -split-input-file --loweringdialectplugin-magic-op %s | \
// RUN: FileCheck %s

// REQUIRES: shared-libs

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {
  tt.func @bar() -> i32 {
    // CHECK: %1 = arith.addi %0, %0 : i32
    %0 = tt.get_program_id x : i32
    %1 = loweringdialectplugin.magic %0 : i32
    tt.return %1 : i32
  }
}  // module
