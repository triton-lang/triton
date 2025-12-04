// RUN: TRITON_PASS_PLUGIN_PATH=%shlibdir/../plugins/libMLIRLoweringDialectPlugin.so triton-opt -split-input-file --loweringdialectplugin-magic-op %s | \
// RUN: FileCheck %s

// REQUIRES: shared-libs

module {
  tt.func @bar() {
    %0 = tt.get_program_id x : i32
    %1 = dialectplugin.magic %0 : i32
    tt.return 
  }
}  // module
