// RUN: LLVM_EXTRACT_DI_LOCAL_VARIABLES=1 triton-opt %s -o - --mlir-print-debuginfo --mlir-use-nameloc-as-prefix --enable-line-info --extract-variable-info | \
// RUN: mlir-translate --mlir-to-llvmir | FileCheck %s

// Regression test for a crash when CallSiteLoc operations coexist with
// NameLoc operations in the same function. When enable-line-info runs with
// LLVM_EXTRACT_DI_LOCAL_VARIABLES=1, it creates a DISubprogramAttr with
// isRecSelf=true and wraps CallSiteLoc ops in DILexicalBlockFileAttr that
// references the isRecSelf=true subprogram. The extract-variable-info pass
// must fix these lexical block scopes to reference the resolved
// (isRecSelf=false) subprogram via fixLexicalBlockScopes, otherwise
// mlir-translate crashes with an assertion in DebugTranslation.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  llvm.func @kernel_with_callsite(%arg0: !llvm.ptr<1> loc(#loc_arg0), %arg1: i32 loc(#loc_arg1), %arg2: !llvm.ptr<1>) {
    // CHECK-DAG: distinct !DISubprogram({{.*}}, retainedNodes:
    // CHECK-DAG: !DILocalVariable(name: "input_ptr", arg: 1, scope:
    // CHECK-DAG: !DILocalVariable(name: "n_elements", arg: 2, scope:

    %c = llvm.mlir.constant(42 : i32) : i32

    // This op has a CallSiteLoc, triggering DILexicalBlockFileAttr creation
    // in add_di_scope. Without fixLexicalBlockScopes, the stale isRecSelf=true
    // reference would cause an assertion failure in mlir-translate.
    %sum = llvm.add %arg1, %c : i32 loc(#loc_callsite1)

    // This op has a NameLoc, triggering DILocalVariable creation in
    // fuseDILocalVariable.
    // CHECK-DAG: !DILocalVariable(name: "result", scope:
    %result = llvm.mul %sum, %c : i32 loc(#loc_result)

    llvm.return
  }
}
#loc_base = loc("kernel.py":10:0)
#loc_arg0 = loc("input_ptr"(#loc_base))
#loc_arg1 = loc("n_elements"(#loc_base))
#loc_callee = loc("helper.py":5:3)
#loc_caller = loc("kernel.py":20:5)
#loc_callsite1 = loc(callsite(#loc_callee at #loc_caller))
#loc_result_base = loc("kernel.py":25:10)
#loc_result = loc("result"(#loc_result_base))
