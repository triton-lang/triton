#ifndef DIALECTPLUGIN_PASS
#define DIALECTPLUGIN_PASS

include "mlir/Pass/PassBase.td"


def DialectPluginMagicOp: Pass<"dialectplugin-magic-op", "mlir::ModuleOp"> {
  let summary = "Converts a MagicOp into an mlir::arith::addIOp";
  let description = [{
    Lowers a MagicOp into an mlir::arith::addIOp:
    ```
    tt.func @bar() {
      %0 = tt.get_program_id x : i32
      %1 = dialectplugin.magic %0 : i32
      tt.return
    }
    tt.func @bar() {
      %0 = tt.get_program_id x : i32
      %1 = arith.addi %0, %0 : i32
      tt.return
    }
    ```
  }];
    let constructor = "createDialectPluginMagicOpPass()";
  let dependentDialects = ["mlir::triton::TritonDialect"];
}

#endif // DIALECTPLUGIN_PASS
