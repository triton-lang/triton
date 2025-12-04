#ifndef LOWERINGDIALECTPLUGIN_PASS
#define LOWERINGDIALECTPLUGIN_PASS

include "mlir/Pass/PassBase.td"


def LoweringDialectPluginMagicOp: Pass<"loweringdialectplugin-magic-op", "mlir::ModuleOp"> {
  let summary = "Lowers a MagicOp into an mlir::arith::addIOp";
  let description = [{
    Lowers a MagicOp into an mlir::arith::addIOp:
    ```
    tt.func @bar() -> i32 {
      %0 = tt.get_program_id x : i32
      %1 = loweringdialectplugin.magic %0 : i32
      tt.return %1 : i32
    }
    tt.func @bar() -> i32 {
      %0 = tt.get_program_id x : i32
      %1 = arith.addi %0, %0 : i32
      tt.return %1 : i32
    }
    ```
  }];
    let constructor = "createLoweringDialectPluginMagicOpPass()";
  let dependentDialects = ["mlir::triton::TritonDialect"];
}

#endif // LOWERINGDIALECTPLUGIN_PASS
