#ifndef LOWERINGDIALECTPLUGIN_PASS
#define LOWERINGDIALECTPLUGIN_PASS

include "mlir/Pass/PassBase.td"

def LoweringDialectPluginMagicOpInserter: Pass<"loweringdialectplugin-magic-op-inserter", "::mlir::ModuleOp"> {
  let summary = "Finds a tt.get_program_id and inserts a LoweringDialectPluginMagicOp in-between it and its users.";
  let description = [{
    Finds a tt.get_program_id and inserts a LoweringDialectPluginMagicOp in-between it and its users:
    ```
    tt.func @f() -> i32 {
      %0 = tt.get_program_id x : i32
      tt.return %0 : i32
    }

    // Gets transformed to:
    tt.func @f() -> i32 {
      %0 = tt.get_program_id x : i32
      %1 = loweringdialectplugin.magic %0 : i32
      tt.return %1 : i32
    }
    ```
  }];
}

def LoweringDialectPluginMagicOp: Pass<"loweringdialectplugin-magic-op", "::mlir::ModuleOp"> {
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
}

#endif // LOWERINGDIALECTPLUGIN_PASS
