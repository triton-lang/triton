#ifndef LOWERINGDIALECTPLUGIN_PASS
#define LOWERINGDIALECTPLUGIN_PASS

include "mlir/Pass/PassBase.td"

def LoweringDialectPluginSwitchBarFoo: Pass<"loweringdialectplugin-switch-bar-foo", "::mlir::ModuleOp"> {
  let summary = "Switches the name of a FuncOp named `bar` to `foo` and folds.";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    func.func @bar() {
      return
    }
    // Gets transformed to:
    func.func @foo() {
      return
    }
    ```
  }];
}

def LoweringDialectPluginLowerFooOp: Pass<"loweringdialectplugin-lower-foo-op", "::mlir::ModuleOp"> {
  let summary = "Lowers a foo op to an arith";
  let description = [{
      Lowers a FooOp to Arith:
    ```
    func.func @bar() {
      return
    }
    // Gets transformed to:
    func.func @foo() {
      return
    }
    ```
  }];
}

#endif // LOWERINGDIALECTPLUGIN_PASS
