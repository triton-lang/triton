//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TRITONXPU_DIALECT
#define TRITONXPU_DIALECT

include "mlir/IR/DialectBase.td"


//===----------------------------------------------------------------------===//
// TRITONXPU dialect definitions
//===----------------------------------------------------------------------===//

def TritonXPU_Dialect : Dialect {
  let name = "triton_xpu";

  let cppNamespace = "::mlir::triton::xpu";

  let hasOperationAttrVerify = 1;

  let description = [{
    TRITON XPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
  ];

  let useDefaultAttributePrinterParser = 1;
}

#endif // TRITONXPU_DIALECT
