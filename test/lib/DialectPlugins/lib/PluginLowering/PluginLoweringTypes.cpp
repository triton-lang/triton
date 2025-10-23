//===- PluginLoweringTypes.cpp - PluginLowering dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PluginLowering/PluginLoweringTypes.h"

#include "PluginLowering/PluginLoweringDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::pluginlowering;

#define GET_TYPEDEF_CLASSES
#include "PluginLowering/PluginLoweringOpsTypes.cpp.inc"

void PluginLoweringDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "PluginLowering/PluginLoweringOpsTypes.cpp.inc"
      >();
}
