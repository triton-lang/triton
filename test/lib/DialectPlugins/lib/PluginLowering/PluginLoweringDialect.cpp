//===- PluginLoweringDialect.cpp - PluginLowering dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PluginLowering/PluginLoweringDialect.h"
#include "PluginLowering/PluginLoweringOps.h"
#include "PluginLowering/PluginLoweringTypes.h"

using namespace mlir;
using namespace mlir::pluginlowering;

// #include "PluginLowering/PluginLoweringOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// PluginLowering dialect.
//===----------------------------------------------------------------------===//

void PluginLoweringDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PluginLowering/PluginLoweringOps.cpp.inc"
      >();
  registerTypes();
}
