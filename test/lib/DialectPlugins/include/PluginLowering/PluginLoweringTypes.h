//===- PluginLoweringTypes.h - PluginLowering dialect types -------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PLUGINLOWERING_PLUGINLOWERINGTYPES_H
#define PLUGINLOWERING_PLUGINLOWERINGTYPES_H

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "PluginLowering/PluginLoweringOpsTypes.h.inc"

#endif // PLUGINLOWERING_PLUGINLOWERINGTYPES_H
