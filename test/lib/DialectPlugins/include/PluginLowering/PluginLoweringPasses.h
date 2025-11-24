//===- PluginLoweringPasses.h - PluginLowering passes  ------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef PLUGINLOWERING_PLUGINLOWERINGPASSES_H
#define PLUGINLOWERING_PLUGINLOWERINGPASSES_H

#include "PluginLowering/PluginLoweringDialect.h"
#include "PluginLowering/PluginLoweringOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace pluginlowering {
#define GEN_PASS_DECL
#include "PluginLowering/PluginLoweringPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "PluginLowering/PluginLoweringPasses.h.inc"
} // namespace pluginlowering
} // namespace mlir

#endif
