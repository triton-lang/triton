//===- PluginLoweringDialect.td - PluginLowering dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PLUGINLOWERING_DIALECT
#define PLUGINLOWERING_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PluginLowering dialect definition.
//===----------------------------------------------------------------------===//

def PluginLowering_Dialect : Dialect {
    let name = "pluginlowering";
    let summary = "A pluginlowering out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::pluginlowering";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base pluginlowering operation definition.
//===----------------------------------------------------------------------===//

class PluginLowering_Op<string mnemonic, list<Trait> traits = []> :
        Op<PluginLowering_Dialect, mnemonic, traits>;

#endif // PLUGINLOWERING_DIALECT
