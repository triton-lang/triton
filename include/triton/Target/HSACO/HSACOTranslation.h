/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef TRITON_TARGET_HSACOTRANSLATION_H
#define TRITON_TARGET_HSACOTRANSLATION_H

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace mlir {
class ModuleOp;
}

namespace llvm {
class Module;
class LLVMContext;
} // namespace llvm

namespace mlir {
namespace triton {

// add external libs to modules
void addExternalLibsROCM(mlir::ModuleOp &module,
                         const std::vector<std::string> &names,
                         const std::vector<std::string> &paths);

// Translate Triton dialect to TritonGPU, return null if failed.
void translateTritonToTritonGPUROCM(mlir::ModuleOp &module,
                                    int computeCapability, int numWarps,
                                    int numStages);

// Translate Triton GPU to mlir LLVM dialect, return null if failed.
void translateTritonGPUROCMToLLVMDialect(mlir::ModuleOp &module,
                                         int computeCapability, bool isROCM);

// Translate mlir LLVM dialect to LLVMIR, return null if failed.
std::unique_ptr<llvm::Module>
translateLLVMDialectToLLVMIR(llvm::LLVMContext *llvmContext,
                             mlir::ModuleOp module, bool isROCM);

// Translate LLVMIR to HSACO code.
std::string translateLLVMIRToHSACO(llvm::Module &module, std::string gfx_arch,
                                   std::string gfx_triple,
                                   std::string gfx_features);

std::string translateTritonIRToHSACO(mlir::ModuleOp module,
                                     std::string gfx_arch,
                                     std::string gfx_triple,
                                     std::string gfx_features, int numWarps,
                                     int numStages,
                                     const std::vector<std::string> &names,
                                     const std::vector<std::string> &paths);

} // namespace triton
} // namespace mlir

#endif
