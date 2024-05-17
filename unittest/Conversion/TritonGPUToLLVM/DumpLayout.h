/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
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

#ifndef TRITON_UNITTEST_CONVERSION_TRITONGPU_TO_LLVM_DUMP_LAYOUT_H
#define TRITON_UNITTEST_CONVERSION_TRITONGPU_TO_LLVM_DUMP_LAYOUT_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

// Try to "interpret" the MLIR Value into an integer.
int evalValue(Value value, int ctaid, int tid);

std::string dumpDistributedLayout(Attribute layout,
                                  llvm::ArrayRef<int64_t> shape, bool multiCTA);

std::string dumpSharedLayout(Attribute layout, llvm::ArrayRef<int64_t> shape,
                             Type elemTy, bool multiCTA);

} // namespace gpu
} // namespace triton
} // namespace mlir

#endif
