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

#ifndef TRITON_NVIDIA_GPU_IR_TRAITS_H_
#define TRITON_NVIDIA_GPU_IR_TRAITS_H_

#include "mlir/IR/OpDefinition.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace OpTrait {

// These functions are out-of-line implementations of the methods in the
// corresponding trait classes.  This avoids them being template
// instantiated/duplicated.
namespace impl {
LogicalResult verifySource1IsSharedEncoding(Operation *op);
} // namespace impl

template <typename ConcreteType>
class Source1IsSharedEncoding
    : public TraitBase<ConcreteType, Source1IsSharedEncoding> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySource1IsSharedEncoding(op);
  }
};
} // namespace OpTrait
} // namespace mlir

#endif
