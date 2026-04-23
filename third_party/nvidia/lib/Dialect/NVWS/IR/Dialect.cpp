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

#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "Dialect/NVWS/IR/Dialect.h"
#include "Dialect/NVWS/IR/Dialect.cpp.inc"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`
// clang-format on

using namespace mlir;
using namespace mlir::triton::nvws;

static int getSemaphoreDepthFromBaseType(Type type) {
  if (auto memTy = dyn_cast<triton::gpu::MemDescType>(type)) {
    if (isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
            memTy.getEncoding()))
      return 1;
    auto shape = memTy.getShape();
    if (shape.empty())
      return -1;
    return shape.front();
  }
  if (auto rankedTy = dyn_cast<RankedTensorType>(type)) {
    auto shape = rankedTy.getShape();
    if (shape.empty())
      return -1;
    return shape.front();
  }
  return -1;
}

void mlir::triton::nvws::NVWSDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/NVWS/IR/NVWSAttrDefs.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/NVWS/IR/Types.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/NVWS/IR/Ops.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "Dialect/NVWS/IR/NVWSAttrDefs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/NVWS/IR/Types.cpp.inc"

LogicalResult
SemaphoreType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                      TypeArrayAttr baseType) {
  int inferredNumStages = 1;
  for (Type baseTy : baseType) {
    int depth = getSemaphoreDepthFromBaseType(baseTy);
    if (depth <= 1)
      continue;
    if (inferredNumStages == 1) {
      inferredNumStages = depth;
      continue;
    }
    if (inferredNumStages != depth) {
      return emitError() << "inconsistent semaphore baseType depths: expected "
                         << inferredNumStages << ", got " << depth;
    }
  }
  return success();
}

int SemaphoreType::getNumStages() const {
  int inferredNumStages = 1;
  for (Type baseTy : getBaseType()) {
    int depth = getSemaphoreDepthFromBaseType(baseTy);
    if (depth > inferredNumStages)
      inferredNumStages = depth;
  }
  return inferredNumStages;
}
