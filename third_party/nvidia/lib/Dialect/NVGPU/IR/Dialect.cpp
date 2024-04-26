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

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "Dialect/NVGPU/IR/Dialect.h"
#include "Dialect/NVGPU/IR/Dialect.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::triton::nvgpu;

void LoadDSmemOp::build(OpBuilder &builder, OperationState &state,
                        Type resultTy, Value addr, Value ctaId) {
  unsigned vec, bitwidth;
  if (auto structTy = dyn_cast<LLVM::LLVMStructType>(resultTy)) {
    auto types = structTy.getBody();
    assert(types.size() > 0 && "Invalid result type of LoadDSmemOp");
    vec = types.size();
    for (unsigned i = 0; i < vec; ++i)
      assert(types[0] == types[i]);
    bitwidth = types[0].getIntOrFloatBitWidth();
  } else {
    vec = 1;
    bitwidth = resultTy.getIntOrFloatBitWidth();
  }
  build(builder, state, resultTy, addr, ctaId, bitwidth, vec);
}

void LoadDSmemOp::build(OpBuilder &builder, OperationState &state, Value addr,
                        Value ctaId, unsigned bitwidth, unsigned vec) {
  Type resultTy = builder.getIntegerType(bitwidth);
  if (vec > 1) {
    SmallVector<Type> types(vec, resultTy);
    resultTy = LLVM::LLVMStructType::getLiteral(builder.getContext(), types);
  }
  build(builder, state, resultTy, addr, ctaId, bitwidth, vec);
}

void LoadDSmemOp::build(OpBuilder &builder, OperationState &state, Value addr,
                        Value ctaId, unsigned bitwidth) {
  build(builder, state, addr, ctaId, bitwidth, /*vec*/ 1);
}

void StoreDSmemOp::build(OpBuilder &builder, OperationState &state, Value addr,
                         Value ctaId, Value value, Value pred) {
  SmallVector<Value> values = {value};
  build(builder, state, addr, ctaId, values, pred);
}

unsigned StoreDSmemOp::getBitwidth() {
  auto addrTy = getAddr().getType();
  assert(isa<LLVM::LLVMPointerType>(addrTy) && "addr must be a pointer type");
  if (getValues().empty())
    return 0;
  auto elemTy = getValues().back().getType();
  return elemTy.getIntOrFloatBitWidth();
}

unsigned StoreDSmemOp::getVec() { return getValues().size(); }

static LogicalResult verify(mlir::triton::nvgpu::WGMMAOp op) {
  return success();
}

void mlir::triton::nvgpu::NVGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/NVGPU/IR/NVGPUAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/NVGPU/IR/Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Dialect/NVGPU/IR/Ops.cpp.inc"
#include "Dialect/NVGPU/IR/OpsEnums.cpp.inc"
