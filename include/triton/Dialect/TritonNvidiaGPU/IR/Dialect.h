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

#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_IR_DIALECT_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

// TritonNvidiaGPU depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h.inc"

namespace mlir::triton::nvidia_gpu::impl {
LogicalResult verifyMMAv5Op(Operation *op);
} // namespace mlir::triton::nvidia_gpu::impl

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.h.inc"

#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/Ops.h.inc"

namespace mlir::triton::nvidia_gpu {

constexpr static char AttrTwoCTAsName[] = "ttng.two-ctas";

inline bool getModuleTwoCTAs(ModuleOp mod) {
  auto attr = mod->getAttrOfType<BoolAttr>(AttrTwoCTAsName);
  return attr ? attr.getValue() : false;
}

inline bool getModuleTwoCTAs(Operation *op) {
  return getModuleTwoCTAs(op->getParentOfType<ModuleOp>());
}

struct TensorMemory : public SideEffects::Resource::Base<TensorMemory> {
  StringRef getName() final { return "<TensorMemory>"; }
};

struct TMemAllocation {
  TMemAllocation(int numRows, int numCols)
      : numRows(numRows), numCols(numCols) {}
  int numRows;
  int numCols;
};

// Used to describe the layout of the TMEM load/store instructions
enum class TMemAccessAtom { I32x32b, I16x64b, I16x128b, I16x256b, I16x32bx2 };

inline int getElementsPerThread(TMemAccessAtom atom) {
  switch (atom) {
  case TMemAccessAtom::I32x32b:
  case TMemAccessAtom::I16x64b:
  case TMemAccessAtom::I16x32bx2:
    return 1;
  case TMemAccessAtom::I16x128b:
    return 2;
  case TMemAccessAtom::I16x256b:
    return 4;
  }
  llvm_unreachable("Unknown TMemAccessAtom");
}

inline const char *getOpShape(TMemAccessAtom atom) {
  switch (atom) {
  case TMemAccessAtom::I32x32b:
    return "32x32b";
  case TMemAccessAtom::I16x64b:
    return "16x64b";
  case TMemAccessAtom::I16x128b:
    return "16x128b";
  case TMemAccessAtom::I16x256b:
    return "16x256b";
  case TMemAccessAtom::I16x32bx2:
    return "16x32bx2";
  }
  llvm_unreachable("Unknown TMemAccessAtom");
}

LinearLayout getTileLayout(MLIRContext *ctx, TMemAccessAtom atom, bool unpacked,
                           bool withWarp);

TMemAllocation getTmemAllocSizes(gpu::MemDescType memDescType);

SmallVector<gpu::DistributedEncodingTrait>
getTmemCompatibleLayouts(gpu::MemDescType memType, unsigned numWarps,
                         ArrayRef<int64_t> ctaSplit = {1, 1});

std::optional<gpu::DistributedEncodingTrait>
getTmemLoadLayoutSplitLongM(RankedTensorType tensorType,
                            gpu::MemDescType memType, int numWarps);

SmallVector<gpu::DistributedEncodingTrait>
getTmemCompatibleLayouts(Operation *op, RankedTensorType tensorType,
                         gpu::MemDescType memType);

bool isDistributedLayoutTMemCompatible(Operation *op,
                                       RankedTensorType tensorType,
                                       gpu::MemDescType memType);

gpu::DistributedEncodingTrait
getDefaultLayoutForTmemLdSt(gpu::MemDescType memType, unsigned numWarps,
                            gpu::CGAEncodingAttr cgaLayout);

std::optional<LinearLayout>
getDistributedLayoutForTmemLdSt(gpu::MemDescType memType, TMemAccessAtom atom,
                                unsigned numWarps,
                                gpu::CGAEncodingAttr cgaLayout);

} // namespace mlir::triton::nvidia_gpu

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_IR_DIALECT_H_
