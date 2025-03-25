//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TRITONXPU_ANALYSIS_UTILITY_H
#define TRITONXPU_ANALYSIS_UTILITY_H

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "triton/Dialect/TritonXPU/IR/Dialect.h"

namespace mlir {

#define XPU_MEMORY_OP                                                          \
  triton::xpu::GM2LMOp, triton::xpu::LM2GMOp, triton::xpu::SM2GMOp

template <class T> struct is_xpu_memory_op {
  static const bool value = false;
};
template <> struct is_xpu_memory_op<triton::xpu::GM2LMOp> {
  static const bool value = true;
};
template <> struct is_xpu_memory_op<triton::xpu::LM2GMOp> {
  static const bool value = true;
};
template <> struct is_xpu_memory_op<triton::xpu::SM2GMOp> {
  static const bool value = true;
};

#define ARITH_PTR_UNARY_OP arith::ExtSIOp

#define ARITH_PTR_BINARY_OP                                                    \
  arith::DivSIOp, arith::RemSIOp, arith::MulIOp, arith::AddIOp, arith::SubIOp

#define XPU_VVECTORIZED_BINARY_OP                                              \
  triton::xpu::VvaddFOp, triton::xpu::VvmulFOp, triton::xpu::VvsubFOp,         \
      triton::xpu::VvmaxFOp

#define XPU_SVECTORIZED_BINARY_OP                                              \
  triton::xpu::SvaddFOp, triton::xpu::SvmulFOp, triton::xpu::SvsubFOp,         \
      triton::xpu::SvmaxFOp

enum class OffsetState {
  Unknown = -1,
  DiscreteSame = 0,
  Continuous = 1,
  Discrete = 2,
  LocallyContinuous = 3
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const OffsetState &state);

enum class AtomicMaskCond {
  PostiveCond = 1,
  NegativeCond = -1,
  NonActivate = 0,
};

enum class AtomicMaskType {
  NaiveMask = 1,
  OptimizationMask = 2,
};

enum class XPUArch { XPU2 = 2, XPU3 = 3 };

enum class MemCpyType { GM2LM = 0, LM2GM = 1, GM2SM = 2, SM2GM = 3 };

class SMHelper {
public:
  explicit SMHelper(Operation *op) : op(op) {}

  void setOffset(int64_t offset) { smOffsetMap[op] = offset; }

  int64_t getOffset() {
    int64_t offset = 0;
    if (hasOffset()) {
      offset = smOffsetMap[op];
    }
    return offset;
  }

  bool hasOffset() { return smOffsetMap.find(op) != smOffsetMap.end(); }

private:
  Operation *op;
  static std::map<Operation *, int64_t> smOffsetMap;
};

Type addrspaceCast(Type type, int addressSpace);

bool inOpChain(llvm::SetVector<Operation *> &opChain, Operation *op);

void getOpChainBwd(llvm::SetVector<Operation *> &opChain, Operation *op);
void getOpChainFwd(llvm::SetVector<Operation *> &opChain, Operation *op);
void getOpTreeBwd(llvm::SetVector<Operation *> &opTree,
                  llvm::SetVector<Operation *> &visitedOps, Operation *op);
void getOpTreeBwd(llvm::SetVector<Operation *> &opTree,
                  llvm::SetVector<Operation *> &visitedOps, Operation *op,
                  Block *block);

llvm::SmallVector<Operation *>
sortOpTreeBwd(llvm::SmallVector<Operation *> &opTree);
llvm::SetVector<Operation *>
sortOpTreeBwd(llvm::SetVector<Operation *> &opTree);
llvm::SetVector<Operation *> sortOpTree(llvm::SetVector<Operation *> &opTree);

bool inSameSCFIfBlock(llvm::SetVector<Operation *> &storeOps,
                      Operation *storeOp);

template <typename opType>
Operation *findUserOpImpl(Operation *op,
                          llvm::SetVector<Operation *> &visitedOps) {
  if (!op || op->use_empty() || visitedOps.contains(op))
    return nullptr;

  visitedOps.insert(op);

  if (isa<opType>(op)) {
    return op;
  }

  for (Operation *user : op->getUsers()) {
    Operation *userOp = findUserOpImpl<opType>(user, visitedOps);
    if (userOp) {
      return userOp;
    }
  }

  return nullptr;
}

template <typename opType> Operation *findUserOp(Operation *op) {
  llvm::SetVector<Operation *> visitedOps;
  return findUserOpImpl<opType>(op, visitedOps);
}

template <typename opType> Operation *findDefOpBwd(const Value &val) {
  if (!val || !val.getDefiningOp()) {
    return nullptr;
  }
  auto op = val.getDefiningOp();
  if (op && isa<opType>(op)) {
    return op;
  }
  for (auto operand : op->getOperands()) {
    op = findDefOpBwd<opType>(operand);
    if (op) {
      return op;
    }
  }
  return nullptr;
}

} // namespace mlir

#endif // TRITONXPU_ANALYSIS_UTILITY_H
