//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "triton/Analysis/UtilityXPU.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

std::map<Operation *, int64_t> SMHelper::smOffsetMap;

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const OffsetState &state) {
  switch (state) {
  case OffsetState::Unknown:
    os << "Unknown";
    break;
  case OffsetState::DiscreteSame:
    os << "Discrete Same";
    break;
  case OffsetState::Continuous:
    os << "Continuous";
    break;
  case OffsetState::Discrete:
    os << "Discrete";
    break;
  case OffsetState::LocallyContinuous:
    os << "Locally Continuous";
    break;
  default:
    os << "Invalid State";
    break;
  }
  return os;
}

Type addrspaceCast(Type type, int addressSpace) {
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
    auto elemTy = tensorType.getElementType();
    auto ptrTy = mlir::cast<triton::PointerType>(elemTy);
    auto valTy = ptrTy.getPointeeType();
    auto ptrTyWithNewAS = triton::PointerType::get(valTy, addressSpace);
    return RankedTensorType::get(tensorType.getShape(), ptrTyWithNewAS,
                                 tensorType.getEncoding());
  } else if (auto pointerType = mlir::dyn_cast<triton::PointerType>(type)) {
    auto valTy = pointerType.getPointeeType();
    auto ptrTyWithNewAS = triton::PointerType::get(valTy, addressSpace);
    return ptrTyWithNewAS;
  } else {
    llvm_unreachable("`type` must be a PointerType or RankedTensorType whose "
                     "element type is PointerType.");
  }
}

bool inOpChain(llvm::SetVector<Operation *> &opChain, Operation *op) {
  if (!op || opChain.empty())
    return false;
  for (int i = 0; i < opChain.size(); ++i) {
    if (op == opChain[i]) {
      return true;
    }
  }
  return false;
}

void getOpChainBwd(llvm::SetVector<Operation *> &opChain, Operation *op) {
  if (!op) {
    return;
  }
  opChain.insert(op);

  int noDefCnt = 0;
  for (auto operand : op->getOperands()) {
    if (!operand.getDefiningOp()) {
      noDefCnt++;
    }
  }

  if (isa<mlir::arith::ConstantOp>(op) || isa<triton::xpu::StoreOp>(op) ||
      noDefCnt == op->getNumOperands()) {
    return;
  }

  for (auto operand : op->getOperands()) {
    getOpChainBwd(opChain, operand.getDefiningOp());
  }
}

void getOpChainFwd(llvm::SetVector<Operation *> &opChain, Operation *op) {
  opChain.insert(op);

  if (isa<triton::xpu::LM2GMOp>(op)) {
    return;
  }

  for (auto userOp : op->getUsers()) {
    if (!opChain.contains(userOp)) {
      getOpChainFwd(opChain, userOp);
    }
  }
}

void getOpTreeBwd(llvm::SetVector<Operation *> &opTree,
                  llvm::SetVector<Operation *> &visitedOps, Operation *op) {
  if (!op) {
    return;
  }
  visitedOps.insert(op);
  opTree.insert(op);

  if (isa<mlir::arith::ConstantOp>(op)) {
    // Do nothing
  } else {
    for (auto operand : op->getOperands()) {
      if (!visitedOps.contains(operand.getDefiningOp())) {
        getOpTreeBwd(opTree, visitedOps, operand.getDefiningOp());
      }
    }
  }

  for (auto userOp : op->getUsers()) {
    if (!visitedOps.contains(userOp)) {
      getOpTreeBwd(opTree, visitedOps, userOp);
    }
  }

  return;
}

void getOpTreeBwd(llvm::SetVector<Operation *> &opTree,
                  llvm::SetVector<Operation *> &visitedOps, Operation *op,
                  Block *block) {
  if (!op) {
    return;
  }
  visitedOps.insert(op);

  if (isa<mlir::arith::ConstantOp>(op)) {
    // Do nothing
  } else {
    if (auto forOp = dyn_cast<mlir::scf::ForOp>(op)) {
      forOp->walk([&](Operation *innerOp) {
        if (!visitedOps.contains(innerOp)) {
          getOpTreeBwd(opTree, visitedOps, innerOp, block);
        }
      });
    }
    for (auto operand : op->getOperands()) {
      if (!visitedOps.contains(operand.getDefiningOp())) {
        getOpTreeBwd(opTree, visitedOps, operand.getDefiningOp(), block);
      }
    }
  }

  for (auto userOp : op->getUsers()) {
    if (!visitedOps.contains(userOp)) {
      getOpTreeBwd(opTree, visitedOps, userOp, block);
    }
  }
  if (op->getBlock() == block) {
    opTree.insert(op);
  }
  return;
}

llvm::SmallVector<Operation *>
sortOpTreeBwd(llvm::SmallVector<Operation *> &opTree) {
  auto compareOps = [](Operation *op1, Operation *op2) {
    auto *block1 = op1->getBlock();
    auto *block2 = op2->getBlock();

    if (block1 == block2) {
      return op2->isBeforeInBlock(op1);
    }

    auto *region = block1->getParent();
    assert(region == block2->getParent() &&
           "Operations are in different regions!");
    return std::distance(region->begin(), Region::iterator(block1)) >
           std::distance(region->begin(), Region::iterator(block2));
  };
  auto sortedOpTree = opTree;
  llvm::stable_sort(sortedOpTree, compareOps);
  return sortedOpTree;
}

llvm::SetVector<Operation *>
sortOpTreeBwd(llvm::SetVector<Operation *> &opTree) {
  auto compareOps = [](Operation *op1, Operation *op2) {
    auto *block1 = op1->getBlock();
    auto *block2 = op2->getBlock();

    if (block1 == block2) {
      return op2->isBeforeInBlock(op1);
    }

    auto *region = block1->getParent();
    assert(region == block2->getParent() &&
           "Operations are in different regions!");
    return std::distance(region->begin(), Region::iterator(block1)) >
           std::distance(region->begin(), Region::iterator(block2));
  };
  llvm::SmallVector<Operation *> opTreeVec;
  for (auto op : opTree) {
    opTreeVec.emplace_back(op);
  }
  llvm::stable_sort(opTreeVec, compareOps);
  llvm::SetVector<Operation *> sortedOpTree;
  for (auto op : opTreeVec) {
    sortedOpTree.insert(op);
  }
  return sortedOpTree;
}

llvm::SetVector<Operation *> sortOpTree(llvm::SetVector<Operation *> &opTree) {
  auto compareOps = [](Operation *op1, Operation *op2) {
    auto *parentOp1 = op1;
    auto *parentOp2 = op2;
    for (mlir::Block *block1 = parentOp1->getBlock(); block1 != nullptr;) {
      for (mlir::Block *block2 = parentOp2->getBlock(); block2 != nullptr;) {
        if (block1 == block2) {
          return parentOp1->isBeforeInBlock(parentOp2);
        }
        parentOp2 = block2->getParentOp();
        if (parentOp2 == nullptr) {
          break;
        }
        block2 = parentOp2->getBlock();
      }
      parentOp2 = op2; // reset for next iteration
      parentOp1 = block1->getParentOp();
      if (parentOp1 == nullptr) {
        break;
      }
      block1 = parentOp1->getBlock();
    }
    assert(0 && "Sort Op Tree Failed!");
    return false;
  };
  llvm::SmallVector<Operation *> opTreeVec;
  for (auto op : opTree) {
    opTreeVec.emplace_back(op);
  }
  llvm::stable_sort(opTreeVec, compareOps);
  llvm::SetVector<Operation *> sortedOpTree;
  for (auto op : opTreeVec) {
    sortedOpTree.insert(op);
  }
  return sortedOpTree;
}

// Only Create Loop Once If StoreOp in SCF.IF
bool inSameSCFIfBlock(llvm::SetVector<Operation *> &storeOps,
                      Operation *storeOp) {
  auto block1 = storeOp->getBlock();
  Operation *parentOp1 = block1->getParentOp();

  for (auto otherStoreOp : storeOps) {
    auto block2 = otherStoreOp->getBlock();
    Operation *parentOp2 = block2->getParentOp();
    if (parentOp1 == parentOp2 && dyn_cast<scf::IfOp>(parentOp1) &&
        dyn_cast<scf::IfOp>(parentOp2)) {
      return true;
    }
  }
  return false;
}

} // namespace mlir
