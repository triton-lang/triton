//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// TODO: Pass Description
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Analysis/UtilityXPU.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-offset-analysis"

namespace mlir {

using OffsetStateTransitionTable =
    std::map<OffsetState, std::map<OffsetState, OffsetState>>;

namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUOFFSETANALYSIS
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUOffsetAnalysisPass
    : public impl::TritonXPUOffsetAnalysisBase<TritonXPUOffsetAnalysisPass> {

public:
  using impl::TritonXPUOffsetAnalysisBase<
      TritonXPUOffsetAnalysisPass>::TritonXPUOffsetAnalysisBase;

  struct MockData {
    Operation *mockOp;
    int mockVal;
    SmallVector<int> mockVals;

    MockData(Operation *op, int val, SmallVector<int> &vals) {
      mockOp = op;
      mockVal = val;
      mockVals = vals;
    }

    std::string getToken() {
      std::string prefix = "";
      TypeSwitch<Operation *>(mockOp)
          .Case<arith::IndexCastOp>(
              [&](auto indexCastOp) { prefix = "index_"; })
          .Case<mlir::gpu::ThreadIdOp>(
              [&](auto threadIdOp) { prefix = "coreId_"; })
          .Case<triton::GetProgramIdOp>(
              [&](auto getProgramIdOp) { prefix = "clusterId_"; })
          .Case<triton::xpu::GM2LMOp>([&](auto gm2lmOp) { prefix = "gm2lm_"; });
      return prefix + std::to_string(mockVal);
    }
  };

  template <class T, std::enable_if_t<is_xpu_memory_op<T>::value, bool> = true>
  void legalizeOffset(T memoryOp) {
    Value ptr = memoryOp.getPtr();
    Operation *ptrOp = findDefOpBwd<triton::AddPtrOp>(ptr);
    if (ptrOp) {
      if (auto memAddPtrOp = dyn_cast<triton::AddPtrOp>(ptrOp)) {
        SetVector<Operation *> ptrOpChain;
        getPtrChainBwd(ptrOpChain, memAddPtrOp, memAddPtrOp);
        if (ptrOpChain.size() < 2) {
          return;
        }
        Value addIRes = memAddPtrOp.getOffset();
        OpBuilder builder(memAddPtrOp);
        auto loc = memAddPtrOp.getLoc();
        for (auto op : ptrOpChain) {
          if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
            auto addPtrDefOp = addPtrOp.getPtr().getDefiningOp();
            if (addPtrDefOp) {
              if (auto preAddPtrOp = dyn_cast<triton::AddPtrOp>(addPtrDefOp)) {
                if (getElementTypeOrSelf(addIRes).getIntOrFloatBitWidth() ==
                        32 &&
                    getElementTypeOrSelf(preAddPtrOp.getOffset())
                            .getIntOrFloatBitWidth() == 64) {
                  auto extIntOp = builder.create<arith::ExtSIOp>(
                      loc, preAddPtrOp.getOffset().getType(), addIRes);
                  auto addIOp = builder.create<arith::AddIOp>(
                      loc, preAddPtrOp.getOffset().getType(), extIntOp,
                      preAddPtrOp.getOffset());
                  addIRes = addIOp.getResult();
                } else if (getElementTypeOrSelf(addIRes)
                                   .getIntOrFloatBitWidth() == 64 &&
                           getElementTypeOrSelf(preAddPtrOp.getOffset())
                                   .getIntOrFloatBitWidth() == 32) {
                  auto extIntOp = builder.create<arith::ExtSIOp>(
                      loc, addIRes.getType(), preAddPtrOp.getOffset());
                  auto addIOp = builder.create<arith::AddIOp>(
                      loc, addIRes.getType(), addIRes, extIntOp);
                  addIRes = addIOp.getResult();
                } else {
                  auto addIOp = builder.create<arith::AddIOp>(
                      loc, addPtrOp.getOffset().getType(), addIRes,
                      preAddPtrOp.getOffset());
                  addIRes = addIOp.getResult();
                }
              } else {
                memAddPtrOp.setOperand(0, addPtrOp.getPtr());
              }
            } else {
              memAddPtrOp.setOperand(0, addPtrOp.getPtr());
            }
          }
        }
        memAddPtrOp.setOperand(1, addIRes);
      }
    }
  }

  void getPtrChainBwd(llvm::SetVector<Operation *> &opChain,
                      triton::AddPtrOp startOp, Operation *op) {
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

    if (isa<mlir::arith::ConstantOp>(op) || noDefCnt == op->getNumOperands()) {
      return;
    }

    if (auto addptrOp = dyn_cast<triton::AddPtrOp>(op)) {
      if (startOp.getResult().getType() == addptrOp.getResult().getType()) {
        getPtrChainBwd(opChain, startOp, addptrOp.getPtr().getDefiningOp());
      }
    }
    return;
  }

  void getOpChainBwdBFS(llvm::SetVector<Operation *> &opChain,
                        Operation *startOp) {
    if (!startOp) {
      return;
    }
    std::queue<Operation *> opQueue;
    opQueue.push(startOp);

    while (!opQueue.empty()) {
      Operation *currentOp = opQueue.front();
      opQueue.pop();

      if (!opChain.insert(currentOp)) {
        continue;
      }

      if (isa<triton::xpu::InterleaveOp>(currentOp) ||
          isa<triton::MakeRangeOp>(currentOp) ||
          isa<arith::ConstantOp>(currentOp) ||
          (isa<triton::xpu::GM2LMOp>(currentOp) && currentOp != startOp)) {
        continue;
      }

      if (isa<triton::xpu::LoadOp>(currentOp)) {
        opQueue.push(currentOp->getOperands()[0].getDefiningOp());
        continue;
      }

      for (auto operand : currentOp->getOperands()) {
        if (Operation *operandOp = operand.getDefiningOp()) {
          opQueue.push(operandOp);
        }
      }
    }
  }

  llvm::SetVector<Operation *>
  sortByLine(llvm::SetVector<Operation *> &opChain) {
    auto compareOpsByLine = [&](mlir::Operation *op1, mlir::Operation *op2) {
      return op2Line[op1] > op2Line[op2];
    };

    llvm::SmallVector<Operation *> opChainVec;
    for (auto op : opChain) {
      opChainVec.emplace_back(op);
    }
    llvm::sort(opChainVec, compareOpsByLine);
    llvm::SetVector<Operation *> sortedOpChain;
    for (auto op : opChainVec) {
      sortedOpChain.insert(op);
    }
    return sortedOpChain;
  }

  void dumpOpChain(SetVector<Operation *> &opChain) {
    for (auto it = opChain.rbegin(), eit = opChain.rend(); it != eit; ++it) {
      Operation *op = *it;
      if (op)
        op->dump();
    }
  }

  SmallVector<MockData> getMockDataItems(SetVector<Operation *> opChain) {
    auto getProgramIdMockVals = []() {
      SmallVector<int> mockVals(12);
      std::iota(mockVals.begin(), mockVals.end(), 0);
      return mockVals;
    };

    auto getThreadIdMockVals = []() {
      SmallVector<int> mockVals(/*core_num*/ 64);
      std::iota(mockVals.begin(), mockVals.end(), 0);
      return mockVals;
    };

    auto getIndexMockVals = [](Operation *indexCastOp) {
      // Get forOp
      Value arg = indexCastOp->getOperand(0); // only have one operand
      BlockArgument blockArg = mlir::dyn_cast<BlockArgument>(arg);
      Block *block = blockArg.getOwner();
      auto forOp = cast<scf::ForOp>(block->getParentOp());

      // Get Induction Vars
      auto lowerBoundOp =
          forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();
      auto stepOp = forOp.getStep().getDefiningOp<arith::ConstantOp>();
      auto upperBoundOp =
          forOp.getUpperBound().getDefiningOp<arith::ConstantOp>();

      // Get mockVals
      SmallVector<int> mockVals;
      if (lowerBoundOp && stepOp && upperBoundOp) {
        auto lowerBoundVal = mlir::cast<IntegerAttr>(lowerBoundOp.getValue())
                                 .getValue()
                                 .getZExtValue();
        auto stepVal = mlir::cast<IntegerAttr>(stepOp.getValue())
                           .getValue()
                           .getZExtValue();
        auto upperBoundVal = mlir::cast<IntegerAttr>(upperBoundOp.getValue())
                                 .getValue()
                                 .getZExtValue();

        for (auto i = lowerBoundVal; i < upperBoundVal; i += stepVal)
          mockVals.emplace_back(i);
      }

      return mockVals;
    };

    auto getGM2LMOpMockVals = []() {
      SmallVector<int> mockVals(1, 1);
      return mockVals;
    };

    SmallVector<MockData> mockDataItems;
    for (auto it = opChain.rbegin(), eit = opChain.rend(); it != eit; ++it) {
      TypeSwitch<Operation *>(*it)
          .Case<arith::IndexCastOp>([&](auto indexCastOp) {
            SmallVector<int> mockVals = getIndexMockVals(indexCastOp);
            mockDataItems.emplace_back(MockData(indexCastOp, 0, mockVals));
          })
          .Case<mlir::gpu::ThreadIdOp>([&](auto threadIdOp) {
            SmallVector<int> mockVals = getThreadIdMockVals();
            mockDataItems.emplace_back(MockData(threadIdOp, 0, mockVals));
          })
          .Case<triton::GetProgramIdOp>([&](auto getProgramIdOp) {
            SmallVector<int> mockVals = getProgramIdMockVals();
            mockDataItems.emplace_back(MockData(getProgramIdOp, 0, mockVals));
          })
          .Case<triton::xpu::GM2LMOp>([&](auto gm2lmOp) {
            SmallVector<int> mockVals = getGM2LMOpMockVals();
            mockDataItems.emplace_back(MockData(gm2lmOp, 0, mockVals));
          });
    }
    return mockDataItems;
  };

  SmallVector<int>
  constOpCalFunc(arith::ConstantOp op,
                 DenseMap<Operation *, SmallVector<int>> &op2OffsetVal) {
    auto type = op.getResult().getType();
    int intValue;

    if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) { // tensor
      auto shape = tensorType.getShape();
      unsigned rank = shape.size();
      unsigned numElems = shape[rank - 1];

      auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(op.getValue());

      auto elementType = tensorType.getElementType();
      if (elementType.isF32()) {
        intValue = *denseAttr.getValues<float>().begin();
      } else if (elementType.isInteger(32)) {
        intValue = *denseAttr.getValues<int32_t>().begin();
      } else if (elementType.isInteger(64)) {
        intValue = *denseAttr.getValues<int64_t>().begin();
      } else if (elementType.isInteger(1)) {
        intValue = *denseAttr.getValues<bool>().begin();
      } else {
        llvm_unreachable(
            "[Offset Analysis] Unsupported Element Type in ConstOp");
      }

      return SmallVector<int>(numElems, intValue);
    }

    if (type.isF32()) {
      auto doubleVal = mlir::cast<FloatAttr>(op.getValue()).getValueAsDouble();
      intValue = static_cast<int>(doubleVal);
    } else {
      intValue =
          mlir::cast<IntegerAttr>(op.getValue()).getValue().getZExtValue();
    }

    return SmallVector<int>(1, intValue);
  }

  SmallVector<int>
  indexCastOpCalFunc(arith::IndexCastOp op,
                     DenseMap<Operation *, SmallVector<int>> &op2OffsetVal,
                     int mockVal = 0) {
    return SmallVector<int>(1, mockVal);
  }

  SmallVector<int>
  threadIdOpCalFunc(mlir::gpu::ThreadIdOp op,
                    DenseMap<Operation *, SmallVector<int>> &op2OffsetVal,
                    int mockVal = 0) {
    return SmallVector<int>(1, mockVal);
  }

  SmallVector<int>
  getProgramIdOpCalFunc(triton::GetProgramIdOp op,
                        DenseMap<Operation *, SmallVector<int>> &op2OffsetVal,
                        int mockVal = 0) {
    return SmallVector<int>(1, mockVal);
  }

  SmallVector<int>
  xpuGm2lmOpCalFunc(triton::xpu::GM2LMOp op,
                    DenseMap<Operation *, SmallVector<int>> &op2OffsetVal,
                    int mockVal = 0) {

    auto offsetState = static_cast<OffsetState>(op.getOffsetState());
    int32_t op_lrie = op.getLrie();
    assert((offsetState == OffsetState::DiscreteSame || op_lrie > 1) &&
           "Mocked GM2LMOp must be DiscreteSame");

    auto result = op.getResult();
    auto resTy = mlir::dyn_cast<RankedTensorType>(result.getType());
    auto resShape = resTy.getShape();
    unsigned numElems = resShape[resShape.size() - 1];

    return SmallVector<int>(numElems, mockVal);
  }

  SmallVector<int>
  xpuLoadOpCalFunc(triton::xpu::LoadOp op,
                   DenseMap<Operation *, SmallVector<int>> &op2OffsetVal,
                   int mockVal = 0) {
    auto ptr = op.getPtr();
    auto ptrOp = ptr.getDefiningOp();
    return op2OffsetVal[ptrOp];
  }

  SmallVector<int>
  makeRangeOpCalFunc(triton::MakeRangeOp op,
                     DenseMap<Operation *, SmallVector<int>> &op2OffsetVal) {

    auto start = op.getStart();
    auto end = op.getEnd();

    SmallVector<int> res;
    for (size_t i = start; i < end; ++i) {
      res.push_back(i);
    }
    return res;
  }

  SmallVector<int>
  splatOpCalFunc(triton::SplatOp op,
                 DenseMap<Operation *, SmallVector<int>> &op2OffsetVal) {

    auto operand = op.getOperand();
    Operation *operandOp;
    SmallVector<int> prevValue;
    if (mlir::isa<BlockArgument>(operand)) {
      prevValue.push_back(0);
    } else {
      operandOp = operand.getDefiningOp();
      assert(operandOp && op2OffsetVal.find(operandOp) != op2OffsetVal.end() &&
             "Operands must be present in op2OffsetVal map");
      prevValue = op2OffsetVal[operandOp];
    }

    assert(prevValue.size() == 1 && "[splatOpCalFunc] Only support 1->N splat");

    auto src = op.getSrc();
    auto res = op.getResult();
    auto srcTy = mlir::dyn_cast<RankedTensorType>(src.getType());
    auto resTy = mlir::dyn_cast<RankedTensorType>(res.getType());
    auto resShape = resTy.getShape();

    unsigned rank = resShape.size();
    unsigned numElems = resShape[rank - 1];

    return SmallVector<int>(numElems, prevValue[0]);
  }

  SmallVector<int>
  expandDimsOpCalFunc(triton::ExpandDimsOp op,
                      DenseMap<Operation *, SmallVector<int>> &op2OffsetVal) {

    auto operand = op.getOperand();
    auto operandOp = operand.getDefiningOp();
    assert(operandOp && op2OffsetVal.find(operandOp) != op2OffsetVal.end() &&
           "Operands must be present in op2OffsetVal map");

    auto src = op.getSrc();
    auto res = op.getResult();
    auto srcTy = mlir::dyn_cast<RankedTensorType>(src.getType());
    auto resTy = mlir::dyn_cast<RankedTensorType>(res.getType());
    auto srcShape = srcTy.getShape();
    auto resShape = resTy.getShape();

    if (srcShape.size() == 1 && resShape.size() == 2 &&
        resShape[resShape.size() - 1] == 1) { // xmask make_range
      return SmallVector<int>(1, 0);
    }

    return op2OffsetVal[operandOp];
  }

  SmallVector<int>
  broadcastOpCalFunc(triton::xpu::BroadcastOp op,
                     DenseMap<Operation *, SmallVector<int>> &op2OffsetVal) {

    auto operand = op.getOperand();
    auto operandOp = operand.getDefiningOp();
    assert(operandOp && op2OffsetVal.find(operandOp) != op2OffsetVal.end() &&
           "Operands must be present in op2OffsetVal map");

    auto src = op.getSrc();
    auto res = op.getResult();
    auto srcTy = mlir::dyn_cast<RankedTensorType>(src.getType());
    auto resTy = mlir::dyn_cast<RankedTensorType>(res.getType());
    auto srcShape = srcTy.getShape();
    auto resShape = resTy.getShape();

    if (!srcTy && resTy) { // [f32 -> 1xf32]
      unsigned numElems = resShape[resShape.size() - 1];
      assert(op2OffsetVal[operandOp].size() == 1 &&
             "[broadcastOpCalFunc] Error Input Shape [f32 -> "
             "1xf32]");
      return SmallVector<int>(numElems, op2OffsetVal[operandOp][0]);
    }

    if (srcShape.size() != resShape.size()) {
      return op2OffsetVal[operandOp];
    }

    int broadNum = 0;
    for (size_t i = 0; i < srcShape.size(); ++i) {
      if (srcShape[i] != resShape[i]) {
        if (++broadNum > 1) { // [1x1xf32 -> NxNxf32]
          llvm_unreachable("[broadcastOpCalFunc] Unsupported broadcast 2 dims");
        }
      }
    }

    for (size_t i = 0; i < srcShape.size(); ++i) {
      if (srcShape[i] != resShape[i]) {
        if (srcShape[i] == 1) { // [1x1xf32 -> 1xNxf32]
          unsigned numElems = resShape[resShape.size() - 1];
          if (i == srcShape[srcShape.size() - 1])
            return SmallVector<int>(numElems, op2OffsetVal[operandOp][0]);
          else
            return op2OffsetVal[operandOp];
        } else { // [1x2xf32 -> 1xNxf32]
          llvm_unreachable("[broadcastOpCalFunc] Only support broadcast 1->N");
        }
      }
    }

    return op2OffsetVal[operandOp];
  }

  SmallVector<int> xpuConvertLayoutOpCalFunc(
      triton::xpu::ConvertLayoutOp op,
      DenseMap<Operation *, SmallVector<int>> &op2OffsetVal) {

    auto operand = op.getOperand();
    auto operandOp = operand.getDefiningOp();
    assert(operandOp && op2OffsetVal.find(operandOp) != op2OffsetVal.end() &&
           "Operands must be present in op2OffsetVal map");

    return op2OffsetVal[operandOp];
  }

  template <typename T>
  SmallVector<int>
  unaryOpCalFunc(T op, DenseMap<Operation *, SmallVector<int>> &op2OffsetVal) {
    auto operand = op.getOperand();

    auto unaryOp = operand.getDefiningOp();
    if (!unaryOp) {
      LLVM_DEBUG(llvm::dbgs() << "Operand Must Be Defined by Operations\n");
      findUnsupportedOp = true;
      return {};
    }
    assert(op2OffsetVal.find(unaryOp) != op2OffsetVal.end() &&
           "Operand Must Be Present in op2OffsetVal map");

    SmallVector<int> operandVal = op2OffsetVal[unaryOp];

    SmallVector<int> res;
    if constexpr (std::is_same_v<T, arith::ExtSIOp>) {
      for (size_t i = 0; i < operandVal.size(); ++i) {
        res.push_back(operandVal[i]);
      }
    } else {
      llvm_unreachable("Unknown binOpCalFunc Type");
    }
    return res;
  }

  template <typename T>
  SmallVector<int>
  binOpCalFunc(T op, DenseMap<Operation *, SmallVector<int>> &op2OffsetVal) {
    auto operands = op.getOperands();
    assert(operands.size() == 2 &&
           "Expected binary operation with two operands");

    auto lhsOp = operands[0].getDefiningOp();
    auto rhsOp = operands[1].getDefiningOp();
    if (!lhsOp || !rhsOp) {
      findUnsupportedOp = true;
      return {};
    }
    assert(op2OffsetVal.find(lhsOp) != op2OffsetVal.end() &&
           op2OffsetVal.find(rhsOp) != op2OffsetVal.end() &&
           "Operands must be present in op2OffsetVal map");

    SmallVector<int> lhs = op2OffsetVal[lhsOp];
    SmallVector<int> rhs = op2OffsetVal[rhsOp];
    if (lhs.size() != rhs.size()) {
      LLVM_DEBUG(llvm::dbgs() << "lhs.size(): " << lhs.size()
                              << "/ rhs.size(): " << rhs.size() << "\n");
    }
    assert(lhs.size() == rhs.size() &&
           "Two operands size must be equal"); // TODO: splat for binOp

    SmallVector<int> res;
    if constexpr (std::is_same_v<T, arith::AddIOp>) {
      for (size_t i = 0; i < lhs.size(); ++i) {
        res.push_back(lhs[i] + rhs[i]);
      }
    } else if constexpr (std::is_same_v<T, arith::SubIOp>) {
      for (size_t i = 0; i < lhs.size(); ++i) {
        res.push_back(lhs[i] - rhs[i]);
      }
    } else if constexpr (std::is_same_v<T, arith::DivSIOp>) {
      for (size_t i = 0; i < lhs.size(); ++i) {
        if (rhs[i] == 0) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Div 0, Return OffsetState::Unknown for Protection\n");
          findUnsupportedOp = true;
          return {};
        }
        res.push_back(lhs[i] / rhs[i]);
      }
    } else if constexpr (std::is_same_v<T, arith::MulIOp>) {
      for (size_t i = 0; i < lhs.size(); ++i) {
        res.push_back(lhs[i] * rhs[i]);
      }
    } else if constexpr (std::is_same_v<T, arith::RemSIOp>) {
      for (size_t i = 0; i < lhs.size(); ++i) {
        if (rhs[i] == 0) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Rem 0, Return OffsetState::Unknown for Protection\n");
          findUnsupportedOp = true;
          return {};
        }
        res.push_back(lhs[i] % rhs[i]);
      }
    } else {
      llvm_unreachable("Unknown binOpCalFunc Type");
    }
    return res;
  }

  SmallVector<int> getOffset(const SetVector<Operation *> &opChain,
                             Operation *offsetDefineOp,
                             const SmallVector<MockData> &mockDataItems) {
    auto hasDynamicInput = [](Operation *op) -> bool {
      for (auto operand : op->getOperands()) {
        if (mlir::isa<BlockArgument>(operand)) {
          continue;
        }
        auto operandOp = operand.getDefiningOp();
        if (!operandOp) {
          return true;
        }
      }
      return false;
    };

    bool findGM2LMOp =
        llvm::find_if(opChain, [](Operation *op) {
          if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMOp>(op)) {
            auto offsetState =
                static_cast<OffsetState>(gm2lmOp.getOffsetState());
            // We can mock DiscreteSame OffsetState
            return offsetState != OffsetState::DiscreteSame;
          }
          return false;
        }) != opChain.end();

    // Step 1. Pruning
    if (opChain.empty() || findGM2LMOp)
      return {};

    // Step 2. collect mock value
    DenseMap<Operation *, int> op2MockVal;
    for (auto mockData : mockDataItems)
      op2MockVal[mockData.mockOp] = mockData.mockVal;

    // Step 3. get offset result
    DenseMap<Operation *, SmallVector<int>> op2OffsetVal;
    for (auto it = opChain.rbegin(), eit = opChain.rend(); it != eit; ++it) {
      if (op2OffsetVal.find(*it) == op2OffsetVal.end()) {
        TypeSwitch<Operation *>(*it)
            .Case<arith::IndexCastOp>([&](auto indexCastOp) {
              auto mockVal = op2MockVal[indexCastOp];
              op2OffsetVal[indexCastOp] =
                  indexCastOpCalFunc(indexCastOp, op2OffsetVal, mockVal);
            })
            .Case<mlir::gpu::ThreadIdOp>([&](auto threadIdOp) {
              auto mockVal = op2MockVal[threadIdOp];
              op2OffsetVal[threadIdOp] =
                  threadIdOpCalFunc(threadIdOp, op2OffsetVal, mockVal);
            })
            .Case<triton::GetProgramIdOp>([&](auto getProgramIdOp) {
              auto mockVal = op2MockVal[getProgramIdOp];
              op2OffsetVal[getProgramIdOp] =
                  getProgramIdOpCalFunc(getProgramIdOp, op2OffsetVal, mockVal);
            })
            .Case<triton::xpu::GM2LMOp>([&](auto xpuGm2lmOp) {
              auto mockVal = op2MockVal[xpuGm2lmOp];
              op2OffsetVal[xpuGm2lmOp] =
                  xpuGm2lmOpCalFunc(xpuGm2lmOp, op2OffsetVal, mockVal);
            })
            .Case<triton::xpu::LoadOp>([&](auto xpuLoadOp) {
              auto mockVal = op2MockVal[xpuLoadOp];
              op2OffsetVal[xpuLoadOp] =
                  xpuLoadOpCalFunc(xpuLoadOp, op2OffsetVal, mockVal);
            })
            .Case<arith::ConstantOp>([&](auto constOp) {
              op2OffsetVal[constOp] = constOpCalFunc(constOp, op2OffsetVal);
            })
            .Case<ARITH_PTR_UNARY_OP>([&](auto unaryOp) {
              op2OffsetVal[unaryOp] = unaryOpCalFunc(unaryOp, op2OffsetVal);
            })
            .Case<ARITH_PTR_BINARY_OP>([&](auto binOp) {
              op2OffsetVal[binOp] = binOpCalFunc(binOp, op2OffsetVal);
            })
            .Case<triton::MakeRangeOp>([&](auto makeRangeOp) {
              op2OffsetVal[makeRangeOp] =
                  makeRangeOpCalFunc(makeRangeOp, op2OffsetVal);
            })
            .Case<triton::SplatOp>([&](auto splatOp) {
              if (hasDynamicInput(splatOp)) {
                findUnsupportedOp = true;
                return;
              }
              op2OffsetVal[splatOp] = splatOpCalFunc(splatOp, op2OffsetVal);
            })
            .Case<triton::ExpandDimsOp>([&](auto expandDimsOp) {
              op2OffsetVal[expandDimsOp] =
                  expandDimsOpCalFunc(expandDimsOp, op2OffsetVal);
            })
            .Case<triton::xpu::BroadcastOp>([&](auto broadcastOp) {
              if (hasDynamicInput(broadcastOp)) {
                findUnsupportedOp = true;
                return;
              }
              op2OffsetVal[broadcastOp] =
                  broadcastOpCalFunc(broadcastOp, op2OffsetVal);
            })
            .Case<triton::xpu::ConvertLayoutOp>([&](auto convertLayoutOp) {
              op2OffsetVal[convertLayoutOp] =
                  xpuConvertLayoutOpCalFunc(convertLayoutOp, op2OffsetVal);
            })
            .Default([&](auto &op) {
              findUnsupportedOp = true;
              LLVM_DEBUG(llvm::dbgs()
                         << "[OffsetState]: Unsupported Operation Type: "
                         << op->getName().getStringRef()
                         << ". Return OffsetState::Unknown for Protection.\n");
            });
        if (findUnsupportedOp) {
          return {};
        }
      }
    }

    assert(op2OffsetVal.find(offsetDefineOp) != op2OffsetVal.end() &&
           "Operands Must Be Present in op2OffsetVal Map\n");

    SmallVector<int> res = op2OffsetVal[offsetDefineOp];
    return res;
  }

  void getAllOffset(
      std::unordered_map<std::string, SmallVector<int>> &allOffsetResults,
      SmallVector<std::string> &sortedTokens,
      SmallVector<MockData> &mockDataItems, size_t curOpIndex,
      std::string token, Operation *offsetDefineOp,
      const SetVector<Operation *> &opChain) {

    if (curOpIndex == mockDataItems.size()) {
      sortedTokens.emplace_back(token);
      allOffsetResults[token] =
          getOffset(opChain, offsetDefineOp, mockDataItems);
      return;
    }

    for (int val : mockDataItems[curOpIndex].mockVals) {
      mockDataItems[curOpIndex].mockVal = val;
      getAllOffset(allOffsetResults, sortedTokens, mockDataItems,
                   curOpIndex + 1,
                   token + mockDataItems[curOpIndex].getToken() + "-",
                   offsetDefineOp, opChain);
    }
  }

  OffsetState memoryStateTransfer(const OffsetState &state1,
                                  const OffsetState &state2) {
    auto state1_it = offsetstate_transition_table.find(state1);
    if (state1_it != offsetstate_transition_table.end()) {
      auto state2_it = state1_it->second.find(state2);
      if (state2_it != state1_it->second.end()) {
        return state2_it->second;
      }
    }
    llvm_unreachable("Invalid OffsetState Transition");
    return OffsetState::Unknown;
  }

  OffsetState checkOffset(const SmallVector<int> &res,
                          const unsigned &numElems) {

    auto isEquallyStride = [](const SmallVector<int> &res,
                              size_t numElems) -> bool {
      int step = res[1] - res[0];
      for (size_t start = 0; start < res.size(); start += numElems) {
        for (size_t i = 2; i < numElems; ++i) {
          if (start + i < res.size())
            if ((res[start + i] - res[start + i - 1]) != step) {
              return false;
            }
        }
      }
      return true;
    };

    if (numElems == 1 || res.empty())
      return OffsetState::Unknown;

    SmallVector<int> offsets(res.size());
    bool multiBank = false;
    for (size_t start = 0; start < res.size(); start += numElems) {
      for (size_t i = 0; i < numElems; ++i) {
        if (start + i < res.size()) {
          offsets[start + i] = res[start + i] - res[start];
          // check online
          if (offsets[start + i] >= numElems) {
            multiBank = true;
          }
          if (offsets[start + i] < 0) {
            LLVM_DEBUG(llvm::dbgs()
                       << "[OffsetState]: The 0th Address Is Not the Beginning "
                          "of the Bank.\n");
            fixedStride = -1;
            return OffsetState::Unknown;
          }
        }
      }
    }

    if (multiBank) {
      if (isEquallyStride(res, numElems)) {
        fixedStride = res[1] - res[0];
        LLVM_DEBUG(llvm::dbgs()
                   << "[OffsetState]: Addresses Are Not in the Same Bank, "
                      "but Have Fixed Stride-"
                   << fixedStride << ".\n");
        return OffsetState::Unknown;
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "[OffsetState]: Addresses Are Not in the Same Bank.\n");
        fixedStride = -1;
        return OffsetState::Unknown;
      }
    }

    bool hasDiscreateSame = false;
    bool hasDiscrete = false;
    for (size_t start = 0; start < res.size(); start += numElems) {
      SmallVector<int> coreOffset;
      for (size_t i = 0; i < numElems; ++i) {
        if (start + i < res.size()) {
          if (offsets[start + i] != 0 && offsets[start + i] != i)
            return OffsetState::Discrete;

          coreOffset.push_back(offsets[start + i]);
        }
      }
      auto allZeros = std::all_of(coreOffset.begin(), coreOffset.end(),
                                  [](int num) { return num == 0; });
      if (allZeros) {
        hasDiscreateSame = true;
      } else {
        hasDiscrete = true;
      }
    }

    if (hasDiscreateSame && hasDiscrete)
      return OffsetState::Discrete;

    fixedStride = res[1] - res[0];
    return offsets[1] == 1 ? OffsetState::Continuous
                           : OffsetState::DiscreteSame;
  }

  void lrieDiscreteSameAnalysis(const SmallVector<int> &res) {
    analysisFlag = true;

    if (res.size() == 0)
      return;

    SmallVector<int> seq_lens;

    auto getSeqLens = [&]() {
      int cur_val = res[0];
      int count = 1;

      for (size_t i = 1; i < res.size(); ++i) {
        if (res[i] == cur_val) {
          ++count;
        } else {
          seq_lens.push_back(count);
          cur_val = res[i];
          count = 1;
        }
      }
      seq_lens.push_back(count);
    };
    // Step 1. Get All Same Sequence Lengths
    getSeqLens();

    // for (int i = 0; i < seq_lens.size(); ++i) {
    //   LLVM_DEBUG(llvm::dbgs() << "seq_lens[" << i << "]: " << seq_lens[i] <<
    //   "\n");
    // }

    // Step 2. Get Sequence Lengths' GCD
    auto gcd = [](int a, int b) {
      while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
      }
      return a;
    };

    lrie = seq_lens[0];
    for (size_t i = 1; i < seq_lens.size(); i++) {
      lrie = gcd(lrie, seq_lens[i]);
      if (lrie == 1)
        break;
    }

    if (dumpFlag && lrie > 1) {
      LLVM_DEBUG(llvm::dbgs() << "lrie: " << lrie << "\n");
    }

    if (lrie > 256 && lrie % 256 == 0)
      lrie = 256; // Control LM BufferSize to Avoid Memory Exceed
    else
      assert("lrie is not the multiple of 256");

    return;
  }

  OffsetState locallyContinity(
      std::unordered_map<std::string, SmallVector<int>> &allOffsetResults,
      SmallVector<std::string> &sortedTokens, const int64_t numElems) {

    auto isLocallyContinuous = [](const SmallVector<int> &res,
                                  const int64_t numElems, int64_t &rowLen,
                                  int64_t &rowStride) -> bool {
      if (res.size() < 2) {
        return false;
      }
      int64_t step = res[1] - res[0];
      if (step != 1) {
        return false;
      }
      int64_t currRowLen = 2;
      int64_t currRowStride = 1;
      bool isFirst = true;
      for (int64_t i = 2; i < res.size(); i++) {
        if (res[i] - res[i - 1] == 1) {
          currRowLen++;
        } else {
          currRowStride = res[i] - res[i - 1] + currRowLen - 1;
          if (currRowStride < 0) {
            return false;
          }

          if (isFirst) {
            rowLen = currRowLen;
            rowStride = currRowStride;
            isFirst = false;
          } else {
            if (rowStride == -1 && currRowStride % rowLen != 0) {
              rowLen = -1;
              return false;
            }
            if (currRowStride != rowStride) {
              rowStride = -1;
            }

            auto gcd = [](int64_t a, int64_t b) {
              while (b != 0) {
                int t = b;
                b = a % b;
                a = t;
              }
              return a;
            };
            rowLen = gcd(rowLen, currRowLen);
          }
          currRowLen = 1;
          currRowStride = 1;
        }
      }
      bool _isLocallyContinuous = false;
      if (rowLen >= 2 && rowLen % numElems != 0) {
        _isLocallyContinuous = true;
      }
      return _isLocallyContinuous;
    };

    SmallVector<int> _allOffsets;
    for (auto token : sortedTokens) {
      for (auto offset : allOffsetResults[token]) {
        _allOffsets.emplace_back(offset);
      }
    }
    if (isLocallyContinuous(_allOffsets, numElems, rowLen, rowStride)) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "[OffsetState]: The Address is Locally Continuous, rowLen is "
          << rowLen << ", rowStride is " << rowStride << "\n");
      return OffsetState::LocallyContinuous;
    }
    return OffsetState::Unknown;
  }

  // -1         for Unknown
  // 0          for DiscreteSame
  // 1          for Continuous
  // 2          for Discrete
  // 3          for LocallyContinuous
  template <class T, std::enable_if_t<is_xpu_memory_op<T>::value, bool> = true>
  OffsetState getOffsetState(T memoryOp) {
    Value ptr = memoryOp.getPtr();
    Operation *ptrOp = ptr.getDefiningOp();
    if (!ptrOp) {
      // Case 1. inptr -> gm2lm
      return OffsetState::Continuous;
    } else if (isa<triton::BitcastOp>(ptrOp) || isa<triton::SplatOp>(ptrOp)) {
      // Case 2. inptr -> cal -> addptr -> bitcast -> gm2lm
      // Case 3. inptr -> cal -> addptr -> splat -> gm2lm
      Value prevVal = ptrOp->getOperand(0);
      ptrOp = prevVal.getDefiningOp();
      if (!ptrOp)
        return OffsetState::Continuous;
    } else if (!isa<triton::AddPtrOp>(ptrOp)) {
      // Case 4. inptr -> unknown -> gm2lm
      LLVM_DEBUG(
          llvm::dbgs()
          << "[OffsetAnalysis]: Unsupported Offset Calculation Pattern\n");
      return OffsetState::Unknown;
    }

    if (!isa<triton::AddPtrOp>(ptrOp)) {
      // Case 5. inptr -> bitcast/splat -> unknown -> gm2lm
      LLVM_DEBUG(llvm::dbgs()
                 << "[OffsetAnalysis]: Unsupported Offset Calculation "
                    "Pattern with Bitcast/Splat\n");
      return OffsetState::Unknown;
    }

    // Case Normal. inptr -> cal -> addptr -> gm2lm
    Value offset = ptrOp->getOperand(1);
    Operation *offsetDefineOp = offset.getDefiningOp();

    // Step 1. Get Offset opChain
    SetVector<Operation *> opChain;
    getOpChainBwdBFS(opChain, offsetDefineOp);
    opChain = sortByLine(opChain);

    if (dumpFlag)
      LLVM_DEBUG(dumpOpChain(opChain));

    // Step 2. Cal offsetMock
    // Step 2.1. Get All Mockdata Items
    // [arith::IndexCastOp, mlir::gpu::ThreadIdOp, triton::GetProgramIdOp]
    SmallVector<MockData> mockDataItems = getMockDataItems(opChain);

    // Step 2.2. Check && Pruning Before getAllOffset
    auto checkMockDataList = [](SmallVector<MockData> &mockDataItems) {
      for (auto mockData : mockDataItems) {
        if (mockData.mockVals.empty())
          return false;
      }
      return true;
    };
    if (!checkMockDataList(mockDataItems))
      return OffsetState::Unknown;

    // Step 2.3. Get allOffsetResults With Own Token
    // token(key)-offsetResult(value)
    std::unordered_map<std::string, SmallVector<int>> allOffsetResults;
    SmallVector<std::string> sortedTokens;
    getAllOffset(allOffsetResults, sortedTokens, mockDataItems, 0, "",
                 offsetDefineOp, opChain);
    if (findUnsupportedOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Operands Must Be Defined By operations. "
                    "Check If Operand Is an Input Argument.\n"
                    "Return OffsetState::Unknown for Protection.\n");
    }

    // Step 3. Get OffsetState by Offset
    // Step 3.1. calculate numElems
    bool atomicSim = memoryOp.getAtomicSim();
    unsigned numElems = 1;
    if (auto offsetTy = mlir::dyn_cast<RankedTensorType>(offset.getType())) {
      auto offsetShape = offsetTy.getShape();
      unsigned rank = offsetShape.size();
      auto gEncoding =
          mlir::cast<triton::xpu::ClusterLayoutAttr>(offsetTy.getEncoding());
      auto sizePerCore = gEncoding.getSizePerCore();
      auto coresPerGroup = gEncoding.getCoresPerGroup();
      auto groupsPerCluster = gEncoding.getGroupsPerCluster();

      if (atomicSim) {
        numElems = lrie > 1 ? lrie : 1;
      } else {
        numElems = product(sizePerCore);
      }

      // We Can Only Check 1st Row Ptr Offset While Small ColSize Opt Hit
      if (memoryOp.getTensorColSize() != -1)
        numElems = std::min(numElems, (unsigned)offsetShape[rank - 1]);
    }

    // Step 3.2. Calculate all the offsetStates for each token, and then
    // combine these offsetStates through memoryStateTransfer.
    OffsetState memoryState =
        checkOffset(allOffsetResults[sortedTokens[0]], numElems);
    std::unordered_map<std::string, OffsetState> allOffsetStateResult;
    for (auto token : sortedTokens) {
      auto offsetMock = allOffsetResults[token];
      allOffsetStateResult[token] = checkOffset(offsetMock, numElems);
      if (dumpFlag)
        LLVM_DEBUG(llvm::dbgs() << "\n"
                                << token << ":" << allOffsetStateResult[token]
                                << ", offsetMock.size() = " << offsetMock.size()
                                << ", numElems = " << numElems << "\n");
      memoryState =
          memoryStateTransfer(memoryState, allOffsetStateResult[token]);
    }

    if (atomicSim && !analysisFlag) {
      // analysis only once
      lrieDiscreteSameAnalysis(allOffsetResults[sortedTokens[0]]);
      if (lrie > 1) {
        fixedStride = 0;
        return OffsetState::DiscreteSame;
      }
    }

    // Step 4. Optimize to OffsetState::LocallyContinuous
    if (memoryState == OffsetState::Unknown &&
        memoryOp.getTensorColSize() == -1) {
      return locallyContinity(allOffsetResults, sortedTokens, numElems);
    }

    return memoryState;
  }

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    mlir::ModuleOp mod = getOperation();

    mod.walk([&](mlir::Operation *op) {
      TypeSwitch<const Operation *>(op).Case<XPU_MEMORY_OP>(
          [&](auto memoryOp) { legalizeOffset(memoryOp); });
    });

    mod.walk([&](mlir::Operation *op) { op2Line[op] = line++; });

    mod.walk([&](triton::xpu::GM2LMOp gm2lmOp) {
      if (dumpFlag)
        LLVM_DEBUG(llvm::dbgs()
                   << "\n=======================================\n");
      OffsetState offsetState = getOffsetState(gm2lmOp);
      if (dumpFlag) {
        LLVM_DEBUG(llvm::dbgs()
                   << "\n"
                   << gm2lmOp << "\n[OffsetState]: " << offsetState
                   << "\n=======================================\n");
      }
      // In case `fixedStride` being modified by cluster(s) whose
      // OffsetState is Continuous.
      if (offsetState == OffsetState::Discrete) {
        fixedStride = -1;
      } else if (offsetState == OffsetState::Unknown &&
                 (fixedStride == 1 | fixedStride == 0)) {
        // Multi Memory State Like (Unknown & Continuous)
        fixedStride = -1;
      }

      OpBuilder builder(gm2lmOp);
      int32_t offsetStateInt = static_cast<int32_t>(offsetState);
      gm2lmOp->setAttr("offsetState",
                       builder.getSI32IntegerAttr(offsetStateInt));
      gm2lmOp->setAttr("fixedStride", builder.getSI32IntegerAttr(fixedStride));
      gm2lmOp->setAttr("rowLen", builder.getIntegerAttr(
                                     builder.getIntegerType(64, true), rowLen));
      gm2lmOp->setAttr(
          "rowStride",
          builder.getIntegerAttr(builder.getIntegerType(64, true), rowStride));
      gm2lmOp->setAttr("lrie", builder.getSI32IntegerAttr(lrie));
      auto loadOp = cast<triton::xpu::LoadOp>(gm2lmOp->getNextNode());
      loadOp->setOperand(0, gm2lmOp);
      loadOp->setAttr("stride", builder.getSI32IntegerAttr(fixedStride));
      loadOp->setAttr("isDiscrete", builder.getBoolAttr(offsetState ==
                                                        OffsetState::Discrete));
      fixedStride = -1; // reset
      rowLen = -1;
      rowStride = -1;
      findUnsupportedOp = false;
    });

    mod.walk([&](triton::xpu::LM2GMOp lm2gmOp) {
      if (dumpFlag)
        LLVM_DEBUG(llvm::dbgs()
                   << "\n=======================================\n");
      OffsetState offsetState = getOffsetState(lm2gmOp);
      // Only able to handle continuous and unknown cases.
      if (offsetState != OffsetState::Continuous &&
          offsetState != OffsetState::LocallyContinuous) {
        offsetState = OffsetState::Unknown;
      }
      if (dumpFlag) {
        LLVM_DEBUG(llvm::dbgs()
                   << "\n"
                   << lm2gmOp << "\n[OffsetState]: " << offsetState
                   << "\n=======================================\n");
      }
      OpBuilder builder(lm2gmOp);
      int32_t offsetStateInt = static_cast<int32_t>(offsetState);
      lm2gmOp->setAttr("offsetState",
                       builder.getSI32IntegerAttr(offsetStateInt));
      lm2gmOp->setAttr("rowLen", builder.getIntegerAttr(
                                     builder.getIntegerType(64, true), rowLen));
      lm2gmOp->setAttr(
          "rowStride",
          builder.getIntegerAttr(builder.getIntegerType(64, true), rowStride));
      findUnsupportedOp = false; // reset
      rowLen = -1;
      rowStride = -1;
    });

    mod.walk([&](triton::xpu::SM2GMOp sm2gmOp) {
      // TODO: Deal with other offset states in SM2GM, especially 2D case in
      // reduction
      OffsetState offsetState = OffsetState::Continuous;
      if (dumpFlag) {
        LLVM_DEBUG(llvm::dbgs()
                   << "\n=======================================\n"
                   << sm2gmOp << "\n[OffsetState]: " << offsetState
                   << "\n=======================================\n");
      }

      OpBuilder builder(sm2gmOp);
      int32_t offsetStateInt = static_cast<int32_t>(offsetState);
      sm2gmOp->setAttr("offsetState",
                       builder.getSI32IntegerAttr(offsetStateInt));
      findUnsupportedOp = false; // reset
    });
  }

private:
  llvm::DenseMap<mlir::Operation *, int32_t> op2Line;
  int32_t line = 0;
  int32_t fixedStride = -1;
  int64_t rowLen = -1;
  int64_t rowStride = -1;
  int32_t lrie = -1; // Longest Run Of Identical Elements
  bool analysisFlag = false;
  bool findUnsupportedOp = false;

  /*                   OffsetState Transition Table
   *
   * +---------------+----------+----------+--------------+-----------+
   * |               | Unknown  | Discrete | Discrete Same| Continuous|
   * +---------------+----------+----------+--------------+-----------+
   * | Unknown       | Unknown  | Unknown  | Unknown      | Unknown   |
   * +---------------+----------+----------+--------------+-----------+
   * | Discrete      | Unknown  | Discrete | Discrete     | Discrete  |
   * +---------------+----------+----------+--------------+-----------+
   * | Discrete Same | Unknown  | Discrete | Discrete Same| Discrete  |
   * +---------------+----------+----------+--------------+-----------+
   * | Continuous    | Unknown  | Discrete | Discrete     | Continuous|
   * +---------------+----------+----------+--------------+-----------+
   *
   * Unknown Case:
   *   1. The Offset Mock Result's Size <= 1
   *   2. The 0th Address Is Not the Beginning of the Bank.
   *   3. Addresses Are Not in the Same Bank.
   *   4. MockDataList Check Failed (Empty MockData)
   */

  OffsetStateTransitionTable offsetstate_transition_table = {
      {OffsetState::Unknown,
       {
           {OffsetState::Unknown, OffsetState::Unknown},
           {OffsetState::Discrete, OffsetState::Unknown},
           {OffsetState::DiscreteSame, OffsetState::Unknown},
           {OffsetState::Continuous, OffsetState::Unknown},
           {OffsetState::LocallyContinuous, OffsetState::Unknown},
       }},
      {OffsetState::Discrete,
       {
           {OffsetState::Unknown, OffsetState::Unknown},
           {OffsetState::Discrete, OffsetState::Discrete},
           {OffsetState::DiscreteSame, OffsetState::Discrete},
           {OffsetState::Continuous, OffsetState::Discrete},
           {OffsetState::LocallyContinuous, OffsetState::Discrete},
       }},
      {OffsetState::DiscreteSame,
       {
           {OffsetState::Unknown, OffsetState::Unknown},
           {OffsetState::Discrete, OffsetState::Discrete},
           {OffsetState::DiscreteSame, OffsetState::DiscreteSame},
           {OffsetState::Continuous, OffsetState::Discrete},
           {OffsetState::LocallyContinuous, OffsetState::Discrete},
       }},
      {OffsetState::Continuous,
       {
           {OffsetState::Unknown, OffsetState::Unknown},
           {OffsetState::Discrete, OffsetState::Discrete},
           {OffsetState::DiscreteSame, OffsetState::Discrete},
           {OffsetState::Continuous, OffsetState::Continuous},
           {OffsetState::LocallyContinuous, OffsetState::LocallyContinuous},
       }},
      {OffsetState::LocallyContinuous,
       {
           {OffsetState::Unknown, OffsetState::Unknown},
           {OffsetState::Discrete, OffsetState::Discrete},
           {OffsetState::DiscreteSame, OffsetState::Discrete},
           {OffsetState::Continuous, OffsetState::LocallyContinuous},
           {OffsetState::LocallyContinuous, OffsetState::LocallyContinuous},
       }}};
};

} // namespace xpu
} // namespace triton
} // namespace mlir
