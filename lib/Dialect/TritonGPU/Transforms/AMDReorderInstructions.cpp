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

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;

class TritonAMDGPUReorderInstructionsPass
    : public TritonAMDGPUReorderInstructionsBase<
          TritonAMDGPUReorderInstructionsPass> {
public:
  TritonAMDGPUReorderInstructionsPass() = default;

  void sortOperandsByDominance(OperandRange operands,
                               SmallVector<Value> &operandsSorted) {
    ModuleOp m = getOperation();
    mlir::DominanceInfo dom(m);

    for (auto operand : operands) {
      // Sort only operands for which defining op can be fetched. This will
      // exclude, for example, block arguments.
      if (operand.getDefiningOp()) {
        operandsSorted.push_back(operand);
      }
    }

    if (operandsSorted.size() == 1) {
      return;
    }

    std::sort(operandsSorted.begin(), operandsSorted.end(),
              [&](const Value &a, const Value &b) {
                Operation *operandA = a.getDefiningOp();
                Operation *operandB = b.getDefiningOp();
                assert(operandA && operandB);
                return dom.dominates(operandA, operandB);
              });
  }

  void moveAfter(Operation *lhs, Operation *rhs) {
    auto lhsId = getWSRoleId(lhs);
    auto rhsId = getWSRoleId(rhs);
    if (lhsId == rhsId)
      lhs->moveAfter(rhs);
  }

  void moveBefore(Operation *lhs, Operation *rhs) {
    auto lhsId = getWSRoleId(lhs);
    auto rhsId = getWSRoleId(rhs);
    if (lhsId == rhsId)
      lhs->moveBefore(rhs);
  }

  bool isFAChainDot(tt::DotOp &dotOp) const {
    SetVector<Operation *> slices;
    getForwardSlice((Operation *)dotOp, &slices);

    for (Operation *op : slices) {
      if (isa<tt::DotOp>(op) && (op != dotOp)) {
        auto operandA = op->getOperand(0).getDefiningOp();
        auto containsOperandA =
            std::find(slices.begin(), slices.end(), operandA) != slices.end();
        if (containsOperandA) {
          return true;
        }
      }
    }
    return false;
  }

  void moveImmediatelyAfterOperands(Operation *op,
                                    SmallVector<Operation *> &movedOperations) {

    if (std::find(movedOperations.begin(), movedOperations.end(), op) !=
        movedOperations.end()) {
      return;
    }
    auto operands = op->getOperands();
    if (operands.empty()) {
      return;
    }
    ModuleOp m = getOperation();
    mlir::DominanceInfo dom(m);

    for (auto operandVal : operands) {
      Operation *argOp = operandVal.getDefiningOp();
      if (!argOp) {
        continue;
      }
      moveImmediatelyAfterOperands(argOp, movedOperations);
    }

    SmallVector<Value> operandsSorted;
    sortOperandsByDominance(operands, operandsSorted);

    if (!operandsSorted.empty()) {
      auto dominantOperandOp =
          operandsSorted[operandsSorted.size() - 1].getDefiningOp();
      if (dominantOperandOp) {
        moveAfter(op, dominantOperandOp);
        assert(succeeded(mlir::verify(m)));
      }
    }

    movedOperations.push_back(op);
  }

  // Moves Q tensor in Flash Attention algorithm out of the
  // "main" flash attention loop. Since Q tensor is the loop invariant, this way
  // we ensure that loading of Q tensor, Q tensor transformations and related
  // layout conversions happen only once.
  void moveQTensorOutOfTheLoop(ModuleOp m) {
    m.walk([&](tt::DotOp dotOp) {
      if (isFAChainDot(dotOp)) {
        Operation *operandA = dotOp->getOperand(0).getDefiningOp();
        SmallVector<Operation *> movedOperations;
        moveImmediatelyAfterOperands(operandA, movedOperations);
        return;
      }
    });
  }

  bool contains(const SmallVector<Operation *> &vec, Operation *element) {
    return std::find(vec.begin(), vec.end(), element) != vec.end();
  }

  bool containsInAnyChain(SmallVector<SmallVector<Operation *>> dotChains,
                          Operation *element) {
    for (auto chain : dotChains) {
      if (contains(chain, element)) {
        return true;
      }
    }
    return false;
  }

  bool isLDSWrite(Operation *op) {
    auto cvtLayoutOp = dyn_cast<ttg::ConvertLayoutOp>(op);
    if (!cvtLayoutOp) {
      return false;
    }
    auto srcType = cvtLayoutOp.getOperand().getType().cast<RankedTensorType>();
    auto dstType = cvtLayoutOp.getResult().getType().cast<RankedTensorType>();
    auto srcEncoding = srcType.getEncoding();
    auto dstEncoding = dstType.getEncoding();
    if (srcEncoding.isa<triton::gpu::BlockedEncodingAttr>() &&
        dstEncoding.isa<triton::gpu::SharedEncodingAttr>())
      return true;
    return false;
  }

  bool isLDSRead(Operation *op) {
    auto cvtLayoutOp = dyn_cast<ttg::ConvertLayoutOp>(op);
    if (!cvtLayoutOp) {
      return false;
    }
    auto srcType = cvtLayoutOp.getOperand().getType().cast<RankedTensorType>();
    auto dstType = cvtLayoutOp.getResult().getType().cast<RankedTensorType>();
    auto srcEncoding = srcType.getEncoding();
    auto dstEncoding = dstType.getEncoding();
    if (srcEncoding.isa<triton::gpu::SharedEncodingAttr>() &&
        dstEncoding.isa<triton::gpu::DotOperandEncodingAttr>())
      return true;
    return false;
  }

  void moveLoadStoreBeforeDot(Operation *currDot, Operation *moveBeforeDot,
                              SmallVector<Operation *> &operations,
                              int operandIdx) {
    auto operandB = currDot->getOperand(operandIdx).getDefiningOp();
    Operation *currOp = operandB;
    Operation *moveBeforeOp = moveBeforeDot;

    auto moveOp = [&](Operation *op, Operation *&opType) {
      if (opType) {
        moveAfter(op, opType);
      } else {
        moveBefore(op, moveBeforeOp);
      }
      opType = op;
    };

    for (int i = 0; !isa<ttg::ViewSliceOp>(currOp); i++) {
      moveOp(currOp, operations[i]);
      moveBeforeOp = currOp;
      currOp = currOp->getOperand(0).getDefiningOp();
    }
    moveOp(currOp, operations[operations.size() - 1]);
  }

  void initOperations(Operation *currOp, SmallVector<Operation *> &vec,
                      int operandIdx) {
    while (!isa<ttg::ViewSliceOp>(currOp)) {
      if (operandIdx == 0) {
        vec.push_back(currOp);
      } else {
        vec.push_back(nullptr);
      }
      currOp = currOp->getOperand(0).getDefiningOp();
    }
    if (operandIdx == 0) {
      vec.push_back(currOp);
    } else {
      vec.push_back(nullptr);
    }
  }

  void processStage(Operation *currDot, Operation *moveBeforeDot,
                    SmallVector<Operation *> &operations, bool init,
                    int operandIdx) {
    if (init) {
      initOperations(currDot->getOperand(operandIdx).getDefiningOp(),
                     operations, operandIdx);
      if (operandIdx == 0) {
        return;
      }
    }
    moveLoadStoreBeforeDot(currDot, moveBeforeDot, operations, operandIdx);
  }

  unsigned getNumUsers(Value value) {
    return std::distance(value.user_begin(), value.user_end());
  }

  // Rearrange instructions of dot chain in pipelining manner.
  // Note that not only load instruction
  // will be hoisted, but all instructions starting from load to cvt(shared,
  // dot). Let's say there are two dots:
  //
  // k0 = load(k0_ptr)
  // k0_shared = cvt(k0, blocked, shared)
  // k0_dot = cvt(k0_shared, shared, dot)
  // dot0 = dot(..., k0_dot, 0)
  //
  // k1 = load(k1_ptr)
  // k1_shared = cvt(k0, blocked, shared)
  // k1_dot = cvt(k0_shared, shared, dot)
  // dot1 = dot(..., k1_dot, dot1)
  //
  // doPipelining will rearrange instructions in following manner:
  //
  // k0 = load(k0_ptr)
  // k1 = load(k1_ptr)
  //
  // k0_shared = cvt(k0, blocked, shared)
  // k1_shared = cvt(k0, blocked, shared)
  //
  // k1_dot = cvt(k0_shared, shared, dot)
  // k0_dot = cvt(k0_shared, shared, dot)
  //
  // dot0 = dot(..., k0_dot, 0)
  // dot1 = dot(..., k1_dot, dot1)
  void doPipelining(SmallVector<SmallVector<Operation *>> &dotChains,
                    int pipelineStages) {
    for (auto chain : dotChains) {
      for (int i = 0; i < (chain.size() - 1) / pipelineStages; i++) {
        SmallVector<Operation *> operations;
        SmallVector<Operation *> operationsIdx0;
        int startStageIdx = i == 0 ? 0 : 1;
        for (int j = startStageIdx; j <= pipelineStages; j++) {
          processStage(/*currDot*/ chain[i * pipelineStages + j],
                       /*moveBeforeDot*/ chain[i * pipelineStages],
                       operationsIdx0, j == startStageIdx, /*operandIdx*/ 0);
          processStage(/*currDot*/ chain[i * pipelineStages + j],
                       /*moveBeforeDot*/ chain[i * pipelineStages], operations,
                       j == startStageIdx, /*operandIdx*/ 1);
        }
      }

      int startDotIdx = ((chain.size() - 1) / pipelineStages) * pipelineStages;
      SmallVector<Operation *> operations;
      SmallVector<Operation *> operationsIdx0;
      for (int i = 1; i <= (chain.size() - 1) % pipelineStages; i++) {
        processStage(chain[startDotIdx + i], chain[startDotIdx], operationsIdx0,
                     i == 1, 0);
        processStage(chain[startDotIdx + i], chain[startDotIdx], operations,
                     i == 1, 1);
      }
    }
  }

  void findDotChains(ModuleOp &m,
                     SmallVector<SmallVector<Operation *>> &dotChains) {
    m.walk([&](tt::DotOp dotOp) {
      if (!containsInAnyChain(dotChains, dotOp)) {
        SmallVector<Operation *> newChain;
        Operation *currOp = dotOp;
        newChain.push_back(currOp);

        if (getNumUsers(dotOp->getResult(0)) == 1) {
          auto user = *currOp->getUsers().begin();
          while (isa<tt::DotOp>(user)) {
            newChain.push_back(user);
            if (getNumUsers(user->getResult(0)) > 1) {
              break;
            }
            // TODO: check that  user is accumulator
            // of the dot.
            user = *user->getUsers().begin();
          }
        }
        if (newChain.size() >= 2) {
          dotChains.push_back(newChain);
        }
      }
    });
  }

  void sinkLDSConverts(SmallVector<SmallVector<Operation *>> &dotChains,
                       bool sinkLDSWr) {
    for (auto chain : dotChains) {
      for (int i = 0; i < chain.size(); i++) {
        Operation *dotOp = chain[i];
        Operation *ldsRd = dotOp->getOperand(1).getDefiningOp();
        assert(isLDSRead(ldsRd));
        moveBefore(ldsRd, dotOp);
        if (sinkLDSWr) {
          Operation *ldsWr = ldsRd->getOperand(0).getDefiningOp();
          assert(isLDSWrite(ldsWr));
          moveBefore(ldsWr, ldsRd);
        }
      }
    }
  }

  void interleaveLoadsAndLDS(SmallVector<SmallVector<Operation *>> &dotChains) {
    for (auto chain : dotChains) {
      for (int i = 1; i < chain.size(); i++) {
        Operation *dotOp = chain[i - 1];
        Operation *ldsRd = dotOp->getOperand(1).getDefiningOp();
        assert(isLDSRead(ldsRd));

        Operation *dotOpCurr = chain[i];
        Operation *curr = dotOpCurr->getOperand(1).getDefiningOp();
        while (!isa<tt::LoadOp>(curr)) {
          curr = curr->getOperand(0).getDefiningOp();
        }
        moveBefore(curr, ldsRd);
      }
    }
  }

  void scheduleSlicedDot(ModuleOp m, int stages, bool sinkLDSRd, bool sinkLDSWr,
                         bool interleaveLoadWithLDSOps) {
    SmallVector<SmallVector<Operation *>> dotChains;
    int pipelineLoads = stages - 1;

    findDotChains(m, dotChains);

    if (stages > 1)
      doPipelining(dotChains, pipelineLoads);

    if (sinkLDSRd)
      sinkLDSConverts(dotChains, sinkLDSWr);

    // Arrange ops in CK-like manner.
    // k0 = load k0_ptrs
    // k0_shared = cvt(k0, blocked, shared)
    // k1 = load k1_ptrs
    // k0_dot = cvt(k0, shared, dot)
    // dot0 = dot(q0_dot, k0_dot, 0
    //
    // k1_shared = cvt(k1, blocked, shared)
    // k2 = load k2_ptrs
    // k1_dot = cvt(k1, shared, dot)
    // dot1 = dot(q1_dot, k1_dot, dot0)
    if (interleaveLoadWithLDSOps && stages == 2) {
      interleaveLoadsAndLDS(dotChains);
    }
  }

  void runOnOperation() override {
    SmallVector<Operation *> movedOperations;
    ModuleOp m = getOperation();

    moveQTensorOutOfTheLoop(m);
    // TODO: Add some of these variables as autotunable parameters if needed.
    // At present, the optimal performance of FA fwd pass is achieved using the
    // following setup:
    int stages = 4;
    bool sinkLDSRd = false;
    bool sinkLDSWr = false;
    bool interleaveLoadWithLDSOps = false;

    // For CK-like FA fwd pass schedule of sliced dots use following configuration:
    // int stages = 2;
    // bool sinkLDSRd = true;
    // bool sinkLDSWr = true;
    // bool interleaveLoadWithLDSOps = true;
    scheduleSlicedDot(m, stages, sinkLDSRd, sinkLDSWr,
                      interleaveLoadWithLDSOps);
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
