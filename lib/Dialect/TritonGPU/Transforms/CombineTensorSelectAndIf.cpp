#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include <memory>

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUCOMBINETENSORSELECTANDIF
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class CombineTensorSelectAndIfPass
    : public mlir::triton::gpu::impl::TritonGPUCombineTensorSelectAndIfBase<
          CombineTensorSelectAndIfPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    DominanceInfo dom(m);

    // Go over the arith.select ops, look if there is an if
    // with the same condition.
    SmallVector<std::pair<arith::SelectOp, scf::IfOp>, 4> selectIfCandidates;
    m.walk([&](arith::SelectOp selectOp) {
      // Look if there is an if in the same block, with the same condition.
      auto *parentBlock = selectOp->getBlock();
      Value condition = selectOp.getOperand(0);
      // Get condition's users
      for (auto user : condition.getUsers()) {
        if (auto ifOp = dyn_cast<scf::IfOp>(user)) {
          if (ifOp->getBlock() == parentBlock) {
            selectIfCandidates.emplace_back(selectOp, ifOp);
            continue;
          }
        }
      }
    });

    llvm::MapVector<arith::SelectOp, scf::IfOp> selectToIf;

    // Go over the candidates and check if the select and if can be combined.
    for (auto [selectOp, ifOp] : selectIfCandidates) {
      if (selectToIf.count(selectOp) > 0) {
        continue;
      }

      // If needs to be dominated by the select.
      if (!dom.dominates(selectOp.getOperation(), ifOp.getOperation())) {
        continue;
      }
      // If needs to dominate all the select's users.
      bool allUsersDominated = true;
      for (auto user : selectOp.getResult().getUsers()) {
        if (!dom.dominates(ifOp, user)) {
          allUsersDominated = false;
          break;
        }
      }

      if (allUsersDominated) {
        selectToIf[selectOp] = ifOp;
      }
    }

    for (auto [selectOp, ifOp] : selectToIf) {
      // Add new return value to the if (and create else block if necessary),
      // then yield the select value in the then block and the else block.
      OpBuilder builder(ifOp);
      auto loc = ifOp.getLoc();
      auto thenValue = selectOp.getTrueValue();
      auto elseValue = selectOp.getFalseValue();
      // Create an scf::IfOp with extra return value.
      SmallVector<Type> newResultTypes = {ifOp.getResultTypes().begin(),
                                          ifOp.getResultTypes().end()};
      newResultTypes.push_back(selectOp.getResult().getType());
      auto newIfOp = builder.create<scf::IfOp>(
          loc, newResultTypes, ifOp.getCondition(), /*hasElse*/ true);
      // Move the existing blocks to the new if.
      newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());

      if (ifOp.elseBlock()) {
        newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
      } else {
        // Create an empty yield
        auto yieldOp = newIfOp.getElseBodyBuilder().create<scf::YieldOp>(loc);
      }

      // Update yields
      auto appendToYield = [&](scf::YieldOp yield, Value value) {
        SmallVector<Value> operands(yield.getOperands());
        operands.append({value});
        builder.setInsertionPoint(yield);
        builder.create<scf::YieldOp>(loc, operands);
        yield.erase();
      };
      appendToYield(newIfOp.thenYield(), thenValue);
      appendToYield(newIfOp.elseYield(), elseValue);

      // Replace old if with the new one.
      for (auto result : ifOp.getResults()) {
        result.replaceAllUsesWith(newIfOp->getResult(result.getResultNumber()));
      }
      // Replace the select with the new return value.
      selectOp.replaceAllUsesWith(
          newIfOp->getResult(newIfOp->getNumResults() - 1));
      selectOp.erase();
      ifOp.erase();
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
