//===----------------------------------------------------------------------===//
//
// Copyright (c) Triton Project Contributors.
//
//===----------------------------------------------------------------------===//

#include "triton/Analysis/UseAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace triton;
using namespace dataflow;

#define DEBUG_TYPE "triton-use-analysis"

//===----------------------------------------------------------------------===//
// Use Analysis
// Note that logic below should evolve with triton-to-affine pass
//===----------------------------------------------------------------------===//
void triton::UseAnalysis::visitOperation(Operation *op,
                                         ArrayRef<UseInfo *> operands,
                                         ArrayRef<const UseInfo *> results) {
  // If an op only produces pointer, all its operands are used as meta data.
  // This accounts for scenarios such as addptr in a loop whose result is
  // yielded. In this case, if the loop returns data tensors, addptr will be
  // marked correctly as meta use.
  if (op->getResults().size() == 1) {
    auto resultType = op->getResult(0).getType().dyn_cast<ShapedType>();
    if (resultType && isa<triton::PointerType>(resultType.getElementType())) {
      for (auto opnd : operands)
        propagateUse(opnd, UseType::MetaUse);
    }
  }

  TypeSwitch<Operation *>(op)
      .Case<triton::LoadOp>([&](auto load) {
        propagateUse(operands[0], UseType::MetaUse);
        auto mask = load.getMask();
        auto other = load.getOther();
        if (mask) {
          assert(mask != other && "mask and other cannot be the same");
          propagateUse(operands[1], UseType::MetaUse);
        }
        if (other) {
          // TODO:
          // More complicated patterns that generate other is unsupported.
          propagateUse(operands[2], UseType::MetaUse);
        }
      })
      .Case<triton::StoreOp>([&](auto store) {
        propagateUse(operands[0], UseType::MetaUse);
        propagateUse(operands[1], UseType::DataUse);
        auto value = store.getValue();
        auto mask = store.getMask();
        if (mask) {
          assert(mask != value && "mask and data cannot be the same");
          propagateUse(operands[2], UseType::MetaUse);
        }
      })
      .Case<triton::DotOp>([&](auto dot) {
        propagateResults(operands[0], results);
        propagateResults(operands[1], results);

        auto opc = dot.getC();
        triton::SplatOp splat;
        if (opc)
          splat = opc.template getDefiningOp<triton::SplatOp>();

        if (opc && splat && splat.getSrc().getDefiningOp<arith::ConstantOp>())
          propagateUse(operands[2], UseType::MetaUse);
        else
          propagateUse(operands[2], UseType::DataUse);
      })
      .Default([&](Operation *op) {
        // this condition account for tt.addptr
        for (auto operand : operands) {
          propagateResults(operand, results);
        }
      });
}

LogicalResult triton::runUseAnalysis(triton::FuncOp &funcOp) {
  MLIRContext *context = funcOp.getContext();
  SymbolTableCollection symbolTable;

  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();
  solver.load<UseAnalysis>(symbolTable);
  if (failed(solver.initializeAndRun(funcOp)))
    return failure();

  // Walk the func op, convert tags on operands to tags on operations
  funcOp.walk([&](Operation *op) {
    UseType useType = UseType::Undefined;
    for (auto result : op->getResults()) {
      auto use = solver.lookupState<UseInfo>(result);
      assert(use && "Lattice value not found");
      auto thisUseType = use->type;
      if (thisUseType == UseType::Undefined)
        continue;
      if (useType == UseType::Undefined)
        useType = thisUseType;
      if (thisUseType == UseType::MixUse || thisUseType != useType) {
        useType = UseType::MixUse;
        break;
      }
    }

    if (useType == UseType::Undefined) {
      LLVM_DEBUG({ op->setAttr("Undefined", UnitAttr::get(context)); });
      return;
    } else if (useType == UseType::MetaUse) {
      assert(op->getNumResults() == 1 &&
             "Ops used for meta computation are expected to have one result");
      // Only set the tag if the operation uses tensors
      if (op->getResult(0).getType().isa<ShapedType>()) {
        // Setting tag for erasing op later
        op->setAttr("MetaUse", UnitAttr::get(context));
      }
      return;
    } else if (useType == UseType::DataUse) {
      LLVM_DEBUG({ op->setAttr("DataUse", UnitAttr::get(context)); });
      return;
    }

    assert(useType == UseType::MixUse);

    // If the operation only produces scalars, no need to clone it
    bool shapedResult = true;
    for (auto result : op->getResults())
      shapedResult &= result.getType().isa<ShapedType>();
    if (!shapedResult) {
      LLVM_DEBUG({ op->setAttr("MixUse", UnitAttr::get(context)); });
      return;
    }

    // Value has MixUse. However, the operation may or may not have direct
    // MetaUse. E.g., it may only have MixUse, or only have MixUse and
    // DataUse.
    // - If the operation has direct MetaUse, clone it, tag the clone as
    // MetaUse only and point meta users to use the clone.
    // - If not, do nothing; this operation will still be materlized.
    llvm::SetVector<Operation *> metaUsers;
    for (auto result : op->getResults()) {
      for (auto user : result.getUsers()) {
        TypeSwitch<Operation *>(user)
            .Case<triton::LoadOp>([&](auto load) {
              auto ptr = load.getPtr();
              auto mask = load.getMask();
              auto other = load.getOther();
              if (result == ptr || result == mask || result == other)
                metaUsers.insert(user);
            })
            .Case<triton::StoreOp>([&](auto store) {
              auto ptr = store.getPtr();
              auto mask = store.getMask();
              if (result == ptr || result == mask)
                metaUsers.insert(user);
            })
            .Case<triton::DotOp>([&](auto dot) {
              auto opc = dot.getC();
              triton::SplatOp splat;
              if (opc)
                splat = opc.template getDefiningOp<triton::SplatOp>();

              if (opc && splat &&
                  splat.getSrc().getDefiningOp<arith::ConstantOp>())
                metaUsers.insert(user);
            })
            .Default([&](Operation *op) {
              // if all output of user are used as meta data, user is a meta
              // user. This condition account for addptr, or an addi whose
              // output only feeds into addptr
              bool allMeta = true;
              for (auto res : op->getResults()) {
                auto resUse = solver.lookupState<UseInfo>(res);
                if (resUse->type != UseType::MetaUse) {
                  allMeta = false;
                  break;
                }
              }
              if (allMeta)
                metaUsers.insert(user);
            });
      }
    }

    // If the operation doesn't have direct meta users, no need to clone it
    if (metaUsers.empty()) {
      LLVM_DEBUG({ op->setAttr("MixUse", UnitAttr::get(context)); });
      return;
    }

    // Clone the operation; switch all meta users to use the clone
    OpBuilder builder(op);
    auto clone = builder.clone(*op);
    LLVM_DEBUG({ op->setAttr("MixUse", UnitAttr::get(context)); });

    // Setting tag for erasing op later
    clone->setAttr("MetaUse", UnitAttr::get(context));

    for (auto [res_i, result] : llvm::enumerate(op->getResults()))
      for (auto user : metaUsers)
        for (auto &operand : user->getOpOperands())
          if (operand.get() == result)
            operand.set(clone->getResult(res_i));
  });

  return success();
}
