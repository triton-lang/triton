//===----------------------------------------------------------------------===//
//
// Copyright (c) Triton Project Contributors.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ANALYSIS_USEANALYSIS_H
#define TRITON_ANALYSIS_USEANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

std::unique_ptr<Pass> createTritonUseAnalysisPass();

enum class UseType {
  Undefined, // Initial state
  DataUse,   // value used for tensor computation only
  MetaUse,   // value used for metadata only
  MixUse     // value used for both tensor computation and metadata
};

struct UseInfo : public dataflow::AbstractSparseLattice {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UseInfo)
  using AbstractSparseLattice::AbstractSparseLattice;

  // Lattice state transfer function
  ChangeResult meetUseType(const UseType &other) {
    if (other == UseType::Undefined)
      return ChangeResult::NoChange;

    switch (type) {
    case UseType::Undefined:
      type = other;
      return ChangeResult::Change;
    case UseType::DataUse:
    case UseType::MetaUse:
      if (type == other) {
        return ChangeResult::NoChange;
      } else {
        type = UseType::MixUse;
        return ChangeResult::Change;
      }
    case UseType::MixUse:
      return ChangeResult::NoChange;
    }
  }

  ChangeResult meet(const AbstractSparseLattice &other) override {
    auto rhs = reinterpret_cast<const UseInfo *>(&other);
    return meetUseType(rhs->type);
  }

  void print(raw_ostream &os) const override {
    switch (type) {
    case UseType::DataUse:
      os << "DataUse";
      break;
    case UseType::MetaUse:
      os << "MetaUse";
      break;
    case UseType::MixUse:
      os << "MixUse";
      break;
    default:
      os << "Undefined";
    }
  }

  UseType type = UseType::Undefined;
};

class UseAnalysis : public dataflow::SparseBackwardDataFlowAnalysis<UseInfo> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;
  void visitOperation(Operation *op, ArrayRef<UseInfo *> operands,
                      ArrayRef<const UseInfo *> results) override;

  void visitBranchOperand(OpOperand &operand) override { return; }

  void setToExitState(UseInfo *lattice) override {
    lattice->type = UseType::Undefined;
  }

private:
  void propagateUse(UseInfo *lattice, const UseType &type) {
    auto changed = lattice->meetUseType(type);
    propagateIfChanged(lattice, changed);
  }

  void propagateResults(UseInfo *lattice, ArrayRef<const UseInfo *> results) {
    auto changed = ChangeResult::NoChange;
    for (auto result : results)
      changed |= lattice->meet(*result);
    propagateIfChanged(lattice, changed);
  }
};

// Use SparseBackwardDataAnalysis to identify operations whose results are used
// as data tensor operations, meta operations (address calculation,
// broadcasting/splating constant, etc.), or both. For operations used as both
// purposes, clone them so that the remaining pass built on
// ConversionPatternRewriter can replace all tensor producers cleanly and simply
// delete meta data producers.
LogicalResult runUseAnalysis(triton::FuncOp &funcOp);

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOAFFINE_TRITONUSEANALYSIS_H
