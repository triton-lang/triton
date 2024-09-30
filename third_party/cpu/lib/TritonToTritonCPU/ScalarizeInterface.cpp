#include "cpu/include/ScalarizePass/ScalarizeInterfaceImpl.h"

#include "cpu/include/ScalarizePass/ScalarizeInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

#include "cpu/include/ScalarizePass/ScalarizeInterface.cpp.inc"

using namespace mlir::triton;
using namespace mlir::triton::cpu;

Value mlir::triton::cpu::computeScalarValue(Operation *scalarizationOp,
                                            Value vals,
                                            ArrayRef<int64_t> indices,
                                            PatternRewriter &rewriter) {
  auto scalarized = cast<ScalarizeInterface>(scalarizationOp);
  return scalarized.computeScalarValue(vals, indices, rewriter);
}

Value mlir::triton::cpu::computeScalarValue(Operation *scalarizationOp,
                                            Value vals, ValueRange indices,
                                            PatternRewriter &rewriter) {
  auto scalarized = cast<ScalarizeInterface>(scalarizationOp);
  return scalarized.computeScalarValueForLoop(vals, indices, rewriter);
}

bool mlir::triton::cpu::canComputeScalarValue(Value vals) {
  auto def = vals.getDefiningOp();
  if (!def)
    return false;
  auto scalarized = dyn_cast<ScalarizeInterface>(def);
  if (!scalarized)
    return false;
  return scalarized.canComputeScalarValue(vals);
}

namespace {

namespace detail {

template <typename T> struct value_type_trait {
  using type = typename T::value_type;
};

template <> struct value_type_trait<ValueRange> {
  using type = Value;
};

template <typename T>
T createZeroIndex(mlir::Location loc, PatternRewriter &rewriter) {
  llvm_unreachable("Default implementation should be overwritten.");
}

template <>
int64_t createZeroIndex(mlir::Location loc, PatternRewriter &rewriter) {
  return 0;
}

template <>
Value createZeroIndex(mlir::Location loc, PatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantIndexOp>(loc, 0);
}

} // namespace detail

// Using ScalariztionFunctor class to partially specialize helper method
template <typename OpTy> struct ScalariztionFunctor {
  template <typename T>
  static Value getScalarValue(OpTy operation, Value vals, T indices,
                              PatternRewriter &rewriter) {
    auto def = vals.getDefiningOp<OpTy>();
    OperationState newState(def->getLoc(), def->getName());
    for (auto operand : def->getOperands()) {
      newState.operands.push_back(computeScalarValue(
          operand.getDefiningOp(), operand, indices, rewriter));
    }
    assert(def->getResults().size() == 1 &&
           "[Unsupported] Opearation have multiple outputs.");
    newState.types.push_back(
        cast<ShapedType>(def->getResultTypes()[0]).getElementType());
    newState.attributes = def->getAttrs();
    return rewriter.create(newState)->getResult(0);
  }
};

/// External model implementation of ScalarizeInterface for TritonOps. An
/// external model implementation is used for now till the use of
/// `ScalarizeInterface` is on-par with the current ScalarizeUsingForOp. This
/// allows to register this Interface for all required ops depending on it's
/// type.
template <typename OpTy>
struct TritonOpScalarizeInterface
    : public ScalarizeInterface::ExternalModel<TritonOpScalarizeInterface<OpTy>,
                                               OpTy> {
  bool canComputeScalarValue(Operation *op, Value vals) const {
    for (auto operand : op->getOperands()) {
      if (isa<BlockArgument>(operand)) {
        return false;
      }
      auto scalarized = dyn_cast<ScalarizeInterface>(operand.getDefiningOp());
      if (!scalarized) {
        return false;
      }
      if (!scalarized.canComputeScalarValue(operand)) {
        return false;
      }
    }
    return true;
  }

  Value computeScalarValue(Operation *op, Value vals, ArrayRef<int64_t> indices,
                           PatternRewriter &rewriter) const {
    OpTy def = vals.getDefiningOp<OpTy>();
    return ScalariztionFunctor<OpTy>().getScalarValue(def, vals, indices,
                                                      rewriter);
  }

  Value computeScalarValueForLoop(Operation *op, Value vals, ValueRange indices,
                                  PatternRewriter &rewriter) const {
    OpTy def = vals.getDefiningOp<OpTy>();
    return ScalariztionFunctor<OpTy>().getScalarValue(def, vals, indices,
                                                      rewriter);
  }
};
template <> struct ScalariztionFunctor<SplatOp> {
  template <typename T>
  Value getScalarValue(SplatOp def, Value vals, T indices,
                       PatternRewriter &rewriter) {

    return def.getSrc();
  }
};

template <>
bool TritonOpScalarizeInterface<SplatOp>::canComputeScalarValue(
    Operation *op, Value vals) const {
  return true;
}

template <>
struct TritonOpScalarizeInterface<MakeRangeOp>
    : public ScalarizeInterface::ExternalModel<
          TritonOpScalarizeInterface<MakeRangeOp>, MakeRangeOp> {

  bool canComputeScalarValue(Operation *op, Value vals) const { return true; }

  Value computeScalarValue(Operation *op, Value vals, ArrayRef<int64_t> indices,
                           PatternRewriter &rewriter) const {
    MakeRangeOp def = vals.getDefiningOp<MakeRangeOp>();
    int32_t start = static_cast<int32_t>(def.getStart());
    assert(indices.size() == 1);
    Type elemTy = cast<RankedTensorType>(def.getType()).getElementType();
    return rewriter.create<arith::ConstantOp>(
        def.getLoc(), elemTy,
        rewriter.getIntegerAttr(elemTy, start + indices[0]));
  }

  Value computeScalarValueForLoop(Operation *op, Value vals, ValueRange indices,
                                  PatternRewriter &rewriter) const {
    MakeRangeOp def = vals.getDefiningOp<MakeRangeOp>();
    assert(indices.size() == 1);
    int32_t start = static_cast<int32_t>(def.getStart());
    Type elemTy = cast<RankedTensorType>(def.getType()).getElementType();
    Value startVal = rewriter.create<arith::ConstantOp>(
        def.getLoc(), elemTy, rewriter.getIntegerAttr(elemTy, start));
    Value index = indices[0];
    if (!elemTy.isIndex())
      index =
          rewriter.create<arith::IndexCastUIOp>(def.getLoc(), elemTy, index);
    return rewriter.create<arith::AddIOp>(def.getLoc(), elemTy, startVal,
                                          index);
  }
};

template <> struct ScalariztionFunctor<BroadcastOp> {
  template <typename T>
  Value getScalarValue(BroadcastOp operation, Value vals, T indices,
                       PatternRewriter &rewriter) {
    BroadcastOp def = operation;
    using UnderlyingIndicesType = typename detail::value_type_trait<T>::type;
    // Find broadcasted dimensions and replace indices for those
    // dimensions with 0 (broadcasted dimension has always size 1).
    SmallVector<UnderlyingIndicesType> newIndices;
    auto sourceTy = cast<RankedTensorType>(def.getSrc().getType());
    auto targetTy = cast<RankedTensorType>(def.getType());
    assert(sourceTy.getRank() == indices.size() && "Mismatched rank");
    for (int64_t i = 0; i < sourceTy.getRank(); ++i) {
      if (sourceTy.getShape()[i] != targetTy.getShape()[i])
        newIndices.push_back(detail::createZeroIndex<UnderlyingIndicesType>(
            std::move(def.getLoc()), rewriter));
      else
        newIndices.push_back(indices[i]);
    }
    Value src = def.getSrc();
    return computeScalarValue(src.getDefiningOp(), src, newIndices, rewriter);
  }
};

template <> struct ScalariztionFunctor<ExpandDimsOp> {
  template <typename T>
  Value getScalarValue(ExpandDimsOp def, Value vals, T indices,
                       PatternRewriter &rewriter) {
    using UnderlyingIndicesType = typename detail::value_type_trait<T>::type;
    // Remove index at expanded dimension.
    SmallVector<UnderlyingIndicesType> newIndices(indices);
    newIndices.erase(newIndices.begin() + def.getAxis());
    Value src = def.getSrc();
    return computeScalarValue(src.getDefiningOp(), src, newIndices, rewriter);
  }
};

template <> struct ScalariztionFunctor<arith::ConstantOp> {
  template <typename T>
  Value getScalarValue(arith::ConstantOp def, Value vals, T indices,
                       PatternRewriter &rewriter) {
    auto denseVal = cast<DenseElementsAttr>(def.getValue());
    assert(denseVal.isSplat());
    auto scalarAttr = denseVal.getSplatValue<TypedAttr>();
    Value res = rewriter.create<arith::ConstantOp>(
        def.getLoc(), scalarAttr.getType(), scalarAttr);
    return res;
  }
};

template <>
bool TritonOpScalarizeInterface<arith::ConstantOp>::canComputeScalarValue(
    Operation *op, Value vals) const {
  auto cst = static_cast<arith::ConstantOp>(op);
  if (auto denseVal = dyn_cast<DenseElementsAttr>(cst.getValue())) {
    return denseVal.isSplat();
  }
  return false;
}

template <> struct ScalariztionFunctor<TransOp> {
  template <typename T>
  Value getScalarValue(TransOp def, Value vals, T indices,
                       PatternRewriter &rewriter) {

    using UnderlyingIndicesType = typename detail::value_type_trait<T>::type;

    // Permute indices.
    SmallVector<UnderlyingIndicesType> newIndices;
    auto order = def.getOrder();
    assert(indices.size() == order.size() && "Mismatched rank");
    for (auto idx : order)
      newIndices.push_back(indices[idx]);
    Value src = def.getSrc();
    return computeScalarValue(src.getDefiningOp(), src, newIndices, rewriter);
  }
};

} // namespace

template <typename OpType> static void registerOne(MLIRContext *ctx) {
  OpType::template attachInterface<TritonOpScalarizeInterface<OpType>>(*ctx);
}

template <typename... OpTypes> static void registerAll(MLIRContext *ctx) {
  (registerOne<OpTypes>(ctx), ...);
}

void mlir::triton::cpu::registerTritonOpScalarizeExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, TritonDialect *dialect) {
    registerAll<AddPtrOp, BroadcastOp, ExpandDimsOp, TransOp, SplatOp,
                MakeRangeOp>(ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    registerAll<arith::AddFOp, arith::AddIOp, arith::CmpFOp, arith::CmpIOp,
                arith::DivFOp, arith::DivSIOp, arith::MulIOp, arith::MulFOp,
                arith::RemFOp, arith::RemUIOp, arith::RemSIOp,
                arith::ConstantOp>(ctx);
  });
}
