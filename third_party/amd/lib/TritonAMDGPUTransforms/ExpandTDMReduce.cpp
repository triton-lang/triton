#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
namespace mlir {
#define GEN_PASS_DEF_TRITONAMDGPUEXPANDTDMREDUCE
#define GEN_PASS_DEF_TRITONAMDGPULOWERGLOBALREDUCE
#include "TritonAMDGPUTransforms/Passes.h.inc"
} // namespace mlir

namespace {
SmallVector<Type> descriptorFieldTypes(TensorDescType type) {
  auto *ctx = type.getContext();
  unsigned rank = type.getShape().size();
  SmallVector<Type> types{PointerType::get(
      type.getSignlessBlockType().getElementType(), PtrAddrSpace::Global)};
  types.append(rank, IntegerType::get(ctx, 32));
  types.append(rank, IntegerType::get(ctx, 64));
  types.push_back(IntegerType::get(ctx, 32));
  return types;
}

struct ExtractedDescriptorFields {
  Value base;
  ValueRange bounds;
  SmallVector<Value> strides;
  Value predicate;
};

ExtractedDescriptorFields extract(OpBuilder &b, Location loc, Value desc) {
  auto op = amdgpu::ExtractTensorDescriptorFieldsOp::create(
      b, loc, descriptorFieldTypes(cast<TensorDescType>(desc.getType())), desc);
  unsigned rank = cast<TensorDescType>(desc.getType()).getShape().size();
  auto fields = op.getResults();
  return {fields.front(), fields.slice(1, rank),
          llvm::to_vector_of<Value>(fields.slice(1 + rank, rank)),
          fields.back()};
}

std::optional<RMWOp> reductionKind(DescriptorReduceOp op) {
  Type type = op.getDesc().getType().getElementType();
  switch (op.getKind()) {
  case DescriptorReduceKind::ADD:
    return type.isInteger() ? RMWOp::ADD : RMWOp::FADD;
  case DescriptorReduceKind::MIN:
    if (type.isUnsignedInteger())
      return RMWOp::UMIN;
    if (type.isSignedInteger())
      return RMWOp::MIN;
    return {};
  case DescriptorReduceKind::MAX:
    if (type.isUnsignedInteger())
      return RMWOp::UMAX;
    if (type.isSignedInteger())
      return RMWOp::MAX;
    return {};
  case DescriptorReduceKind::AND:
    return RMWOp::AND;
  case DescriptorReduceKind::OR:
    return RMWOp::OR;
  case DescriptorReduceKind::XOR:
    return RMWOp::XOR;
  default:
    return {};
  }
}

struct ExpandTDMReduce
    : mlir::impl::TritonAMDGPUExpandTDMReduceBase<ExpandTDMReduce> {
  void runOnOperation() override {
    SmallVector<DescriptorReduceOp> reductions;
    getOperation().walk(
        [&](DescriptorReduceOp op) { reductions.push_back(op); });
    for (auto op : reductions) {
      auto kind = reductionKind(op);
      if (!kind) {
        op.emitError("unsupported descriptor reduction element type");
        return signalPassFailure();
      }
      OpBuilder b(op);
      auto loc = op.getLoc();
      auto srcType = op.getSrc().getType();
      auto shape = srcType.getShape();
      unsigned rank = shape.size();
      auto encoding = srcType.getEncoding();
      auto [base, bounds, strides, descriptorPredicate] =
          extract(b, loc, op.getDesc());
      auto ptrType = srcType.clone(base.getType());
      auto maskType = srcType.clone(b.getI1Type());
      Value ptr = SplatOp::create(b, loc, ptrType, base);
      Value zero32 = arith::ConstantIntOp::create(b, loc, 0, 32);
      Value pred = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                         descriptorPredicate, zero32);
      // Native TDM treats a negative starting offset as a fully OOB tile.
      for (Value offset : op.getIndices()) {
        Value nonNegative = arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::sge, offset, zero32);
        pred = arith::AndIOp::create(b, loc, pred, nonNegative);
      }
      Value mask = SplatOp::create(b, loc, maskType, pred);
      auto offsetType = srcType.clone(b.getI64Type());
      Value elementOffsetSum = SplatOp::create(
          b, loc, offsetType, arith::ConstantIntOp::create(b, loc, 0, 64));
      // TDM descriptors do not encode the innermost stride because it is
      // always 1.
      strides.back() = arith::ConstantIntOp::create(b, loc, 1, 64);
      for (unsigned dim = 0; dim < rank; ++dim) {
        // Slice the source layout down to this dimension. Remove dimensions
        // in descending order so their indices remain valid. The ExpandDims
        // operations below restore the original dimensions.
        Attribute axisEncoding = encoding;
        for (int other = rank - 1; other >= 0; --other)
          if (unsigned(other) != dim)
            axisEncoding = SliceEncodingAttr::get(
                b.getContext(), other,
                cast<DistributedEncodingTrait>(axisEncoding));
        auto axisType =
            RankedTensorType::get({shape[dim]}, b.getI32Type(), axisEncoding);
        Value range = MakeRangeOp::create(b, loc, axisType, 0, shape[dim]);
        auto axis64Type = axisType.clone(b.getI64Type());
        Value coordinate = arith::ExtSIOp::create(b, loc, axis64Type, range);
        Value offset = arith::ExtSIOp::create(b, loc, b.getI64Type(),
                                              op.getIndices()[dim]);
        coordinate = arith::AddIOp::create(
            b, loc, coordinate, SplatOp::create(b, loc, axis64Type, offset));
        Value bound =
            arith::ExtUIOp::create(b, loc, b.getI64Type(), bounds[dim]);
        Value valid =
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, coordinate,
                                  SplatOp::create(b, loc, axis64Type, bound));
        Value stride = SplatOp::create(b, loc, axis64Type, strides[dim]);
        Value elementOffset = arith::MulIOp::create(b, loc, coordinate, stride);
        for (unsigned other = 0; other < rank; ++other) {
          if (other == dim)
            continue;
          elementOffset = ExpandDimsOp::create(b, loc, elementOffset, other);
          valid = ExpandDimsOp::create(b, loc, valid, other);
        }
        elementOffset = BroadcastOp::create(b, loc, offsetType, elementOffset);
        valid = BroadcastOp::create(b, loc, maskType, valid);
        elementOffsetSum =
            arith::AddIOp::create(b, loc, elementOffsetSum, elementOffset);
        mask = arith::AndIOp::create(b, loc, mask, valid);
      }
      ptr = AddPtrOp::create(b, loc, ptrType, ptr, elementOffsetSum);
      amdgpu::GlobalReduceOp::create(b, loc, *kind, ptr, op.getSrc(), mask);
      op.erase();
    }
  }
};

struct LowerGlobalReduce
    : mlir::impl::TritonAMDGPULowerGlobalReduceBase<LowerGlobalReduce> {
  void runOnOperation() override {
    SmallVector<Operation *> operations;
    bool hasReduction = false;
    getOperation().walk([&](Operation *op) {
      bool isReduction = isa<amdgpu::GlobalReduceOp>(op);
      hasReduction |= isReduction;
      if (isReduction ||
          isa<WarpSpecializeOp, WarpYieldOp, WarpReturnOp, CallOpInterface>(op))
        operations.push_back(op);
    });
    if (!hasReduction)
      return;
    for (Operation *op : operations) {
      OpBuilder b(op);
      auto loc = op->getLoc();
      // Drain tensor work in the waves that issued it, before a reduction,
      // call, or change in execution-region participants.
      amdgpu::AsyncTDMIntrinsicWait::create(b, loc, ValueRange{}, 0);
      BarrierOp::create(b, loc, AddrSpace::None);
      auto reduce = dyn_cast<amdgpu::GlobalReduceOp>(op);
      if (!reduce)
        continue;
      AtomicRMWOp::create(b, loc, reduce.getVal().getType(),
                          reduce.getAtomicRmwOp(), reduce.getPtr(),
                          reduce.getVal(), reduce.getMask(),
                          MemSemantic::RELEASE, MemSyncScope::GPU);
      // Complete the reduction before any following tensor access.
      BarrierOp::create(b, loc, AddrSpace::GlobalWrite);
      reduce.erase();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUExpandTDMReduce() {
  return std::make_unique<ExpandTDMReduce>();
}
std::unique_ptr<Pass> mlir::createTritonAMDGPULowerGlobalReduce() {
  return std::make_unique<LowerGlobalReduce>();
}
