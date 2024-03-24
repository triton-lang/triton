#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
// This file implements the dot slicing pass. It slices along the k dimension
// in a dot operation (for dot A, B with shape(A) = (m,k), shape(B) = (k,n)).
// The slice size along the k dimension is set by 'sliceKTile'.
// The algorithm includes three main steps:
//
// 1) Modify load instruction layouts for dot operands.
//    - This ensures tensor slices have the same layout as the original tensor,
//      allowing slicing without data exchange between threads.
//
// 2) Slice the dot operands.
//   - Here, the original dot operands are sliced, starting from the load
//     instruction. This helps in hiding latency in global memory transactions.
//     Currently, the algorithm assumes the original tensor undergoes only
//     elementwise operations before being used as a dot operand. In other
//     words, the algorithm expects only elementwise operations and convert
//     layout operations to occur between the load and dot instructions.
//     However, It can be extended to handle other operations if necessary.
//
// 3) Slice the dot operation along the k dimension.
//   - This involves calculating the number of slices from 'sliceKTile', and
//     then creating a new dot operation for each slice using the operands
//     sliced in the previous step. Sliced dots are concatenated so the result
//     of each dot is used in the next, leveraging the slicing along the k dim.
//===----------------------------------------------------------------------===//

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

bool isElementwiseOp(Operation *op) {
  if (llvm::isa<arith::AddFOp, arith::AddIOp, arith::AndIOp, arith::CeilDivSIOp,
                arith::CeilDivUIOp, arith::DivFOp, arith::DivSIOp,
                arith::DivUIOp, arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp,
                arith::FloorDivSIOp, arith::FPToSIOp, arith::FPToUIOp,
                arith::MaximumFOp, arith::MaxSIOp, arith::MaxUIOp,
                arith::MinimumFOp, arith::MinSIOp, arith::MinUIOp,
                arith::MulFOp, arith::MulIOp, arith::NegFOp, arith::OrIOp,
                arith::RemFOp, arith::RemSIOp, arith::RemUIOp, arith::ShLIOp,
                arith::ShRSIOp, arith::ShRUIOp, arith::SIToFPOp, arith::SubFOp,
                arith::SubIOp, arith::TruncFOp, arith::TruncIOp,
                arith::UIToFPOp, arith::XOrIOp>(op))
    return true;
  if (llvm::isa<math::AbsFOp, math::AbsIOp, math::AtanOp, math::Atan2Op,
                math::CeilOp, math::CopySignOp, math::CosOp, math::SinOp,
                math::CountLeadingZerosOp, math::CountTrailingZerosOp,
                math::CtPopOp, math::ErfOp, math::ExpOp, math::Exp2Op,
                math::ExpM1Op, math::FloorOp, math::FmaOp, math::LogOp,
                math::Log10Op, math::Log1pOp, math::Log2Op, math::PowFOp,
                math::RsqrtOp, math::SqrtOp, math::TanhOp>(op))
    return true;
  if (llvm::isa<tt::IntToPtrOp, tt::PtrToIntOp, tt::BitcastOp, tt::FpToFpOp,
                tt::AddPtrOp>(op))
    return true;
  if (auto externElementwiseOp = dyn_cast<tt::ExternElementwiseOp>(op))
    return externElementwiseOp.getPure();
  if (llvm::isa<arith::CmpIOp, arith::CmpFOp, arith::SelectOp>(op))
    return true;
  return false;
}
} // anonymous namespace

struct TritonAMDGPUDotSlicingPass
    : public TritonAMDGPUDotSlicingBase<TritonAMDGPUDotSlicingPass> {
  TritonAMDGPUDotSlicingPass() = default;

  TritonAMDGPUDotSlicingPass(int sliceKTile) { this->sliceKTile = sliceKTile; }

  // Find user of the currOp that affects dotOperand calculation.
  // We assume here that there is only one such user.
  Operation *getUserThatAffectsDotOperand(Operation *currOp,
                                          Operation *dotOperand) {
    SetVector<Operation *> forwardSlices;
    SmallVector<Operation *> usersThatAffectDot;
    for (auto *user : currOp->getUsers()) {
      forwardSlices.clear();
      getForwardSlice(user, &forwardSlices);

      if (user == dotOperand) {
        usersThatAffectDot.push_back(user);
        continue;
      }
      if (std::find(forwardSlices.begin(), forwardSlices.end(), dotOperand) !=
          forwardSlices.end()) {
        usersThatAffectDot.push_back(user);
      }
    }
    assert(usersThatAffectDot.size() == 1);
    return usersThatAffectDot[0];
  }

  Value getSlicedDotOperand(Operation *firstOpToSlice, tt::DotOp dotOp,
                            int operandIdx, int loopIter, int sliceSizeK,
                            OpBuilder builder,
                            SmallVector<Operation *> &eraseOps) {
    auto dotOperand = dotOp.getOperand(operandIdx);
    auto dotOperandTy = dotOp.getType().cast<RankedTensorType>();

    SmallVector<int64_t> sliceSizes;
    SmallVector<int64_t> sliceOffsets;
    SmallVector<int64_t> sliceStrides{1, 1};
    if (operandIdx == 0) {
      sliceSizes.push_back(dotOperandTy.getShape()[0]);
      sliceSizes.push_back(sliceSizeK);
      sliceOffsets.push_back(0);
      sliceOffsets.push_back(loopIter * sliceSizeK);
    } else {
      assert(operandIdx == 1);
      sliceSizes.push_back(sliceSizeK);
      sliceSizes.push_back(dotOperandTy.getShape()[1]);
      sliceOffsets.push_back(loopIter * sliceSizeK);
      sliceOffsets.push_back(0);
    }

    // Begin with the load instruction and proceed to slice the operations
    // along the execution path of the dotOperand.
    IRMapping mapping;
    ttg::ViewSliceOp viewPtr;
    for (auto i = 0; i < firstOpToSlice->getOperands().size(); i++) {
      auto arg = firstOpToSlice->getOperand(i);
      if (auto tensorType = arg.getType().dyn_cast<RankedTensorType>()) {
        auto slice = builder.create<ttg::ViewSliceOp>(
            dotOp.getLoc(),
            RankedTensorType::get(sliceSizes, tensorType.getElementType(),
                                  tensorType.getEncoding()),
            arg, ValueRange({}), ValueRange({}), ValueRange({}), sliceOffsets,
            sliceSizes, sliceStrides);
        mapping.map(arg, slice);
        if (i == 0)
          viewPtr = slice;
      }
    }

    Operation *currOp = firstOpToSlice;
    Operation *slicedOp = nullptr;
    while (true) {
      if (loopIter == 0) {
        eraseOps.push_back(currOp);
      }
      slicedOp = builder.clone(*currOp, mapping);

      // The 'load', 'convert_layout', and 'elementwise' operations each have
      // one result. This limitation can be removed if necessary.
      assert(currOp->getNumResults() == 1);
      // Convert the operation's results to sliced types.
      for (auto [currRes, slicedRes] :
           llvm::zip(currOp->getResults(), slicedOp->getResults())) {
        auto slicedType = RankedTensorType::get(
            viewPtr.getType().cast<RankedTensorType>().getShape(),
            currRes.getType().cast<RankedTensorType>().getElementType(),
            currRes.getType().cast<RankedTensorType>().getEncoding());
        slicedRes.setType(slicedType);
      }

      mapping.map(currOp, slicedOp);
      if (currOp == dotOperand.getDefiningOp()) {
        break;
      }
      assert(llvm::isa<tt::LoadOp>(currOp) ||
             llvm::isa<ttg::ConvertLayoutOp>(currOp) ||
             isElementwiseOp(currOp));

      // If currOp has more then one user, proceed with the one that is "on a
      // path" of dot operand calculation. We expect there is only one such
      // user.
      auto currOpUser =
          getUserThatAffectsDotOperand(currOp, dotOperand.getDefiningOp());

      // The currOpUser operation can have multiple operands, such as in any
      // binary elementwise op. In such cases, we slice all of the operands
      // using the same sliceOffsets, sliceSizes, and sliceStrides. This
      // approach is valid only under the assumption that currOpUser is an
      // elementwise operation. For non-elementwise operations with multiple
      // operands, slicing should potentially be handled differently.
      for (auto operandVal : currOpUser->getOperands()) {
        auto nonSlicedOperand = operandVal.getDefiningOp();
        if (nonSlicedOperand == currOp) {
          continue;
        }
        auto nonSlicedOperandTy = nonSlicedOperand->getResults()[0]
                                      .getType()
                                      .cast<RankedTensorType>();

        auto slicedTy = RankedTensorType::get(
            sliceSizes, nonSlicedOperandTy.getElementType(),
            nonSlicedOperandTy.getEncoding());

        auto slicedOperand = builder.create<ttg::ViewSliceOp>(
            nonSlicedOperand->getLoc(), slicedTy,
            nonSlicedOperand->getResults()[0], ValueRange({}), ValueRange({}),
            ValueRange({}), sliceOffsets, sliceSizes, sliceStrides);
        mapping.map(nonSlicedOperand->getResults()[0], slicedOperand);
      }

      currOp = currOpUser;
    }

    assert(llvm::isa<ttg::ConvertLayoutOp>(slicedOp));
    return slicedOp->getResults()[0];
  }

  static Type getNewType(Type type, Attribute encoding) {
    RankedTensorType tensorType = type.cast<RankedTensorType>();
    return RankedTensorType::get(tensorType.getShape(),
                                 tensorType.getElementType(), encoding);
  }

  // Same as coalesceOp function in Coalesce.cpp.
  void convertLayout(Attribute encoding, Operation *op) {
    OpBuilder builder(op);
    // Convert operands
    // For load/store with tensor pointers, we don't have to change the
    // operands' type, we do this by changing the outputs' type of
    // `make_tensor_ptr`
    SmallVector<Value, 4> newArgs;
    for (auto operand : op->getOperands()) {
      auto tensorType = operand.getType().dyn_cast<RankedTensorType>();
      if (tensorType &&
          !tensorType.getEncoding().isa<ttg::SharedEncodingAttr>()) {
        Type newType = getNewType(tensorType, encoding);
        newArgs.push_back(builder.create<ttg::ConvertLayoutOp>(
            op->getLoc(), newType, operand));
      } else {
        newArgs.push_back(operand);
      }
    }

    // Convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      bool isAsync = isa<ttg::InsertSliceAsyncOp>(op);
      newTypes.push_back(isAsync ? t : getNewType(t, encoding));
    }

    // Construct new op with the new encoding
    Operation *newOp =
        builder.create(op->getLoc(), op->getName().getIdentifier(), newArgs,
                       newTypes, op->getAttrs());

    // Cast the results back to the original layout
    for (size_t i = 0; i < op->getNumResults(); i++) {
      Value newResult = newOp->getResult(i);
      if (newTypes[i] != op->getResultTypes()[i]) {
        newResult = builder.create<ttg::ConvertLayoutOp>(
            op->getLoc(), op->getResult(i).getType(), newResult);
      }
      op->getResult(i).replaceAllUsesWith(newResult);
    }
    op->erase();
  }

  // Return true if layout was changed, else return false.
  bool setBlockedLayout(Operation *firstOpToSlice, ArrayRef<long> shape,
                        ttg::BlockedEncodingAttr blockedEncoding,
                        int operandIdx) {
    auto shapePerCTA = ttg::getShapePerCTATile(blockedEncoding, shape);
    auto sizePerThread = blockedEncoding.getSizePerThread();
    auto threadsPerWarp = blockedEncoding.getThreadsPerWarp();
    auto warpsPerCTA = blockedEncoding.getWarpsPerCTA();
    ModuleOp mod = getOperation();

    // clang-format off
    //
    // Current layout can be used for slicing as is.
    // Example: sliceKTile = 32, slicing along dim 1 (A operand)
    // Layout: #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]>
    //
    // clang-format on
    if (this->sliceKTile % shapePerCTA[1 - operandIdx] == 0) {
      return false;
      // clang-format off
      //
      // Current layout can be used for slicing only by setting warpsPerCTA to 1
      // along slicing dim.
      // Example: sliceKTile = 32, slicing along y dim (A operand)
      // Layout: #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [1, 0]>
      // NewLayout: #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]>
      //
      // clang-format on
    } else if (this->sliceKTile % (shapePerCTA[1 - operandIdx] /
                                   warpsPerCTA[1 - operandIdx]) ==
               0) {
      SmallVector<unsigned> newWarpsPerCTA(2, warpsPerCTA[0] * warpsPerCTA[1]);
      newWarpsPerCTA[1 - operandIdx] = 1;
      auto newBlockedEncoding = ttg::BlockedEncodingAttr::get(
          mod.getContext(), sizePerThread, threadsPerWarp, newWarpsPerCTA,
          blockedEncoding.getOrder(), blockedEncoding.getCTALayout());
      convertLayout(newBlockedEncoding, firstOpToSlice);
      // clang-format off
      //
      // Current layout can be used for slicing by setting warpsPerCTA to 1
      // along slicing dim and changing ThreadsPerWarp parameter.
      // Example: sliceKTile = 32, slicing along y dim (A operand)
      // Layout: #blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [1, 0]>
      // NewLayout: #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]>
      //
      // clang-format on
    } else if (this->sliceKTile % sizePerThread[operandIdx] == 0) {
      SmallVector<unsigned> newWarpsPerCTA(2, warpsPerCTA[0] * warpsPerCTA[1]);
      newWarpsPerCTA[1 - operandIdx] = 1;
      SmallVector<unsigned> newThreadsPerWarp(2, 1);
      newThreadsPerWarp[operandIdx] =
          (threadsPerWarp[0] * threadsPerWarp[1]) /
          (this->sliceKTile / sizePerThread[1 - operandIdx]);
      newThreadsPerWarp[1 - operandIdx] =
          this->sliceKTile / sizePerThread[1 - operandIdx];

      auto newBlockedEncoding = ttg::BlockedEncodingAttr::get(
          mod.getContext(), sizePerThread, newThreadsPerWarp, newWarpsPerCTA,
          blockedEncoding.getOrder(), blockedEncoding.getCTALayout());
      convertLayout(newBlockedEncoding, firstOpToSlice);
      // In other cases, the sizePerThread parameter would need to be changed,
      // which can affect coalescing and thus potentially decrease performance.
    } else {
      assert(false && "Unexpected layout in DotSlicing pass.");
    }
    return true;
  }

  // Return true if layout was changed, else return false.
  bool setLayoutForSlicing(Operation *firstOpToSlice, int operandIdx) {
    auto firstOpToSliceTy =
        firstOpToSlice->getOperand(0).getType().cast<RankedTensorType>();
    auto srcShape = firstOpToSliceTy.getShape();
    auto encoding = firstOpToSliceTy.getEncoding();

    if (auto blockedEncoding = dyn_cast<ttg::BlockedEncodingAttr>(encoding)) {
      return setBlockedLayout(firstOpToSlice, srcShape, blockedEncoding,
                              operandIdx);
    } else if (auto mfmaEncoding = dyn_cast<ttg::MfmaEncodingAttr>(encoding)) {
      auto shapePerCTA = ttg::getShapePerCTATile(mfmaEncoding, srcShape);
      // TODO: Implement changing of mfma layout in case it is not suitable for
      // slicing (similar as in setBlockedLayout).
      assert(this->sliceKTile % shapePerCTA[1] == 0);
    } else {
      assert(false && "Unsupported layout in setLayoutForSlicing.");
    }
    return false;
  }

  tt::LoadOp getLoadInst(tt::DotOp dotOp, int operandIdx) {
    auto dotOperand = dotOp.getOperand(operandIdx);
    SmallVector<tt::LoadOp> loadOpsVec;

    getOperation()->walk([&](tt::LoadOp loadOp) {
      SetVector<Operation *> forwardSlices;
      getForwardSlice((Operation *)loadOp, &forwardSlices);
      if (std::find(forwardSlices.begin(), forwardSlices.end(),
                    dotOperand.getDefiningOp()) != forwardSlices.end()) {
        loadOpsVec.push_back(loadOp);
      }
    });

    // Currently, we expect the dot operand to depend only on one tensor
    // from global memory (applicable for dot ops that don't depend on other dot
    // ops). This condition can be lifted if necessary.
    assert(loadOpsVec.size() == 1);
    return loadOpsVec[0];
  }

  bool dependsOnPreviousDot(tt::DotOp dotOp, int operandIdx) {
    SetVector<Operation *> bwdSlices;
    SmallVector<Operation *> filteredSlices;
    Operation *operand = dotOp.getOperand(operandIdx).getDefiningOp();
    // Seems like getBackwardSlice(dotOp, bwdSlices, filter) doesn't work
    // properly. Do it manually.
    getBackwardSlice(operand, &bwdSlices);
    std::copy_if(bwdSlices.begin(), bwdSlices.end(),
                 std::back_inserter(filteredSlices),
                 [](Operation *op) { return isa<tt::DotOp>(op); });

    if (filteredSlices.empty()) {
      return false;
    }
    return true;
  }

  bool shouldSliceDot(tt::DotOp dotOp) {
    auto dotOperand = dotOp.getOperand(0);
    auto dotATy = dotOperand.getType().cast<RankedTensorType>();
    auto kDim = dotATy.getShape()[1];

    if (this->sliceKTile == 0 || this->sliceKTile == kDim) {
      return false;
    }
    return true;
  }

  void dotSlicingDCE(ArrayRef<Operation *> eraseOps) {
    for (Operation *opToErase : llvm::reverse(eraseOps)) {
      assert(opToErase);
      bool hasUses = false;
      for (auto result : opToErase->getResults()) {
        if (!result.use_empty()) {
          hasUses = true;
        }
      }
      if (hasUses) {
        continue;
      }
      opToErase->erase();
    }
  }

  Operation *getFirstOpToSlice(tt::DotOp dotOp, int operandIdx) {
    if (dependsOnPreviousDot(dotOp, operandIdx)) {
      return dotOp.getOperand(operandIdx).getDefiningOp();
    }
    return getLoadInst(dotOp, operandIdx);
  }

  void runOnOperation() override {
    getOperation()->walk([&](tt::DotOp dotOp) {
      if (!shouldSliceDot(dotOp)) {
        return;
      }

      OpBuilder builder(dotOp);
      SmallVector<Operation *> eraseOps;

      auto dotResTy = dotOp.getType().cast<RankedTensorType>();
      auto dotOperand = dotOp.getOperand(0);
      auto dotATy = dotOperand.getType().cast<RankedTensorType>();
      auto dotAShape = dotATy.getShape();
      int64_t numSlices = dotAShape[1] / this->sliceKTile;
      Value slicedAcc = dotOp.getOperand(2);

      auto firstOpToSliceA = getFirstOpToSlice(dotOp, 0);
      auto firstOpToSliceB = getFirstOpToSlice(dotOp, 1);

      if (setLayoutForSlicing(firstOpToSliceA, /*operandIdx*/ 0)) {
        firstOpToSliceA = getFirstOpToSlice(dotOp, /*operandIdx*/ 0);
      }

      if (setLayoutForSlicing(firstOpToSliceB, /*operandIdx*/ 1)) {
        firstOpToSliceB = getFirstOpToSlice(dotOp, /*operandIdx*/ 1);
      }

      for (int i = 0; i < numSlices; i++) {
        auto slicedOperandA = getSlicedDotOperand(
            firstOpToSliceA, dotOp, 0, i, this->sliceKTile, builder, eraseOps);
        auto slicedOperandB = getSlicedDotOperand(
            firstOpToSliceB, dotOp, 1, i, this->sliceKTile, builder, eraseOps);

        auto slicedDot = builder.create<tt::DotOp>(
            dotOp.getLoc(), dotResTy, slicedOperandA, slicedOperandB, slicedAcc,
            dotOp.getAllowTF32(), dotOp.getMaxNumImpreciseAcc());
        slicedAcc = slicedDot;
      }

      eraseOps.push_back((Operation *)dotOp);
      dotOp.replaceAllUsesWith(slicedAcc);
      dotSlicingDCE(eraseOps);
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUDotSlicingPass(int sliceKTile) {
  return std::make_unique<TritonAMDGPUDotSlicingPass>(sliceKTile);
}
