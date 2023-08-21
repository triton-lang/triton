/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
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

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include <queue>

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

// TODO: use ConvertLayoutOp
using CastOp = ::mlir::UnrealizedConversionCastOp;

unsigned getNumUsers(Value value) {
  return std::distance(value.user_begin(), value.user_end());
}

Type replaceLayout(const Type &type, const Attribute &newLayout) {
  Type curType = type;
  auto ptrTy = curType.dyn_cast<triton::PointerType>();
  if (ptrTy)
    curType = ptrTy.getPointeeType();
  if (auto tensorTy = curType.dyn_cast<RankedTensorType>())
    curType = RankedTensorType::get(tensorTy.getShape(),
                                    tensorTy.getElementType(), newLayout);
  if (ptrTy)
    curType = triton::PointerType::get(curType, ptrTy.getAddressSpace());
  return curType;
}

Attribute replaceCTALayout(Attribute layout, llvm::ArrayRef<int64_t> shape,
                           const ttg::CTALayoutAttr &newCTALayout) {
  if (auto blockedLayout = layout.dyn_cast<ttg::BlockedEncodingAttr>()) {
    return ttg::BlockedEncodingAttr::get(
        layout.getContext(), shape, blockedLayout.getSizePerThread(),
        blockedLayout.getOrder(), ttg::getNumWarpsPerCTA(layout), 32,
        newCTALayout);
  } else if (auto sliceLayout = layout.dyn_cast<ttg::SliceEncodingAttr>()) {
    return ttg::SliceEncodingAttr::get(
        layout.getContext(), sliceLayout.getDim(),
        replaceCTALayout(sliceLayout.getParent(), shape, newCTALayout));
  } else {
    // Other layouts are generated by passes after PlanCTAPass
    assert(0 && "replaceCTALayout not implemented");
  }
}

class CTAPlanner {
public:
  CTAPlanner(ttng::ClusterInfo *clusterInfo_);
  ~CTAPlanner();

  void run(triton::FuncOp &funcOp);

private:
  CastOp markBackward(CastOp cast) const;
  CastOp markForward(CastOp cast) const;
  bool isBackward(CastOp cast) const;
  bool isForward(CastOp cast) const;

  void setTiling(llvm::ArrayRef<unsigned> CTAsPerCGA);
  bool processDot(triton::FuncOp &funcOp);
  bool processReduce(triton::FuncOp &funcOp);
  void processStoreLikeOps(triton::FuncOp &funcOp);

  bool propagate(CastOp cast);
  bool propagateBackward(CastOp cast);
  bool propagateForward(CastOp cast);

  void eraseCastOp(CastOp cast);
  void eraseCastOpFromQueue(CastOp cast);
  void eraseCastOpsFromQueue(llvm::ArrayRef<CastOp> casts);

  void insertCasts(Operation *op, llvm::ArrayRef<Attribute> newOperandLayouts,
                   llvm::ArrayRef<Attribute> newResultLayouts);
  void eliminateAdjacentCasts(CastOp cast0, CastOp cast1);

  bool isLoadStoreOp(Operation *op) const;
  bool processLoadStore(Operation *op, Attribute layout);

  bool isElementwiseOp(Operation *op) const;
  bool processElementwise(Operation *op, Attribute layout);

  bool processConstant(arith::ConstantOp constant, Attribute layout);
  bool processSplat(triton::SplatOp splat, Attribute layout);
  bool processMakeRange(triton::MakeRangeOp makeRange, Attribute layout);
  bool processMakeTensorPtr(triton::MakeTensorPtrOp makeTensorPtr,
                            Attribute layout);

  bool processBroadcast(triton::BroadcastOp broadcast, Attribute layout);
  bool processExpandDimsBackward(triton::ExpandDimsOp expandDims,
                                 Attribute newResultLayout);
  bool processExpandDimsForward(triton::ExpandDimsOp expandDims,
                                Attribute newSrcLayout);

  bool processConvertLayoutBackward(ttg::ConvertLayoutOp convertLayout,
                                    CastOp cast);
  bool processConvertLayoutForward(ttg::ConvertLayoutOp convertLayout,
                                   CastOp cast);

  bool processIfOp(scf::IfOp ifOp, int index, const Type &newType);
  bool processForOp(scf::ForOp forOp, int index, const Type &newType);

  bool processIfOpBackward(scf::IfOp ifOp, CastOp cast);
  bool processForOpBackward(scf::ForOp forOp, CastOp cast);
  bool processBlockArgBackward(BlockArgument arg, CastOp cast);
  bool processForOpForward(scf::ForOp forOp, CastOp cast);
  bool processYieldOpForward(scf::YieldOp yieldOp, CastOp cast);

  bool processOpFallback(Operation *op);

  bool processMultiUsersBackward(Value input, CastOp cast);
  bool processMultiUsersForward(Value output, CastOp cast);

  // This flag indicates whether clusterInfo needs to be deleted in the
  // destructor of CTAPlanner. The flag `ownInfo` is set to false when a
  // non-null pointer to clusterInfo is passed to the constructor of CTAPlanner.
  // Otherwise, a self-managed ClusterInfo will be created and the ownInfo will
  // be set to true.
  bool ownInfo;
  ttng::ClusterInfo *clusterInfo;
  bool tiled;
  unsigned step;
  unsigned stepUnchanged;
  std::queue<CastOp> queue;
};

CTAPlanner::CTAPlanner(ttng::ClusterInfo *clusterInfo_)
    : ownInfo(false), clusterInfo(clusterInfo_), tiled(false), step(0),
      stepUnchanged(0) {
  if (clusterInfo == nullptr) {
    clusterInfo = new ttng::ClusterInfo();
    ownInfo = true;
  }
}

CTAPlanner::~CTAPlanner() {
  if (ownInfo) {
    delete clusterInfo;
    // Actually not necessary but safer
    ownInfo = false;
    clusterInfo = nullptr;
  }
}

void CTAPlanner::run(triton::FuncOp &funcOp) {
  assert(!tiled && "Please create a new CTAPlanner");
  static const unsigned maxSteps = 10000;

  auto nextStep = [&]() {
    ++step;
    assert(step < maxSteps && "Maximum number of steps exceeded");
  };

  processDot(funcOp);
  nextStep();

  processReduce(funcOp);
  nextStep();

  if (!tiled) {
    processStoreLikeOps(funcOp);
    nextStep();
  }

  while (!queue.empty()) {
    CastOp cast = queue.front();
    queue.pop();
    bool changed = propagate(cast);
    if (changed) {
      stepUnchanged = 0;
    } else {
      queue.push(cast);
      ++stepUnchanged;
    }
    nextStep();
  }
}

CastOp CTAPlanner::markBackward(CastOp cast) const {
  cast->setAttr("direction", StringAttr::get(cast.getContext(), "backward"));
  return cast;
}

CastOp CTAPlanner::markForward(CastOp cast) const {
  cast->setAttr("direction", StringAttr::get(cast.getContext(), "forward"));
  return cast;
}

bool CTAPlanner::isBackward(CastOp cast) const {
  return cast->getAttrOfType<StringAttr>("direction") == "backward";
}

bool CTAPlanner::isForward(CastOp cast) const {
  return cast->getAttrOfType<StringAttr>("direction") == "forward";
}

void CTAPlanner::setTiling(llvm::ArrayRef<unsigned> CTAsPerCGA) {
  assert(!tiled && "CTA tiling is already determinted");
  assert(clusterInfo && "ClusterInfo pointer is null");
  assert(CTAsPerCGA.size() <= 3 && "setTiling not implemented");
  if (CTAsPerCGA.size() > 0)
    clusterInfo->clusterDimX = CTAsPerCGA[0];
  if (CTAsPerCGA.size() > 1)
    clusterInfo->clusterDimY = CTAsPerCGA[1];
  if (CTAsPerCGA.size() > 2)
    clusterInfo->clusterDimZ = CTAsPerCGA[2];
  tiled = true;
}

bool CTAPlanner::processDot(triton::FuncOp &funcOp) {
  // TODO: This is a naive implementation and should be refactored
  auto getCTATiling = [](int64_t M, int64_t N, int64_t K,
                         unsigned numCTAs) -> std::pair<unsigned, unsigned> {
    unsigned splitM = std::clamp<unsigned>(M / 64, 1, numCTAs);
    unsigned splitN = numCTAs / splitM;
    return {splitM, splitN};
  };

  funcOp.walk([&](triton::DotOp dot) {
    MLIRContext *ctx = dot.getContext();

    auto aTy = dot.getA().getType().cast<RankedTensorType>();
    auto bTy = dot.getB().getType().cast<RankedTensorType>();
    auto dTy = dot.getD().getType().cast<RankedTensorType>();

    assert(aTy.getEncoding().isa<ttg::DotOperandEncodingAttr>() &&
           bTy.getEncoding().isa<ttg::DotOperandEncodingAttr>() &&
           dTy.getEncoding().isa<ttg::BlockedEncodingAttr>() &&
           "PlanCTAPass should follow immediately after CoalescePass");

    auto aLayout = aTy.getEncoding().cast<ttg::DotOperandEncodingAttr>();
    auto bLayout = bTy.getEncoding().cast<ttg::DotOperandEncodingAttr>();
    auto dLayout = dTy.getEncoding().cast<ttg::BlockedEncodingAttr>();

    unsigned M = dTy.getShape()[0];
    unsigned N = dTy.getShape()[1];
    unsigned K = aTy.getShape()[1];

    unsigned splitM, splitN;
    std::tie(splitM, splitN) = getCTATiling(M, N, K, ttg::getNumCTAs(dLayout));
    // FIXME: Should consider IR with more than one DotOps
    setTiling({splitM, splitN, 1});

    auto newCTALayout = ttg::CTALayoutAttr::get(ctx, {splitM, splitN},
                                                {splitM, splitN}, {1, 0});
    auto newDLayout = ttg::BlockedEncodingAttr::get(
        ctx, dTy.getShape(), dLayout.getSizePerThread(), dLayout.getOrder(),
        ttg::getNumWarpsPerCTA(dLayout), 32, newCTALayout);
    auto newALayout = ttg::DotOperandEncodingAttr::get(ctx, aLayout.getOpIdx(),
                                                       newDLayout, 0);
    auto newBLayout = ttg::DotOperandEncodingAttr::get(ctx, bLayout.getOpIdx(),
                                                       newDLayout, 0);

    insertCasts(dot.getOperation(), {newALayout, newBLayout, newDLayout},
                {newDLayout});
  });

  return true;
}

bool CTAPlanner::processReduce(triton::FuncOp &funcOp) {
  ModuleOp mod = funcOp->getParentOfType<ModuleOp>();
  unsigned numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);

  funcOp.walk([&](triton::ReduceOp reduce) {
    MLIRContext *context = reduce.getContext();
    Value src = reduce.getOperands()[0];
    unsigned axis = reduce.getAxis();

    auto srcTy = src.getType().cast<RankedTensorType>();
    auto srcShape = srcTy.getShape();
    auto srcLayout = srcTy.getEncoding();

    auto rank = srcShape.size();
    auto order = ttg::getOrder(srcLayout);
    auto sizePerThread =
        ttg::getSizePerThread(srcLayout, ttg::getShapePerCTA(srcTy));
    auto CTAOrder = ttg::getCTAOrder(srcLayout);

    llvm::SmallVector<unsigned> CTAsPerCGA(rank, 0);
    unsigned remainingCTAs = numCTAs;
    for (int i = rank - 1; i >= 0; --i) {
      unsigned dim = order[i];
      if (dim == axis) {
        CTAsPerCGA[dim] = 1;
      } else {
        CTAsPerCGA[dim] = std::min<unsigned>(srcShape[dim] / sizePerThread[dim],
                                             remainingCTAs);
        remainingCTAs /= CTAsPerCGA[dim];
      }
    }

    for (int i = rank - 1; i >= 0; --i) {
      unsigned dim = order[i];
      if (dim != axis) {
        CTAsPerCGA[dim] *= remainingCTAs;
        break;
      }
    }

    llvm::SmallVector<unsigned> CTASplitNum = CTAsPerCGA;

    // If numCTAs > 1 and the only dimension is the reduced dimension, after the
    // above two for-loops, CTAsPerCGA = [0] and remainingCTAs = numCTAs. We set
    // CTAsPerCGA[0] = numCTAs and keep CTASplitNum[0] = 1 to ensure that no
    // cross-CTA reduction is required, although this will introduce duplicated
    // calculation
    if (remainingCTAs > 0)
      CTAsPerCGA[order[rank - 1]] *= remainingCTAs;

    auto CTALayout =
        ttg::CTALayoutAttr::get(context, CTAsPerCGA, CTASplitNum, CTAOrder);
    if (!tiled)
      setTiling(CTALayout.getCTAsPerCGA());
    auto newSrcLayout = replaceCTALayout(srcLayout, srcShape, CTALayout);
    auto newResultLayout =
        ttg::SliceEncodingAttr::get(context, axis, newSrcLayout);
    unsigned numOperands = reduce.getNumOperands();
    SmallVector<Attribute> newSrcLayoutVec(numOperands, newSrcLayout);
    SmallVector<Attribute> newResultLayoutVec(numOperands, newResultLayout);

    insertCasts(reduce.getOperation(), newSrcLayoutVec, newResultLayoutVec);
  });
  return true;
}

void CTAPlanner::processStoreLikeOps(triton::FuncOp &funcOp) {
  assert(!tiled && "CTA tiling is already determinted");

  llvm::SmallVector<Operation *> stores;
  funcOp.walk([&](Operation *op) {
    if (llvm::isa<triton::StoreOp, triton::AtomicRMWOp, triton::AtomicCASOp>(
            op))
      stores.push_back(op);
  });
  assert(stores.size() > 0 && "Cannot find store-like ops");

  ttg::CTALayoutAttr CTALayout;
  for (Operation *store : stores) {
    if (auto tensorTy =
            store->getOperand(0).getType().dyn_cast<RankedTensorType>()) {
      if (!tiled) {
        // Use CTA tiling of the first store-like op as global CTA tiling
        CTALayout = ttg::getCTALayout(tensorTy.getEncoding());
        setTiling(CTALayout.getCTAsPerCGA());
      }
      auto newLayout = replaceCTALayout(tensorTy.getEncoding(),
                                        tensorTy.getShape(), CTALayout);
      processElementwise(store, newLayout);
    }
  }

  // If all store-like ops are processing scalar values and no ReduceOp is
  // found, we can conclude that this is an all-scalar computation, since
  // ReduceOp is the only op that converts tensor values to scalar values.
  if (!tiled)
    setTiling({1, 1, 1});
}

bool CTAPlanner::propagate(CastOp cast) {
  return isBackward(cast) ? propagateBackward(cast) : propagateForward(cast);
}

bool CTAPlanner::propagateBackward(CastOp cast) {
  Value input = cast.getOperand(0);
  Value output = cast.getResult(0);
  unsigned numUsers = getNumUsers(input);
  if (numUsers == 0) {
    assert(0 && "Unreachable branch");
  } else if (numUsers == 1) {
    Type outTy = output.getType();
    if (auto ptrTy = outTy.dyn_cast<triton::PointerType>())
      outTy = ptrTy.getPointeeType();
    Attribute layout = outTy.cast<RankedTensorType>().getEncoding();
    Operation *op = input.getDefiningOp();
    if (op == nullptr) {
      assert(input.isa<BlockArgument>() &&
             "Unexpected Value without defining op");
      processBlockArgBackward(input.cast<BlockArgument>(), cast);
    } else if (auto prevCast = llvm::dyn_cast<CastOp>(op)) {
      eliminateAdjacentCasts(prevCast, cast);
    } else if (isLoadStoreOp(op)) {
      processLoadStore(op, layout);
    } else if (isElementwiseOp(op)) {
      processElementwise(op, layout);
    } else if (auto constant = llvm::dyn_cast<arith::ConstantOp>(op)) {
      processConstant(constant, layout);
    } else if (auto splat = llvm::dyn_cast<triton::SplatOp>(op)) {
      processSplat(splat, layout);
    } else if (auto makeRange = llvm::dyn_cast<triton::MakeRangeOp>(op)) {
      processMakeRange(makeRange, layout);
    } else if (auto makeTensorPtr =
                   llvm::dyn_cast<triton::MakeTensorPtrOp>(op)) {
      processMakeTensorPtr(makeTensorPtr, layout);
    } else if (llvm::isa<triton::AdvanceOp>(op)) {
      // ptr operand and result have the same layout, while other operands are
      // scalar values
      processElementwise(op, layout);
    } else if (auto broadcast = llvm::dyn_cast<triton::BroadcastOp>(op)) {
      processBroadcast(broadcast, layout);
    } else if (auto expandDims = llvm::dyn_cast<triton::ExpandDimsOp>(op)) {
      processExpandDimsBackward(expandDims, layout);
    } else if (auto ifOp = llvm::dyn_cast<scf::IfOp>(op)) {
      processIfOpBackward(ifOp, cast);
    } else if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
      processForOpBackward(forOp, cast);
    } else if (auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(op)) {
      return processConvertLayoutBackward(convertLayout, cast);
    } else {
      // Keep original layouts. This may result in a loss of performance.
      return processOpFallback(op);
    }
    return true;
  } else {
    return processMultiUsersBackward(input, cast);
  }
}

bool CTAPlanner::propagateForward(CastOp cast) {
  Value input = cast.getOperand(0);
  Value output = cast.getResult(0);
  unsigned numUsers = getNumUsers(output);
  if (numUsers == 0) {
    cast.erase();
  } else if (numUsers == 1) {
    Type inTy = input.getType();
    if (auto ptrTy = inTy.dyn_cast<triton::PointerType>())
      inTy = ptrTy.getPointeeType();
    Attribute layout = inTy.cast<RankedTensorType>().getEncoding();
    Operation *op = *output.user_begin();
    if (auto nextCast = llvm::dyn_cast<CastOp>(op)) {
      eliminateAdjacentCasts(cast, nextCast);
    } else if (isLoadStoreOp(op)) {
      processLoadStore(op, layout);
    } else if (isElementwiseOp(op)) {
      processElementwise(op, layout);
    } else if (llvm::isa<triton::AdvanceOp>(op)) {
      // ptr operand and result have the same layout, while other operands are
      // scalar values
      processElementwise(op, layout);
    } else if (auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(op)) {
      return processConvertLayoutForward(convertLayout, cast);
    } else if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
      processForOpForward(forOp, cast);
    } else if (auto yieldOp = llvm::dyn_cast<scf::YieldOp>(op)) {
      processYieldOpForward(yieldOp, cast);
    } else {
      // Keep original layouts. This may result in a loss of performance.
      return processOpFallback(op);
    }
  } else {
    processMultiUsersForward(output, cast);
  }
  return true;
}

void CTAPlanner::eraseCastOp(CastOp cast) {
  Value output = cast.getResult(0);
  assert(getNumUsers(output) == 0 &&
         "Cannot erase CastOp because it is still in use");
  cast.erase();
}

void CTAPlanner::eraseCastOpFromQueue(CastOp cast) {
  eraseCastOpsFromQueue({cast});
}

void CTAPlanner::eraseCastOpsFromQueue(llvm::ArrayRef<CastOp> casts) {
  llvm::DenseSet<CastOp> erased;
  for (CastOp cast : casts) {
    eraseCastOp(cast);
    erased.insert(cast);
  }

  decltype(queue) tempQueue;
  std::swap(queue, tempQueue);

  // This is only a naive implementation. Should refactor with linked-list.
  while (!tempQueue.empty()) {
    auto cast = tempQueue.front();
    tempQueue.pop();
    if (!erased.contains(cast))
      queue.push(cast);
  }
}

void CTAPlanner::insertCasts(Operation *op,
                             llvm::ArrayRef<Attribute> newOperandLayouts,
                             llvm::ArrayRef<Attribute> newResultLayouts) {
  assert(op->getNumOperands() == newOperandLayouts.size() &&
         "NumOperands mismatched");
  assert(op->getNumResults() == newResultLayouts.size() &&
         "NumResults mismatched");

  Location loc = op->getLoc();
  OpBuilder builder(op->getContext());

  builder.setInsertionPoint(op);
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    Value operand = op->getOperand(i);
    auto operandTy = operand.getType();
    if (triton::isTensorOrTensorPointerType(operandTy)) {
      operandTy = replaceLayout(operandTy, newOperandLayouts[i]);
      auto cast = markBackward(builder.create<CastOp>(loc, operandTy, operand));
      op->setOperand(i, cast.getResult(0));
      queue.push(cast);
    }
  }

  builder.setInsertionPointAfter(op);
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    Value result = op->getResult(i);
    auto resultTy = result.getType();
    if (triton::isTensorOrTensorPointerType(resultTy)) {
      resultTy = replaceLayout(resultTy, newResultLayouts[i]);
      auto cast =
          markForward(builder.create<CastOp>(loc, result.getType(), result));
      result.setType(resultTy);
      result.replaceAllUsesExcept(cast.getResult(0), cast.getOperation());
      queue.push(cast);
    }
  }
}

void CTAPlanner::eliminateAdjacentCasts(CastOp cast0, CastOp cast1) {
  assert(cast0.getResult(0) == cast1.getOperand(0) &&
         "The two casts are not adjacent");
  assert(isForward(cast0) && isBackward(cast1) &&
         "Expected pattern of adjacent casts: forward + backward");

  Value input = cast0.getOperand(0);
  Value output = cast1.getResult(0);

  if (input.getType() == output.getType()) {
    output.replaceAllUsesWith(input);
    eraseCastOpsFromQueue({cast1, cast0});
  } else {
    OpBuilder builder(cast1.getOperation());
    auto cvt = builder.create<ttg::ConvertLayoutOp>(cast1.getLoc(),
                                                    output.getType(), input);
    output.replaceAllUsesWith(cvt.getResult());
    eraseCastOpsFromQueue({cast1, cast0});
  }
}

bool CTAPlanner::isLoadStoreOp(Operation *op) const {
  return llvm::isa<triton::LoadOp, triton::StoreOp, triton::AtomicRMWOp,
                   triton::AtomicCASOp>(op);
}

bool CTAPlanner::processLoadStore(Operation *op, Attribute layout) {
  // Special logic for:
  //     LoadOp -> SliceLayout
  // Transform to:
  //     LoadOp -> originalLayout -> ConvertLayout(DSmem) -> SliceLayout
  if (auto sliceLayout = layout.dyn_cast<ttg::SliceEncodingAttr>()) {
    auto dim = sliceLayout.getDim();
    auto CTAsPerCGA = ttg::getCTAsPerCGA(sliceLayout.getParent());
    if (CTAsPerCGA[dim] > 1) {
      // Find an input or output value of LoadOp or StoreOp to get its layout
      Value val =
          op->getNumResults() > 0 ? op->getResult(0) : op->getOperand(0);
      Attribute originalLayout =
          val.getType().cast<RankedTensorType>().getEncoding();
      // Insert casts using originalLayout. Adjacent casts will be eliminated
      // and generate a ConvertLayoutOp with DSmem access
      return processLoadStore(op, originalLayout);
    }
  }

  auto CTALayout = ttg::getCTALayout(layout);

  llvm::SmallVector<Attribute> newOperandLayouts;
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    auto type = op->getOperand(i).getType();
    if (auto ptrTy = type.dyn_cast<triton::PointerType>())
      type = ptrTy.getPointeeType();
    auto tensorTy = type.cast<RankedTensorType>();
    auto newLayout = replaceCTALayout(tensorTy.getEncoding(),
                                      tensorTy.getShape(), CTALayout);
    newOperandLayouts.push_back(newLayout);
  }

  llvm::SmallVector<Attribute> newResultLayouts;
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    auto type = op->getResult(i).getType();
    if (auto ptrTy = type.dyn_cast<triton::PointerType>())
      type = ptrTy.getPointeeType();
    auto tensorTy = type.cast<RankedTensorType>();
    auto newLayout = replaceCTALayout(tensorTy.getEncoding(),
                                      tensorTy.getShape(), CTALayout);
    newResultLayouts.push_back(newLayout);
  }

  insertCasts(op, newOperandLayouts, newResultLayouts);
  return true;
}

bool CTAPlanner::isElementwiseOp(Operation *op) const {
  if (llvm::isa<arith::AddFOp, arith::AddIOp, arith::AndIOp, arith::CeilDivSIOp,
                arith::CeilDivUIOp, arith::DivFOp, arith::DivSIOp,
                arith::DivUIOp, arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp,
                arith::FloorDivSIOp, arith::FPToSIOp, arith::FPToUIOp,
                arith::MaxFOp, arith::MaxSIOp, arith::MaxUIOp, arith::MinFOp,
                arith::MinSIOp, arith::MinUIOp, arith::MulFOp, arith::MulIOp,
                arith::NegFOp, arith::OrIOp, arith::RemFOp, arith::RemSIOp,
                arith::RemUIOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp,
                arith::SIToFPOp, arith::SubFOp, arith::SubIOp, arith::TruncFOp,
                arith::TruncIOp, arith::UIToFPOp, arith::XOrIOp>(op))
    return true;
  if (llvm::isa<math::AbsFOp, math::AbsIOp, math::AtanOp, math::Atan2Op,
                math::CeilOp, math::CopySignOp, math::CosOp, math::SinOp,
                math::CountLeadingZerosOp, math::CountTrailingZerosOp,
                math::CtPopOp, math::ErfOp, math::ExpOp, math::Exp2Op,
                math::ExpM1Op, math::FloorOp, math::FmaOp, math::LogOp,
                math::Log10Op, math::Log1pOp, math::Log2Op, math::PowFOp,
                math::RsqrtOp, math::SqrtOp, math::TanhOp>(op))
    return true;
  if (llvm::isa<triton::IntToPtrOp, triton::PtrToIntOp, triton::BitcastOp,
                triton::FpToFpOp, triton::AddPtrOp>(op))
    return true;
  if (auto externElementwiseOp = dyn_cast<triton::ExternElementwiseOp>(op))
    return externElementwiseOp.getPure();
  if (llvm::isa<ttg::CmpIOp, ttg::CmpFOp, ttg::SelectOp>(op))
    return true;
  return false;
}

bool CTAPlanner::processElementwise(Operation *op, Attribute layout) {
  llvm::SmallVector<Attribute> newOperandLayouts(op->getNumOperands(), layout);
  llvm::SmallVector<Attribute> newResultLayouts(op->getNumResults(), layout);
  insertCasts(op, newOperandLayouts, newResultLayouts);
  return true;
}

bool CTAPlanner::processConstant(arith::ConstantOp constant, Attribute layout) {
  if (auto tensorTy =
          constant.getResult().getType().dyn_cast<RankedTensorType>()) {
    if (auto attr = constant.getValue().dyn_cast<SplatElementsAttr>()) {

      auto newTensorTy = RankedTensorType::get(
          tensorTy.getShape(), tensorTy.getElementType(), layout);
      constant.setValueAttr(
          SplatElementsAttr::get(newTensorTy, attr.getSplatValue<Attribute>()));
    }
  }
  insertCasts(constant.getOperation(), {}, {layout});
  return true;
}

bool CTAPlanner::processSplat(triton::SplatOp splat, Attribute layout) {
  insertCasts(splat.getOperation(), {{}}, {layout});
  return true;
}

bool CTAPlanner::processMakeRange(triton::MakeRangeOp makeRange,
                                  Attribute layout) {
  insertCasts(makeRange.getOperation(), {}, {layout});
  return true;
}

bool CTAPlanner::processMakeTensorPtr(triton::MakeTensorPtrOp makeTensorPtr,
                                      Attribute layout) {
  // All inputs of `makeTensorPtr` are scalar types
  llvm::SmallVector<Attribute> dummyInAttrs(makeTensorPtr.getNumOperands(), {});
  insertCasts(makeTensorPtr.getOperation(), dummyInAttrs, {layout});
  return true;
}

bool CTAPlanner::processBroadcast(triton::BroadcastOp broadcast,
                                  Attribute layout) {
  insertCasts(broadcast.getOperation(), {layout}, {layout});
  return true;
}

bool CTAPlanner::processExpandDimsBackward(triton::ExpandDimsOp expandDims,
                                           Attribute newResultLayout) {
  auto newSrcLayout = ttg::SliceEncodingAttr::get(
      newResultLayout.getContext(), expandDims.getAxis(), newResultLayout);
  insertCasts(expandDims.getOperation(), {newSrcLayout}, {newResultLayout});
  return true;
}

bool CTAPlanner::processExpandDimsForward(triton::ExpandDimsOp expandDims,
                                          Attribute newSrcLayout) {
  assert(0 && "processExpandDimsForward not implemented yet");
  return true;
}

bool CTAPlanner::processConvertLayoutBackward(
    ttg::ConvertLayoutOp convertLayout, CastOp cast) {
  Value src = convertLayout.getSrc();
  Value result = convertLayout.getResult();
  assert(getNumUsers(result) == 1 &&
         "Expect to call processMultiUsersBackward first");
  result.replaceAllUsesWith(src);
  convertLayout.erase();
  queue.push(cast);
  return true;
}

bool CTAPlanner::processConvertLayoutForward(ttg::ConvertLayoutOp convertLayout,
                                             CastOp cast) {
  Value src = convertLayout.getSrc();
  Value result = convertLayout.getResult();
  assert(getNumUsers(src) == 1 &&
         "Expect to call processMultiUsersForward first");
  src.setType(result.getType());
  result.replaceAllUsesWith(src);
  convertLayout.erase();
  queue.push(cast);
  return true;
}

bool CTAPlanner::processIfOp(scf::IfOp ifOp, int index, const Type &newType) {
  // Check index
  assert(index < ifOp.getNumResults() && "Invalid result index of IfOp");
  assert(index < ifOp.thenYield().getNumOperands() &&
         "Invalid operand index of YieldOp");
  assert(index < ifOp.elseYield().getNumOperands() &&
         "Invalid operand index of YieldOp");

  Location loc = ifOp.getLoc();
  OpBuilder builder(ifOp.getContext());

  // Insert forward cast after ifOp
  Value result = ifOp.getResult(index);
  builder.setInsertionPointAfter(ifOp.getOperation());
  auto newCast =
      markForward(builder.create<CastOp>(loc, result.getType(), result));
  result.setType(newType);
  result.replaceAllUsesExcept(newCast.getResult(0), newCast.getOperation());
  queue.push(newCast);

  // Insert backward casts before yield
  for (scf::YieldOp yield : {ifOp.thenYield(), ifOp.elseYield()}) {
    Value yieldSrc = yield.getOperand(index);
    builder.setInsertionPoint(yield.getOperation());
    newCast = markBackward(builder.create<CastOp>(loc, newType, yieldSrc));
    yield->setOperand(index, newCast.getResult(0));
    queue.push(newCast);
  }

  return true;
}

bool CTAPlanner::processForOp(scf::ForOp forOp, int index,
                              const Type &newType) {
  Block *body = forOp.getBody();
  auto yield = llvm::cast<scf::YieldOp>(forOp.getBody()->getTerminator());

  // Check index
  assert(index + forOp.getNumControlOperands() < forOp.getNumOperands() &&
         "Invalid operand index of ForOp");
  assert(index + forOp.getNumInductionVars() < body->getNumArguments() &&
         "Invalid block arg index of ForOp");
  assert(index < yield.getNumOperands() && "Invalid operand index of YieldOp");
  assert(index < forOp.getNumResults() && "Invalid result index of IfOp");

  Location loc = forOp.getLoc();
  OpBuilder builder(forOp.getContext());

  // Insert backward cast before forOp
  OpOperand &operand =
      forOp->getOpOperand(index + forOp.getNumControlOperands());
  builder.setInsertionPoint(forOp.getOperation());
  auto newCast =
      markBackward(builder.create<CastOp>(loc, newType, operand.get()));
  operand.set(newCast.getResult(0));
  queue.push(newCast);

  // Insert forward cast after block arg
  Value arg = body->getArgument(index + forOp.getNumInductionVars());
  builder.setInsertionPointToStart(body);
  newCast = markForward(builder.create<CastOp>(loc, arg.getType(), arg));
  arg.setType(newType);
  arg.replaceAllUsesExcept(newCast.getResult(0), newCast.getOperation());
  queue.push(newCast);

  // Insert backward cast before yield
  Value yieldSrc = yield.getOperand(index);
  builder.setInsertionPoint(yield.getOperation());
  newCast = markBackward(builder.create<CastOp>(loc, newType, yieldSrc));
  yield->setOperand(index, newCast.getResult(0));
  queue.push(newCast);

  // Insert forward cast after forOp
  Value result = forOp.getResult(index);
  builder.setInsertionPointAfter(forOp.getOperation());
  newCast = markForward(builder.create<CastOp>(loc, result.getType(), result));
  result.setType(newType);
  result.replaceAllUsesExcept(newCast.getResult(0), newCast.getOperation());
  queue.push(newCast);

  return true;
}

int findResultIndex(Operation *op, Value result) {
  for (int i = 0; i < op->getNumResults(); ++i)
    if (op->getResult(i) == result)
      return i;
  assert(0 && "Invalid index of op result");
  return -1;
}

bool CTAPlanner::processIfOpBackward(scf::IfOp ifOp, CastOp cast) {
  int index = findResultIndex(ifOp.getOperation(), cast.getOperand(0));
  auto newType = cast.getResult(0).getType();
  return processIfOp(ifOp, index, newType);
}

bool CTAPlanner::processForOpBackward(scf::ForOp forOp, CastOp cast) {
  int index = findResultIndex(forOp.getOperation(), cast.getOperand(0));
  auto newType = cast.getResult(0).getType();
  return processForOp(forOp, index, newType);
}

bool CTAPlanner::processBlockArgBackward(BlockArgument arg, CastOp cast) {
  if (auto forOp = llvm::dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
    int index = int(arg.getArgNumber()) - forOp.getNumInductionVars();
    auto newType = cast.getResult(0).getType();
    return processForOp(forOp, index, newType);
  } else {
    assert(0 && "Unexpected parent op of block argument");
    return true;
  }
}

bool CTAPlanner::processForOpForward(scf::ForOp forOp, CastOp cast) {
  int index = cast.getResult(0).use_begin()->getOperandNumber() -
              forOp.getNumControlOperands();
  auto newType = cast.getOperand(0).getType();
  return processForOp(forOp, index, newType);
}

bool CTAPlanner::processYieldOpForward(scf::YieldOp yieldOp, CastOp cast) {
  int index = cast.getResult(0).use_begin()->getOperandNumber();
  auto newType = cast.getOperand(0).getType();
  if (auto ifOp = llvm::dyn_cast<scf::IfOp>(yieldOp->getParentOp()))
    return processIfOp(ifOp, index, newType);
  else if (auto forOp = llvm::dyn_cast<scf::ForOp>(yieldOp->getParentOp()))
    return processForOp(forOp, index, newType);
  else
    assert(0 && "Unexpected parent op of YieldOp");
  return true;
}

bool CTAPlanner::processOpFallback(Operation *op) {
  Location loc = op->getLoc();
  OpBuilder builder(op->getContext());

  builder.setInsertionPoint(op);
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    Value operand = op->getOperand(i);
    auto operandTy = operand.getType();
    if (triton::isTensorOrTensorPointerType(operandTy)) {
      auto cast = markBackward(builder.create<CastOp>(loc, operandTy, operand));
      op->setOperand(i, cast.getResult(0));
      queue.push(cast);
    }
  }

  builder.setInsertionPointAfter(op);
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    Value result = op->getResult(i);
    auto resultTy = result.getType();
    if (triton::isTensorOrTensorPointerType(resultTy)) {
      auto cast = markForward(builder.create<CastOp>(loc, resultTy, result));
      result.replaceAllUsesExcept(cast.getResult(0), cast.getOperation());
      queue.push(cast);
    }
  }

  return true;
}

bool CTAPlanner::processMultiUsersBackward(Value input, CastOp cast) {
  Location loc = input.getLoc();
  OpBuilder builder(input.getContext());

  llvm::DenseMap<Type, llvm::SmallVector<CastOp>> typeToIndices;
  for (OpOperand &operand : input.getUses()) {
    auto brotherCast = llvm::dyn_cast<CastOp>(operand.getOwner());
    if (!brotherCast) {
      if (stepUnchanged <= queue.size())
        return false;
      builder.setInsertionPoint(operand.getOwner());
      brotherCast = markBackward(
          builder.create<CastOp>(loc, cast.getResult(0).getType(), input));
      auto newCast = markForward(builder.create<CastOp>(
          loc, input.getType(), brotherCast.getResult(0)));
      operand.set(newCast.getResult(0));
      queue.push(brotherCast);
      queue.push(newCast);
    }
    auto type = brotherCast.getResult(0).getType();
    typeToIndices[type].push_back(brotherCast);
  }

  bool first = true;
  for (auto it : typeToIndices) {
    Type &type = it.first;
    llvm::SmallVector<CastOp> &casts = it.second;
    Value newInput = input;
    if (!first) {
      if (Operation *defOp = input.getDefiningOp()) {
        builder.setInsertionPointAfter(defOp);
        Operation *clonedOp = builder.clone(*defOp);
        newInput = clonedOp->getResult(0);
      } else {
        assert(0 && "Layout conflict for block arg"); // TODO
      }
    }
    first = false;
    if (Operation *defOp = newInput.getDefiningOp()) {
      builder.setInsertionPointAfter(defOp);
    } else {
      assert(newInput.isa<BlockArgument>() &&
             "Unexpected Value without defining op");
      builder.setInsertionPointToStart(
          newInput.cast<BlockArgument>().getOwner());
    }
    auto newCast = markBackward(builder.create<CastOp>(loc, type, newInput));
    queue.push(newCast);
    auto newResult = newCast.getResult(0);
    for (CastOp &brotherCast : casts) {
      brotherCast.getResult(0).replaceAllUsesWith(newResult);
      eraseCastOpFromQueue(brotherCast);
    }
  }
  return true;
}

bool CTAPlanner::processMultiUsersForward(Value castResult, CastOp cast) {
  Value castSrc = cast.getOperand(0);

  Location loc = cast.getLoc();
  OpBuilder builder(cast.getContext());
  builder.setInsertionPointAfter(cast.getOperation());

  while (!castResult.use_empty()) {
    auto newCast =
        markForward(builder.create<CastOp>(loc, castResult.getType(), castSrc));
    castResult.use_begin()->set(newCast.getResult(0));
    queue.push(newCast);
  }

  eraseCastOp(cast);
  return true;
}

struct PlanCTAPass : public TritonGPUPlanCTAPassBase<PlanCTAPass> {
  PlanCTAPass(ttng::ClusterInfo *clusterInfo_ = nullptr)
      : clusterInfo(clusterInfo_) {}

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Skip PlanCTAPass when numCTAs == 1
    if (ttg::TritonGPUDialect::getNumCTAs(mod) == 1)
      return;

    mod.walk([&](triton::FuncOp funcOp) {
      CTAPlanner planner(clusterInfo);
      planner.run(funcOp);

      // FIXME: Clone funcOp so that the IR change can be identified after
      // PlanCTAPass. Without this, the change after PlanCTAPass will not be
      // displayed when MLIR_ENABLE_DUMP=1. This is not reasonable and should
      // be fixed later.
      OpBuilder builder(funcOp);
      builder.clone(*funcOp.getOperation());
      funcOp.erase();
    });
  }

  ttng::ClusterInfo *clusterInfo;
};

} // namespace

std::unique_ptr<Pass>
mlir::createTritonNvidiaGPUPlanCTAPass(ttng::ClusterInfo *clusterInfo) {
  return std::make_unique<PlanCTAPass>(clusterInfo);
}

/* TODO
 * - Use ConvertLayoutOp instead of UnrealizedConversionCastOp.
 * - Move PlanCTAPass to the front of CoalescePass.
 * - Design better tiling strategy for DotOp and ReduceOp.
 * - Consider cases where there are more than one DotOps.
 * - Use better data structure for erasing CastOps from queue (linked list?).
 * - Process eliminable CastOps in higher priority.
 * - Fix the clone func bug in PlanCTAPass::runOnOperation.
 * - Add some comments to introduce the overall idea of this pass.
 * - Add some lit tests for this pass.
 */
