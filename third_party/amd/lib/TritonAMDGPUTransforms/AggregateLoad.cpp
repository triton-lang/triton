#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"

using llvm::MapVector;
using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = triton::gpu;

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

int64_t getAllocSize(ShapedType type) {
  int64_t numel = ShapedType::getNumElements(type.getShape());
  int64_t bitwidth = type.getElementType().getIntOrFloatBitWidth();
  // TODO: Remove this, only using this to simplify shared memory calculation
  //       of using 65536 "bytes". If we change it to bit that can handle
  //       practically anything.
  assert(bitwidth >= 8 && "Only support bitwidth > 1 byte");
  int64_t allocSize = numel * (bitwidth / 8);
  return allocSize;
}

std::optional<std::array<int64_t, 3>>
canHoistAndAggregate(triton::LoadOp loadOp, int ub,
                     int64_t &currentSharedMemoryUsage) {
  auto loadTensorType =
      dyn_cast<RankedTensorType>(loadOp.getResult().getType());
  auto shape = loadTensorType.getShape();
  // Assume each load has one use, which is the convert_layout #blocked to
  // #dotOp
  Operation *use = *loadOp.getResult().getUsers().begin();
  auto cvt = llvm::dyn_cast<ttg::ConvertLayoutOp>(use);
  auto tensorType = dyn_cast<RankedTensorType>(cvt.getResult().getType());
  auto dotOpEnc =
      dyn_cast<ttg::DotOperandEncodingAttr>(tensorType.getEncoding());
  auto opIdx = dotOpEnc.getOpIdx();
  // Assume we don't want to hoist load for operand B
  if (opIdx == 1)
    return std::nullopt;

  const int64_t kMaxSharedMemory = 65536;
  assert(currentSharedMemoryUsage <= kMaxSharedMemory &&
         "Even without hoisting, block size is too large.");

  int64_t loadLDS = getAllocSize(loadTensorType);
  int64_t nonLDSSharedMemSize = currentSharedMemoryUsage - loadLDS;
  int64_t expandableMemory = kMaxSharedMemory - nonLDSSharedMemSize;
  // Case where already using max amount of shared mem.
  if (loadLDS >= expandableMemory) {
    llvm::outs() << "Already maxed out!\n";
    return std::nullopt;
  }
  // Assume m-dim is dim0 and only hoist in K.
  int64_t globalKDim = shape[1] * ub;
  int64_t bitWidth = tensorType.getElementType().getIntOrFloatBitWidth();
  int64_t byteWidth = bitWidth / 8;
  int64_t largestK = expandableMemory / (shape[0] * byteWidth);
  // Hoist entire tensor A if there is space.
  if (globalKDim <= largestK) {
    int64_t newMemoryUsed = shape[0] * globalKDim * byteWidth;
    int64_t memoryLeft = currentSharedMemoryUsage - (newMemoryUsed - loadLDS);
    return std::array<int64_t, 3>({globalKDim, 1, memoryLeft});
  }

  // Determine largest hoist K size, by getting largest divisor of globalK
  // that is <= largestK.
  // TODO: Need to add assert to make sure %2 == 0 and or K needs to be power of
  // 2 aligned.
  int64_t alignedHoistK = -1;
  for (int i = globalKDim; i > 0; i /= 2) {
    if (i <= largestK) {
      alignedHoistK = i;
      break;
    }
  }
  assert(alignedHoistK > 0 && "Cannot determine hoisted-K size.");
  int64_t hoistFactor = globalKDim / alignedHoistK;
  int64_t newMemoryUsed = shape[0] * alignedHoistK * byteWidth;
  int64_t memoryLeft = currentSharedMemoryUsage - (newMemoryUsed - loadLDS);

  return std::array<int64_t, 3>({alignedHoistK, hoistFactor, memoryLeft});
}

void findValidLoads(scf::ForOp forOp, SetVector<Operation *> &validLoads,
                    SmallVector<std::pair<int64_t, int64_t>> &hoistLoopSpecs,
                    int ub, int64_t totalSharedMemoryUsage) {
  int64_t currentSharedMemoryUsage = totalSharedMemoryUsage;
  for (Operation &op : forOp) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(&op)) {
      std::optional<std::array<int64_t, 3>> hoistKSpecs =
          canHoistAndAggregate(loadOp, ub, currentSharedMemoryUsage);
      // TODO: Store the hoistKSize, hoistFactor S.T we can generate scf.for
      // loop around.
      if (hoistKSpecs.has_value()) {
        auto [hoistKSize, hoistFactor, currentSharedMemoryUsage] =
            hoistKSpecs.value();
        hoistLoopSpecs.push_back({hoistKSize, hoistFactor});
        validLoads.insert(loadOp);
      }
    }
  }
}

int isUpperBoundConstant(scf::ForOp forOp) {
  auto ub = forOp.getUpperBound();
  if (auto constant = dyn_cast<arith::ConstantOp>(ub.getDefiningOp())) {
    return cast<IntegerAttr>(constant.getValue()).getInt();
  } else {
    llvm::outs() << "Non constant upper bound??\n";
    return 0;
  }
}

// make a new make_range op that extend the existing end to be end*ub
// and update tensor shape accordingly
Value extendMakeRange(OpBuilder &builder, triton::MakeRangeOp makeRangeOp,
                      int64_t hoistDimSize) {
  auto tensorTy = dyn_cast<RankedTensorType>(makeRangeOp.getType());
  auto tensorShape = tensorTy.getShape();
  assert(tensorShape.size() == 1 && "make_range should be 1D");
  auto elemTy = tensorTy.getElementType();
  auto enc = tensorTy.getEncoding();
  int makeRangeNewEnd = hoistDimSize;
  SmallVector<int64_t> newTensorShape(1, makeRangeNewEnd);
  RankedTensorType newTensorTy =
      RankedTensorType::get(newTensorShape, elemTy, enc);
  Value range = builder.create<triton::MakeRangeOp>(
      makeRangeOp.getLoc(), newTensorTy, 0, makeRangeNewEnd);
  return range;
}

Value extendBroadcast(OpBuilder &builder, Operation *op, int dim,
                      int64_t hoistedDimSize, Value newSrc) {
  auto broadcastOp = dyn_cast<triton::BroadcastOp>(op);
  assert(broadcastOp && "We are not starting with a broadcast op");
  auto bTensorTy = dyn_cast<RankedTensorType>(broadcastOp.getType());
  auto bShape = bTensorTy.getShape();
  SmallVector<int64_t> newShape(bShape.begin(), bShape.end());
  newShape[dim] = hoistedDimSize;
  RankedTensorType newBTensorTy = RankedTensorType::get(
      newShape, bTensorTy.getElementType(), bTensorTy.getEncoding());
  return builder.create<triton::BroadcastOp>(broadcastOp.getLoc(), newBTensorTy,
                                             newSrc);
}

Value expandPathBcastM(Operation *bcastM, OpBuilder &builder,
                       int64_t hoistDimSize) {
  // Assume the following chain of IRs
  // %0 = make_range {0, 128}
  // %1 = expand_dims %0: -> tensor<1x128>
  // %2 = broadcast %1: --> tensor<16x128>
  // bcastM is the broadcast op
  auto broadcastOp = dyn_cast<triton::BroadcastOp>(bcastM);
  assert(broadcastOp && "We are not starting with a broadcast op");
  auto bcastKParentOp = broadcastOp.getSrc().getDefiningOp();
  auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(bcastKParentOp);
  assert(expandDimsOp && "broadcast's parent must be a expand_dims op");
  auto expandDimsOpParent = expandDimsOp.getSrc().getDefiningOp();
  auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(expandDimsOpParent);
  assert(makeRangeOp && "expandDims' parent must be a make_range op");

  // new make_range {0, 128*ub}
  auto newMakeRangeValue = extendMakeRange(builder, makeRangeOp, hoistDimSize);
  // new expand_dims 1x{128*ub}
  int expandDim = expandDimsOp.getAxisAttr().getInt();
  auto newExpandDimsValue = builder.create<triton::ExpandDimsOp>(
      expandDimsOp.getLoc(), newMakeRangeValue, expandDim);

  // erase ops
  // makeRangeOp.erase();
  // expandDimsOp.erase();

  // new broadcast
  return extendBroadcast(builder, bcastM, /*which dim to extend*/ 1,
                         hoistDimSize, newExpandDimsValue);
}

Value expandPathBcastK(Operation *bcastK, OpBuilder &builder, int ub) {
  // %27 = tt.broadcast %24 : tensor<16x1x!tt.ptr<f16>, #blocked> ->
  // tensor<16x128x!tt.ptr<f16>, #blocked> extend shape[1] with shape[1]*ub

  auto broadcastOp = dyn_cast<triton::BroadcastOp>(bcastK);

  return broadcastOp.getResult();
}

Value createLocalAlloc(OpBuilder &builder, Location loc, Value loadVal) {
  auto tensorTy = dyn_cast<RankedTensorType>(loadVal.getType());
  SmallVector<int64_t> bufferShape(tensorTy.getShape().begin(),
                                   tensorTy.getShape().end());
  Type eType = tensorTy.getElementType();
  auto CTALayout = ttg::getCTALayout(tensorTy.getEncoding());
  auto sharedEnc = ttg::SharedEncodingAttr::get(
      tensorTy.getContext(), /*vec*/ 1, /*perPhase*/ 1, /*maxPhase*/ 1,
      ttg::getOrder(tensorTy.getEncoding()), CTALayout);
  auto ldsBufferType = triton::MemDescType::get(
      bufferShape, eType, sharedEnc,
      triton::gpu::SharedMemorySpaceAttr::get(tensorTy.getContext()),
      /*mutableMemory=*/true);
  return builder.create<ttg::LocalAllocOp>(loc, ldsBufferType, loadVal);
}

Value hoistLoad(scf::ForOp forOp, Operation *op, int64_t hoistKSize, int ub) {
  triton::LoadOp loadOp = dyn_cast<triton::LoadOp>(op);
  // Here I assume
  // 1. There is no mask along k dim in the loadOp
  // 2. The ptr of loadOp comes from a block arg of the loop
  OpBuilder builder(forOp);
  builder.setInsertionPoint(forOp);

  // Dealing with mask
  Value maskM = loadOp.getMask();
  Value newMaskVal, newOtherVal;
  if (maskM) {
    // We assume the mask along the M dim is NOT loop carried
    assert(maskM.getParentRegion() != forOp.getRegion() &&
           "load mask should not be loop carried");
    Operation *maskOp = maskM.getDefiningOp();
    auto bcastMask = dyn_cast<tt::BroadcastOp>(maskOp);
    assert(bcastMask && "load mask does not come from a broadcast op");

    newMaskVal = extendBroadcast(builder, maskOp, /*dim*/ 1,
                                 /*hoist size*/ hoistKSize, bcastMask.getSrc());

    // Dealing with other
    Value other = loadOp.getOther();
    auto otherConstant = dyn_cast<arith::ConstantOp>(other.getDefiningOp());
    llvm::outs() << "OTHER OG:" << other << "\n";
    llvm::outs() << "OTHER CONST:" << otherConstant << "\n";
    // auto attr = otherConstant.getValue();
    auto denseAttr =
        dyn_cast<DenseFPElementsAttr>(otherConstant.getValueAttr());
    auto ty = dyn_cast<RankedTensorType>(denseAttr.getType());
    SmallVector<int64_t> newShape(ty.getShape().begin(), ty.getShape().end());
    newShape[1] = hoistKSize;
    auto newTy =
        RankedTensorType::get(newShape, ty.getElementType(), ty.getEncoding());
    assert(denseAttr.isSplat() &&
           "The attribute of the constantOp is not a splat");
    auto reshapedAttr = denseAttr.resizeSplat(newTy);
    newOtherVal =
        builder.create<arith::ConstantOp>(forOp.getLoc(), newTy, reshapedAttr);
  }

  // Dealing with ptr
  auto blockArg = dyn_cast<BlockArgument>(loadOp.getOperand(0));
  assert(blockArg && "ptr is not a block arg");

  OpOperand &operand = *forOp.getTiedLoopInit(blockArg);
  // This is assumed to be the addptr op to compute the final aptrs for loadOp
  // say %29 = tt.addptr %27, %28 : tensor<16x128x!tt.ptr<f16>, #blocked>
  Operation *aPtrs = operand.get().getDefiningOp();
  // We assume the operands of this addptr come from broadcast
  Operation *bcastK, *bcastM;
  for (Value ptrOperand : aPtrs->getOperands()) {
    Operation *broadcastOp = ptrOperand.getDefiningOp();
    Value bcastSrc = broadcastOp->getOperand(0);
    auto srcShape = dyn_cast<RankedTensorType>(bcastSrc.getType()).getShape();
    if (srcShape[1] == 1)
      bcastK = broadcastOp;
    else // srcShape[0] == 1
      bcastM = broadcastOp;
  }

  // addptr has form: res = addptr ptr, offset
  // bcastM refers to the broadcast along M dim, which is assumed to be the
  // offset of the addptr. So its shape is <1 x BLOCK_K> --> <BLOCK_M x BLOCK_K>
  // bcastK refers to the broadcast along K dim, which is assumed to be the
  // ptr of the addptr. So its shape is <BLOCK_M x 1> --> <BLOCK_M x BLOCK_K>
  //
  // We also assume that bcastM comes from the chain of the following ops
  // 1. make_range <BLOCK_K>
  // 2. expand_dims <BLOCK_K> --> <1xBLOCK_K>
  // 3. broadcast <1 x BLOCK_K> --> <BLOCK_M x BLOCK_K>
  // Therefore, we need to go all the way to make_range and extend BLOCK_K
  // to BLOCK_K*ub
  //
  // For bcastK, we only need to extend the broadcast to be
  // <BLOCK_M x 1> --> <BLOCK_M x {BLOCK_K*ub}>

  auto newBcastMVal = expandPathBcastM(bcastM, builder, hoistKSize);
  auto newBcastKVal =
      extendBroadcast(builder, bcastK, /*which dim to extend*/ 1, hoistKSize,
                      dyn_cast<triton::BroadcastOp>(bcastK).getSrc());

  // After expanding BLOCK_K to BLOCK_K*ub, we create the new addptr
  // with the new broadcast values: addptr newBcastKVal, newBcastMVal
  auto newPtrVal = builder.create<triton::AddPtrOp>(
      aPtrs->getLoc(), newBcastKVal.getType(), newBcastKVal, newBcastMVal);

  // The we create the aggregated load with the "fat" pointer
  // create: load newPtr
  Value aggregatedLoadVal;
  if (maskM)
    aggregatedLoadVal = builder.create<triton::LoadOp>(
        forOp.getLoc(), newPtrVal, newMaskVal, newOtherVal, loadOp.getCache(),
        loadOp.getEvict(), loadOp.getIsVolatile());
  else
    aggregatedLoadVal = builder.create<triton::LoadOp>(
        forOp.getLoc(), newPtrVal, loadOp.getCache(), loadOp.getEvict(),
        loadOp.getIsVolatile());

  // Store loaded tensor into LDS
  auto localAllocVal =
      createLocalAlloc(builder, forOp.getLoc(), aggregatedLoadVal);

  return localAllocVal;
}

void processLoopBody(scf::ForOp forOp, Operation *op, Value localAllocVal) {
  // Now we have hoisted the loadOp out of the loop and make it "fat"
  // We have also inserted a local_alloc op right after the load to put
  // everything into LDS.
  // Now we need to process the forOp:
  // The current loop body has the following chain:
  // 1. res = load ptrs
  // 2. opA = convert_layout res (#blocked --> #dotOperand)
  // 3. acc = dot opA, opB
  //
  // What we need is to replace the above with
  // 1. bufOff = i * BLOCK_K
  // 2. localBuf = memdesc_subview ldsBuffer[0, bufOff]
  // 3. opA = local_load localBuf
  // 4. acc = dot opA, opB
  OpBuilder builder(forOp);
  builder.setInsertionPoint(op);
  Location loc = forOp.getLoc();
  // step 1: bufOff = i * BLOCK_K
  auto loadOp = dyn_cast<tt::LoadOp>(op);
  llvm::outs() << "ORIGINAL LOAD:" << loadOp << "\n";
  auto subviewShape = dyn_cast<RankedTensorType>(loadOp.getType()).getShape();

  Value BLOCK_K =
      builder.create<arith::ConstantIntOp>(loc, subviewShape[1], 32);
  auto forOpIV = forOp.getInductionVar();
  auto bufOffVal = builder.create<arith::MulIOp>(loc, forOpIV, BLOCK_K);

  // step 2: localBuf = memdesc_subview ldsBuffer[0, bufOff]
  SmallVector<Value> localBufOff(2);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  localBufOff[0] = zero;      // along M dim
  localBufOff[1] = bufOffVal; // along K dim

  tt::MemDescType allocTy = cast<tt::MemDescType>(localAllocVal.getType());
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  tt::MemDescType subviewTy = tt::MemDescType::get(
      subviewShape, allocTy.getElementType(), allocTy.getEncoding(),
      sharedMemorySpace, /*mutableMemory=*/true);
  auto ldsSubview = builder.create<ttg::MemDescSubviewOp>(
      loc, subviewTy, localAllocVal, localBufOff);

  // step 3: local_load
  Operation *use = *loadOp.getResult().getUsers().begin();
  auto cvt = dyn_cast<ttg::ConvertLayoutOp>(use);
  auto localLoadVal =
      builder.create<ttg::LocalLoadOp>(loc, cvt.getType(), ldsSubview);

  // Step 4: replace opA in dot
  cvt.getResult().replaceAllUsesWith(localLoadVal);
  cvt.erase();

  // step 5: cleanup
  auto blockArg = dyn_cast<BlockArgument>(loadOp.getOperand(0));
  for (OpOperand &operand : blockArg.getUses()) {
    auto user = operand.getOwner();
    // Skip the loadOp, which will be removed later
    if (user != loadOp) {
      // We will not update the blockArg (ptr) of the hoisted load
      // So we replace all uses of the updated ptr with the original one
      user->getResult(0).replaceAllUsesWith(blockArg);
      user->erase();
    }
  }
  loadOp.erase();
}

void generateOuterLoop(scf::ForOp forOp, Value localAllocVal,
                       int64_t hoistFactor) {

  // Set up ops/info required to build outer loop.
  auto localAllocOp =
      llvm::cast<ttg::LocalAllocOp>(localAllocVal.getDefiningOp());
  auto loadOp =
      llvm::dyn_cast<triton::LoadOp>(localAllocOp.getSrc().getDefiningOp());
  assert(loadOp &&
         "Expected src of local alloc to be loadOp to generate outer loop.");

  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  Value lb =
      builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(0));
  Value ub = builder.create<arith::ConstantOp>(
      loc, builder.getI32IntegerAttr(hoistFactor));
  Value step =
      builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(1));
  Value init = forOp.getInits()[0];
  int innerUB = isUpperBoundConstant(forOp);
  Value newInnerUB = builder.create<arith::ConstantOp>(
      loc, builder.getI32IntegerAttr(innerUB / hoistFactor));
  auto outerDimLoop = builder.create<scf::ForOp>(
      loc, lb, ub, step, ValueRange{init},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        Operation *newLoadOp = builder.clone(*loadOp.getOperation());
        auto newLocalAllocOp = builder.create<ttg::LocalAllocOp>(
            loc, localAllocVal.getType(), newLoadOp->getResults()[0]);
        IRMapping mapping;
        mapping.map(loadOp.getResult(), newLoadOp->getResults()[0]);
        mapping.map(localAllocVal, newLocalAllocOp.getResult());
        mapping.map(init, args[0]);
        // TODO: Need to hoist B-matrix local_alloc to be out of this loop.
        // TODO: Setup matmul test with 2048.
        // TODO: Generalize figuring out/(error out when not the case) that old
        // for loop[0] is the one being used.
        Operation *newInnerLoop = builder.clone(*forOp.getOperation(), mapping);
        auto newInnerForOp = llvm::cast<scf::ForOp>(newInnerLoop);
        newInnerForOp.setUpperBound(newInnerUB);
        builder.create<scf::YieldOp>(loc,
                                     ValueRange{newInnerLoop->getResults()[0]});
      });
  forOp.getResults()[0].replaceAllUsesWith(outerDimLoop.getResults()[0]);
  forOp.erase();

  if (localAllocOp->use_empty()) {
    localAllocOp.erase();
  }

  if (loadOp->use_empty()) {
    loadOp.erase();
  }
}

// Stream Pipeline
struct AggregateLoad : public TritonAMDGPUAggregateLoadBase<AggregateLoad> {
  AggregateLoad() = default;

  void runOnOperation() override {

    int64_t totalSharedMemoryUsage = 0;
    bool foundDotOp = false;
    getOperation()->walk([&](triton::DotOp dotOp) -> void {
      Value lhs = dotOp.getA();
      Value rhs = dotOp.getB();
      // Get LHS and RHS size.
      auto lhsType = llvm::cast<ShapedType>(lhs.getType());
      auto rhsType = llvm::cast<ShapedType>(rhs.getType());
      assert(lhsType.hasStaticShape() &&
             "Expected tt.dot to have A as static shape.");
      assert(rhsType.hasStaticShape() &&
             "Expected tt.dot to have B as static shape.");
      int64_t lhsAllocSize = getAllocSize(lhsType);
      int64_t rhsAllocSize = getAllocSize(rhsType);
      totalSharedMemoryUsage += lhsAllocSize + rhsAllocSize;
      assert(!foundDotOp && "Currently only support a single dot operation.");
      foundDotOp = true;
    });
    llvm::outs() << "TOTAL SHARED MEM:" << totalSharedMemoryUsage << "\n";

    // Do the pipelining
    getOperation()->walk([&](scf::ForOp forOp) -> void {
      // We need K to be constant, i.e. the upper bound of the loop is a
      // constant
      auto ub = isUpperBoundConstant(forOp);
      if (!ub)
        return;
      SetVector<Operation *> validLoads;
      SmallVector<std::pair<int64_t, int64_t>> hoistLoopSpecs;
      findValidLoads(forOp, validLoads, hoistLoopSpecs, ub,
                     totalSharedMemoryUsage);
      for (auto [index, loadOp] : llvm::enumerate(validLoads)) {
        auto [hoistKSize, hoistFactor] = hoistLoopSpecs[index];
        auto localAllocValue = hoistLoad(forOp, loadOp, hoistKSize, ub);
        processLoopBody(forOp, loadOp, localAllocValue);
        if (hoistFactor > 1)
          generateOuterLoop(forOp, localAllocValue, hoistFactor);
      }
    });
    // llvm::outs() << "after:" << *getOperation() << "\n";
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUAggregateLoadPass() {
  return std::make_unique<AggregateLoad>();
}
