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

bool canHoistAndAggregate(triton::LoadOp loadOp, int ub) {
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
    return false;

  // Assume fp16 element type
  // Also assume operand B does not use LDS
  int total_LDS = shape[0] * shape[1] * ub * 2;
  assert(total_LDS <= 65536 && "BLOCK_M * K is too large");

  return true;
}

void findValidLoads(scf::ForOp forOp, SetVector<Operation *> &validLoads,
                    int ub) {
  for (Operation &op : forOp) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(&op)) {
      if (canHoistAndAggregate(loadOp, ub))
        validLoads.insert(loadOp);
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
                      int ub) {
  auto tensorTy = dyn_cast<RankedTensorType>(makeRangeOp.getType());
  auto tensorShape = tensorTy.getShape();
  assert(tensorShape.size() == 1 && "make_range should be 1D");
  auto elemTy = tensorTy.getElementType();
  auto enc = tensorTy.getEncoding();
  int makeRangeNewEnd = tensorShape[0] * ub;
  SmallVector<int64_t> newTensorShape(1, makeRangeNewEnd);
  RankedTensorType newTensorTy =
      RankedTensorType::get(newTensorShape, elemTy, enc);
  Value range = builder.create<triton::MakeRangeOp>(
      makeRangeOp.getLoc(), newTensorTy, 0, makeRangeNewEnd);
  return range;
}

Value extendBroadcast(OpBuilder &builder, Operation *op, int dim, int factor,
                      Value newSrc) {
  auto broadcastOp = dyn_cast<triton::BroadcastOp>(op);
  assert(broadcastOp && "We are not starting with a broadcast op");
  auto bTensorTy = dyn_cast<RankedTensorType>(broadcastOp.getType());
  auto bShape = bTensorTy.getShape();
  SmallVector<int64_t> newShape(bShape.begin(), bShape.end());
  newShape[dim] *= factor;
  RankedTensorType newBTensorTy = RankedTensorType::get(
      newShape, bTensorTy.getElementType(), bTensorTy.getEncoding());
  return builder.create<triton::BroadcastOp>(broadcastOp.getLoc(), newBTensorTy,
                                             newSrc);
}

Value expandPathBcastM(Operation *bcastM, OpBuilder &builder, int ub) {
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
  auto newMakeRangeValue = extendMakeRange(builder, makeRangeOp, ub);
  // new expand_dims 1x{128*ub}
  int expandDim = expandDimsOp.getAxisAttr().getInt();
  auto newExpandDimsValue = builder.create<triton::ExpandDimsOp>(
      expandDimsOp.getLoc(), newMakeRangeValue, expandDim);

  // erase ops
  // makeRangeOp.erase();
  // expandDimsOp.erase();

  // new broadcast
  return extendBroadcast(builder, bcastM, /*which dim to extend*/ 1, ub,
                         newExpandDimsValue);
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

Value hoistLoad(scf::ForOp forOp, Operation *op, int ub) {
  triton::LoadOp loadOp = dyn_cast<triton::LoadOp>(op);
  // Here I assume
  // 1. There is no mask in the loadOp
  // 2. The ptr of loadOp comes from a block arg of the loop
  IRMapping prologueMap;
  OpBuilder builder(forOp);
  auto numOperands = loadOp.getNumOperands();
  assert(numOperands == 1 && "Does not assume to have mask for now");
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

  builder.setInsertionPoint(aPtrs);

  auto newBcastMVal = expandPathBcastM(bcastM, builder, ub);
  auto newBcastKVal =
      extendBroadcast(builder, bcastK, /*which dim to extend*/ 1, ub,
                      dyn_cast<triton::BroadcastOp>(bcastK).getSrc());

  // After expanding BLOCK_K to BLOCK_K*ub, we create the new addptr
  // with the new broadcast values: addptr newBcastKVal, newBcastMVal
  auto newPtrVal = builder.create<triton::AddPtrOp>(
      aPtrs->getLoc(), newBcastKVal.getType(), newBcastKVal, newBcastMVal);

  // The we create the aggregated load with the "fat" pointer
  // create: load newPtr
  auto aggregatedLoadVal = builder.create<triton::LoadOp>(
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
  auto forOpIV = forOp.getInductionVar();
  auto forOpStep = forOp.getStep();
  auto bufOffVal = builder.create<arith::MulIOp>(loc, forOpIV, forOpStep);

  // step 2: localBuf = memdesc_subview ldsBuffer[0, bufOff]
  SmallVector<Value> localBufOff(2);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  localBufOff[0] = zero;      // along M dim
  localBufOff[1] = bufOffVal; // along K dim

  auto loadOp = dyn_cast<tt::LoadOp>(op);
  auto subviewShape = dyn_cast<RankedTensorType>(loadOp.getType()).getShape();
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

// Stream Pipeline
struct AggregateLoad : public TritonAMDGPUAggregateLoadBase<AggregateLoad> {
  AggregateLoad() = default;

  void runOnOperation() override {

    // Do the pipelining
    getOperation()->walk([&](scf::ForOp forOp) -> void {
      // We need K to be constant, i.e. the upper bound of the loop is a
      // constant
      auto ub = isUpperBoundConstant(forOp);
      if (!ub)
        return;
      SetVector<Operation *> validLoads;
      findValidLoads(forOp, validLoads, ub);
      for (auto loadOp : validLoads) {
        auto localAllocValue = hoistLoad(forOp, loadOp, ub);
        processLoopBody(forOp, loadOp, localAllocValue);
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUAggregateLoadPass() {
  return std::make_unique<AggregateLoad>();
}
