#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <optional>

namespace mlir::triton::instrument {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONINSTRUMENTGLOBALSANITIZER
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

namespace {

static constexpr const char kGSanGlobalStateArgAttr[] = "tti.gsan_global_state";
static constexpr const char kGSanStreamClockArgAttr[] = "tti.gsan_stream_clock";
static constexpr const char kGSanKernelIdArgAttr[] = "tti.gsan_kernel_id";
static constexpr const char kGSanMBarrierScratchArgAttr[] =
    "tti.gsan_mbarrier_scratch";
static constexpr const char kDisableSetMaxRegisterAttr[] =
    "tti.disable_setmaxregister";
static constexpr int64_t kGSanClusterBarrierScratchBytes = 128;
static constexpr int64_t kGSanMBarrierTableHeaderBytes = 16;
static constexpr int64_t kGSanMBarrierRecordBytes = 176;

struct DescriptorInfo {
  Value base;
  SmallVector<Value> shape;
  SmallVector<Value> strides;
};

static void setTMAPtrAxisHints(OpBuilder &builder, Value ptr) {
  auto ptrTy = cast<RankedTensorType>(ptr.getType());

  Operation *def = ptr.getDefiningOp();
  if (!def)
    return;

  auto rank = ptrTy.getRank();
  SmallVector<int32_t> contiguity(rank, 1);
  contiguity.back() = ptrTy.getShape().back();
  SmallVector<int32_t> divisibility(rank, 1);
  divisibility.back() = 16;
  auto attrTy = RankedTensorType::get({rank}, builder.getI32Type());
  def->setDiscardableAttr("tt.contiguity",
                          DenseIntElementsAttr::get(attrTy, contiguity));
  def->setDiscardableAttr("tt.divisibility",
                          DenseIntElementsAttr::get(attrTy, divisibility));
}

static Value castToI64(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isInteger(64))
    return value;
  return builder.createOrFold<arith::ExtSIOp>(loc, builder.getI64Type(), value);
}

static SmallVector<Value> castToI64(OpBuilder &builder, Location loc,
                                    ValueRange values) {
  SmallVector<Value> result;
  result.reserve(values.size());
  for (Value value : values)
    result.push_back(castToI64(builder, loc, value));
  return result;
}

static ttg::BlockedEncodingAttr
getInstrumentationEncoding(OpBuilder &builder, ArrayRef<int64_t> shape,
                           Type elemType) {
  int numWarps = ttg::lookupNumWarps(builder.getInsertionBlock()->getParent());
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numCTAs = ttg::lookupNumCTAs(builder.getInsertionBlock()->getParentOp());
  auto base = ttg::getDefaultBlockedEncoding(builder.getContext(), shape,
                                             numWarps, threadsPerWarp, numCTAs);
  SmallVector<unsigned> order = llvm::to_vector(base.getOrder());
  SmallVector<unsigned> warpsPerCTA = llvm::to_vector(base.getWarpsPerCTA());
  SmallVector<unsigned> sizePerThread(shape.size(), 1);
  unsigned elemBits = elemType.getIntOrFloatBitWidth();
  unsigned maxElems = std::max(128u / elemBits, 1u);
  if (!order.empty()) {
    unsigned dim = order.front();
    // Distribute last dim to maximize contiguity within a thread
    if (order.size() > 1 && warpsPerCTA[dim] > 1) {
      warpsPerCTA[order[1]] *= warpsPerCTA[dim];
      warpsPerCTA[dim] = 1;
    }

    auto threadsOnDim = base.getThreadsPerWarp()[dim] * warpsPerCTA[dim];
    auto numUniqueElems = ceil(static_cast<unsigned>(shape[dim]), threadsOnDim);
    sizePerThread[dim] = std::min(maxElems, numUniqueElems);
  }
  return ttg::BlockedEncodingAttr::get(builder.getContext(), sizePerThread,
                                       base.getThreadsPerWarp(), warpsPerCTA,
                                       order, base.getCGALayout());
}

static DescriptorInfo getDescriptorInfo(Value desc, OpBuilder &builder) {
  if (!isa<tt::TensorDescType>(desc.getType())) {
    std::string msg;
    llvm::raw_string_ostream stream(msg);
    stream << "GSan: Unsupported descriptor type" << desc.getType();
    llvm::report_fatal_error(msg.c_str());
  }
  auto descTy = cast<tt::TensorDescType>(desc.getType());

  auto elemTy = descTy.getSignlessBlockType().getElementType();
  auto basePtrTy = tt::getPointerType(elemTy);
  unsigned rank = descTy.getShape().size();
  SmallVector<Type> resultTypes;
  resultTypes.reserve(1 + 2 * rank);
  resultTypes.push_back(basePtrTy);
  resultTypes.append(rank, builder.getI64Type());
  resultTypes.append(rank, builder.getI64Type());

  auto info = ExperimentalGSanTensorDescInfoOp::create(builder, desc.getLoc(),
                                                       resultTypes, desc);
  auto results = info->getResults();

  DescriptorInfo descriptorInfo;
  descriptorInfo.base = results.front();
  descriptorInfo.shape.assign(results.begin() + 1, results.begin() + 1 + rank);
  descriptorInfo.strides.assign(results.begin() + 1 + rank, results.end());
  return descriptorInfo;
}

static Value createExpandedOffsetRange(OpBuilder &builder, Location loc,
                                       RankedTensorType fullI64Type,
                                       Value offset, unsigned dim) {
  int64_t dimSize = fullI64Type.getShape()[dim];

  auto sliceI32Type = getSlicedTensorType(fullI64Type, {static_cast<int>(dim)},
                                          builder.getI32Type());
  auto sliceI64Type = getSlicedTensorType(fullI64Type, {static_cast<int>(dim)},
                                          builder.getI64Type());

  Value range = tt::MakeRangeOp::create(builder, loc, sliceI32Type, 0, dimSize);
  Value rangeI64 = arith::ExtSIOp::create(builder, loc, sliceI64Type, range);
  Value offsetI64 = castToI64(builder, loc, offset);
  Value offsetSplat =
      tt::SplatOp::create(builder, loc, sliceI64Type, offsetI64);
  Value result =
      arith::AddIOp::create(builder, loc, sliceI64Type, offsetSplat, rangeI64);
  return reshapeAndBroadcast(builder, loc, result, {static_cast<int>(dim)},
                             fullI64Type);
}

static Value convertAndBroadcast(OpBuilder &builder, Location loc, Value tensor,
                                 int dim, RankedTensorType dstType) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto sliceType =
      getSlicedTensorType(dstType, {dim}, tensorType.getElementType());
  tensor = ttg::ConvertLayoutOp::create(builder, loc, sliceType, tensor);
  return reshapeAndBroadcast(builder, loc, tensor, {dim}, dstType);
}

static Value createMaskFromRanges(OpBuilder &builder, Location loc,
                                  const DescriptorInfo &desc,
                                  ArrayRef<Value> offsetRanges,
                                  RankedTensorType fullI64Type) {
  auto maskType = RankedTensorType::get(
      fullI64Type.getShape(), builder.getI1Type(), fullI64Type.getEncoding());
  Value zero = createConstIntTensor(builder, loc, 0, fullI64Type,
                                    /*isSigned=*/true);

  Value mask;
  for (auto [dim, offsets] : llvm::enumerate(offsetRanges)) {
    Value upperBound =
        tt::SplatOp::create(builder, loc, fullI64Type, desc.shape[dim]);
    Value lower = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                        offsets, zero);
    Value upper = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                        offsets, upperBound);
    Value dimMask = arith::AndIOp::create(builder, loc, lower, upper);
    dimMask = cast<RankedTensorType>(dimMask.getType()) == maskType
                  ? dimMask
                  : tt::BroadcastOp::create(builder, loc, maskType, dimMask);
    mask = mask ? arith::AndIOp::create(builder, loc, mask, dimMask) : dimMask;
  }
  return mask;
}

static Value createPtrFromRanges(OpBuilder &builder, Location loc,
                                 const DescriptorInfo &desc,
                                 ArrayRef<Value> offsetRanges,
                                 RankedTensorType fullI64Type) {
  auto ptrTensorType = RankedTensorType::get(
      fullI64Type.getShape(), desc.base.getType(), fullI64Type.getEncoding());
  Value ptr = tt::SplatOp::create(builder, loc, ptrTensorType, desc.base);
  for (auto [dim, offsets] : llvm::enumerate(offsetRanges)) {
    Value stride =
        tt::SplatOp::create(builder, loc, fullI64Type, desc.strides[dim]);
    Value offsetWithStride =
        arith::MulIOp::create(builder, loc, fullI64Type, offsets, stride);
    ptr = tt::AddPtrOp::create(builder, loc, ptrTensorType, ptr,
                               offsetWithStride);
  }
  return ptr;
}

static std::pair<Value, Value>
createTiledAccess(OpBuilder &builder, Location loc, const DescriptorInfo &desc,
                  ArrayRef<int64_t> blockShape, ValueRange offsets,
                  std::optional<Value> pred) {
  auto encoding = getInstrumentationEncoding(
      builder, blockShape,
      cast<tt::PointerType>(desc.base.getType()).getPointeeType());
  auto fullI64Type =
      RankedTensorType::get(blockShape, builder.getI64Type(), encoding);

  SmallVector<Value> offsetRanges;
  offsetRanges.reserve(offsets.size());
  for (auto [dim, offset] : llvm::enumerate(offsets)) {
    offsetRanges.push_back(
        createExpandedOffsetRange(builder, loc, fullI64Type, offset, dim));
  }

  Value ptr =
      createPtrFromRanges(builder, loc, desc, offsetRanges, fullI64Type);
  Value mask =
      createMaskFromRanges(builder, loc, desc, offsetRanges, fullI64Type);
  if (pred) {
    auto maskType = cast<RankedTensorType>(mask.getType());
    Value predTensor = tt::SplatOp::create(builder, loc, maskType, *pred);
    mask = arith::AndIOp::create(builder, loc, mask, predTensor);
  }
  setTMAPtrAxisHints(builder, ptr);
  return std::make_pair(ptr, mask);
}

static std::pair<Value, Value> createGatherScatterAccess(
    OpBuilder &builder, Location loc, const DescriptorInfo &desc,
    ArrayRef<int64_t> blockShape, Value xOffsets, Value yOffset) {
  auto encoding = getInstrumentationEncoding(
      builder, blockShape,
      cast<tt::PointerType>(desc.base.getType()).getPointeeType());
  auto fullI64Type =
      RankedTensorType::get(blockShape, builder.getI64Type(), encoding);

  auto xOffsetsTy = cast<RankedTensorType>(xOffsets.getType());
  auto xOffsetsI64Ty = RankedTensorType::get(
      xOffsetsTy.getShape(), builder.getI64Type(), xOffsetsTy.getEncoding());
  Value xOffsetsI64 =
      arith::ExtSIOp::create(builder, loc, xOffsetsI64Ty, xOffsets);
  Value xRange =
      convertAndBroadcast(builder, loc, xOffsetsI64, /*dim=*/0, fullI64Type);
  Value yRange =
      createExpandedOffsetRange(builder, loc, fullI64Type, yOffset, /*dim=*/1);
  SmallVector<Value> offsetRanges = {xRange, yRange};
  auto ptrs =
      createPtrFromRanges(builder, loc, desc, offsetRanges, fullI64Type);
  auto mask =
      createMaskFromRanges(builder, loc, desc, offsetRanges, fullI64Type);
  setTMAPtrAxisHints(builder, ptrs);
  return std::make_pair(ptrs, mask);
}

static void instrumentAsyncTMALoad(ttng::AsyncTMACopyGlobalToLocalOp op) {
  if (isa<ttng::TensorDescIm2ColType>(op.getDesc().getType()))
    return;

  OpBuilder builder(op);
  auto desc = getDescriptorInfo(op.getDesc(), builder);
  auto blockShape = op.getDesc().getType().getShape();

  auto offsets = castToI64(builder, op.getLoc(), op.getCoord());
  auto access = createTiledAccess(builder, op.getLoc(), desc, blockShape,
                                  offsets, op.getPred());
  ExperimentalGSanTensorAccessOp::create(builder, op.getLoc(), access.first,
                                         access.second, /*isStore=*/false);
}

static void instrumentAsyncTMAStore(Operation *op, Value descValue,
                                    ValueRange coords) {
  OpBuilder builder(op);
  auto desc = getDescriptorInfo(descValue, builder);
  auto blockShape = cast<tt::TensorDescType>(descValue.getType()).getShape();

  auto offsets = castToI64(builder, op->getLoc(), coords);
  auto access = createTiledAccess(builder, op->getLoc(), desc, blockShape,
                                  offsets, std::nullopt);
  ExperimentalGSanTensorAccessOp::create(builder, op->getLoc(), access.first,
                                         access.second, /*isStore=*/true);
}

static void instrumentAsyncTMAReduce(ttng::AsyncTMAReduceOp op) {
  OpBuilder builder(op);
  auto desc = getDescriptorInfo(op.getDesc(), builder);
  auto blockShape = op.getDesc().getType().getShape();

  auto offsets = castToI64(builder, op.getLoc(), op.getCoord());
  auto access = createTiledAccess(builder, op.getLoc(), desc, blockShape,
                                  offsets, std::nullopt);
  ExperimentalGSanAtomicTensorAccessOp::create(
      builder, op.getLoc(), access.first, access.second, MemSemantic::RELAXED,
      MemSyncScope::GPU);
}

static void instrumentAtomicPoll(tt::AtomicPollOp op) {
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  ExperimentalGSanAtomicPollOp::create(builder, op.getLoc(), op.getPtr(),
                                       op.getResult(), op.getSem(),
                                       op.getScope());
  if (ttg::lookupNumCTAs(op) == 1)
    ttg::BarrierOp::create(builder, op.getLoc(), ttg::AddrSpace::Local);
  else
    ttng::ClusterBarrierOp::create(builder, op.getLoc());
}

static void instrumentAsyncTMAGather(ttng::AsyncTMAGatherOp op) {
  OpBuilder builder(op);
  auto desc = getDescriptorInfo(op.getDesc(), builder);

  auto access = createGatherScatterAccess(builder, op.getLoc(), desc,
                                          op.getResult().getType().getShape(),
                                          op.getXOffsets(), op.getYOffset());
  auto maskType = cast<RankedTensorType>(access.second.getType());
  Value predTensor =
      tt::SplatOp::create(builder, op.getLoc(), maskType, op.getPred());
  Value mask =
      arith::AndIOp::create(builder, op.getLoc(), access.second, predTensor);
  ExperimentalGSanTensorAccessOp::create(builder, op.getLoc(), access.first,
                                         mask, /*isStore=*/false);
}

static void instrumentAsyncTMAScatter(ttng::AsyncTMAScatterOp op) {
  OpBuilder builder(op);
  auto desc = getDescriptorInfo(op.getDesc(), builder);

  auto access = createGatherScatterAccess(builder, op.getLoc(), desc,
                                          op.getSrc().getType().getShape(),
                                          op.getXOffsets(), op.getYOffset());
  ExperimentalGSanTensorAccessOp::create(builder, op.getLoc(), access.first,
                                         access.second, /*isStore=*/true);
}

static Value getValueForOp(Operation *op, Value value) {
  auto partitions = op->getParentOfType<ttg::WarpSpecializePartitionsOp>();
  if (!partitions)
    return value;

  auto captures = partitions.getExplicitCaptures();
  auto capture = llvm::find(captures, value);
  unsigned captureIdx = std::distance(captures.begin(), capture);
  if (capture == captures.end()) {
    partitions->insertOperands(captureIdx, value);
    for (Region &region : partitions.getPartitionRegions())
      region.addArgument(value.getType(), op->getLoc());
  }

  Region *partitionRegion = op->getParentRegion();
  while (partitionRegion->getParentOp() != partitions.getOperation())
    partitionRegion = partitionRegion->getParentRegion();
  return partitionRegion->getArgument(captureIdx);
}

static Value getFuncArgumentWithAttr(tt::FuncOp func, StringRef attrName) {
  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    if (func.getArgAttr(i, attrName))
      return func.getArgument(i);
  }
  llvm_unreachable("missing attributed GSan function argument");
}

struct MBarrierArrivalInfo {
  Value barrier;
  Value pred;
  int32_t count;
  SmallVector<int32_t> multicastMasks;
  bool multicast;
  int32_t sourceBroadcastMask;
  bool publishClock;
};

static Value trueValue(OpBuilder &builder, Location loc) {
  return arith::ConstantIntOp::create(builder, loc, 1, 1);
}

static Value andPredicates(OpBuilder &builder, Location loc, Value lhs,
                           Value rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return arith::AndIOp::create(builder, loc, lhs, rhs);
}

static SmallVector<int32_t> getMMACompletionMulticastMasks(ValueRange descs,
                                                           bool twoCTAs) {
  SmallVector<int32_t> masks;
  for (uint16_t mask : ttng::getCTABroadcastMasks(twoCTAs, descs))
    masks.push_back(mask);
  return masks;
}

static SmallVector<MBarrierArrivalInfo>
getMBarrierArrivals(Operation *op, OpBuilder &builder) {
  SmallVector<MBarrierArrivalInfo> arrivals;
  Location loc = op->getLoc();
  int numCTAs = ttg::lookupNumCTAs(op);

  if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op)) {
    Value pred = arrive.getPred() ? arrive.getPred() : trueValue(builder, loc);
    SmallVector<int32_t> multicastMasks;
    bool multicast = false;
    int32_t sourceBroadcastMask = 0;
    if (arrive.isMulticast()) {
      multicastMasks.push_back(arrive.getMulticastCTA());
      multicast = true;
    } else if (std::optional<uint32_t> fromCTA = arrive.getFromCTA();
               fromCTA && *fromCTA != static_cast<uint32_t>(numCTAs - 1)) {
      uint32_t broadcastMask = ~*fromCTA & (numCTAs - 1);
      multicastMasks.push_back(broadcastMask);
      multicast = true;
      sourceBroadcastMask = broadcastMask;
    }
    arrivals.push_back({arrive.getBarrier(), pred,
                        static_cast<int32_t>(arrive.getCount()),
                        std::move(multicastMasks), multicast,
                        sourceBroadcastMask, /*publishClock=*/true});
    return arrivals;
  }

  if (auto expect = dyn_cast<ttng::BarrierExpectOp>(op)) {
    Value pred = expect.getPred();
    SmallVector<int32_t> multicastMasks;
    bool multicast = false;
    int32_t sourceBroadcastMask = 0;
    if (std::optional<uint32_t> fromCTA = expect.getFromCTA();
        fromCTA && *fromCTA != static_cast<uint32_t>(numCTAs - 1)) {
      uint32_t broadcastMask = ~*fromCTA & (numCTAs - 1);
      multicastMasks.push_back(broadcastMask);
      multicast = true;
      sourceBroadcastMask = broadcastMask;
    }
    arrivals.push_back({expect.getBarrier(), pred, /*count=*/1,
                        std::move(multicastMasks), multicast,
                        sourceBroadcastMask, /*publishClock=*/true});
    return arrivals;
  }

  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op)) {
    bool twoCTAs = mma.getTwoCtas();
    auto masks =
        getMMACompletionMulticastMasks(mma.getCompletionDescs(), twoCTAs);
    for (auto [barrier, barrierPred] : llvm::zip(
             mma.getCompletionBarriers(), mma.getCompletionBarrierPreds())) {
      Value pred = andPredicates(builder, loc, mma.getPredicate(), barrierPred);
      arrivals.push_back({barrier, pred, /*count=*/1, masks,
                          /*multicast=*/!masks.empty(),
                          /*sourceBroadcastMask=*/twoCTAs ? 1 : 0,
                          /*publishClock=*/false});
    }
    return arrivals;
  }

  if (auto commit = dyn_cast<ttng::TCGen5CommitOp>(op)) {
    bool twoCTAs = ttng::getModuleTwoCTAs(op);
    auto masks = getMMACompletionMulticastMasks(commit.getDescs(), twoCTAs);
    bool multicast = !masks.empty();
    Value pred = commit.getPred() ? commit.getPred() : trueValue(builder, loc);
    arrivals.push_back({commit.getBarrier(), pred, /*count=*/1,
                        std::move(masks), multicast,
                        /*sourceBroadcastMask=*/twoCTAs ? 1 : 0,
                        /*publishClock=*/false});
  }
  return arrivals;
}

static int64_t getMBarrierCapacity(ModuleOp module) {
  // Upper bound to the number of simultaneously active mbarriers
  int64_t capacity = 0;
  module.walk([&](ttng::InitBarrierOp op) {
    capacity += op.getAlloc().getType().getNumElements();
  });
  return capacity;
}

static void
instrumentMBarrierOps(ModuleOp module,
                      const DenseMap<tt::FuncOp, Value> &scratchMap) {
  SmallVector<Operation *> ops;
  module.walk([&](Operation *op) { ops.push_back(op); });
  for (Operation *op : ops) {
    auto func = op->getParentOfType<tt::FuncOp>();
    auto scratchIt = scratchMap.find(func);
    if (scratchIt == scratchMap.end())
      continue;
    Value scratch = getValueForOp(op, scratchIt->second);

    if (auto init = dyn_cast<ttng::InitBarrierOp>(op)) {
      OpBuilder builder(op);
      builder.setInsertionPointAfter(op);
      auto barrierTy = init.getAlloc().getType();
      int32_t expectedCount =
          static_cast<int32_t>(init.getCount() * ttg::lookupNumCTAs(op) /
                               barrierTy.getNumElements());
      ExperimentalGSanMBarrierInitOp::create(builder, op->getLoc(), scratch,
                                             init.getBarrier(), expectedCount);
      continue;
    }

    if (auto wait = dyn_cast<ttng::WaitBarrierOp>(op)) {
      OpBuilder builder(op);
      builder.setInsertionPointAfter(op);
      Value pred =
          wait.getPred() ? wait.getPred() : trueValue(builder, op->getLoc());
      ExperimentalGSanMBarrierWaitOp::create(builder, op->getLoc(), scratch,
                                             wait.getBarrier(), wait.getPhase(),
                                             pred);
      ttg::BarrierOp::create(builder, op->getLoc(), ttg::AddrSpace::Local);
      continue;
    }

    OpBuilder builder(op);
    auto arrivals = getMBarrierArrivals(op, builder);
    if (arrivals.empty())
      continue;
    for (const MBarrierArrivalInfo &arrival : arrivals) {
      auto masks = builder.getDenseI32ArrayAttr(arrival.multicastMasks);
      ExperimentalGSanMBarrierArriveOp::create(
          builder, op->getLoc(), scratch, arrival.barrier, arrival.pred,
          arrival.count, masks, arrival.multicast, arrival.sourceBroadcastMask,
          arrival.publishClock);
    }
    if (llvm::any_of(arrivals, [](const MBarrierArrivalInfo &arrival) {
          return arrival.publishClock;
        })) {
      // Publish before the hardware arrival and hold every thread in the
      // current partition until the elected thread has finished writing the
      // release. Completion-only signals do not need this synchronization.
      ttg::BarrierOp::create(builder, op->getLoc(), ttg::AddrSpace::Local);
    }
  }
}

struct ClusterSyncPoint {
  Operation *op;
  bool before;
  bool materializeBarrier;
};

static bool atomicResultNeedsClusterBarrier(Operation *op) {
  if (op->getResult(0).use_empty())
    return false;
  auto tensorTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!tensorTy)
    return true;
  auto kBlock = StringAttr::get(op->getContext(), "block");
  return ttg::toLinearLayout(tensorTy).getFreeVariableMasks().lookup(kBlock);
}

static bool hasAcquireSemantics(MemSemantic sem) {
  return sem == MemSemantic::ACQUIRE || sem == MemSemantic::ACQUIRE_RELEASE;
}

static bool hasReleaseSemantics(MemSemantic sem) {
  return sem == MemSemantic::RELEASE || sem == MemSemantic::ACQUIRE_RELEASE;
}

static void instrumentClusterBarrierEquivalents(ModuleOp module) {
  DenseMap<Region *, SmallVector<ClusterSyncPoint>> pointsByGroup;
  SmallVector<Region *> groups;
  auto addPoint = [&](Operation *op, bool before, bool materializeBarrier) {
    Region *group = getClusterBarrierGroupRegion(op);
    auto [it, inserted] = pointsByGroup.try_emplace(group);
    if (inserted)
      groups.push_back(group);
    it->second.push_back({op, before, materializeBarrier});
  };

  SmallVector<Operation *> ops;
  module.walk([&](Operation *op) { ops.push_back(op); });
  for (Operation *op : ops) {
    if (auto barrier = dyn_cast<ttng::ClusterBarrierOp>(op)) {
      if (!barrier.getRelaxed())
        addPoint(op, /*before=*/false, /*materializeBarrier=*/false);
      continue;
    }

    if (isa<tt::AtomicPollOp>(op)) {
      // instrumentAtomicPoll materializes the poll's post-operation cluster
      // barrier before this scan. Instrument that barrier instead.
      continue;
    }

    if (auto atomic = dyn_cast<tt::AtomicOpInterface>(op)) {
      if (ttg::lookupNumCTAs(op) == 1)
        continue;
      auto sem = atomic.getMemSemantic();
      if (hasReleaseSemantics(sem))
        addPoint(op, /*before=*/true, /*materializeBarrier=*/true);
      if ((hasAcquireSemantics(sem) && !op->hasAttr("allocation.offset")) ||
          atomicResultNeedsClusterBarrier(op)) {
        addPoint(op, /*before=*/false, /*materializeBarrier=*/true);
      }
      continue;
    }

    if (ttng::needsClusterBarrier(op))
      addPoint(op, /*before=*/false, /*materializeBarrier=*/true);
  }

  for (Region *group : groups) {
    auto &points = pointsByGroup[group];
    Location loc = points.front().op->getLoc();
    Block &entry = group->front();
    OpBuilder initBuilder(&entry, entry.begin());
    Type ptrTy = tt::PointerType::get(initBuilder.getI8Type(), 1);
    Value scratch = createThirdPartyScratchAlloc(
        initBuilder, loc, ptrTy, kGSanClusterBarrierScratchBytes,
        /*alignment=*/16, /*sharedClusterState=*/true);
    ExperimentalGSanClusterBarrierInitOp::create(initBuilder, loc, scratch);
    ttng::ClusterBarrierOp::create(initBuilder, loc, /*relaxed=*/true);

    for (const ClusterSyncPoint &point : points) {
      OpBuilder syncBuilder(point.op);
      if (!point.before)
        syncBuilder.setInsertionPointAfter(point.op);
      if (point.materializeBarrier)
        ttng::ClusterBarrierOp::create(syncBuilder, point.op->getLoc());
      ExperimentalGSanClusterBarrierSyncOp::create(syncBuilder,
                                                   point.op->getLoc(), scratch);
      ttng::ClusterBarrierOp::create(syncBuilder, point.op->getLoc());
    }
  }
}

class GlobalSanitizerPass
    : public impl::TritonInstrumentGlobalSanitizerBase<GlobalSanitizerPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module);
    Type gsanStatePtrTy = tt::PointerType::get(builder.getI8Type(), 1);
    Type streamClockPtrTy = tt::PointerType::get(builder.getI32Type(), 1);
    Type kernelIdTy = builder.getI64Type();
    auto launchPdl = module->getAttrOfType<IntegerAttr>("tti.gsan_launch_pdl");
    bool acquireStreamClock = !launchPdl || launchPdl.getInt() == 0;
    DenseSet<StringRef> calledFuncs;
    module.walk(
        [&](tt::CallOp callOp) { calledFuncs.insert(callOp.getCallee()); });
    int64_t mBarrierCapacity =
        ttg::lookupNumCTAs(module) > 1 ? getMBarrierCapacity(module) : 0;
    const bool instrumentMBarriers = mBarrierCapacity > 0;
    DenseMap<tt::FuncOp, Value> mbarrierScratch;

    SmallVector<tt::FuncOp> funcs;
    module.walk([&](tt::FuncOp func) { funcs.push_back(func); });
    for (tt::FuncOp func : funcs) {
      bool isEntry = !calledFuncs.contains(func.getSymName());
      auto funcTy = func.getFunctionType();
      SmallVector<Type> inputTys(funcTy.getInputs().begin(),
                                 funcTy.getInputs().end());
      if (instrumentMBarriers && !isEntry)
        inputTys.push_back(gsanStatePtrTy);
      inputTys.push_back(gsanStatePtrTy);
      inputTys.push_back(streamClockPtrTy);
      inputTys.push_back(kernelIdTy);
      func.setType(FunctionType::get(module.getContext(), inputTys,
                                     funcTy.getResults()));

      SmallVector<std::pair<BlockArgument, StringRef>> hiddenArgs;
      auto addHiddenArg = [&](Type type, StringRef attrName) {
        BlockArgument arg = func.getBody().addArgument(type, func.getLoc());
        hiddenArgs.emplace_back(arg, attrName);
      };
      if (instrumentMBarriers && !isEntry)
        addHiddenArg(gsanStatePtrTy, kGSanMBarrierScratchArgAttr);
      addHiddenArg(gsanStatePtrTy, kGSanGlobalStateArgAttr);
      addHiddenArg(streamClockPtrTy, kGSanStreamClockArgAttr);
      addHiddenArg(kernelIdTy, kGSanKernelIdArgAttr);
      SmallVector<Attribute> newArgAttrs;
      if (auto argAttrs = func.getAllArgAttrs())
        newArgAttrs.append(argAttrs.begin(), argAttrs.end());
      while (newArgAttrs.size() < func.getNumArguments()) {
        newArgAttrs.push_back(DictionaryAttr::get(module.getContext()));
      }
      if (!newArgAttrs.empty())
        func.setAllArgAttrs(newArgAttrs);
      for (auto [arg, attrName] : hiddenArgs)
        func.setArgAttr(arg.getArgNumber(), attrName, builder.getUnitAttr());
      if (instrumentMBarriers && !isEntry) {
        mbarrierScratch.try_emplace(
            func, getFuncArgumentWithAttr(func, kGSanMBarrierScratchArgAttr));
      }

      if (isEntry) {
        OpBuilder b(&func.front(), func.front().begin());
        ExperimentalGSanInitOp::create(b, func.getLoc(), acquireStreamClock);
        if (instrumentMBarriers) {
          int64_t scratchBytes = kGSanMBarrierTableHeaderBytes +
                                 mBarrierCapacity * kGSanMBarrierRecordBytes;
          Value scratch = createThirdPartyScratchAlloc(
              b, func.getLoc(), gsanStatePtrTy, scratchBytes,
              /*alignment=*/16, /*sharedClusterState=*/true);
          ExperimentalGSanMBarrierTableInitOp::create(b, func.getLoc(), scratch,
                                                      mBarrierCapacity);
          ttng::ClusterBarrierOp::create(b, func.getLoc(), /*relaxed=*/true);
          mbarrierScratch.try_emplace(func, scratch);
        }
        func.walk([&](tt::ReturnOp returnOp) {
          OpBuilder returnBuilder(returnOp);
          ExperimentalGSanKernelExitOp::create(returnBuilder,
                                               returnOp.getLoc());
        });
      }
    }

    SmallVector<tt::CallOp> callOps;
    module.walk([&](tt::CallOp op) { callOps.push_back(op); });
    for (tt::CallOp callOp : callOps) {
      auto caller = callOp->getParentOfType<tt::FuncOp>();
      assert(caller && caller.getNumArguments() >= 3 &&
             "expected triton.call to be nested under a Triton function");

      SmallVector<Value> operands(callOp.getOperands().begin(),
                                  callOp.getOperands().end());
      Value gsanState =
          getFuncArgumentWithAttr(caller, kGSanGlobalStateArgAttr);
      Value streamClock =
          getFuncArgumentWithAttr(caller, kGSanStreamClockArgAttr);
      Value kernelId = getFuncArgumentWithAttr(caller, kGSanKernelIdArgAttr);
      if (instrumentMBarriers) {
        auto scratchIt = mbarrierScratch.find(caller);
        assert(scratchIt != mbarrierScratch.end() &&
               "missing GSan mbarrier scratch for caller");
        operands.push_back(
            getValueForOp(callOp.getOperation(), scratchIt->second));
      }
      operands.push_back(getValueForOp(callOp.getOperation(), gsanState));
      operands.push_back(getValueForOp(callOp.getOperation(), streamClock));
      operands.push_back(getValueForOp(callOp.getOperation(), kernelId));

      OpBuilder b(callOp);
      auto newCallOp =
          tt::CallOp::create(b, callOp.getLoc(), callOp.getCallee(),
                             callOp.getResultTypes(), operands);
      newCallOp->setAttrs(callOp->getAttrs());
      callOp->replaceAllUsesWith(newCallOp->getResults());
      callOp.erase();
    }

    module.walk([&](Operation *op) {
      IRRewriter b(op);
      mlir::TypeSwitch<Operation *>(op)
          .Case([&](tt::LoadOp op) {
            ExperimentalGSanTensorAccessOp::create(
                b, op.getLoc(), op.getPtr(), op.getMask(), /*isStore=*/false);
          })
          .Case([&](tt::StoreOp op) {
            ExperimentalGSanTensorAccessOp::create(
                b, op.getLoc(), op.getPtr(), op.getMask(), /*isStore=*/true);
          })
          .Case([&](ttg::AsyncCopyGlobalToLocalOp op) {
            ExperimentalGSanTensorAccessOp::create(
                b, op.getLoc(), op.getSrc(), op.getMask(), /*isStore=*/false);
          })
          .Case([&](ttng::AsyncTMACopyGlobalToLocalOp op) {
            instrumentAsyncTMALoad(op);
          })
          .Case(
              [&](ttng::AsyncTMAGatherOp op) { instrumentAsyncTMAGather(op); })
          .Case([&](ttng::AsyncTMACopyLocalToGlobalOp op) {
            instrumentAsyncTMAStore(op, op.getDesc(), op.getCoord());
          })
          .Case(
              [&](ttng::AsyncTMAReduceOp op) { instrumentAsyncTMAReduce(op); })
          .Case([&](ttng::AsyncTMAScatterOp op) {
            instrumentAsyncTMAScatter(op);
          })
          .Case([&](tt::AtomicRMWOp op) {
            auto newOp = ExperimentalGSanAtomicRMWOp::create(
                b, op.getLoc(), op.getType(), op.getAtomicRmwOp(), op.getPtr(),
                op.getVal(), op.getMask(), op.getSem(), op.getScope());
            newOp->setAttrs(op->getAttrs());
            b.replaceOp(op, newOp);
          })
          .Case([&](tt::AtomicCASOp op) {
            auto newOp = ExperimentalGSanAtomicCASOp::create(
                b, op.getLoc(), op.getType(), op.getPtr(), op.getCmp(),
                op.getVal(), op.getSem(), op.getScope());
            newOp->setAttrs(op->getAttrs());
            b.replaceOp(op, newOp);
          })
          .Case([&](tt::AtomicPollOp op) { instrumentAtomicPoll(op); })
          .Case([&](tt::GridDependencyWaitOp op) {
            b.setInsertionPointAfter(op);
            ExperimentalGSanGridDependencyWaitOp::create(b, op.getLoc());
          })
          .Case([&](ttg::WarpSpecializeOp op) {
            op->setAttr(kDisableSetMaxRegisterAttr, builder.getUnitAttr());
          });
    });

    if (instrumentMBarriers)
      instrumentMBarrierOps(module, mbarrierScratch);
    instrumentClusterBarrierEquivalents(module);
  }
};

} // namespace

} // namespace mlir::triton::instrument
