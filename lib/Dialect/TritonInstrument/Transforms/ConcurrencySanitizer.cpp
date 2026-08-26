#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/FunctionBuilder.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonInstrument/Transforms/ConSanTargetHooks.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/Sys/GetEnv.h"

namespace mlir {
namespace triton {
namespace instrument {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tti = mlir::triton::instrument;

#define GEN_PASS_DEF_TRITONINSTRUMENTCONCURRENCYSANITIZER
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

static llvm::StringMap<ConSanHooksFactory> &getHooksRegistry() {
  static llvm::StringMap<ConSanHooksFactory> registry;
  return registry;
}

void registerConSanHooks(llvm::StringRef key, ConSanHooksFactory factory) {
  getHooksRegistry()[key] = std::move(factory);
}

std::unique_ptr<ConSanTargetHooks> createConSanHooks(llvm::StringRef key) {
  auto it = getHooksRegistry().find(key);
  if (it != getHooksRegistry().end())
    return it->second();
  return nullptr;
}

namespace {

// OpBuilder listener tracking operations added to the builder to be wrapped
// with a lock acquire/release pair.
class CriticalSectionListener : public ImplicitLocOpBuilder::Listener {
public:
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint /*previous*/) override {
    if (firstOp == nullptr) {
      firstOp = op;
    }
    lastOp = op;
  }
  void maybeWrapWithCriticalSection(ImplicitLocOpBuilder &b,
                                    AuxDataMap &auxData, Value pred) {
    Operation *_firstOp = firstOp;
    Operation *_lastOp = lastOp;
    if (firstOp != nullptr && lastOp != nullptr) {
      assert(firstOp->getParentRegion() == lastOp->getParentRegion());
      b.setInsertionPoint(_firstOp);
      tti::ExperimentalLockAcquireOp::create(b, auxData.lock.at(_firstOp).value,
                                             pred);
      b.setInsertionPointAfter(_lastOp);
      tti::ExperimentalLockReleaseOp::create(b, auxData.lock.at(_firstOp).value,
                                             pred);
    }
  }

private:
  Operation *firstOp = nullptr;
  Operation *lastOp = nullptr;
};

bool isTensorCoreOp(Operation *op) {
  return isa<ttng::MMAv5OpInterface, ttng::TCGen5CommitOp, ttng::TMEMCopyOp>(
      op);
}

std::optional<int> maybeGetPartitionIdx(Operation *op) {
  Operation *parent = op->getParentOp();
  if (!parent)
    return std::nullopt;
  if (isa<ttg::WarpSpecializePartitionsOp>(parent))
    return op->getParentRegion()->getRegionNumber();
  return maybeGetPartitionIdx(parent);
}

int getCurrentThread(Operation *op, const ConSanTargetHooks &hooks,
                     const AuxDataMap::ThreadLayout &threadLayout) {
  // Default partition is 0, other partitions are idx + 1
  int thread = maybeGetPartitionIdx(op).value_or(-1) + 1;
  if (hooks.isTMAOp(op)) {
    assert(threadLayout.hasTMAThreads() &&
           "TMA thread class must exist when instrumenting a TMA op");
    thread += threadLayout.tmaThreadOffset;
    return thread;
  }
  if (isTensorCoreOp(op)) {
    assert(threadLayout.hasTCThreads() &&
           "TC thread class must exist when instrumenting a tensor-core op");
    thread += threadLayout.tcThreadOffset;
    return thread;
  }
  if (hooks.isCLCOp(op)) {
    assert(threadLayout.hasCLCThreads() &&
           "CLC thread class must exist when instrumenting a CLC op");
    thread += threadLayout.clcThreadOffset;
    return thread;
  }
  return thread;
}

int getBaseThread(int thread, const AuxDataMap::ThreadLayout &threadLayout) {
  return thread % threadLayout.numBaseThreads;
}

// Peer threads are the equivalent threads in the TMA, TC, CLC and normal
// thread classes.
// If a thread is a base thread, return the mask with the peers, otherwise
// return the mask with the thread itself.
uint64_t getThreadPeersMask(int thread,
                            const AuxDataMap::ThreadLayout &threadLayout) {
  uint64_t mask = 1ULL << thread;
  if (thread < threadLayout.numBaseThreads) {
    if (threadLayout.hasTMAThreads())
      mask |= 1ULL << (thread + threadLayout.tmaThreadOffset);
    if (threadLayout.hasTCThreads())
      mask |= 1ULL << (thread + threadLayout.tcThreadOffset);
    if (threadLayout.hasCLCThreads())
      mask |= 1ULL << (thread + threadLayout.clcThreadOffset);
  }
  return mask;
}

int getActiveMask(ttg::WarpSpecializeOp wsOp) {
  int activeMask = 1;
  for (Region *region : wsOp.getNonEmptyPartitionRegions())
    activeMask |= 1 << (region->getRegionNumber() + 1);
  return activeMask;
}

Value currentCTAMask(ImplicitLocOpBuilder &b) {
  Value ctaId = tti::ExperimentalClusterCTAIdOp::create(b, b.getLoc());
  return arith::ShLIOp::create(b, arith::ConstantIntOp::create(b, 1, 32),
                               ctaId);
}

Value createStateMaskConstant(ImplicitLocOpBuilder &b,
                              RankedTensorType maskType,
                              const llvm::SmallBitVector &mask) {
  assert(mask.size() <= static_cast<size_t>(maskType.getNumElements()));
  SmallVector<APInt> bits(maskType.getNumElements(), APInt(1, 0));
  for (unsigned bit = 0; bit < mask.size(); ++bit)
    if (mask.test(bit))
      bits[bit] = APInt(1, 1);
  return arith::ConstantOp::create(b, maskType,
                                   DenseElementsAttr::get(maskType, bits));
}

FailureOr<Value> createBufferStateMask(ImplicitLocOpBuilder &b,
                                       AuxDataMap &auxData, MemType memType,
                                       Value buffer, Value runtimeBase,
                                       const BufferStateCandidates &candidates,
                                       Operation *op,
                                       SmallVectorImpl<Value> &predicates) {
  int index = static_cast<int>(memType);
  const BufferStatePlan &plan = auxData.bufferStatePlans[index];
  if (plan.numLanes == 0 || auxData.writeVisibility[index].empty()) {
    op->emitError("active memdesc has no ConSan state plan");
    return failure();
  }
  if (!candidates.unknown && candidates.cases.empty()) {
    op->emitError("active memdesc has no buffer-state candidates");
    return failure();
  }

  auto writeVisibilityType =
      cast<RankedTensorType>(auxData.writeVisibility[index].at(op).type);
  RankedTensorType maskType =
      tti::getSlicedTensorType(writeVisibilityType, {1}, b.getI1Type());

  if (candidates.unknown)
    return createStateMaskConstant(b, maskType, plan.unknownMask);

  if (candidates.cases.size() > 1) {
    if (!runtimeBase) {
      op->emitError("dynamic buffer-state cases require a runtime base");
      return failure();
    }
    Value resolved = arith::ConstantIntOp::create(b, 0, 1);
    auto &candidateMap = auxData.bufferCandidates[index];
    for (const BufferStateCandidate &candidate : candidates.cases) {
      Value base = tti::ExperimentalMemoryOffsetToI32Op::create(
          b, candidate.baseOffset, memType);
      Value predicate =
          arith::CmpIOp::create(b, arith::CmpIPredicate::eq, runtimeBase, base);
      Value selectedValue = buffer;
      while (auto select = selectedValue.getDefiningOp<arith::SelectOp>()) {
        auto matchesArm = [&](Value arm) {
          auto it = candidateMap.find(arm);
          return it == candidateMap.end() || it->second.unknown ||
                 llvm::any_of(
                     it->second.cases,
                     [&](const BufferStateCandidate &armCandidate) {
                       return armCandidate.baseOffset == candidate.baseOffset &&
                              armCandidate.ctaMask == candidate.ctaMask &&
                              armCandidate.mask.anyCommon(candidate.mask);
                     });
        };
        bool selectedOnTrue = matchesArm(select.getTrueValue());
        bool selectedOnFalse = matchesArm(select.getFalseValue());
        if (selectedOnTrue == selectedOnFalse)
          break;
        Value selected = select.getCondition();
        if (!selectedOnTrue)
          selected = arith::XOrIOp::create(
              b, selected, arith::ConstantIntOp::create(b, 1, 1));
        predicate = arith::AndIOp::create(b, predicate, selected);
        selectedValue =
            selectedOnTrue ? select.getTrueValue() : select.getFalseValue();
      }
      predicates.push_back(predicate);
      resolved = arith::OrIOp::create(b, resolved, predicate);
    }
    tti::createAssertInThread(
        b, resolved,
        "internal ConSan error: active memdesc resolved to no buffer state");
  }

  if (candidates.cases.size() == 1)
    return createStateMaskConstant(b, maskType, candidates.cases.front().mask);

  Value result =
      createStateMaskConstant(b, maskType, llvm::SmallBitVector(plan.numLanes));
  for (auto [state, predicate] : llvm::zip(candidates.cases, predicates)) {
    Value candidate = createStateMaskConstant(b, maskType, state.mask);
    Value predicateTensor = tt::SplatOp::create(b, maskType, predicate);
    candidate = arith::AndIOp::create(b, candidate, predicateTensor);
    result = arith::OrIOp::create(b, result, candidate);
  }
  return result;
}

Value allCTAsMask(ImplicitLocOpBuilder &b) {
  int numCTAs = ttg::lookupNumCTAs(b);
  assert(numCTAs <= 16 && "ConSan CTA bitsets assume at most 16 CTAs");
  return arith::ConstantIntOp::create(b, (1u << numCTAs) - 1, 32);
}

bool shouldInitializeAllocations() {
  std::string envValue = tt::tools::getStrEnv("TRITON_CONSAN_INIT_ALLOCATIONS");
  if (envValue.empty())
    return true;
  if (auto enabled = tt::tools::isEnvValueBool(envValue))
    return *enabled;
  llvm::report_fatal_error("TRITON_CONSAN_INIT_ALLOCATIONS must be a boolean");
}

llvm::APInt getIntegerNaNPattern(unsigned bitWidth) {
  switch (bitWidth) {
  case 16:
    // 0x7FC0 is a NaN in both bfloat16 and float16 interpretations.
    return llvm::APInt(16, 0x7FC0);
  case 32:
    return llvm::APFloat::getNaN(llvm::APFloat::IEEEsingle()).bitcastToAPInt();
  case 64:
    return llvm::APFloat::getNaN(llvm::APFloat::IEEEdouble()).bitcastToAPInt();
  default:
    return llvm::APInt::getAllOnes(bitWidth);
  }
}

Value createPoisonTensor(ImplicitLocOpBuilder &b,
                         ttg::MemDescType memDescType) {
  auto region = b.getInsertionBlock()->getParent();
  Type elementType = memDescType.getElementType();
  RankedTensorType poisonType;
  if (isa<ttng::TensorMemorySpaceAttr>(memDescType.getMemorySpace())) {
    auto encoding = ttng::getDefaultLayoutForTmemLdSt(
        memDescType, ttg::lookupNumWarps(region));
    poisonType =
        RankedTensorType::get(memDescType.getShape(), elementType, encoding);
  } else {
    auto encoding = ttg::getDefaultBlockedEncoding(
        b.getContext(), memDescType.getShape(), ttg::lookupNumWarps(region),
        ttg::lookupThreadsPerWarp(b), ttg::lookupNumCTAs(b));
    encoding = ttg::BlockedEncodingAttr::get(
        b.getContext(), encoding.getSizePerThread(),
        encoding.getThreadsPerWarp(), encoding.getWarpsPerCTA(),
        encoding.getOrder(), ttg::getCGALayout(memDescType.getEncoding()));
    poisonType =
        RankedTensorType::get(memDescType.getShape(), elementType, encoding);
  }

  DenseElementsAttr poison;
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    poison = DenseElementsAttr::get(
        poisonType, llvm::APFloat::getNaN(floatType.getFloatSemantics()));
  } else if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    poison = DenseElementsAttr::get(
        poisonType, getIntegerNaNPattern(integerType.getWidth()));
  } else {
    llvm::report_fatal_error(
        "ConSan allocation initialization expects integer or float elements");
  }
  return arith::ConstantOp::create(b, b.getLoc(), poisonType, poison);
}

Value createSingleBufferView(ImplicitLocOpBuilder &b, Value alloc,
                             int64_t buffer) {
  auto allocType = cast<ttg::MemDescType>(alloc.getType());
  SmallVector<int64_t> shape(allocType.getShape().begin() + 1,
                             allocType.getShape().end());
  auto viewType = ttg::MemDescType::get(
      shape, allocType.getElementType(), allocType.getEncoding(),
      allocType.getMemorySpace(), allocType.getMutableMemory());
  Value index = arith::ConstantIntOp::create(b, buffer, 32);
  return ttg::MemDescIndexOp::create(b, b.getLoc(), viewType, alloc, index);
}

void initializeAllocation(ImplicitLocOpBuilder &b, Value alloc) {
  auto allocType = cast<ttg::MemDescType>(alloc.getType());
  SmallVector<Value> leaves;
  unsigned storeRank = allocType.getRank();
  if (isa<ttng::TensorMemorySpaceAttr>(allocType.getMemorySpace())) {
    storeRank = 2;
  } else {
    auto encoding = dyn_cast<ttg::LayoutEncodingTrait>(allocType.getEncoding());
    assert(encoding && "shared allocation must have a layout encoding");
    storeRank = encoding.getRank();
  }

  if (allocType.getRank() == storeRank) {
    leaves.push_back(alloc);
  } else {
    assert(allocType.getRank() == storeRank + 1 &&
           "only single-dimension multibuffer allocations are supported");
    for (int64_t buffer = 0; buffer < allocType.getDimSize(0); ++buffer)
      leaves.push_back(createSingleBufferView(b, alloc, buffer));
  }

  bool isTensorMemory =
      isa<ttng::TensorMemorySpaceAttr>(allocType.getMemorySpace());
  ttg::AddrSpace barrierSpace =
      isTensorMemory
          ? (ttg::AddrSpace::TensorRead | ttg::AddrSpace::TensorWrite)
          : ttg::AddrSpace::Local;
  // Synchronize warps, so in case of re-used memory we won't start poisoning
  // memory that is still being used, and finish poisoning before the kernel's
  // first real use of the allocation.
  ttg::BarrierOp::create(b, b.getLoc(), barrierSpace);
  for (Value leaf : leaves) {
    auto leafType = cast<ttg::MemDescType>(leaf.getType());
    Value poison = createPoisonTensor(b, leafType);
    if (isTensorMemory) {
      Value pred = arith::ConstantIntOp::create(b, 1, 1);
      ttng::TMEMStoreOp::create(b, leaf, poison, pred);
    } else {
      ttg::LocalStoreOp::create(b, poison, leaf);
    }
  }
  ttg::BarrierOp::create(b, b.getLoc(), barrierSpace);
}

bool canInitializeAllocation(Value alloc) {
  auto allocType = cast<ttg::MemDescType>(alloc.getType());
  if (!isa<ttng::TensorMemorySpaceAttr>(allocType.getMemorySpace()))
    return true;
  unsigned numWarps = ttg::lookupNumWarps(alloc.getDefiningOp());
  return numWarps % 4 == 0;
}

uint16_t getBlockBroadcastMask(Value alloc) {
  auto allocTy = cast<ttg::MemDescType>(alloc.getType());
  auto kBlock = StringAttr::get(alloc.getContext(), "block");
  return toLinearLayout(allocTy).getFreeVariableMasks().lookup(kBlock);
}

Value createCTABitset(ImplicitLocOpBuilder &b, uint32_t pattern,
                      uint32_t baseMask) {
  // Create a CTA bitset by shifting `pattern` by the non-broadcast CTA bits of
  // the current CTA.
  Value ctaId = tti::ExperimentalClusterCTAIdOp::create(b, b.getLoc());
  Value base = arith::AndIOp::create(
      b, ctaId, arith::ConstantIntOp::create(b, baseMask, 32));
  return arith::ShLIOp::create(b, arith::ConstantIntOp::create(b, pattern, 32),
                               base);
}

Value getMulticastRecipientCTAs(ImplicitLocOpBuilder &b, Value alloc) {
  // Return the CTA rows touched by an alloc: current CTA for
  // non-broadcast allocs, or all CTAs in the current multicast group.
  uint16_t broadcastMask = getBlockBroadcastMask(alloc);
  if (!broadcastMask)
    return currentCTAMask(b);
  int numCTAs = ttg::lookupNumCTAs(b);
  auto encoding = ttng::getTMAMulticastMaskEncoding(numCTAs, broadcastMask);
  return createCTABitset(b, encoding.pattern, encoding.fixedBits);
}

Value getLeaderCTA(ImplicitLocOpBuilder &b, Value barrier) {
  uint16_t broadcastMask = getBlockBroadcastMask(barrier);
  if (!broadcastMask)
    return currentCTAMask(b);
  int numCTAs = ttg::lookupNumCTAs(b);
  auto encoding = ttng::getTMAMulticastMaskEncoding(numCTAs, broadcastMask);
  return createCTABitset(b, /*pattern=*/1, encoding.fixedBits);
}

Value getAsyncSharedStoreBarrierRecipientCTAs(ImplicitLocOpBuilder &b,
                                              Value barrier,
                                              Value recipientCTAs) {
  uint16_t broadcastMask = getBlockBroadcastMask(barrier);
  if (!broadcastMask)
    return recipientCTAs;

  int numCTAs = ttg::lookupNumCTAs(b);
  uint32_t leaderMask =
      ttng::getTMAMulticastMaskEncoding(numCTAs, ~broadcastMask).pattern;
  for (int bit = 1; bit < numCTAs; bit <<= 1) {
    if (!(broadcastMask & bit))
      continue;
    Value shifted = b.createOrFold<arith::ShRUIOp>(
        recipientCTAs, arith::ConstantIntOp::create(b, bit, 32));
    recipientCTAs = b.createOrFold<arith::OrIOp>(recipientCTAs, shifted);
  }
  return b.createOrFold<arith::AndIOp>(
      recipientCTAs, arith::ConstantIntOp::create(b, leaderMask, 32));
}

Value getMulticastBarrierRecipientCTAs(ImplicitLocOpBuilder &b, Value result,
                                       Value barrier) {
  uint32_t resultBroadcastMask = getBlockBroadcastMask(result);
  uint32_t barrierBroadcastMask = getBlockBroadcastMask(barrier);
  int numCTAs = ttg::lookupNumCTAs(b);
  uint32_t recipientBroadcastMask =
      resultBroadcastMask & ~barrierBroadcastMask & (numCTAs - 1);
  auto encoding =
      ttng::getTMAMulticastMaskEncoding(numCTAs, recipientBroadcastMask);
  uint32_t baseMask =
      ~(resultBroadcastMask | barrierBroadcastMask) & (numCTAs - 1);
  return createCTABitset(b, encoding.pattern, baseMask);
}

Value getRecipientCTAsForBroadcastMasks(ImplicitLocOpBuilder &b,
                                        ArrayRef<uint16_t> broadcastMasks) {
  if (broadcastMasks.empty())
    return currentCTAMask(b);

  int numCTAs = ttg::lookupNumCTAs(b);
  Value ctaId = tti::ExperimentalClusterCTAIdOp::create(b, b.getLoc());
  Value recipientCTAs = arith::ConstantIntOp::create(b, 0, 32);
  // Match eager tcgen05_commit lowering in
  // DotOpToLLVM/MMAv5.cpp:createMMACommit: build one concrete recipient bitset
  // per descriptor, then OR those bitsets.
  for (uint16_t broadcastBits : broadcastMasks) {
    // Compute the map that goes from cta_id to lead_cta_id (fixedBits)
    // and the pattern that goes from cta_0 to its multicast group (pattern).
    auto encoding = ttng::getTMAMulticastMaskEncoding(numCTAs, broadcastBits);
    Value fixedBitsVal =
        arith::ConstantIntOp::create(b, encoding.fixedBits, 32);
    Value base = arith::AndIOp::create(b, ctaId, fixedBitsVal);
    Value patternVal = arith::ConstantIntOp::create(b, encoding.pattern, 32);
    Value descRecipientCTAs = arith::ShLIOp::create(b, patternVal, base);
    recipientCTAs = arith::OrIOp::create(b, recipientCTAs, descRecipientCTAs);
  }
  return recipientCTAs;
}

SmallVector<uint16_t> getTensorCoreBarrierBroadcastMasks(Operation *op) {
  assert(isTensorCoreOp(op) && "expected a tensor-core op");
  bool twoCTAs = ttng::getModuleTwoCTAs(op);
  SmallVector<Value> commitDescs;
  if (auto commitOp = dyn_cast<ttng::TCGen5CommitOp>(op)) {
    llvm::append_range(commitDescs, commitOp.getDescs());
  } else if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(op)) {
    commitDescs = mmaOp.getCompletionDescs();
  } else if (isa<ttng::TMEMCopyOp>(op)) {
    // TMEMCopy does not have descs (empty)
  } else {
    llvm_unreachable("unknown tensor-core op");
  }
  return ttng::getCTABroadcastMasks(twoCTAs, commitDescs);
}

void extendXorSpan(uint32_t &span, uint32_t basis, int numCTAs) {
  uint32_t oldSpan = span;
  for (int value = 0; value < numCTAs; ++value) {
    if (!(oldSpan & (1u << value)))
      continue;
    uint32_t extended = value ^ basis;
    assert(extended < static_cast<uint32_t>(numCTAs) &&
           "CTA basis exceeds the cluster size");
    span |= 1u << extended;
  }
}

LinearLayout getLocalLoadStoreConversion(ttg::MemDescType memDescTy,
                                         RankedTensorType regTy) {
  return invertAndComposeBlockLocal(
      ttg::toLinearLayoutIgnoringPadding(memDescTy),
      ttg::toLinearLayout(regTy));
}

LinearLayout getLocalGatherScatterConversion(ttg::MemDescType memDescTy,
                                             RankedTensorType regTy,
                                             unsigned axis) {
  MLIRContext *ctx = memDescTy.getContext();
  LinearLayout sharedLayout = ttg::toLinearLayoutIgnoringPadding(memDescTy);
  SmallVector<StringAttr> allDims =
      standardOutDimNames(ctx, memDescTy.getRank());
  StringAttr axisDim = allDims[axis];
  LinearLayout regLayout = ttg::toLinearLayout(regTy).transposeOuts(allDims);
  SmallVector<StringAttr> nonIndexedDims = allDims;
  nonIndexedDims.erase(nonIndexedDims.begin() + axis);
  LinearLayout indexedLayout =
      regLayout.sublayout(llvm::to_vector(regLayout.getInDimNames()),
                          nonIndexedDims) *
      LinearLayout::identity1D(sharedLayout.getOutDimSize(axisDim), axisDim,
                               axisDim);
  indexedLayout = indexedLayout.transposeOuts(allDims);
  return invertAndComposeBlockLocal(sharedLayout, indexedLayout);
}

uint32_t getXorImageMask(const LinearLayout &layout, StringAttr outDim,
                         int numCTAs) {
  uint32_t image = 1;
  for (StringAttr inDim : layout.getInDimNames()) {
    for (int bit = 0; bit < layout.getInDimSizeLog2(inDim); ++bit)
      extendXorSpan(image, layout.getBasis(inDim, bit, outDim), numCTAs);
  }
  return image;
}

uint32_t translateXorMask(uint32_t mask, uint32_t translation, int numCTAs) {
  uint32_t translated = 0;
  for (int value = 0; value < numCTAs; ++value) {
    if (!(mask & (1u << value)))
      continue;
    uint32_t target = value ^ translation;
    assert(target < static_cast<uint32_t>(numCTAs) &&
           "target CTA exceeds the cluster size");
    translated |= 1u << target;
  }
  return translated;
}

Value getLocalMemoryRecipientCTAs(ImplicitLocOpBuilder &b,
                                  const LinearLayout &conversion) {
  MLIRContext *ctx = b.getContext();

  StringAttr kBlock = StringAttr::get(ctx, "block");
  int numCTAs = ttg::lookupNumCTAs(b);
  assert(conversion.hasInDim(kBlock) && conversion.hasOutDim(kBlock) &&
         conversion.getInDimSize(kBlock) == numCTAs &&
         conversion.getOutDimSize(kBlock) == numCTAs &&
         "expected conversion to preserve the cluster dimensions");

  // Span every non-issuer input basis that lowering can map into the block
  // output. For gather/scatter this includes the independent runtime-index
  // input, whose value is unavailable to this pass. The resulting image is a
  // conservative recipient set for the full BufferRegion effect.
  SmallVector<StringAttr> varyingInputs =
      llvm::to_vector(conversion.getInDimNames());
  llvm::erase(varyingInputs, kBlock);
  LinearLayout varyingInputsToTarget =
      conversion.sublayout(varyingInputs, {kBlock});
  uint32_t targetSpan = getXorImageMask(varyingInputsToTarget, kBlock, numCTAs);

  LinearLayout issuerToTarget = conversion.sublayout({kBlock}, {kBlock});

  SmallVector<uint32_t> recipientMasks;
  recipientMasks.reserve(numCTAs);
  for (int issuer = 0; issuer < numCTAs; ++issuer) {
    auto outputs = issuerToTarget.apply({{kBlock, issuer}});
    assert(outputs.size() == 1 && outputs.front().first == kBlock &&
           "expected block output dimension");
    recipientMasks.push_back(
        translateXorMask(targetSpan, outputs.front().second, numCTAs));
  }

  bool currentCTAOnly =
      llvm::all_of(llvm::enumerate(recipientMasks), [](auto entry) {
        return entry.value() == (1u << entry.index());
      });
  if (currentCTAOnly)
    return currentCTAMask(b);
  if (llvm::all_of(recipientMasks, [&](uint32_t mask) {
        return mask == recipientMasks.front();
      }))
    return arith::ConstantIntOp::create(b, recipientMasks.front(), 32);

  Value ctaId = tti::ExperimentalClusterCTAIdOp::create(b, b.getLoc());
  Value recipients =
      arith::ConstantIntOp::create(b, recipientMasks.front(), 32);
  for (int issuer = 1; issuer < numCTAs; ++issuer) {
    Value isIssuer =
        arith::CmpIOp::create(b, arith::CmpIPredicate::eq, ctaId,
                              arith::ConstantIntOp::create(b, issuer, 32));
    recipients = arith::SelectOp::create(
        b, isIssuer,
        arith::ConstantIntOp::create(b, recipientMasks[issuer], 32),
        recipients);
  }
  return recipients;
}

Value getLocalLoadStoreRecipientCTAs(ImplicitLocOpBuilder &b,
                                     ttg::MemDescType memDescTy,
                                     RankedTensorType regTy) {
  // Layout-less tensors can appear in intermediate/test IR but cannot encode a
  // cross-CTA ownership mapping. Preserve the existing current-CTA behavior.
  if (!regTy.getEncoding())
    return currentCTAMask(b);
  return getLocalMemoryRecipientCTAs(
      b, getLocalLoadStoreConversion(memDescTy, regTy));
}

Value getScratchReadCTAs(ImplicitLocOpBuilder &b, Operation *op,
                         Value ownerCTAs) {
  if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(op)) {
    LinearLayout srcLayout = ttg::toLinearLayout(convert.getSrc().getType());
    LinearLayout dstLayout = ttg::toLinearLayout(convert.getType());
    Value loadCTAs = getLocalMemoryRecipientCTAs(
        b, invertAndComposeBlockLocal(srcLayout, dstLayout));
    return arith::OrIOp::create(b, ownerCTAs, loadCTAs);
  }

  if (auto reduce = dyn_cast<tt::ReduceOp>(op)) {
    LinearLayout srcLayout = ttg::toLinearLayout(reduce.getInputTypes()[0]);
    auto block = StringAttr::get(op->getContext(), "block");
    auto axis = *(srcLayout.getOutDimNames().begin() + reduce.getAxis());
    uint16_t groupMask = getInputBasisMask(srcLayout, block, {axis});
    if (!groupMask)
      return ownerCTAs;
    if (groupMask == ttg::lookupNumCTAs(op) - 1)
      return allCTAsMask(b);
    return getRecipientCTAsForBroadcastMasks(b, {groupMask});
  }

  return ownerCTAs;
}

Value getMemEffectCTAs(ImplicitLocOpBuilder &b, Operation *op) {
  if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
    return getLocalLoadStoreRecipientCTAs(b, load.getSrc().getType(),
                                          load.getType());
  }
  if (auto store = dyn_cast<ttg::LocalStoreOp>(op)) {
    return getLocalLoadStoreRecipientCTAs(b, store.getDst().getType(),
                                          store.getSrc().getType());
  }
  if (auto store = dyn_cast<ttng::AsyncSharedStoreOp>(op)) {
    return getLocalLoadStoreRecipientCTAs(b, store.getDst().getType(),
                                          store.getSrc().getType());
  }
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op); alloc && alloc.getSrc()) {
    return getLocalLoadStoreRecipientCTAs(b, alloc.getType(),
                                          alloc.getSrc().getType());
  }
  if (auto gather = dyn_cast<ttg::LocalGatherOp>(op)) {
    return getLocalMemoryRecipientCTAs(
        b, getLocalGatherScatterConversion(gather.getSrc().getType(),
                                           gather.getType(), gather.getAxis()));
  }
  if (auto scatter = dyn_cast<ttg::LocalScatterOp>(op)) {
    return getLocalMemoryRecipientCTAs(
        b, getLocalGatherScatterConversion(scatter.getDst().getType(),
                                           scatter.getValues().getType(),
                                           scatter.getAxis()));
  }
  if (auto atomic = dyn_cast<ttg::LocalAtomicScatterRMWOp>(op)) {
    return getLocalMemoryRecipientCTAs(
        b, getLocalGatherScatterConversion(atomic.getDst().getType(),
                                           atomic.getValues().getType(),
                                           atomic.getAxis()));
  }
  if (auto tmaLoad = dyn_cast<ttng::TMALoadLikeOpInterface>(op)) {
    if (tmaLoad.getMulticast())
      return getMulticastRecipientCTAs(b, tmaLoad.getResult());
    return currentCTAMask(b);
  }
  if (isa<ttng::CLCTryCancelOp>(op))
    return allCTAsMask(b);
  if (isa<ttng::MMAv5OpInterface, ttng::TMEMCopyOp>(op))
    return getRecipientCTAsForBroadcastMasks(
        b, ttng::getCTABroadcastMasks(ttng::getModuleTwoCTAs(op), {}));
  return currentCTAMask(b);
}

Value getMemEffectCTAs(ImplicitLocOpBuilder &b, Value recipients,
                       uint32_t ownerMask) {
  if (ownerMask == 1)
    return recipients;

  int numCTAs = ttg::lookupNumCTAs(b);
  APInt constant;
  if (matchPattern(recipients, m_ConstantInt(&constant))) {
    uint32_t result = 0;
    for (int offset = 0; offset < numCTAs; ++offset)
      if (ownerMask & (1u << offset))
        result |= translateXorMask(constant.getZExtValue(), offset, numCTAs);
    return arith::ConstantIntOp::create(b, result, 32);
  }

  Value result = arith::ConstantIntOp::create(b, 0, 32);
  for (int source = 0; source < numCTAs; ++source) {
    uint32_t targets = translateXorMask(ownerMask, source, numCTAs);
    Value sourceBit = arith::ConstantIntOp::create(b, 1u << source, 32);
    Value present =
        arith::CmpIOp::create(b, arith::CmpIPredicate::ne,
                              arith::AndIOp::create(b, recipients, sourceBit),
                              arith::ConstantIntOp::create(b, 0, 32));
    Value targetMask = arith::SelectOp::create(
        b, present, arith::ConstantIntOp::create(b, targets, 32),
        arith::ConstantIntOp::create(b, 0, 32));
    result = arith::OrIOp::create(b, result, targetMask);
  }
  return result;
}

Value getBarrierRecipientCTAs(ImplicitLocOpBuilder &b, Operation *op) {
  if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op);
      arrive && arrive.isMulticast())
    return getRecipientCTAsForBroadcastMasks(
        b, {static_cast<uint16_t>(arrive.getMulticastCTA())});
  if (isa<ttng::BarrierExpectOp, ttng::ArriveBarrierOp>(op)) {
    Value barrier = cast<ttg::MBarrierOpInterface>(op).getBarrier();
    std::optional<uint32_t> fromCTA;
    if (auto expectOp = dyn_cast<ttng::BarrierExpectOp>(op))
      fromCTA = expectOp.getFromCTA();
    else
      fromCTA = cast<ttng::ArriveBarrierOp>(op).getFromCTA();
    if (fromCTA) {
      int numCTAs = ttg::lookupNumCTAs(op);
      uint32_t broadcastBits = ~*fromCTA & (numCTAs - 1);
      auto encoding = ttng::getTMAMulticastMaskEncoding(numCTAs, broadcastBits);
      return createCTABitset(b, encoding.pattern, encoding.fixedBits);
    }
    return getLeaderCTA(b, barrier);
  }
  if (auto arriveOp = dyn_cast<ttng::AsyncCopyMbarrierArriveOp>(op))
    return getLeaderCTA(b, arriveOp.getBarrier());
  if (auto tmaLoad = dyn_cast<ttng::TMALoadLikeOpInterface>(op)) {
    if (tmaLoad.getMulticast())
      return getMulticastBarrierRecipientCTAs(b, tmaLoad.getResult(),
                                              tmaLoad.getBarrier());
    return getLeaderCTA(b, tmaLoad.getBarrier());
  }
  if (isa<ttng::CLCTryCancelOp>(op))
    return allCTAsMask(b);

  if (isTensorCoreOp(op))
    return getRecipientCTAsForBroadcastMasks(
        b, getTensorCoreBarrierBroadcastMasks(op));
  return currentCTAMask(b);
}

class ConcurrencySanitizerImpl {
public:
  ConcurrencySanitizerImpl(ModuleOp module, const ConSanTargetHooks &hooks)
      : module(module), hooks(hooks) {}

  LogicalResult run() {
    SmallVector<tt::FuncOp> publicFuncs =
        llvm::to_vector(llvm::make_filter_range(
            module.getOps<tt::FuncOp>(),
            [](tt::FuncOp func) { return tt::isKernel(func); }));
    if (publicFuncs.size() != 1) {
      module.emitError(
          "ConSan requires exactly one public entrypoint function; "
          "found ")
          << publicFuncs.size();
      return failure();
    }
    entryPoint = publicFuncs.front();
    if (entryPoint.isExternal()) {
      entryPoint.emitError("ConSan entrypoint must have a function body");
      return failure();
    }
    if (failed(validateNonEntryFunctions()))
      return failure();

    tti::FunctionBuilder funcBuilder(module, auxData);
    if (failed(auxData.populateAndPassToWarpSpecialize(module, entryPoint,
                                                       funcBuilder, hooks)))
      return failure();

    ImplicitLocOpBuilder b(entryPoint.getLoc(), entryPoint);
    b.setInsertionPointToStart(&entryPoint.getBody().front());
    if (failed(instrumentMemoryOperations(b, funcBuilder)))
      return failure();
    initializeAllocations();
    return success();
  }

private:
  bool isCTALocalScratch(Operation *op) const {
    if (!op->hasAttr("allocation.size"))
      return true;

    // NVIDIA broadcasts scalar atomic results; AMD executes them in each CTA.
    if (isa<tt::AtomicRMWOp, tt::AtomicCASOp>(op) &&
        !isa<RankedTensorType>(op->getResult(0).getType()))
      return !hooks.hasUnsummarizableCalleeState(op);

    return !tt::hasCrossCTAScratch(op);
  }

  // Non-entry bodies are not instrumented. Their compiler-owned shared memory
  // is covered by the virtual frame on each call, but SSA-visible memory and
  // sanitizer state transitions cannot be represented by that summary.
  LogicalResult validateNonEntryFunctions() {
    SymbolTableCollection symbolTable;
    AuxDataMap emptyAuxData;
    for (tt::FuncOp func : module.getOps<tt::FuncOp>()) {
      WalkResult result = func.walk([&](Operation *op) -> WalkResult {
        if (op == func.getOperation())
          return WalkResult::advance();
        if (auto call = dyn_cast<CallOpInterface>(op)) {
          Operation *resolved = call.resolveCallableInTable(&symbolTable);
          tt::FuncOp callee = dyn_cast_or_null<tt::FuncOp>(resolved);
          if (!callee || callee.isExternal()) {
            call->emitError("ConSan cannot summarize an unresolved or external "
                            "callee in function @")
                << func.getName();
            return WalkResult::interrupt();
          }
        }
        if (func == entryPoint)
          return WalkResult::advance();
        if (isa<CallOpInterface>(op))
          return WalkResult::advance();

        bool hasUnsupportedAllocation =
            isa<ttg::LocalAllocOp, ttng::TMEMAllocOp>(op);
        bool hasOpaqueEffects =
            !isa<ttg::BarrierOp>(op) && hasUnknownEffects(op);
        bool hasUnsupportedResource = false;
        if (auto memoryEffects = dyn_cast<MemoryEffectOpInterface>(op)) {
          SmallVector<MemoryEffects::EffectInstance> effects;
          memoryEffects.getEffects(effects);
          hasUnsupportedResource = llvm::any_of(
              effects, [op](const MemoryEffects::EffectInstance &effect) {
                if (effect.getResource() == tt::GlobalMemory::get())
                  return false;
                // Volatile global loads publish a synthetic default-resource
                // write only to prevent compiler reordering.
                auto load = dyn_cast<tt::LoadOp>(op);
                return !(load && load.getIsVolatile() &&
                         isa<MemoryEffects::Write>(effect.getEffect()) &&
                         effect.getResource() ==
                             SideEffects::DefaultResource::get());
              });
        }

        auto info = hooks.getMemEffectsOpInfo(op);
        bool hasMemoryState =
            info && (!info->operandEffects.empty() || !info->barriers.empty() ||
                     info->implicitCommit);
        bool hasBarrierState = hooks.getBarrierInitInfo(op) ||
                               hooks.getBarrierWaitInfo(op) ||
                               hooks.getBarrierInvalidateInfo(op) ||
                               isa<ttg::MBarrierOpInterface>(op);
        bool hasAsyncState =
            hooks.getAsyncProxyFenceInfo(op) ||
            hooks.getWaitOpInfo(op, emptyAuxData) ||
            op->hasTrait<OpTrait::MemWaitOpTrait>() ||
            isa<ttg::AsyncCommitGroupOp, ttng::WarpGroupDotWaitOp,
                ttg::AsyncWaitOp>(op);
        bool hasControlState =
            isa<ttg::WarpSpecializeOp, ttg::WarpSpecializePartitionsOp,
                ttng::ClusterBarrierOp>(op) ||
            hooks.hasUnsummarizableCalleeState(op);
        if (!hasUnsupportedAllocation && !hasOpaqueEffects &&
            !hasUnsupportedResource && !hasMemoryState && !hasBarrierState &&
            !hasAsyncState && !hasControlState && isCTALocalScratch(op))
          return WalkResult::advance();

        op->emitError("ConSan cannot summarize ")
            << op->getName() << " in non-entry function @" << func.getName()
            << "; inline the function before ConSan or keep its body limited "
               "to register/global-memory operations and compiler-owned "
               "shared scratch";
        return WalkResult::interrupt();
      });
      if (result.wasInterrupted())
        return failure();
    }
    return success();
  }

  void initializeAllocations() {
    if (!shouldInitializeAllocations())
      return;

    SmallVector<Operation *> allocationsToInitialize;
    entryPoint.walk([&](Operation *op) {
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
        if (!alloc.getSrc())
          allocationsToInitialize.push_back(op);
      }
      if (auto alloc = dyn_cast<ttng::TMEMAllocOp>(op)) {
        if (!alloc.getSrc())
          allocationsToInitialize.push_back(op);
      }
    });

    for (Operation *op : allocationsToInitialize) {
      ImplicitLocOpBuilder b(op->getLoc(), op);
      b.setInsertionPointAfter(op);
      Value alloc = op->getResult(0);
      if (canInitializeAllocation(alloc)) {
        initializeAllocation(b, alloc);
        auto allocType = cast<ttg::MemDescType>(alloc.getType());
        bool isShared =
            isa<ttg::SharedMemorySpaceAttr>(allocType.getMemorySpace());
        if (isShared && auxData.hasAsyncProxyFenceTracking)
          ttng::FenceAsyncSharedOp::create(b, /*bCluster=*/false);
      }
    }
  }

  LogicalResult instrumentMemoryOperations(ImplicitLocOpBuilder &b,
                                           tti::FunctionBuilder &funcBuilder) {
    SmallVector<ttng::ClusterBarrierOp> clusterBarriers;
    WalkResult walkResult = entryPoint.walk([&](Operation *op) -> WalkResult {
      CriticalSectionListener listener;
      b.setListener(&listener);

      int thread = getCurrentThread(op, hooks, auxData.threadLayout);
      int baseThread = getBaseThread(thread, auxData.threadLayout);
      b.setLoc(op->getLoc());
      b.setInsertionPoint(op);
      if (isa<ttg::LocalAllocOp, ttng::TMEMAllocOp>(op)) {
        // Place insert point after specific ops:
        // allocs - we want to
        //   check if it is not overwriting any earlier allocation, but the
        //   memref value can be referenced only after it is created.
        b.setInsertionPointAfter(op);
      }

      if (auto info = hooks.getBarrierWaitInfo(op)) {
        // For waits we want to instrument it before and after, so we do it
        // manually inside instrumentBarrierWait (disable the critical section
        // listener and return early)
        b.setListener(nullptr);
        instrumentBarrierWait(op, info->alloc, info->phase, info->pred, thread,
                              baseThread, funcBuilder);
        return WalkResult::advance();
      }

      if (failed(instrumentMemEffects(b, op, thread, funcBuilder))) {
        b.setListener(nullptr);
        return WalkResult::interrupt();
      }
      b.setLoc(op->getLoc());
      if (auto info = hooks.getAsyncProxyFenceInfo(op)) {
        funcBuilder.createFenceProxyAccessesCall(
            b, baseThread, info->cluster, hooks.getIssuerCTAPred(b, op), op);
      }
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        // ConSan helpers can exceed the partition register budgets and make
        // PTXAS-generated dynamic register allocation deadlock.
        wsOp->setAttr("tti.disable_setmaxregister", b.getUnitAttr());
        funcBuilder.createSetActiveMaskCall(b, getActiveMask(wsOp), op);
        auto partitionRegions = wsOp.getNonEmptyPartitionRegions();
        if (!partitionRegions.empty()) {
          uint64_t destMask = 0;
          uint64_t baseDestMask = 0;
          for (Region *region : partitionRegions)
            destMask |= getThreadPeersMask(region->getRegionNumber() + 1,
                                           auxData.threadLayout);
          for (Region *region : partitionRegions)
            baseDestMask |= 1ULL << (region->getRegionNumber() + 1);
          if (destMask) {
            for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
              funcBuilder.createCopyWriteVisibilityCall(b, thread, destMask,
                                                        nullptr, memType, op);
              funcBuilder.createCopyReadVisibilityCall(b, thread, destMask,
                                                       nullptr, memType, op);
            }
          }
          if (baseDestMask)
            funcBuilder.createCopyProxyAccessesCall(b, baseThread, baseDestMask,
                                                    nullptr, op);
          // Lowering joins the partitions with a CTA barrier. Publish facts
          // observed by worker base threads only after that join; async-only
          // effects remain pending until a worker explicitly waits for them.
          b.setListener(nullptr);
          b.setInsertionPointAfter(wsOp);
          Value lock = auxData.lock.at(op).value;
          Value trueVal = arith::ConstantIntOp::create(b, 1, 1);
          tti::ExperimentalLockAcquireOp::create(b, lock, trueVal);
          for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM})
            funcBuilder.createPublishCTAVisibilityCall(
                b, baseDestMask,
                getThreadPeersMask(baseThread, auxData.threadLayout), memType,
                op);
          tti::ExperimentalLockReleaseOp::create(b, lock, trueVal);
          b.setInsertionPoint(wsOp);
          b.setListener(&listener);
        }
      }
      if (auto info = hooks.getBarrierInitInfo(op)) {
        Value pred = hooks.getIssuerCTAPred(b, op);
        if (!hooks.barrierWritesInvalidate())
          funcBuilder.createVerifyBarrierCanInitCall(b, info->alloc, pred, op,
                                                     currentCTAMask(b));
        funcBuilder.createInitBarrierStateCall(b, info->alloc, info->count,
                                               pred, op);
      }
      if (auto info = hooks.getBarrierInvalidateInfo(op)) {
        Value barrier = info->alloc;
        Value pred = hooks.getIssuerCTAPred(b, op);
        funcBuilder.createInvalidateBarrierStateCall(b, barrier, pred, op);
      }
      if (auto asyncCommitGroupOp = dyn_cast<ttg::AsyncCommitGroupOp>(op)) {
        if (!auxData.commits[CommitKind::AsyncCp].empty())
          funcBuilder.createCommitAccessesCall(b, thread, nullptr,
                                               CommitKind::AsyncCp, op);
      }
      if (auto wgmmaWaitOp = dyn_cast<ttng::WarpGroupDotWaitOp>(op)) {
        funcBuilder.createClearOutstandingCommitsTransferReadsCall(
            b, baseThread, getThreadPeersMask(thread, auxData.threadLayout),
            wgmmaWaitOp.getPendings(), nullptr, CommitKind::Wgmma,
            MemType::SHARED_MEM, op);
      }
      if (auto info = hooks.getWaitOpInfo(op, auxData)) {
        if (info->transferWrites && info->transferReads) {
          funcBuilder.createClearOutstandingCommitsTransferBothCall(
              b, baseThread, getThreadPeersMask(thread, auxData.threadLayout),
              info->pendingCount, nullptr, info->commitKind,
              MemType::SHARED_MEM, op);
        } else if (info->transferWrites) {
          funcBuilder.createClearOutstandingCommitsTransferWritesCall(
              b, baseThread, getThreadPeersMask(thread, auxData.threadLayout),
              info->pendingCount, nullptr, info->commitKind,
              MemType::SHARED_MEM, op);
        } else if (info->transferReads) {
          funcBuilder.createClearOutstandingCommitsTransferReadsCall(
              b, baseThread, getThreadPeersMask(thread, auxData.threadLayout),
              info->pendingCount, nullptr, info->commitKind,
              MemType::SHARED_MEM, op);
        }
      } else if (auto asyncWaitOp = dyn_cast<ttg::AsyncWaitOp>(op)) {
        funcBuilder.createClearOutstandingCommitsTransferWritesCall(
            b, baseThread, getThreadPeersMask(thread, auxData.threadLayout),
            asyncWaitOp.getNum(), nullptr, CommitKind::AsyncCp,
            MemType::SHARED_MEM, op);
      }
      if (auto clusterBarrier = dyn_cast<ttng::ClusterBarrierOp>(op)) {
        if (!llvm::is_contained(auxData.internalClusterBarriers, op))
          clusterBarriers.push_back(clusterBarrier);
      }

      if (isa<ttg::WarpYieldOp, ttg::WarpReturnOp>(op) &&
          !auxData.activeMasks.empty()) {
        auto wsOp = op->getParentOfType<ttg::WarpSpecializeOp>();
        bool shouldRetire =
            isa<ttg::WarpYieldOp>(op) ||
            llvm::is_contained(wsOp.getNonEmptyPartitionRegions(),
                               op->getParentRegion());
        if (shouldRetire) {
          b.setListener(nullptr);
          b.setLoc(wsOp.getLoc());
          Value lock = auxData.lock.at(op).value;
          Value trueVal = arith::ConstantIntOp::create(b, 1, 1);
          tti::ExperimentalLockAcquireOp::create(b, lock, trueVal);
          funcBuilder.createRetireActiveThreadCall(b, baseThread, op);
          Value ok =
              funcBuilder.createCheckAllActiveWaitingCall(b, nullptr, op);
          tti::ExperimentalLockReleaseOp::create(b, lock, trueVal);
          tti::createAssertInThread(
              b, ok,
              "Deadlock detected after a warp-specialized thread exited");
          b.setListener(&listener);
        }
      }
      if (isa<tt::ReturnOp>(op) && !auxData.activeMasks.empty()) {
        b.setListener(nullptr);
        Value lock = auxData.lock.at(op).value;
        Value trueVal = arith::ConstantIntOp::create(b, 1, 1);
        tti::ExperimentalLockAcquireOp::create(b, lock, trueVal);
        funcBuilder.createSetActiveMaskCall(b, 0, op);
        Value ok = funcBuilder.createCheckAllActiveWaitingCall(b, nullptr, op);
        tti::ExperimentalLockReleaseOp::create(b, lock, trueVal);
        tti::createAssertInThread(b, ok,
                                  "Deadlock detected when the kernel returned");
        b.setListener(&listener);
      }

      listener.maybeWrapWithCriticalSection(b, auxData, nullptr);
      b.setListener(nullptr);
      if (auto scatter = dyn_cast<ttg::LocalScatterOp>(op))
        instrumentLocalScatter(b, scatter, funcBuilder);
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return failure();

    // Cluster rendezvous polling introduces control-flow blocks, so add it
    // after the operation walk rather than invalidating the walk iterators.
    for (ttng::ClusterBarrierOp clusterBarrier : clusterBarriers) {
      Operation *op = clusterBarrier.getOperation();
      int thread = getCurrentThread(op, hooks, auxData.threadLayout);
      int baseThread = getBaseThread(thread, auxData.threadLayout);
      bool partitionScoped =
          static_cast<bool>(op->getParentOfType<ttg::WarpSpecializeOp>());
      b.setLoc(op->getLoc());
      b.setInsertionPoint(op);
      funcBuilder.createClusterBarrierRendezvousCall(
          b, auxData.getClusterBarrierSlot(op), baseThread,
          getThreadPeersMask(baseThread, auxData.threadLayout), partitionScoped,
          /*publishVisibility=*/!clusterBarrier.getRelaxed(), op);
    }
    return success();
  }

  void instrumentLocalScatter(ImplicitLocOpBuilder &b,
                              ttg::LocalScatterOp scatter,
                              tti::FunctionBuilder &funcBuilder) {
    b.setLoc(scatter.getLoc());
    b.setInsertionPointAfter(scatter);
    bool isMultiCTA = ttg::lookupNumCTAs(scatter) > 1;
    auto synchronize = [&] {
      if (isMultiCTA) {
        SmallVector<Operation *> barriers = hooks.createInitClusterBarrier(b);
        llvm::append_range(auxData.internalClusterBarriers, barriers);
      } else {
        ttg::BarrierOp::create(b, b.getLoc(), ttg::AddrSpace::Local);
      }
    };
    synchronize();
    funcBuilder.createVerifyLocalScatterDestinationsCall(
        b, scatter.getDst(), scatter.getIndices(), scatter.getValues(),
        scatter.getAxis());
    if (isMultiCTA)
      synchronize();
  }

  void instrumentBarrierWait(Operation *op, Value alloc, Value phase,
                             Value pred, int thread, int baseThread,
                             tti::FunctionBuilder &funcBuilder) {
    ImplicitLocOpBuilder wb(op->getLoc(), op);
    pred = tti::maybeAnd(wb, pred, hooks.getIssuerCTAPred(wb, op));
    Value lock = auxData.lock.at(op).value;
    // Pre-wait: mark waiting threads and check for deadlock.
    tti::ExperimentalLockAcquireOp::create(wb, lock, pred);
    funcBuilder.createVerifyBarrierInitializedCall(wb, alloc, pred, op,
                                                   currentCTAMask(wb));
    funcBuilder.createSetWaitingCall(wb, alloc, baseThread, phase, pred, op);
    Value ok = funcBuilder.createCheckAllActiveWaitingCall(wb, pred, op);
    tti::ExperimentalLockReleaseOp::create(wb, lock, pred);
    tti::createAssertInThread(wb, ok,
                              "Deadlock detected while waiting on an mbarrier");
    // Post-wait: transfer the waited phase's visible writes and reads to all
    // peer threads, and clear waiting for this barrier.
    assert(!auxData.barriers.empty() &&
           "barrier descriptors must exist when instrumenting wait");
    wb.setInsertionPointAfter(op);
    tti::ExperimentalLockAcquireOp::create(wb, lock, pred);
    for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
      funcBuilder.createTransferVisibleAccessesCall(
          wb, alloc, phase, getThreadPeersMask(thread, auxData.threadLayout),
          pred, memType, op);
    }
    funcBuilder.createCompleteBarrierWaitCall(wb, alloc, phase, baseThread,
                                              pred, op);
    tti::ExperimentalLockReleaseOp::create(wb, lock, pred);
  }

  LogicalResult instrumentMemEffects(ImplicitLocOpBuilder &b, Operation *op,
                                     int thread,
                                     tti::FunctionBuilder &funcBuilder) {
    int baseThread = getBaseThread(thread, auxData.threadLayout);
    std::optional<MemEffectsOpInfo> opInfo =
        getConSanMemEffectsOpInfo(hooks, op);
    for (const MemoryAccess &access :
         getMemoryAccesses(op, ttg::SharedKind::Barrier)) {
      if (opInfo && llvm::any_of(opInfo->barriers, [&](const auto &barrier) {
            return barrier.barrier == access.value;
          }))
        continue;
      funcBuilder.createVerifyBarrierInitializedCall(
          b, access.value, hooks.getIssuerCTAPred(b, op), op,
          getBarrierRecipientCTAs(b, op));
    }
    if (!opInfo)
      return success();
    bool isBarrierLifecycle = hooks.getBarrierInitInfo(op).has_value() ||
                              hooks.getBarrierInvalidateInfo(op).has_value();
    Value pred = opInfo->pred;
    Value issuerCTAPred = hooks.getIssuerCTAPred(b, op);
    pred = tti::maybeAnd(b, pred, issuerCTAPred);
    Value defaultEffectCTAs = getMemEffectCTAs(b, op);
    struct MaterializedEffect {
      Value bufferMask;
      Value effectCTAs;
      Value barrierCTAs;
      MemType memType;
    };
    SmallVector<MaterializedEffect> materializedEffects;
    materializedEffects.reserve(opInfo->operandEffects.size());

    auto addBarrierOwners = [&](MaterializedEffect &materialized,
                                const llvm::SmallBitVector &bufferMask,
                                Value ownerCTAs, bool selectBarrierStorage) {
      if (!selectBarrierStorage ||
          materialized.memType != MemType::SHARED_MEM ||
          auxData.barrierBufferMasks.empty())
        return;

      auto barrierStatesType =
          cast<RankedTensorType>(auxData.barrierStates.at(op).type);
      RankedTensorType barrierMaskType =
          tti::getSlicedTensorType(barrierStatesType, {1}, b.getI1Type());
      RankedTensorType barrierOwnersType =
          cast<RankedTensorType>(barrierMaskType.clone(b.getI32Type()));
      llvm::SmallBitVector overlaps(barrierMaskType.getNumElements());
      for (auto [index, barrierMask] :
           llvm::enumerate(auxData.barrierBufferMasks))
        if (bufferMask.anyCommon(barrierMask))
          overlaps.set(index);
      if (overlaps.none())
        return;

      Value selected = createStateMaskConstant(b, barrierMaskType, overlaps);
      Value owners = tt::SplatOp::create(b, barrierOwnersType, ownerCTAs);
      Value zero =
          tti::createConstIntTensor(b, b.getLoc(), 0, barrierOwnersType);
      owners = arith::SelectOp::create(b, selected, owners, zero);
      materialized.barrierCTAs =
          materialized.barrierCTAs
              ? arith::OrIOp::create(b, materialized.barrierCTAs, owners)
              : owners;
    };

    auto materializeBuffer =
        [&](Value buf, Value recipients,
            bool selectBarrierStorage) -> FailureOr<MaterializedEffect> {
      MaterializedEffect materialized;
      auto bufType = cast<ttg::MemDescType>(buf.getType());
      materialized.memType = MemType::TENSOR_MEM;
      if (isa<ttg::SharedMemorySpaceAttr>(bufType.getMemorySpace()))
        materialized.memType = MemType::SHARED_MEM;
      auto &candidateMap =
          auxData.bufferCandidates[static_cast<int>(materialized.memType)];
      auto candidateIt = candidateMap.find(buf);
      if (candidateIt == candidateMap.end()) {
        op->emitError("missing buffer-region candidates for memdesc");
        return failure();
      }
      const BufferStateCandidates &candidates = candidateIt->second;
      Value runtimeBase;
      if (!candidates.unknown && candidates.cases.size() > 1)
        runtimeBase = tti::ExperimentalMemDescToI32Op::create(b, buf);
      SmallVector<Value> predicates;
      FailureOr<Value> stateMask =
          createBufferStateMask(b, auxData, materialized.memType, buf,
                                runtimeBase, candidates, op, predicates);
      if (failed(stateMask))
        return failure();
      materialized.bufferMask = *stateMask;

      if (candidates.unknown) {
        materialized.effectCTAs = allCTAsMask(b);
        addBarrierOwners(
            materialized,
            auxData.bufferStatePlans[(int)materialized.memType].unknownMask,
            materialized.effectCTAs, selectBarrierStorage);
        return materialized;
      }

      for (auto [index, candidate] : llvm::enumerate(candidates.cases)) {
        Value ownerCTAs = getMemEffectCTAs(b, recipients, candidate.ctaMask);
        if (!predicates.empty())
          ownerCTAs =
              arith::SelectOp::create(b, predicates[index], ownerCTAs,
                                      arith::ConstantIntOp::create(b, 0, 32));
        materialized.effectCTAs =
            materialized.effectCTAs
                ? arith::OrIOp::create(b, materialized.effectCTAs, ownerCTAs)
                : ownerCTAs;
        addBarrierOwners(materialized, candidate.mask, ownerCTAs,
                         selectBarrierStorage);
      }
      return materialized;
    };

    auto materializeStaticBuffer =
        [&](const MemEffectsOpInfo::Effects::StaticSharedBuffer &buffer,
            bool selectBarrierStorage) -> FailureOr<MaterializedEffect> {
      int index = static_cast<int>(MemType::SHARED_MEM);
      const auto &regions = auxData.bufferRegions[index];
      BufferRegion region = buffer.getRegion(ttg::lookupNumCTAs(op));
      auto it = llvm::lower_bound(regions, region);
      if (it == regions.end() || !(*it == region)) {
        op->emitError("static shared-memory region is absent from the ConSan "
                      "registry");
        return failure();
      }
      unsigned id = std::distance(regions.begin(), it);
      const llvm::SmallBitVector &stateMask =
          auxData.bufferStatePlans[index].regionMasks[id];
      auto writeVisibilityType =
          cast<RankedTensorType>(auxData.writeVisibility[index].at(op).type);
      RankedTensorType maskType =
          tti::getSlicedTensorType(writeVisibilityType, {1}, b.getI1Type());

      MaterializedEffect materialized;
      materialized.memType = MemType::SHARED_MEM;
      materialized.bufferMask = createStateMaskConstant(b, maskType, stateMask);
      // Scratch stores target the issuing CTA; cross-CTA consumers read after
      // intrinsic synchronization.
      materialized.effectCTAs = currentCTAMask(b);
      addBarrierOwners(materialized, stateMask, materialized.effectCTAs,
                       selectBarrierStorage);
      return materialized;
    };

    for (const auto &effect : opInfo->operandEffects) {
      const auto *buf = std::get_if<Value>(&effect.buffer);
      bool isSharedMemory =
          !buf || isa<ttg::SharedMemorySpaceAttr>(
                      cast<ttg::MemDescType>(buf->getType()).getMemorySpace());
      bool invalidatesBarriers =
          isSharedMemory && hooks.barrierWritesInvalidate() &&
          effect.rw == RW::Write &&
          effect.sharedKind == ttg::SharedKind::Generic &&
          thread == baseThread &&
          opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier &&
          (!buf ||
           llvm::none_of(getMemoryAccesses(op), [&](const auto &access) {
             return access.value == *buf &&
                    access.sharedKind == effect.sharedKind && access.isRead;
           }));
      bool selectBarrierStorage = !isBarrierLifecycle || invalidatesBarriers;
      FailureOr<MaterializedEffect> maybeMaterialized =
          buf ? materializeBuffer(*buf, defaultEffectCTAs, selectBarrierStorage)
              : materializeStaticBuffer(
                    std::get<MemEffectsOpInfo::Effects::StaticSharedBuffer>(
                        effect.buffer),
                    selectBarrierStorage);
      if (failed(maybeMaterialized))
        return failure();
      MaterializedEffect materialized = *maybeMaterialized;
      materializedEffects.push_back(materialized);

      Value bufferMask = materialized.bufferMask;
      Value effectCTAs = materialized.effectCTAs;
      Value readCTAs = buf ? effectCTAs : getScratchReadCTAs(b, op, effectCTAs);
      MemType memType = materialized.memType;
      bool invalidatedBarrier = false;
      auto verifyWrite = [&] {
        addWriteChecks(b, funcBuilder, op, bufferMask, pred, memType, thread,
                       effect.operandName, readCTAs, opInfo->commitKind);
        addReadChecks(b, funcBuilder, op, bufferMask, pred, memType, thread,
                      effect.operandName, effectCTAs, opInfo->commitKind);
      };

      if (materialized.barrierCTAs) {
        if (invalidatesBarriers) {
          verifyWrite();
          invalidatedBarrier = true;
          funcBuilder.createInvalidateBarrierStorageCall(
              b, materialized.barrierCTAs, pred, op);
        } else {
          funcBuilder.createVerifyBarrierMemoryAvailableCall(
              b, materialized.barrierCTAs, pred, op);
        }
      }

      if (memType == MemType::SHARED_MEM) {
        if (effect.sharedKind == ttg::SharedKind::Async) {
          funcBuilder.createVerifyProxyAccessCall(b, bufferMask, baseThread,
                                                  effect.operandName, pred, op,
                                                  effectCTAs);
        } else {
          funcBuilder.createSetProxyAccessCall(b, bufferMask, baseThread, pred,
                                               op, readCTAs);
        }
      }
      if (effect.rw == RW::Read) {
        // For op that is reading, we only need to check if anything else
        // is writing to the same buffer.
        addWriteChecks(b, funcBuilder, op, bufferMask, pred, memType, thread,
                       effect.operandName, effectCTAs, opInfo->commitKind);
        if (opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier) {
          funcBuilder.createSetReadVisibilityCall(
              b, bufferMask, thread,
              getThreadPeersMask(thread, auxData.threadLayout), pred, memType,
              op, effectCTAs);
        }
        if (opInfo->trackingKind ==
            MemEffectsOpInfo::TrackingKind::CommitCount) {
          assert(memType == MemType::SHARED_MEM);
          funcBuilder.createStageAccessForCommitCall(
              b, bufferMask, baseThread, pred, memType, opInfo->commitKind, op);
        }
      }
      if (effect.rw == RW::Write) {
        // Op is writing to the buffer, we need to check if anything else
        // is reading or writing to the same buffer.
        if (!invalidatedBarrier)
          verifyWrite();
        if (opInfo->trackingKind == MemEffectsOpInfo::TrackingKind::Barrier) {
          funcBuilder.createPublishWriteVisibilityCall(
              b, bufferMask, getThreadPeersMask(thread, auxData.threadLayout),
              pred, memType, op, effectCTAs);
        }
        if (opInfo->trackingKind ==
            MemEffectsOpInfo::TrackingKind::CommitCount) {
          assert(memType == MemType::SHARED_MEM);
          funcBuilder.createStageAccessForCommitCall(
              b, bufferMask, baseThread, pred, memType, opInfo->commitKind, op);
        }
      }
    }
    for (const auto &barrierInfo : opInfo->barriers) {
      Value barrier = barrierInfo.barrier;
      Value combinedPred = tti::maybeAnd(b, barrierInfo.pred, pred);
      Value recipientCTAs =
          isa<ttng::AsyncSharedStoreOp>(op)
              ? getAsyncSharedStoreBarrierRecipientCTAs(
                    b, barrier, materializedEffects.front().effectCTAs)
              : getBarrierRecipientCTAs(b, op);
      Value completionBufferMask;
      if (thread != baseThread) {
        FailureOr<MaterializedEffect> completion =
            materializeBuffer(barrier, recipientCTAs,
                              /*selectBarrierStorage=*/false);
        if (failed(completion))
          return failure();
        completionBufferMask = completion->bufferMask;
        // A deferred completion may still touch the barrier after the issuing
        // thread advances. Model that future touch as a reader owned by the
        // synthetic engine. Waiting publishes the reader through the existing
        // barrier frontier; invalidation is a generic write and therefore must
        // observe it first.
        funcBuilder.createSetReadVisibilityCall(
            b, completionBufferMask, thread,
            getThreadPeersMask(thread, auxData.threadLayout), combinedPred,
            MemType::SHARED_MEM, op, recipientCTAs);
      }
      if (barrierInfo.count == 0 && barrierInfo.txCount == 0)
        funcBuilder.createVerifyBarrierInitializedCall(b, barrier, combinedPred,
                                                       op, recipientCTAs);
      if (barrierInfo.trackingMode ==
          MemEffectsOpInfo::BarrierTrackingMode::Frontier) {
        // If the op has barriers, we treat it as a commit emitted for each
        // barrier.
        for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
          funcBuilder.createTrackVisibleAccessesCall(
              b, barrier, thread, combinedPred, memType, op, recipientCTAs);
        }
        funcBuilder.createTrackProxyAccessesCall(
            b, barrier, baseThread, combinedPred, op, recipientCTAs);
      } else if (barrierInfo.trackingMode ==
                 MemEffectsOpInfo::BarrierTrackingMode::EffectWrites) {
        for (auto [effect, materialized] :
             llvm::zip(opInfo->operandEffects, materializedEffects)) {
          if (effect.rw != RW::Write)
            continue;
          funcBuilder.createTrackBarrierWriteForBufferCall(
              b, barrier, materialized.bufferMask, combinedPred,
              materialized.memType, op, recipientCTAs, materialized.effectCTAs);
          if (materialized.memType == MemType::SHARED_MEM) {
            funcBuilder.createTrackProxyAccessesForBufferCall(
                b, barrier, materialized.bufferMask, baseThread, combinedPred,
                op, recipientCTAs, materialized.effectCTAs);
          }
        }
        if (completionBufferMask)
          funcBuilder.createTrackVisibleAccessesCall(
              b, barrier, thread, combinedPred, MemType::SHARED_MEM, op,
              recipientCTAs, completionBufferMask);
      }
      if (barrierInfo.count > 0 || barrierInfo.txCount != 0) {
        funcBuilder.createVerifyAndUpdateBarrierStateCall(
            b, barrier, barrierInfo.count, combinedPred, op, recipientCTAs,
            barrierInfo.txCount);
      }
    }
    if (opInfo->implicitCommit) {
      assert(opInfo->trackingKind ==
             MemEffectsOpInfo::TrackingKind::CommitCount);
      funcBuilder.createCommitAccessesCall(b, baseThread, pred,
                                           opInfo->commitKind, op);
    }
    return success();
  }

  void addWriteChecks(ImplicitLocOpBuilder &b,
                      tti::FunctionBuilder &funcBuilder, Operation *op,
                      Value bufferMask, Value pred, MemType memType, int thread,
                      const std::string &operandName, Value effectCTAs,
                      CommitKind::Kind opCommitKind = CommitKind::None) {
    funcBuilder.createVerifyWriteVisibilityCall(
        b, bufferMask, thread, operandName, pred, memType, op, effectCTAs);
    if (hooks.isTMAOp(op) && !hooks.isOrderedCommitKind(CommitKind::TmaStore)) {
      funcBuilder.createVerifyWriteVisibilityCall(
          b, bufferMask, getBaseThread(thread, auxData.threadLayout),
          operandName, pred, memType, op, effectCTAs);
    }
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM) {
      for (const auto &commitKindDesc :
           hooks.getOutstandingWriteCommitKinds()) {
        bool excludeSelf = (opCommitKind == commitKindDesc.kind &&
                            hooks.isOrderedCommitKind(opCommitKind));
        funcBuilder.createCheckOutstandingCommitsCall(
            b, bufferMask, getBaseThread(thread, auxData.threadLayout),
            commitKindDesc.operationDesc, pred, memType, commitKindDesc.kind,
            op, effectCTAs, excludeSelf);
      }
    }
  }

  void addReadChecks(ImplicitLocOpBuilder &b, tti::FunctionBuilder &funcBuilder,
                     Operation *op, Value bufferMask, Value pred,
                     MemType memType, int thread,
                     const std::string &operandName, Value effectCTAs,
                     CommitKind::Kind opCommitKind = CommitKind::None) {
    funcBuilder.createVerifyReadVisibilityCall(
        b, bufferMask, thread, operandName, pred, memType, op, effectCTAs);
    // commit-num-based synchronization is only supported for shared memory
    if (memType == MemType::SHARED_MEM) {
      for (const auto &commitKindDesc :
           hooks.getOutstandingReadCommitKinds(auxData)) {
        bool excludeSelf = (opCommitKind == commitKindDesc.kind &&
                            hooks.isOrderedCommitKind(opCommitKind));
        funcBuilder.createCheckOutstandingCommitsCall(
            b, bufferMask, getBaseThread(thread, auxData.threadLayout),
            commitKindDesc.operationDesc, pred, memType, commitKindDesc.kind,
            op, effectCTAs, excludeSelf);
      }
    }
  }

  ModuleOp module;
  tt::FuncOp entryPoint;
  AuxDataMap auxData;
  const ConSanTargetHooks &hooks;
};

} // namespace

LogicalResult runConcurrencySanitizer(ModuleOp module,
                                      const ConSanTargetHooks &hooks) {
  ConcurrencySanitizerImpl impl(module, hooks);
  return impl.run();
}

class ConcurrencySanitizerPass
    : public impl::TritonInstrumentConcurrencySanitizerBase<
          ConcurrencySanitizerPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto targetAttr = module->getAttrOfType<StringAttr>(ttg::AttrTargetName);
    assert(targetAttr && "module missing ttg.target attribute");
    StringRef target = targetAttr.strref();
    StringRef key = target.starts_with("cuda:")  ? "nvidia"
                    : target.starts_with("hip:") ? "amd"
                                                 : "";
    auto hooks = createConSanHooks(key);
    assert(hooks && "no ConSan hooks registered for target");
    if (failed(runConcurrencySanitizer(module, *hooks)))
      return signalPassFailure();
  }
};

} // namespace instrument
} // namespace triton
} // namespace mlir
