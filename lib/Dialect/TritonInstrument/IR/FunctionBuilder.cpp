#include "triton/Dialect/TritonInstrument/IR/FunctionBuilder.h"

#include <cassert>
#include <optional>

#include "llvm/ADT/DenseSet.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/DebugStringHelper.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton::instrument {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tti = mlir::triton::instrument;

std::string mangleType(Type t) {
  if (auto intType = dyn_cast<IntegerType>(t)) {
    return ("I" + Twine(intType.getWidth())).str();
  }
  if (auto floatType = dyn_cast<FloatType>(t)) {
    return ("F" + Twine(floatType.getWidth())).str();
  }
  if (auto ptrType = dyn_cast<PointerType>(t)) {
    return "P";
  }
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    std::string result = "T";
    llvm::raw_string_ostream os(result);
    for (int s : tensorType.getShape()) {
      os << s << "x";
    }
    os << mangleType(tensorType.getElementType());
    return result;
  }
  // Fallback to hash of the type's string representation.
  return "U" + llvm::utohexstr(llvm::hash_value(mlir::debugString(t)));
}

namespace {

namespace BarrierBits {
constexpr unsigned initCountLsb = 1;
constexpr unsigned currentCountLsb = 21;
constexpr unsigned txCountLsb = 41;
constexpr unsigned countBitWidth = 20;
constexpr unsigned txCountBitWidth = 21;
constexpr uint64_t countMask = (1ull << countBitWidth) - 1;
constexpr uint64_t txCountMask = (1ull << txCountBitWidth) - 1;
constexpr int64_t txCountMin = -(int64_t)countMask;
constexpr int64_t txCountMax = (int64_t)countMask;
} // namespace BarrierBits

namespace WaitingBits {
constexpr unsigned bitsPerThread = 2;
constexpr unsigned flagBit = 0;
constexpr unsigned phaseBit = 1;

uint32_t makeInterleavedMask(unsigned bit, unsigned numBaseThreads) {
  uint32_t mask = 0;
  for (unsigned i = 0; i < numBaseThreads; ++i)
    mask |= 1u << (bitsPerThread * i + bit);
  return mask;
}
} // namespace WaitingBits

namespace ProxyAccessBits {
constexpr unsigned fencedOffset = MAX_NUM_BASE_THREADS;
constexpr uint64_t seenMask = (1ull << MAX_NUM_BASE_THREADS) - 1;
} // namespace ProxyAccessBits

// Information about the optional assert message and tensor type to check.
struct AssertInfo {
  StringRef message;
  Type type;
};

static uint64_t expandActiveMask(uint64_t activeMask, unsigned numBaseThreads) {
  uint64_t expanded = 0;
  for (unsigned i = 0; i < numBaseThreads; ++i) {
    if (activeMask & (1ull << i))
      expanded |=
          1ull << (WaitingBits::bitsPerThread * i + WaitingBits::flagBit);
  }
  return expanded;
}

Value createCmpIntTensorScalar(
    ImplicitLocOpBuilder &b, Value tensor, Value scalar,
    arith::CmpIPredicate predicate = arith::CmpIPredicate::eq) {
  auto tensorTy = cast<RankedTensorType>(tensor.getType());
  Value splat = triton::SplatOp::create(b, tensorTy, scalar);
  return arith::CmpIOp::create(b, predicate, tensor, splat);
}

template <typename OpTy>
Value reduceLastDim(ImplicitLocOpBuilder &b, Value tensor) {
  OpBuilder::InsertionGuard guard(b);
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  assert(tensorType.getRank() > 0 && "cannot reduce a rank-0 tensor");
  int axis = tensorType.getRank() - 1;
  auto reduceOp = triton::ReduceOp::create(b, std::vector<Value>{tensor}, axis);
  auto &region = reduceOp.getRegion();
  auto &block = region.emplaceBlock();
  block.addArguments({tensorType.getElementType(), tensorType.getElementType()},
                     {b.getLoc(), b.getLoc()});
  b.setInsertionPointToStart(&block);
  auto result = OpTy::create(b, block.getArgument(0), block.getArgument(1));
  triton::ReduceReturnOp::create(b, std::vector<Value>{result});
  return reduceOp->getResult(0);
}

template <typename OpTy>
Value reduce(ImplicitLocOpBuilder &b, Value tensor, ArrayRef<int> axes) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  assert(!axes.empty() && "expected at least one reduction axis");

  llvm::SmallDenseSet<int> reducedAxes;
  for (int axis : axes) {
    assert(axis >= 0 && axis < tensorType.getRank() &&
           "invalid reduction axis");
    assert(!reducedAxes.contains(axis) && "duplicate reduction axis");
    reducedAxes.insert(axis);
  }

  SmallVector<int32_t> transposeOrder;
  SmallVector<int64_t> flattenedShape;
  for (int dim = 0; dim < tensorType.getRank(); ++dim) {
    if (reducedAxes.contains(dim))
      continue;
    transposeOrder.push_back(dim);
    flattenedShape.push_back(tensorType.getShape()[dim]);
  }

  int64_t reducedSize = 1;
  for (int axis : axes) {
    transposeOrder.push_back(axis);
    reducedSize *= tensorType.getShape()[axis];
  }
  flattenedShape.push_back(reducedSize);

  tensor = triton::TransOp::create(b, tensor, transposeOrder);
  tensor = triton::ReshapeOp::create(b, flattenedShape, tensor);
  return reduceLastDim<OpTy>(b, tensor);
}

template <typename OpTy>
Value reduceAll(ImplicitLocOpBuilder &b, Value tensor) {
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType || tensorType.getRank() == 0)
    return tensor;
  if (tensorType.getRank() != 1) {
    tensor = triton::ReshapeOp::create(b, {tensorType.getNumElements()}, tensor,
                                       /*allowReorder=*/true);
  }
  return reduceLastDim<OpTy>(b, tensor);
}

FuncOp getOrCreateFunction(
    ModuleOp module, const std::string &name, llvm::ArrayRef<Type> argTypes,
    ManglingArgs specializationArgs, int numWarps, Type assertType,
    std::function<void(ImplicitLocOpBuilder &b, Block *entryBlock)> buildBody) {
  ManglingArgs manglingArgs;
  manglingArgs.append(argTypes);
  manglingArgs.append(specializationArgs);
  if (assertType) {
    manglingArgs.append(assertType);
  }
  std::string funcName = manglingArgs.mangle(name, numWarps);
  if (auto existing = module.lookupSymbol<FuncOp>(funcName)) {
    return existing;
  }

  OpBuilder moduleBuilder(module.getContext());
  moduleBuilder.setInsertionPointToStart(module.getBody());
  Location loc = module.getLoc();
  SmallVector<Type> resultTypes = {};
  if (assertType) {
    resultTypes.push_back(assertType);
  }
  auto funcType = moduleBuilder.getFunctionType(argTypes, resultTypes);
  FuncOp func = FuncOp::create(moduleBuilder, loc, funcName, funcType);
  func.setVisibility(SymbolTable::Visibility::Private);
  func->setAttr(ttg::AttrNumWarpsName,
                moduleBuilder.getI32IntegerAttr(numWarps));
  for (auto [i, argType] : llvm::enumerate(argTypes)) {
    if (isa<PointerType>(argType)) {
      func.setArgAttr(i, "tt.divisibility",
                      moduleBuilder.getI32IntegerAttr(16));
    }
  }
  Block *entryBlock = func.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entryBlock);
  ImplicitLocOpBuilder fb(loc, bodyBuilder);
  buildBody(fb, entryBlock);
  func.walk([&](ttg::ConvertLayoutOp op) {
    op.setForceWarpShuffleAttr(moduleBuilder.getUnitAttr());
  });
  return func;
}

// Create a call to a function with body given by `buildBody`.
// If the function does not exist, it will be created, otherwise the
// existing function will be used.
// If `assertInfo` is provided, the function should return a tensor of the
// given type. The result is asserted unless `emitAssert` is false.
Value createCallToCachedFunction(
    ImplicitLocOpBuilder &b, const std::string &name, ArrayRef<Value> args,
    std::optional<AssertInfo> assertInfo, ManglingArgs specializationArgs,
    std::function<void(ImplicitLocOpBuilder &b, Block *entryBlock)> buildBody,
    bool emitAssert = true) {
  Region *region = b.getInsertionBlock()->getParent();
  ModuleOp module = region->getParentOp()->getParentOfType<ModuleOp>();
  int numWarps = ttg::lookupNumWarps(region);
  SmallVector<Type> argTypes = llvm::to_vector(
      llvm::map_range(args, [](Value v) { return v.getType(); }));
  Type assertType = assertInfo ? assertInfo->type : nullptr;
  triton::FuncOp func =
      getOrCreateFunction(module, name, argTypes, specializationArgs, numWarps,
                          assertType, buildBody);
  SmallVector<Type> resultTypes = {};
  if (assertInfo) {
    resultTypes.push_back(assertInfo->type);
  }
  auto callOp = triton::CallOp::create(b, func.getName(), resultTypes, args);
  if (assertInfo) {
    Value result = callOp->getResult(0);
    if (emitAssert) {
      StringRef message = b.getStringAttr(assertInfo->message);
      createAssertInThread(b, result, message);
    }
    return result;
  }
  return {};
}

Value createBufferDescriptor(ImplicitLocOpBuilder &b, Value offsetI32,
                             Value lengthI32) {
  auto i64Type = b.getI64Type();
  Value offsetI64 = arith::ExtUIOp::create(b, i64Type, offsetI32);
  Value lengthI64 = arith::ExtUIOp::create(b, i64Type, lengthI32);
  Value shiftAmount = arith::ConstantIntOp::create(b, 32, 64);
  Value lengthShifted = arith::ShLIOp::create(b, lengthI64, shiftAmount);
  return arith::OrIOp::create(b, lengthShifted, offsetI64);
}

std::tuple<Block *, Block *, Block *> createIfBlock(ImplicitLocOpBuilder &b,
                                                    Value cnd) {
  // #prevBlock
  // if (condition) {
  //   #ifBlock
  // }
  // #thenBlock
  Block *prevBlock = b.getInsertionBlock();
  Block::iterator insertPoint = b.getInsertionPoint();
  Block *ifBlock = prevBlock->splitBlock(insertPoint);

  // Split a block after the call.
  Block *thenBlock = ifBlock->splitBlock(ifBlock->begin());
  b.setInsertionPointToEnd(ifBlock);
  cf::BranchOp::create(b, thenBlock);
  b.setInsertionPointToEnd(prevBlock);
  cf::CondBranchOp::create(b, cnd, ifBlock, ValueRange{}, thenBlock,
                           ValueRange{});
  b.setInsertionPointToStart(thenBlock);

  return {prevBlock, ifBlock, thenBlock};
}

Value createConvertLayout(ImplicitLocOpBuilder &b, Value tensor,
                          Attribute encoding) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto dstType = tensorType.cloneWithEncoding(encoding);
  return ttg::ConvertLayoutOp::create(b, dstType, tensor);
}

Value convertAndBroadcast(ImplicitLocOpBuilder &b, Value tensor,
                          ArrayRef<int> keptDims, RankedTensorType dstType) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  assert(static_cast<size_t>(tensorType.getRank()) == keptDims.size() &&
         "expected one kept dimension per source tensor rank");
  auto resultType = RankedTensorType::get(
      dstType.getShape(), tensorType.getElementType(), dstType.getEncoding());
  auto sliceType = tti::getSlicedTensorType(resultType, keptDims,
                                            tensorType.getElementType());
  if (tensorType != sliceType)
    tensor = ttg::ConvertLayoutOp::create(b, sliceType, tensor);
  return tti::reshapeAndBroadcast(b, b.getLoc(), tensor, keptDims, resultType);
}

Value adjustIntegerWidth(ImplicitLocOpBuilder &b, Value value,
                         IntegerType targetType) {
  auto srcType = cast<IntegerType>(value.getType());
  if (srcType.getWidth() == targetType.getWidth())
    return value;
  if (srcType.getWidth() < targetType.getWidth())
    return arith::ExtUIOp::create(b, targetType, value);
  return arith::TruncIOp::create(b, targetType, value);
}

Value createThreadColumnMask(ImplicitLocOpBuilder &b, Value threadMask,
                             RankedTensorType tensorType, int columnDim) {
  auto loc = b.getLoc();
  int columns = tensorType.getShape()[columnDim];

  RankedTensorType rangeType =
      tti::getSlicedTensorType(tensorType, {columnDim}, b.getI32Type());
  Value range = triton::MakeRangeOp::create(b, rangeType, 0, columns);

  auto elemType = cast<IntegerType>(tensorType.getElementType());
  RankedTensorType rangeElemType = rangeType.clone(elemType);
  Value rangeElem = range;
  if (elemType.getWidth() != 32)
    rangeElem = arith::ExtUIOp::create(b, rangeElemType, range);

  Value indices = convertAndBroadcast(b, rangeElem, {columnDim}, tensorType);

  Value threadMaskElem = adjustIntegerWidth(b, threadMask, elemType);
  Value maskTensor = triton::SplatOp::create(b, tensorType, threadMaskElem);

  Value shifted = arith::ShRUIOp::create(b, maskTensor, indices);
  Value one = tti::createConstIntTensor(b, loc, 1, tensorType);
  Value bits = arith::AndIOp::create(b, shifted, one);
  Value zero = tti::createConstIntTensor(b, loc, 0, tensorType);
  return arith::CmpIOp::create(b, arith::CmpIPredicate::ne, bits, zero);
}

Value createDimMask(ImplicitLocOpBuilder &b, Value index,
                    RankedTensorType tensorType, int dim) {
  assert(dim >= 0 && dim < tensorType.getRank() && "invalid tensor dimension");
  auto indexType = tti::getSlicedTensorType(tensorType, {dim}, b.getI32Type());
  Value range = triton::MakeRangeOp::create(b, indexType, /*start=*/0,
                                            /*end=*/tensorType.getShape()[dim]);
  Value indexTensor = triton::SplatOp::create(b, indexType, index);
  Value mask1D =
      arith::CmpIOp::create(b, arith::CmpIPredicate::eq, range, indexTensor);
  auto maskType =
      cast<RankedTensorType>(tensorType.cloneWith(std::nullopt, b.getI1Type()));
  return convertAndBroadcast(b, mask1D, {dim}, maskType);
}

Value createDimIndices(ImplicitLocOpBuilder &b, RankedTensorType tensorType,
                       int dim) {
  assert(dim >= 0 && dim < tensorType.getRank() && "invalid tensor dimension");
  auto indexType = tti::getSlicedTensorType(tensorType, {dim}, b.getI32Type());
  Value range = triton::MakeRangeOp::create(b, indexType, /*start=*/0,
                                            /*end=*/tensorType.getShape()[dim]);
  auto fullIndexType = cast<RankedTensorType>(
      tensorType.cloneWith(std::nullopt, b.getI32Type()));
  return convertAndBroadcast(b, range, {dim}, fullIndexType);
}

Value createLoadTrackingPhase(ImplicitLocOpBuilder &b, Value trackingPtr,
                              RankedTensorType trackingType, Value phase) {
  int phaseDim = trackingType.getRank() - 1;
  assert(phaseDim >= 0 && trackingType.getShape()[phaseDim] == 2 &&
         "expected a trailing barrier phase dimension");

  SmallVector<int> keptDims;
  keptDims.reserve(phaseDim);
  for (int dim = 0; dim < phaseDim; ++dim)
    keptDims.push_back(dim);
  RankedTensorType bankType = tti::getSlicedTensorType(
      trackingType, keptDims, trackingType.getElementType());

  Value phaseIndex = adjustIntegerWidth(b, phase, b.getI32Type());
  phaseIndex = arith::AndIOp::create(b, phaseIndex,
                                     arith::ConstantIntOp::create(b, 1, 32));
  Value bankElements =
      arith::ConstantIntOp::create(b, bankType.getNumElements(), /*width=*/32);
  Value offset = arith::MulIOp::create(b, phaseIndex, bankElements);
  Value bankPtr =
      triton::AddPtrOp::create(b, trackingPtr.getType(), trackingPtr, offset);
  return tti::createLoadScratchMemory(b, b.getLoc(), bankPtr, bankType);
}

Value createCurrentCTAMask(ImplicitLocOpBuilder &b) {
  Value ctaId = tti::ExperimentalClusterCTAIdOp::create(b, b.getLoc());
  return arith::ShLIOp::create(b, arith::ConstantIntOp::create(b, 1, 32),
                               ctaId);
}

Value createTrackingOriginPhasePointer(ImplicitLocOpBuilder &b,
                                       Value trackingPtr,
                                       RankedTensorType trackingType,
                                       RankedTensorType bankType,
                                       Value originId, int phase) {
  assert(trackingType.getRank() == 6 && trackingType.getShape()[5] == 2 &&
         "expected origin CTA followed by the two barrier phases");
  int64_t numOrigins = trackingType.getShape()[4];
  Value bankIndex = originId;
  if (phase != 0) {
    Value phaseOriginOffset =
        arith::ConstantIntOp::create(b, phase * numOrigins, 32);
    bankIndex = arith::AddIOp::create(b, bankIndex, phaseOriginOffset);
  }
  Value bankElements =
      arith::ConstantIntOp::create(b, bankType.getNumElements(), 32);
  Value elementOffset = arith::MulIOp::create(b, bankIndex, bankElements);
  return triton::AddPtrOp::create(b, trackingPtr.getType(), trackingPtr,
                                  elementOffset);
}

Value createCTASetMask(ImplicitLocOpBuilder &b, RankedTensorType tensorType,
                       int dim, Value ctas) {
  int numCTAs = ttg::lookupNumCTAs(b);

  // Turn the scalar recipient bitset into a tensor mask over logical CTA rows:
  // build a [0, numCTAs) row-index vector on `dim`, broadcast it to the state
  // tensor shape, and test one bit of `recipientCTAs` per row.
  auto loc = b.getLoc();
  auto rowType = tti::getSlicedTensorType(tensorType, {dim}, b.getI32Type());
  Value rowIdx = triton::MakeRangeOp::create(b, rowType, /*start=*/0,
                                             /*end=*/numCTAs);
  auto indexType = cast<RankedTensorType>(
      tensorType.cloneWith(std::nullopt, b.getI32Type()));
  rowIdx = convertAndBroadcast(b, rowIdx, {dim}, indexType);

  Value recipientBitsTensor = triton::SplatOp::create(b, indexType, ctas);
  Value shifted = arith::ShRUIOp::create(b, recipientBitsTensor, rowIdx);
  Value one = tti::createConstIntTensor(b, loc, 1, indexType);
  Value selectedBit = arith::AndIOp::create(b, shifted, one);
  Value zero = tti::createConstIntTensor(b, loc, 0, indexType);
  return arith::CmpIOp::create(b, arith::CmpIPredicate::ne, selectedBit, zero);
}

Value createBarrierOwnerMask(ImplicitLocOpBuilder &b, Value barrierCTAs,
                             RankedTensorType statesType) {
  auto indexType = cast<RankedTensorType>(statesType.clone(b.getI32Type()));
  auto rowType = tti::getSlicedTensorType(indexType, {0}, b.getI32Type());
  Value row =
      triton::MakeRangeOp::create(b, rowType, 0, statesType.getShape()[0]);
  row = convertAndBroadcast(b, row, {0}, indexType);
  Value owners = convertAndBroadcast(b, barrierCTAs, {1}, indexType);
  Value bit = arith::AndIOp::create(
      b, arith::ShRUIOp::create(b, owners, row),
      tti::createConstIntTensor(b, b.getLoc(), 1, indexType));
  return arith::CmpIOp::create(
      b, arith::CmpIPredicate::ne, bit,
      tti::createConstIntTensor(b, b.getLoc(), 0, indexType));
}

static void createVerifyBarrierStateCall(ImplicitLocOpBuilder &b,
                                         StringRef message, Value offset,
                                         Value length, Value pred,
                                         ValueType barriers, ValueType states,
                                         Value recipientCTAs,
                                         bool initialized) {
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  auto barriersType = cast<RankedTensorType>(barriers.type);
  auto statesType = cast<RankedTensorType>(states.type);
  SmallVector<Value> args = {offset,         length,       pred,
                             barriers.value, states.value, recipientCTAs};
  AssertInfo assertInfo{message, b.getI1Type()};
  createCallToCachedFunction(
      b, initialized ? "verify_barrier_initialized" : "verify_barrier_can_init",
      args, assertInfo, {barriersType, statesType},
      [statesType, initialized](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value offset = entryBlock->getArgument(0);
        Value length = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value barriers = entryBlock->getArgument(3);
        Value statesPtr = entryBlock->getArgument(4);
        Value recipientCTAs = entryBlock->getArgument(5);

        Value descriptor = createBufferDescriptor(fb, offset, length);
        Value selected = createCmpIntTensorScalar(fb, barriers, descriptor);
        selected = convertAndBroadcast(fb, selected, {1}, statesType);

        Value states = tti::createLoadScratchMemory(fb, fb.getLoc(), statesPtr,
                                                    statesType);
        Value zero = tti::createConstIntTensor(fb, fb.getLoc(), 0, statesType);
        Value valid = arith::CmpIOp::create(
            fb,
            initialized ? arith::CmpIPredicate::ne : arith::CmpIPredicate::eq,
            states, zero);
        auto condType = cast<RankedTensorType>(valid.getType());
        Value vTrue = tti::createConstIntTensor(fb, fb.getLoc(), 1, condType);
        valid = arith::SelectOp::create(fb, selected, valid, vTrue);
        Value ctaMask =
            createCTASetMask(fb, condType, /*dim=*/0, recipientCTAs);
        valid = arith::SelectOp::create(fb, ctaMask, valid, vTrue);
        Value predTensor = triton::SplatOp::create(fb, condType, pred);
        valid = arith::SelectOp::create(fb, predTensor, valid, vTrue);
        triton::ReturnOp::create(fb, reduceAll<arith::AndIOp>(fb, valid));
      });
}

Value createLeadCTAEffectMask(ImplicitLocOpBuilder &b,
                              RankedTensorType tensorType, Value effectCTAs) {
  Value lhsMask = createCTASetMask(b, tensorType, /*dim=*/0, effectCTAs);
  Value leadCTAMask =
      createCTASetMask(b, tensorType, /*dim=*/2, createCurrentCTAMask(b));
  return arith::AndIOp::create(b, lhsMask, leadCTAMask);
}

Operation *createMaskedStoreScratchMemory(ImplicitLocOpBuilder &b, Location loc,
                                          Value alloc, Value tensor,
                                          RankedTensorType tensorType,
                                          Value mask) {
  int64_t numCTAs = ttg::lookupNumCTAs(b);
  return tti::createStoreScratchMemory(b, loc, alloc, tensor, tensorType,
                                       /*currentCTAOnly=*/false,
                                       numCTAs > 1 ? mask : Value{});
}

Operation *createCTAScopedStoreScratchMemory(ImplicitLocOpBuilder &b,
                                             Location loc, Value alloc,
                                             Value tensor,
                                             RankedTensorType tensorType,
                                             Value recipientCTAs) {
  return createMaskedStoreScratchMemory(
      b, loc, alloc, tensor, tensorType,
      createCTASetMask(b, tensorType, /*dim=*/0, recipientCTAs));
}

Value createVirtualBarrierMask(ImplicitLocOpBuilder &b, Value barrierIdx,
                               RankedTensorType tensorType) {
  Value barrierMask = createDimMask(b, barrierIdx, tensorType, /*dim=*/1);
  Value leadCTAMask = createCTASetMask(b, tensorType, /*dim=*/0,
                                       arith::ConstantIntOp::create(b, 1, 32));
  return arith::AndIOp::create(b, barrierMask, leadCTAMask);
}

Value arriveVirtualBarrier(ImplicitLocOpBuilder &b, Value statesPtr,
                           RankedTensorType statesType, Value barrierIdx,
                           int count) {
  Value states =
      tti::createLoadScratchMemory(b, b.getLoc(), statesPtr, statesType);
  Value mask = createVirtualBarrierMask(b, barrierIdx, statesType);
  Value one = tti::createConstIntTensor(b, b.getLoc(), 1, statesType);
  Value two = tti::createConstIntTensor(b, b.getLoc(), 2, statesType);
  Value phase = arith::AndIOp::create(b, states, one);
  // Virtual slots encode 2 * arrivals + phase. Completing an epoch resets the
  // arrival count and toggles the low phase bit used by deadlock detection.
  Value nextState = arith::AddIOp::create(b, states, two);
  Value completionState = arith::AddIOp::create(
      b, phase,
      tti::createConstIntTensor(b, b.getLoc(), 2 * count, statesType));
  Value completed =
      arith::AndIOp::create(b, mask,
                            arith::CmpIOp::create(b, arith::CmpIPredicate::eq,
                                                  nextState, completionState));
  Value completedInt = arith::ExtUIOp::create(b, statesType, completed);
  Value nextPhase = arith::XOrIOp::create(b, phase, completedInt);
  nextState = arith::SelectOp::create(b, completed, nextPhase, nextState);
  Value updated = arith::SelectOp::create(b, mask, nextState, states);
  tti::createStoreScratchMemory(b, b.getLoc(), statesPtr, updated, statesType);
  return reduceAll<arith::OrIOp>(b, completed);
}

Value updateWaitingBits(ImplicitLocOpBuilder &b, Value waiting,
                        RankedTensorType waitingType, Value thread, Value phase,
                        Value mask, bool markWaiting) {
  Value bitsPerThread =
      arith::ConstantIntOp::create(b, WaitingBits::bitsPerThread, 32);
  Value flagBit = arith::ConstantIntOp::create(b, WaitingBits::flagBit, 32);
  Value phaseBit = arith::ConstantIntOp::create(b, WaitingBits::phaseBit, 32);
  Value one = arith::ConstantIntOp::create(b, 1, 32);
  Value minusOne = arith::ConstantIntOp::create(b, -1, 32);
  Value baseTimesBits = arith::MulIOp::create(b, thread, bitsPerThread);
  Value flagShift = arith::AddIOp::create(b, baseTimesBits, flagBit);
  Value phaseShift = arith::AddIOp::create(b, baseTimesBits, phaseBit);
  Value flagMask = arith::ShLIOp::create(b, one, flagShift);
  Value phaseMask = arith::ShLIOp::create(b, one, phaseShift);
  Value combinedMask = arith::OrIOp::create(b, flagMask, phaseMask);
  Value clearMask = arith::XOrIOp::create(b, combinedMask, minusOne);
  Value clearMaskTensor = triton::SplatOp::create(b, waitingType, clearMask);
  Value cleared = arith::AndIOp::create(b, waiting, clearMaskTensor);

  if (!markWaiting)
    return arith::SelectOp::create(b, mask, cleared, waiting);

  Value phaseI32 = arith::ExtUIOp::create(b, b.getI32Type(), phase);
  Value phaseShifted = arith::ShLIOp::create(b, phaseI32, phaseShift);
  Value setBits = arith::OrIOp::create(b, flagMask, phaseShifted);
  Value setBitsTensor = triton::SplatOp::create(b, waitingType, setBits);
  Value withWaiting = arith::OrIOp::create(b, cleared, setBitsTensor);
  return arith::SelectOp::create(b, mask, withWaiting, waiting);
}

} // namespace

void FunctionBuilder::createFillGlobalTensorCall(ImplicitLocOpBuilder &b,
                                                 Value ptr,
                                                 RankedTensorType type,
                                                 Value scalar) {
  type = tti::getIntTensorType(b.getInsertionBlock()->getParent(),
                               {type.getNumElements()},
                               type.getElementType().getIntOrFloatBitWidth());
  createCallToCachedFunction(
      b, "fill_global_tensor", {ptr, scalar}, /*assertInfo=*/std::nullopt,
      {type}, [type](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value ptr = entryBlock->getArgument(0);
        Value scalar = entryBlock->getArgument(1);
        Value tensor = triton::SplatOp::create(fb, type, scalar);
        createStoreScratchMemory(fb, fb.getLoc(), ptr, tensor, type,
                                 /*currentCTAOnly=*/false);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createSetWaitingCall(ImplicitLocOpBuilder &b, Value mbar,
                                           int thread, Value phase, Value pred,
                                           Operation *insertPoint) {

  if (auxData.barriers.empty() || auxData.waiting.empty()) {
    return;
  }
  if (!pred) {
    pred = arith::ConstantIntOp::create(b, 1, 1);
  }
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value waitingVal = auxData.waiting.at(insertPoint).value;
  auto waitingType =
      cast<RankedTensorType>(auxData.waiting.at(insertPoint).type);
  uint32_t length = getMemDescLength(mbar);
  Value mbarOffset = tti::ExperimentalMemDescToI32Op::create(b, mbar);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {mbarOffset, lengthVal,   threadVal, phase,
                             pred,       barriersVal, waitingVal};
  createCallToCachedFunction(
      b, "set_waiting", args,
      /*assertInfo=*/std::nullopt, {barriersType, waitingType},
      [waitingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value baseThread = entryBlock->getArgument(2);
        Value phase = entryBlock->getArgument(3);
        Value pred = entryBlock->getArgument(4);

        Value barriers = entryBlock->getArgument(5);
        Value waitingPtr = entryBlock->getArgument(6);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value waiting = tti::createLoadScratchMemory(fb, fb.getLoc(),
                                                     waitingPtr, waitingType);
        Value descriptor = createBufferDescriptor(fb, mbarOffset, lengthVal);
        Value barriersEqBar =
            createCmpIntTensorScalar(fb, barriers, descriptor);
        barriersEqBar =
            convertAndBroadcast(fb, barriersEqBar, {1}, waitingType);
        Value ctaMask =
            createLeadCTAEffectMask(fb, waitingType, createCurrentCTAMask(fb));
        barriersEqBar = arith::AndIOp::create(fb, barriersEqBar, ctaMask);

        Value bitsPerThread =
            arith::ConstantIntOp::create(fb, WaitingBits::bitsPerThread, 32);
        Value flagBit =
            arith::ConstantIntOp::create(fb, WaitingBits::flagBit, 32);
        Value phaseBit =
            arith::ConstantIntOp::create(fb, WaitingBits::phaseBit, 32);
        Value one = arith::ConstantIntOp::create(fb, 1, 32);
        Value minusOne = arith::ConstantIntOp::create(fb, -1, 32);

        Value baseTimesBits =
            arith::MulIOp::create(fb, baseThread, bitsPerThread);
        Value flagShift = arith::AddIOp::create(fb, baseTimesBits, flagBit);
        Value phaseShift = arith::AddIOp::create(fb, baseTimesBits, phaseBit);

        Value flagMaskScalar = arith::ShLIOp::create(fb, one, flagShift);
        Value phaseMaskScalar = arith::ShLIOp::create(fb, one, phaseShift);
        Value combinedMask =
            arith::OrIOp::create(fb, flagMaskScalar, phaseMaskScalar);
        Value clearMaskScalar =
            arith::XOrIOp::create(fb, combinedMask, minusOne);

        Value flagMaskTensor =
            triton::SplatOp::create(fb, waitingType, flagMaskScalar);
        Value clearMaskTensor =
            triton::SplatOp::create(fb, waitingType, clearMaskScalar);
        Value phaseShiftTensor =
            triton::SplatOp::create(fb, waitingType, phaseShift);

        Value clearedWaiting =
            arith::AndIOp::create(fb, waiting, clearMaskTensor);
        Value withFlag =
            arith::OrIOp::create(fb, clearedWaiting, flagMaskTensor);

        Value phaseScalar = arith::AndIOp::create(fb, phase, one);
        Value phaseTensor =
            triton::SplatOp::create(fb, waitingType, phaseScalar);
        Value phaseBits =
            arith::ShLIOp::create(fb, phaseTensor, phaseShiftTensor);
        Value pendingWaiting = arith::OrIOp::create(fb, withFlag, phaseBits);

        auto condType = cast<RankedTensorType>(barriersEqBar.getType());
        Value predTensor = triton::SplatOp::create(fb, condType, pred);
        Value cond = arith::AndIOp::create(fb, barriersEqBar, predTensor);

        Value newWaiting =
            arith::SelectOp::create(fb, cond, pendingWaiting, waiting);
        createMaskedStoreScratchMemory(fb, fb.getLoc(), waitingPtr, newWaiting,
                                       waitingType, ctaMask);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createClearWaitingCall(ImplicitLocOpBuilder &b,
                                             Value mbar, int thread, Value pred,
                                             Operation *insertPoint) {
  if (auxData.barriers.empty() || auxData.waiting.empty()) {
    return;
  }
  if (!pred) {
    pred = arith::ConstantIntOp::create(b, 1, 1);
  }
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);

  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value waitingVal = auxData.waiting.at(insertPoint).value;
  auto waitingType =
      cast<RankedTensorType>(auxData.waiting.at(insertPoint).type);

  uint32_t length = getMemDescLength(mbar);
  Value mbarOffset = tti::ExperimentalMemDescToI32Op::create(b, mbar);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {mbarOffset, lengthVal,   threadVal,
                             pred,       barriersVal, waitingVal};
  createCallToCachedFunction(
      b, "clear_waiting", args,
      /*assertInfo=*/std::nullopt, {barriersType, waitingType},
      [waitingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value baseThread = entryBlock->getArgument(2);
        Value pred = entryBlock->getArgument(3);

        Value barriers = entryBlock->getArgument(4);
        Value waitingPtr = entryBlock->getArgument(5);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value waiting = tti::createLoadScratchMemory(fb, fb.getLoc(),
                                                     waitingPtr, waitingType);
        Value descriptor = createBufferDescriptor(fb, mbarOffset, lengthVal);
        Value barriersEqBar =
            createCmpIntTensorScalar(fb, barriers, descriptor);
        barriersEqBar =
            convertAndBroadcast(fb, barriersEqBar, {1}, waitingType);
        Value ctaMask =
            createLeadCTAEffectMask(fb, waitingType, createCurrentCTAMask(fb));
        barriersEqBar = arith::AndIOp::create(fb, barriersEqBar, ctaMask);

        Value bitsPerThread =
            arith::ConstantIntOp::create(fb, WaitingBits::bitsPerThread, 32);
        Value flagBit =
            arith::ConstantIntOp::create(fb, WaitingBits::flagBit, 32);
        Value phaseBit =
            arith::ConstantIntOp::create(fb, WaitingBits::phaseBit, 32);
        Value one = arith::ConstantIntOp::create(fb, 1, 32);
        Value minusOne = arith::ConstantIntOp::create(fb, -1, 32);

        Value baseTimesBits =
            arith::MulIOp::create(fb, baseThread, bitsPerThread);
        Value flagShift = arith::AddIOp::create(fb, baseTimesBits, flagBit);
        Value phaseShift = arith::AddIOp::create(fb, baseTimesBits, phaseBit);

        Value flagMaskScalar = arith::ShLIOp::create(fb, one, flagShift);
        Value phaseMaskScalar = arith::ShLIOp::create(fb, one, phaseShift);
        Value combinedMask =
            arith::OrIOp::create(fb, flagMaskScalar, phaseMaskScalar);
        Value clearMaskScalar =
            arith::XOrIOp::create(fb, combinedMask, minusOne);

        Value clearMaskTensor =
            triton::SplatOp::create(fb, waitingType, clearMaskScalar);
        Value clearedWaiting =
            arith::AndIOp::create(fb, waiting, clearMaskTensor);

        Value newWaiting =
            arith::SelectOp::create(fb, barriersEqBar, clearedWaiting, waiting);

        createMaskedStoreScratchMemory(fb, fb.getLoc(), waitingPtr, newWaiting,
                                       waitingType, ctaMask);
        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createSetActiveMaskCall(ImplicitLocOpBuilder &b,
                                              int activeMask,
                                              Operation *insertPoint) {
  if (auxData.activeMasks.empty())
    return;
  int64_t expandedActiveMask =
      expandActiveMask(activeMask, auxData.threadLayout.numBaseThreads);
  Value expandedActiveMaskVal =
      arith::ConstantIntOp::create(b, expandedActiveMask, 32);
  Value activeMasksVal = auxData.activeMasks.at(insertPoint).value;
  auto activeMasksType =
      cast<RankedTensorType>(auxData.activeMasks.at(insertPoint).type);
  SmallVector<Value> args = {expandedActiveMaskVal, activeMasksVal};
  createCallToCachedFunction(
      b, "set_active_mask", args,
      /*assertInfo=*/std::nullopt, {activeMasksType},
      [activeMasksType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value expandedActiveMaskVal = entryBlock->getArgument(0);
        Value activeMasksPtr = entryBlock->getArgument(1);

        Value newActiveMasks =
            triton::SplatOp::create(fb, activeMasksType, expandedActiveMaskVal);
        tti::createStoreScratchMemory(fb, fb.getLoc(), activeMasksPtr,
                                      newActiveMasks, activeMasksType,
                                      /*currentCTAOnly=*/true);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createRetireActiveThreadCall(ImplicitLocOpBuilder &b,
                                                   int thread,
                                                   Operation *insertPoint) {
  if (auxData.activeMasks.empty())
    return;
  int64_t threadMask =
      expandActiveMask(1u << thread, auxData.threadLayout.numBaseThreads);
  Value clearMaskVal = arith::ConstantIntOp::create(b, ~threadMask, 32);
  Value activeMasksVal = auxData.activeMasks.at(insertPoint).value;
  auto activeMasksType =
      cast<RankedTensorType>(auxData.activeMasks.at(insertPoint).type);
  SmallVector<Value> args = {clearMaskVal, activeMasksVal};
  createCallToCachedFunction(
      b, "retire_active_thread", args,
      /*assertInfo=*/std::nullopt, {activeMasksType},
      [activeMasksType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value clearMaskVal = entryBlock->getArgument(0);
        Value activeMasksPtr = entryBlock->getArgument(1);

        Value activeMasks = tti::createLoadScratchMemory(
            fb, fb.getLoc(), activeMasksPtr, activeMasksType);
        Value clearMask =
            triton::SplatOp::create(fb, activeMasksType, clearMaskVal);
        Value retiredMasks = arith::AndIOp::create(fb, activeMasks, clearMask);
        Value oneMask =
            tti::createConstIntTensor(fb, fb.getLoc(), 1, activeMasksType);
        Value newActiveMasks =
            arith::MaxUIOp::create(fb, retiredMasks, oneMask);
        tti::createStoreScratchMemory(fb, fb.getLoc(), activeMasksPtr,
                                      newActiveMasks, activeMasksType,
                                      /*currentCTAOnly=*/true);
        triton::ReturnOp::create(fb);
      });
}

Value FunctionBuilder::createCheckAllActiveWaitingCall(ImplicitLocOpBuilder &b,
                                                       Value pred,
                                                       Operation *insertPoint) {
  if (auxData.waiting.empty() || auxData.barrierStates.empty()) {
    return arith::ConstantIntOp::create(b, 1, 1);
  }
  if (!pred) {
    pred = arith::ConstantIntOp::create(b, 1, 1);
  }
  uint32_t flagMask = WaitingBits::makeInterleavedMask(
      WaitingBits::flagBit, auxData.threadLayout.numBaseThreads);
  uint32_t phaseMask = WaitingBits::makeInterleavedMask(
      WaitingBits::phaseBit, auxData.threadLayout.numBaseThreads);
  Value waitingVal = auxData.waiting.at(insertPoint).value;
  auto waitingType =
      cast<RankedTensorType>(auxData.waiting.at(insertPoint).type);
  Value barrierStatesVal = auxData.barrierStates.at(insertPoint).value;
  auto barrierStatesType =
      cast<RankedTensorType>(auxData.barrierStates.at(insertPoint).type);
  Region *region = b.getInsertionBlock()->getParent();
  auto waitingGlobalType = tti::getIntTensorType(
      region, waitingType.getShape(),
      waitingType.getElementType().getIntOrFloatBitWidth());
  auto barrierStatesGlobalType = tti::getIntTensorType(
      region, barrierStatesType.getShape(),
      barrierStatesType.getElementType().getIntOrFloatBitWidth());
  Value activeMasksVal = auxData.activeMasks.at(insertPoint).value;
  auto activeMasksType =
      cast<RankedTensorType>(auxData.activeMasks.at(insertPoint).type);
  auto activeMasksGlobalType = tti::getIntTensorType(
      region, activeMasksType.getShape(),
      activeMasksType.getElementType().getIntOrFloatBitWidth());
  SmallVector<Value> args = {pred, waitingVal, barrierStatesVal,
                             activeMasksVal};
  AssertInfo resultInfo{"", b.getI1Type()};
  return createCallToCachedFunction(
      b, "check_all_active_waiting", args, resultInfo,
      {waitingGlobalType, barrierStatesGlobalType, activeMasksGlobalType},
      [waitingGlobalType, barrierStatesGlobalType, activeMasksGlobalType,
       flagMask, phaseMask](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value pred = entryBlock->getArgument(0);

        Value waitingPtr = entryBlock->getArgument(1);
        Value barrierStatesPtr = entryBlock->getArgument(2);
        Value activeMasksPtr = entryBlock->getArgument(3);

        Value waiting = tti::createLoadScratchMemory(
            fb, fb.getLoc(), waitingPtr, waitingGlobalType);
        Value barrierStates = tti::createLoadScratchMemory(
            fb, fb.getLoc(), barrierStatesPtr, barrierStatesGlobalType);

        Value flagMaskTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), flagMask, waitingGlobalType);
        Value phaseMaskTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), phaseMask, waitingGlobalType);

        Value flags = arith::AndIOp::create(fb, waiting, flagMaskTensor);
        Value phases = arith::AndIOp::create(fb, waiting, phaseMaskTensor);
        Value shiftOneTensor =
            tti::createConstIntTensor(fb, fb.getLoc(), 1, waitingGlobalType);
        Value phasesAligned =
            arith::ShRUIOp::create(fb, phases, shiftOneTensor);

        Value phasesComplement =
            arith::XOrIOp::create(fb, phasesAligned, flagMaskTensor);
        Value waitingPhase0 =
            arith::AndIOp::create(fb, flags, phasesComplement);
        Value waitingPhase1 = arith::AndIOp::create(fb, flags, phasesAligned);

        Value oneState = tti::createConstIntTensor(fb, fb.getLoc(), 1,
                                                   barrierStatesGlobalType);
        Value barrierPhase = arith::AndIOp::create(fb, barrierStates, oneState);
        Value phaseIsOne = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                                 barrierPhase, oneState);

        phaseIsOne =
            convertAndBroadcast(fb, phaseIsOne, {0, 1}, waitingGlobalType);
        Value effectiveWaiting = arith::SelectOp::create(
            fb, phaseIsOne, waitingPhase1, waitingPhase0);
        Value waitingOr = reduce<arith::OrIOp>(fb, effectiveWaiting, {0, 1});
        auto waitingOrType = cast<RankedTensorType>(waitingOr.getType());
        Value activeMasks = tti::createLoadScratchMemory(
            fb, fb.getLoc(), activeMasksPtr, activeMasksGlobalType);
        Value activeMaskTensor =
            createConvertLayout(fb, activeMasks, waitingOrType.getEncoding());
        Value waitingMasked =
            arith::AndIOp::create(fb, waitingOr, activeMaskTensor);
        Value eqPerCTA = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                               waitingMasked, activeMaskTensor);
        Value zeroMask =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, waitingOrType);
        Value noActivePerCTA = arith::CmpIOp::create(
            fb, arith::CmpIPredicate::eq, activeMaskTensor, zeroMask);
        auto deadlockStatusType =
            cast<RankedTensorType>(eqPerCTA.getType()).clone(fb.getI32Type());
        Value waitingBits =
            arith::ExtUIOp::create(fb, deadlockStatusType, eqPerCTA);
        Value noActiveBits =
            arith::ExtUIOp::create(fb, deadlockStatusType, noActivePerCTA);
        Value statusShift =
            tti::createConstIntTensor(fb, fb.getLoc(), 1, deadlockStatusType);
        noActiveBits = arith::ShLIOp::create(fb, noActiveBits, statusShift);
        Value statusBits = arith::OrIOp::create(fb, waitingBits, noActiveBits);
        Value status = reduceAll<arith::AndIOp>(fb, statusBits);
        Value deadlockStatus = arith::ConstantIntOp::create(fb, 1, 32);
        Value deadlocked = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                                 status, deadlockStatus);

        Value vTrue = arith::ConstantOp::create(
            fb, deadlocked.getType(), fb.getIntegerAttr(fb.getI1Type(), 1));
        Value ok = arith::XOrIOp::create(fb, deadlocked, vTrue);
        Value predicatedOk = arith::SelectOp::create(fb, pred, ok, vTrue);
        triton::ReturnOp::create(fb, predicatedOk);
      },
      /*emitAssert=*/false);
}

void FunctionBuilder::createClusterBarrierRendezvousCall(
    ImplicitLocOpBuilder &b, int barrierIdx, int thread,
    uint64_t threadPeersMask, bool partitionScoped, bool publishVisibility,
    Operation *insertPoint) {
  assert(!auxData.waiting.empty() && !auxData.barrierStates.empty() &&
         !auxData.activeMasks.empty() &&
         "cluster rendezvous requires barrier deadlock state");
  Value barrierIdxVal = arith::ConstantIntOp::create(b, barrierIdx, 32);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value statesPtr = auxData.barrierStates.at(insertPoint).value;
  auto statesType =
      cast<RankedTensorType>(auxData.barrierStates.at(insertPoint).type);
  Value waitingPtr = auxData.waiting.at(insertPoint).value;
  auto waitingType =
      cast<RankedTensorType>(auxData.waiting.at(insertPoint).type);
  Value lock = auxData.lock.at(insertPoint).value;
  Value vTrue = arith::ConstantIntOp::create(b, 1, 1);
  int numCTAs = ttg::lookupNumCTAs(insertPoint);

  auto getPhase = [&]() {
    Value states =
        tti::createLoadScratchMemory(b, b.getLoc(), statesPtr, statesType);
    Value mask = createVirtualBarrierMask(b, barrierIdxVal, statesType);
    Value phase = arith::AndIOp::create(
        b, states, tti::createConstIntTensor(b, b.getLoc(), 1, statesType));
    phase = arith::SelectOp::create(
        b, mask, phase,
        tti::createConstIntTensor(b, b.getLoc(), 0, statesType));
    Value result = arith::TruncIOp::create(b, b.getI1Type(),
                                           reduceAll<arith::OrIOp>(b, phase));
    // Every warp must sample this phase before the elected warp can advance
    // it; otherwise sibling warps can wait for opposite rendezvous epochs.
    ttg::BarrierOp::create(b, b.getLoc(), ttg::AddrSpace::GlobalRead);
    return result;
  };
  auto updateWaiting = [&](Value phase, Value pred, bool markWaiting) {
    Value waiting =
        tti::createLoadScratchMemory(b, b.getLoc(), waitingPtr, waitingType);
    Value mask = arith::AndIOp::create(
        b, createDimMask(b, barrierIdxVal, waitingType, /*dim=*/1),
        createLeadCTAEffectMask(b, waitingType,
                                arith::ConstantIntOp::create(b, 1, 32)));
    Value predTensor = triton::SplatOp::create(
        b, cast<RankedTensorType>(mask.getType()), pred);
    mask = arith::AndIOp::create(b, mask, predTensor);
    Value updated = updateWaitingBits(b, waiting, waitingType, threadVal, phase,
                                      mask, markWaiting);
    tti::createStoreScratchMemory(b, b.getLoc(), waitingPtr, updated,
                                  waitingType);
  };

  tti::ExperimentalLockAcquireOp::create(b, lock, vTrue);
  Value savedPhase = getPhase();
  updateWaiting(savedPhase, vTrue, /*markWaiting=*/true);
  Value completed =
      arriveVirtualBarrier(b, statesPtr, statesType, barrierIdxVal, numCTAs);
  if (publishVisibility) {
    for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM})
      createPublishClusterVisibilityCall(b, completed, thread, threadPeersMask,
                                         partitionScoped, memType, insertPoint);
    createPublishClusterProxyAccessesCall(b, completed, thread, partitionScoped,
                                          insertPoint);
  }
  Value ok = createCheckAllActiveWaitingCall(b, vTrue, insertPoint);
  tti::ExperimentalLockReleaseOp::create(b, lock, vTrue);
  tti::createAssertInThread(b, ok, "Deadlock detected at a cluster barrier");

  Block *entryBlock = b.getInsertionBlock();
  Block *continueBlock = entryBlock->splitBlock(b.getInsertionPoint());
  Block *phasePollBlock = new Block();
  entryBlock->getParent()->getBlocks().insert(continueBlock->getIterator(),
                                              phasePollBlock);
  b.setInsertionPointToEnd(entryBlock);
  cf::BranchOp::create(b, phasePollBlock);

  b.setInsertionPointToStart(phasePollBlock);
  tti::ExperimentalLockAcquireOp::create(b, lock, vTrue);
  Value currentPhase = getPhase();
  Value phaseChanged = arith::CmpIOp::create(b, arith::CmpIPredicate::ne,
                                             currentPhase, savedPhase);
  updateWaiting(savedPhase, phaseChanged, /*markWaiting=*/false);
  ok = createCheckAllActiveWaitingCall(b, vTrue, insertPoint);
  tti::ExperimentalLockReleaseOp::create(b, lock, vTrue);
  tti::createAssertInThread(b, ok, "Deadlock detected at a cluster barrier");
  cf::CondBranchOp::create(b, phaseChanged, continueBlock, ValueRange{},
                           phasePollBlock, ValueRange{});
  b.setInsertionPointToStart(continueBlock);
}

void FunctionBuilder::createVerifyBarrierCanInitCall(ImplicitLocOpBuilder &b,
                                                     Value mbar, Value pred,
                                                     Operation *insertPoint,
                                                     Value recipientCTAs) {
  assert(!auxData.barriers.empty() &&
         "barrier descriptors must exist when verifying barrier init");
  assert(!auxData.barrierStates.empty() &&
         "barrier states must exist when verifying barrier init");
  createVerifyBarrierStateCall(
      b, "Barrier re-initialized without prior invalidation",
      tti::ExperimentalMemDescToI32Op::create(b, mbar),
      arith::ConstantIntOp::create(b, getMemDescLength(mbar), 32), pred,
      auxData.barriers.at(insertPoint), auxData.barrierStates.at(insertPoint),
      recipientCTAs, /*initialized=*/false);
}

void FunctionBuilder::createVerifyBarrierInitializedCall(
    ImplicitLocOpBuilder &b, Value mbar, Value pred, Operation *insertPoint,
    Value recipientCTAs) {
  assert(!auxData.barriers.empty() &&
         "barrier descriptors must exist when verifying barrier use");
  assert(!auxData.barrierStates.empty() &&
         "barrier states must exist when verifying barrier use");
  createVerifyBarrierStateCall(
      b, "Barrier used before initialization or after invalidation",
      tti::ExperimentalMemDescToI32Op::create(b, mbar),
      arith::ConstantIntOp::create(b, getMemDescLength(mbar), 32), pred,
      auxData.barriers.at(insertPoint), auxData.barrierStates.at(insertPoint),
      recipientCTAs, /*initialized=*/true);
}

void FunctionBuilder::createVerifyBarrierMemoryAvailableCall(
    ImplicitLocOpBuilder &b, Value barrierCTAs, Value pred,
    Operation *insertPoint) {
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  ValueType states = auxData.barrierStates.at(insertPoint);
  auto statesType = cast<RankedTensorType>(states.type);
  SmallVector<Value> args = {barrierCTAs, pred, states.value};
  AssertInfo assertInfo{"Shared memory reused before barrier invalidation",
                        b.getI1Type()};
  createCallToCachedFunction(
      b, "verify_barrier_can_init", args, assertInfo, {statesType},
      [statesType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value barrierCTAs = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value states = tti::createLoadScratchMemory(
            fb, fb.getLoc(), entryBlock->getArgument(2), statesType);
        Value zero = tti::createConstIntTensor(fb, fb.getLoc(), 0, statesType);
        Value available =
            arith::CmpIOp::create(fb, arith::CmpIPredicate::eq, states, zero);
        Value selected = createBarrierOwnerMask(fb, barrierCTAs, statesType);
        auto maskType = cast<RankedTensorType>(selected.getType());
        Value vTrue = tti::createConstIntTensor(fb, fb.getLoc(), 1, maskType);
        available = arith::SelectOp::create(fb, selected, available, vTrue);
        Value active = triton::SplatOp::create(fb, maskType, pred);
        available = arith::SelectOp::create(fb, active, available, vTrue);
        triton::ReturnOp::create(fb, reduceAll<arith::AndIOp>(fb, available));
      });
}

void FunctionBuilder::createInitBarrierStateCall(ImplicitLocOpBuilder &b,
                                                 Value mbar, int count,
                                                 Value pred,
                                                 Operation *insertPoint) {
  assert(count >= 0 && (uint64_t)count <= BarrierBits::countMask &&
         "barrier init count exceeds barrier state capacity");

  if (auxData.barriers.empty() || auxData.barrierStates.empty()) {
    return;
  }
  if (!pred) {
    pred = arith::ConstantIntOp::create(b, 1, 1);
  }
  Value countVal = arith::ConstantIntOp::create(b, count, 32);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value barrierStatesVal = auxData.barrierStates.at(insertPoint).value;
  auto barrierStatesType =
      cast<RankedTensorType>(auxData.barrierStates.at(insertPoint).type);
  uint32_t length = getMemDescLength(mbar);
  Value mbarOffset = tti::ExperimentalMemDescToI32Op::create(b, mbar);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {mbarOffset, lengthVal,   countVal,
                             pred,       barriersVal, barrierStatesVal};
  createCallToCachedFunction(
      b, "init_barrier_state", args,
      /*assertInfo=*/std::nullopt, {barriersType, barrierStatesType},
      [barrierStatesType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value count = entryBlock->getArgument(2);
        Value pred = entryBlock->getArgument(3);

        Value barriers = entryBlock->getArgument(4);
        Value statesPtr = entryBlock->getArgument(5);

        Value states = tti::createLoadScratchMemory(fb, fb.getLoc(), statesPtr,
                                                    barrierStatesType);
        Value descriptor = createBufferDescriptor(fb, mbarOffset, lengthVal);
        Value mask = createCmpIntTensorScalar(fb, barriers, descriptor);
        mask = convertAndBroadcast(fb, mask, {1}, barrierStatesType);

        Value countWide = adjustIntegerWidth(
            fb, count, cast<IntegerType>(barrierStatesType.getElementType()));
        Value countMask =
            arith::ConstantIntOp::create(fb, BarrierBits::countMask, 64);
        Value maskedCount = arith::AndIOp::create(fb, countWide, countMask);
        Value countTensor =
            triton::SplatOp::create(fb, barrierStatesType, maskedCount);

        Value shiftInitTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::initCountLsb, barrierStatesType);
        Value shiftCurrentTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::currentCountLsb, barrierStatesType);

        Value initField =
            arith::ShLIOp::create(fb, countTensor, shiftInitTensor);
        Value currentField =
            arith::ShLIOp::create(fb, countTensor, shiftCurrentTensor);
        Value newState = arith::OrIOp::create(fb, initField, currentField);

        Value updated = arith::SelectOp::create(fb, mask, newState, states);
        auto condType = cast<RankedTensorType>(mask.getType());
        Value predTensor = triton::SplatOp::create(fb, condType, pred);
        updated = arith::SelectOp::create(fb, predTensor, updated, states);
        createCTAScopedStoreScratchMemory(fb, fb.getLoc(), statesPtr, updated,
                                          barrierStatesType,
                                          createCurrentCTAMask(fb));
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createInvalidateBarrierStateCall(ImplicitLocOpBuilder &b,
                                                       Value mbar, Value pred,
                                                       Operation *insertPoint) {
  Value descriptor = createBufferDescriptor(
      b, tti::ExperimentalMemDescToI32Op::create(b, mbar),
      arith::ConstantIntOp::create(b, getMemDescLength(mbar), 32));
  Value selectedBarriers = createCmpIntTensorScalar(
      b, auxData.barriers.at(insertPoint).value, descriptor);
  createInvalidateBarrierStateCallImpl(b, selectedBarriers, pred, insertPoint,
                                       /*allowUninitialized=*/false);
}

void FunctionBuilder::createInvalidateBarrierStorageCall(
    ImplicitLocOpBuilder &b, Value barrierCTAs, Value pred,
    Operation *insertPoint) {
  auto ownersType = cast<RankedTensorType>(barrierCTAs.getType());
  Value currentCTA =
      triton::SplatOp::create(b, ownersType, createCurrentCTAMask(b));
  Value owners = arith::AndIOp::create(b, barrierCTAs, currentCTA);
  Value selectedBarriers = arith::CmpIOp::create(
      b, arith::CmpIPredicate::ne, owners,
      tti::createConstIntTensor(b, b.getLoc(), 0, ownersType));
  createInvalidateBarrierStateCallImpl(b, selectedBarriers, pred, insertPoint,
                                       /*allowUninitialized=*/true);
}

void FunctionBuilder::createInvalidateBarrierStateCallImpl(
    ImplicitLocOpBuilder &b, Value selectedBarriers, Value pred,
    Operation *insertPoint, bool allowUninitialized) {
  assert(!auxData.barriers.empty() &&
         "barrier descriptors must exist when invalidating a barrier");
  assert(!auxData.barrierStates.empty() &&
         "barrier states must exist when invalidating a barrier");
  assert(!auxData.waiting.empty() &&
         "waiting state must exist when invalidating a barrier");
  if (!pred) {
    pred = arith::ConstantIntOp::create(b, 1, 1);
  }
  Value barrierStatesVal = auxData.barrierStates.at(insertPoint).value;
  auto barrierStatesType =
      cast<RankedTensorType>(auxData.barrierStates.at(insertPoint).type);
  auto selectedType =
      tti::getSlicedTensorType(barrierStatesType, {1}, b.getI1Type());
  if (selectedBarriers.getType() != selectedType)
    selectedBarriers =
        ttg::ConvertLayoutOp::create(b, selectedType, selectedBarriers);
  Value waitingVal = auxData.waiting.at(insertPoint).value;
  auto waitingType =
      cast<RankedTensorType>(auxData.waiting.at(insertPoint).type);
  SmallVector<Value> args = {selectedBarriers, pred, barrierStatesVal,
                             waitingVal};
  AssertInfo statusInfo{"", b.getI32Type()};
  Value status = createCallToCachedFunction(
      b,
      allowUninitialized ? "invalidate_barrier_storage"
                         : "invalidate_barrier_state",
      args, statusInfo, {barrierStatesType, waitingType},
      [barrierStatesType, waitingType,
       allowUninitialized](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value selectedBarriers = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value statesPtr = entryBlock->getArgument(2);
        Value waitingPtr = entryBlock->getArgument(3);

        Value states = tti::createLoadScratchMemory(fb, fb.getLoc(), statesPtr,
                                                    barrierStatesType);
        Value waiting = tti::createLoadScratchMemory(fb, fb.getLoc(),
                                                     waitingPtr, waitingType);
        Value mask =
            convertAndBroadcast(fb, selectedBarriers, {1}, barrierStatesType);
        Value currentCTA = createCurrentCTAMask(fb);
        Value stateCTAMask =
            createCTASetMask(fb, barrierStatesType, /*dim=*/0, currentCTA);
        Value selectedStates = arith::AndIOp::create(fb, mask, stateCTAMask);

        Value zeroState =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, barrierStatesType);
        Value zeroWaiting =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, waitingType);

        Value initialized = arith::CmpIOp::create(fb, arith::CmpIPredicate::ne,
                                                  states, zeroState);
        Value stateTrue = tti::createConstIntTensor(
            fb, fb.getLoc(), 1, cast<RankedTensorType>(initialized.getType()));
        initialized =
            arith::SelectOp::create(fb, selectedStates, initialized, stateTrue);
        initialized = reduceAll<arith::AndIOp>(fb, initialized);
        if (allowUninitialized) {
          Value active = arith::CmpIOp::create(fb, arith::CmpIPredicate::ne,
                                               states, zeroState);
          initialized = reduceAll<arith::OrIOp>(
              fb, arith::AndIOp::create(fb, selectedStates, active));
        }

        Value waitingMask = convertAndBroadcast(fb, mask, {0, 1}, waitingType);
        Value waitingCTAMask =
            createLeadCTAEffectMask(fb, waitingType, currentCTA);
        Value selectedWaiters = arith::AndIOp::create(
            fb, waitingMask,
            createCTASetMask(fb, waitingType, /*dim=*/0, currentCTA));
        Value idle = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                           waiting, zeroWaiting);
        Value waitingTrue = tti::createConstIntTensor(
            fb, fb.getLoc(), 1, cast<RankedTensorType>(idle.getType()));
        idle = arith::SelectOp::create(fb, selectedWaiters, idle, waitingTrue);
        idle = reduceAll<arith::AndIOp>(fb, idle);

        Value valid = allowUninitialized
                          ? idle
                          : arith::AndIOp::create(fb, initialized, idle);
        Value shouldInvalidate = arith::AndIOp::create(fb, pred, valid);

        auto stateCondType = cast<RankedTensorType>(selectedStates.getType());
        Value stateValid =
            triton::SplatOp::create(fb, stateCondType, shouldInvalidate);
        Value stateUpdateMask =
            arith::AndIOp::create(fb, selectedStates, stateValid);
        Value updatedStates =
            arith::SelectOp::create(fb, stateUpdateMask, zeroState, states);
        Value waitingUpdateMask =
            arith::AndIOp::create(fb, waitingMask, waitingCTAMask);
        auto waitingCondType =
            cast<RankedTensorType>(waitingUpdateMask.getType());
        Value waitingValid =
            triton::SplatOp::create(fb, waitingCondType, shouldInvalidate);
        waitingUpdateMask =
            arith::AndIOp::create(fb, waitingUpdateMask, waitingValid);
        Value updatedWaiting = arith::SelectOp::create(fb, waitingUpdateMask,
                                                       zeroWaiting, waiting);
        createCTAScopedStoreScratchMemory(fb, fb.getLoc(), statesPtr,
                                          updatedStates, barrierStatesType,
                                          currentCTA);
        createMaskedStoreScratchMemory(fb, fb.getLoc(), waitingPtr,
                                       updatedWaiting, waitingType,
                                       waitingCTAMask);

        Value vTrue = arith::ConstantIntOp::create(fb, 1, 1);
        initialized = arith::SelectOp::create(fb, pred, initialized, vTrue);
        idle = arith::SelectOp::create(fb, pred, idle, vTrue);
        Value initializedI32 =
            arith::ExtUIOp::create(fb, fb.getI32Type(), initialized);
        Value idleI32 = arith::ExtUIOp::create(fb, fb.getI32Type(), idle);
        idleI32 = arith::ShLIOp::create(
            fb, idleI32, arith::ConstantIntOp::create(fb, 1, 32));
        Value status = arith::OrIOp::create(fb, initializedI32, idleI32);
        triton::ReturnOp::create(fb, ValueRange{status});
      },
      /*emitAssert=*/false);

  Value initialized = arith::TruncIOp::create(b, b.getI1Type(), status);
  if (!allowUninitialized) {
    tti::createAssertInThread(
        b, initialized,
        "Barrier used before initialization or after invalidation");
  }
  Value idle = arith::TruncIOp::create(
      b, b.getI1Type(),
      arith::ShRUIOp::create(b, status,
                             arith::ConstantIntOp::create(b, 1, 32)));
  tti::createAssertInThread(b, idle,
                            "Barrier invalidated while a thread is waiting");
  Value clearPred = pred;
  if (allowUninitialized)
    clearPred =
        tti::maybeAnd(b, pred, arith::AndIOp::create(b, initialized, idle));
  createClearBarrierStorageTrackingCall(b, selectedBarriers, clearPred,
                                        insertPoint);
}

void FunctionBuilder::createVerifyAndUpdateBarrierStateCall(
    ImplicitLocOpBuilder &b, Value mbar, int count, Value pred,
    Operation *insertPoint, Value recipientCTAs, int txCount) {
  assert(count >= 0 && (uint64_t)count <= BarrierBits::countMask &&
         "barrier arrive count exceeds barrier state capacity");
  assert(txCount >= BarrierBits::txCountMin &&
         txCount <= BarrierBits::txCountMax &&
         "barrier tx-count delta exceeds barrier state capacity");

  if (auxData.barriers.empty() || auxData.barrierStates.empty()) {
    return;
  }
  if (!pred) {
    pred = arith::ConstantIntOp::create(b, 1, 1);
  }
  Value countVal = arith::ConstantIntOp::create(b, count, 32);
  Value txCountVal = arith::ConstantIntOp::create(b, txCount, 64);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value barrierStatesVal = auxData.barrierStates.at(insertPoint).value;
  auto barrierStatesType =
      cast<RankedTensorType>(auxData.barrierStates.at(insertPoint).type);
  uint32_t length = getMemDescLength(mbar);
  Value mbarOffset = tti::ExperimentalMemDescToI32Op::create(b, mbar);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {mbarOffset,       lengthVal,    countVal,
                             txCountVal,       pred,         barriersVal,
                             barrierStatesVal, recipientCTAs};
  SmallVector<RankedTensorType> trackingTypes;
  ManglingArgs specializationArgs{barriersType, barrierStatesType};
  auto appendTracking = [&](AuxDataMap::RegionToValueMap &tracking) {
    if (tracking.empty())
      return;
    ValueType value = tracking.at(insertPoint);
    auto type = cast<RankedTensorType>(value.type);
    args.push_back(value.value);
    trackingTypes.push_back(type);
    specializationArgs.append(type);
  };
  for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
    appendTracking(auxData.writeTracking[(int)memType]);
    appendTracking(auxData.readTracking[(int)memType]);
  }
  appendTracking(auxData.proxyAccessTracking);

  AssertInfo statusInfo{"", b.getI32Type()};
  Value status = createCallToCachedFunction(
      b, "verify_and_update_barrier_state", args, statusInfo,
      specializationArgs,
      [barrierStatesType, trackingTypes](ImplicitLocOpBuilder &fb,
                                         Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value count = entryBlock->getArgument(2);
        Value txCount = entryBlock->getArgument(3);
        Value pred = entryBlock->getArgument(4);

        Value barriers = entryBlock->getArgument(5);
        Value statesPtr = entryBlock->getArgument(6);
        Value recipientCTAs = entryBlock->getArgument(7);

        Value states = tti::createLoadScratchMemory(fb, fb.getLoc(), statesPtr,
                                                    barrierStatesType);
        Value descriptor = createBufferDescriptor(fb, mbarOffset, lengthVal);
        Value mask = createCmpIntTensorScalar(fb, barriers, descriptor);
        mask = convertAndBroadcast(fb, mask, {1}, barrierStatesType);

        Value zero32 =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, barrierStatesType);
        Value initialized =
            arith::CmpIOp::create(fb, arith::CmpIPredicate::ne, states, zero32);
        auto condType = cast<RankedTensorType>(initialized.getType());
        Value vTrue = tti::createConstIntTensor(fb, fb.getLoc(), 1, condType);
        initialized = arith::SelectOp::create(fb, mask, initialized, vTrue);
        Value ctaMask =
            createCTASetMask(fb, condType, /*dim=*/0, recipientCTAs);
        initialized = arith::SelectOp::create(fb, ctaMask, initialized, vTrue);
        Value predTensor = triton::SplatOp::create(fb, condType, pred);
        initialized =
            arith::SelectOp::create(fb, predTensor, initialized, vTrue);

        Value maskFF = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::countMask, barrierStatesType);
        Value shiftCurrentTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::currentCountLsb, barrierStatesType);
        Value shiftTxTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::txCountLsb, barrierStatesType);
        Value shiftTxSignTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), 64 - BarrierBits::txCountBitWidth,
            barrierStatesType);

        Value currentCount =
            arith::ShRUIOp::create(fb, states, shiftCurrentTensor);
        currentCount = arith::AndIOp::create(fb, currentCount, maskFF);
        Value currentTxCount =
            arith::ShRUIOp::create(fb, states, shiftTxTensor);
        currentTxCount =
            arith::ShLIOp::create(fb, currentTxCount, shiftTxSignTensor);
        currentTxCount =
            arith::ShRSIOp::create(fb, currentTxCount, shiftTxSignTensor);

        Value countMask =
            arith::ConstantIntOp::create(fb, BarrierBits::countMask, 64);
        Value countWide = adjustIntegerWidth(
            fb, count, cast<IntegerType>(barrierStatesType.getElementType()));
        Value maskedCount = arith::AndIOp::create(fb, countWide, countMask);
        Value arriveCount =
            triton::SplatOp::create(fb, barrierStatesType, maskedCount);
        Value txCountTensor =
            triton::SplatOp::create(fb, barrierStatesType, txCount);

        Value newCurrent = arith::SubIOp::create(fb, currentCount, arriveCount);
        Value newCurrentMasked =
            arith::SelectOp::create(fb, mask, newCurrent, zero32);
        Value newTxCount =
            arith::AddIOp::create(fb, currentTxCount, txCountTensor);
        Value newTxCountMasked =
            arith::SelectOp::create(fb, mask, newTxCount, zero32);
        Value arrivalsNonNegative = arith::CmpIOp::create(
            fb, arith::CmpIPredicate::sge, newCurrentMasked, zero32);
        Value minTxCount = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::txCountMin, barrierStatesType,
            /*isSigned=*/true);
        Value maxTxCount = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::txCountMax, barrierStatesType);
        Value txCountInRange = arith::AndIOp::create(
            fb,
            arith::CmpIOp::create(fb, arith::CmpIPredicate::sge,
                                  newTxCountMasked, minTxCount),
            arith::CmpIOp::create(fb, arith::CmpIPredicate::sle,
                                  newTxCountMasked, maxTxCount));
        Value valid =
            arith::AndIOp::create(fb, arrivalsNonNegative, txCountInRange);
        valid = arith::SelectOp::create(fb, ctaMask, valid, vTrue);
        Value predicatedValid =
            arith::SelectOp::create(fb, predTensor, valid, vTrue);

        auto statusType = condType.clone(fb.getI32Type());
        Value initializedBits =
            arith::ExtUIOp::create(fb, statusType, initialized);
        Value validBits =
            arith::ExtUIOp::create(fb, statusType, predicatedValid);
        Value statusShift =
            tti::createConstIntTensor(fb, fb.getLoc(), 1, statusType);
        validBits = arith::ShLIOp::create(fb, validBits, statusShift);
        Value statusBits = arith::OrIOp::create(fb, initializedBits, validBits);
        Value packedStatus = reduceAll<arith::AndIOp>(fb, statusBits);
        Value allInitialized =
            arith::TruncIOp::create(fb, fb.getI1Type(), packedStatus);
        Value validShift = arith::ConstantIntOp::create(fb, 1, 32);
        Value allValid = arith::TruncIOp::create(
            fb, fb.getI1Type(),
            arith::ShRUIOp::create(fb, packedStatus, validShift));
        Value shouldUpdate = arith::AndIOp::create(
            fb, pred, arith::AndIOp::create(fb, allInitialized, allValid));
        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, shouldUpdate);
        fb.setInsertionPointToStart(ifBlock);

        Value one32 =
            tti::createConstIntTensor(fb, fb.getLoc(), 1, barrierStatesType);
        Value shiftInitTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::initCountLsb, barrierStatesType);
        Value phase = arith::AndIOp::create(fb, states, one32);
        Value initCount = arith::ShRUIOp::create(fb, states, shiftInitTensor);
        initCount = arith::AndIOp::create(fb, initCount, maskFF);
        Value updatedCurrent =
            arith::SelectOp::create(fb, mask, newCurrent, currentCount);
        Value updatedTxCount =
            arith::SelectOp::create(fb, mask, newTxCount, currentTxCount);

        Value zeroCond = arith::AndIOp::create(
            fb,
            arith::CmpIOp::create(fb, arith::CmpIPredicate::eq, updatedCurrent,
                                  zero32),
            arith::CmpIOp::create(fb, arith::CmpIPredicate::eq, updatedTxCount,
                                  zero32));
        zeroCond = arith::AndIOp::create(fb, zeroCond, mask);
        Value zeroCondI32 =
            arith::ExtUIOp::create(fb, barrierStatesType, zeroCond);
        Value newPhase = arith::XOrIOp::create(fb, phase, zeroCondI32);
        Value newCurrentValue =
            arith::SelectOp::create(fb, zeroCond, initCount, updatedCurrent);
        Value newTxCountValue =
            arith::SelectOp::create(fb, zeroCond, zero32, updatedTxCount);

        Value initField = arith::ShLIOp::create(fb, initCount, shiftInitTensor);
        Value currentField =
            arith::ShLIOp::create(fb, newCurrentValue, shiftCurrentTensor);
        Value txCountMask = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::txCountMask, barrierStatesType);
        Value txCountField =
            arith::AndIOp::create(fb, newTxCountValue, txCountMask);
        txCountField = arith::ShLIOp::create(fb, txCountField, shiftTxTensor);
        Value newState = arith::OrIOp::create(fb, newPhase, initField);
        newState = arith::OrIOp::create(fb, newState, currentField);
        newState = arith::OrIOp::create(fb, newState, txCountField);

        Value updated = arith::SelectOp::create(fb, mask, newState, states);
        createCTAScopedStoreScratchMemory(fb, fb.getLoc(), statesPtr, updated,
                                          barrierStatesType, recipientCTAs);

        if (!trackingTypes.empty()) {
          Value recipientMask =
              createCTASetMask(fb, barrierStatesType, /*dim=*/0, recipientCTAs);
          Value completed = arith::AndIOp::create(fb, zeroCond, recipientMask);
          Value anyCompleted = reduceAll<arith::OrIOp>(fb, completed);
          auto [beforeClear, clearBlock, afterClear] =
              createIfBlock(fb, anyCompleted);
          fb.setInsertionPointToStart(clearBlock);

          auto phaseStateType = cast<RankedTensorType>(
              barrierStatesType.cloneWith(std::nullopt, fb.getI32Type()));
          Value nextPhases =
              arith::TruncIOp::create(fb, phaseStateType, newPhase);
          for (auto [index, trackingType] : llvm::enumerate(trackingTypes)) {
            Value trackingPtr = entryBlock->getArgument(8 + index);
            Value completedMask =
                convertAndBroadcast(fb, completed, {2, 3}, trackingType);
            Value nextPhase =
                convertAndBroadcast(fb, nextPhases, {2, 3}, trackingType);
            Value phaseIndices =
                createDimIndices(fb, trackingType, trackingType.getRank() - 1);
            Value phaseMask = arith::CmpIOp::create(
                fb, arith::CmpIPredicate::eq, phaseIndices, nextPhase);
            Value clearMask =
                arith::AndIOp::create(fb, completedMask, phaseMask);
            Value zero =
                tti::createConstIntTensor(fb, fb.getLoc(), 0, trackingType);
            tti::createStoreScratchMemory(fb, fb.getLoc(), trackingPtr, zero,
                                          trackingType,
                                          /*currentCTAOnly=*/false, clearMask);
          }
        }

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb, packedStatus);
      },
      /*emitAssert=*/false);

  Value initialized = arith::TruncIOp::create(b, b.getI1Type(), status);
  tti::createAssertInThread(
      b, initialized,
      "Barrier used before initialization or after invalidation");
  Value shift = arith::ConstantIntOp::create(b, 1, 32);
  Value valid = arith::TruncIOp::create(
      b, b.getI1Type(), arith::ShRUIOp::create(b, status, shift));
  tti::createAssertInThread(b, valid,
                            "Barrier arrive underflow: current count or "
                            "tx-count would become invalid");
}

void FunctionBuilder::createPublishWriteVisibilityCall(
    ImplicitLocOpBuilder &b, Value bufferMask, uint64_t threadMask, Value pred,
    MemType memType, Operation *insertPoint, Value effectCTAs) {
  const bool publishWrite = !auxData.writeVisibility[(int)memType].empty();
  const bool clearWrites = !auxData.writeTracking[(int)memType].empty();
  const bool clearReads = !auxData.readVisibility[(int)memType].empty();
  const bool clearReadTracking = !auxData.readTracking[(int)memType].empty();
  if (!publishWrite && !clearWrites && !clearReads && !clearReadTracking)
    return;

  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value threadMaskVal = arith::ConstantIntOp::create(b, threadMask, 64);
  SmallVector<Value> args = {bufferMask, pred, threadMaskVal, effectCTAs};
  ManglingArgs specializationArgs;
  specializationArgs.append(static_cast<uint64_t>(publishWrite));
  specializationArgs.append(static_cast<uint64_t>(clearWrites));
  specializationArgs.append(static_cast<uint64_t>(clearReads));
  specializationArgs.append(static_cast<uint64_t>(clearReadTracking));

  RankedTensorType writeVisibilityType;
  if (publishWrite) {
    ValueType visibility =
        auxData.writeVisibility[(int)memType].at(insertPoint);
    writeVisibilityType = cast<RankedTensorType>(visibility.type);
    args.push_back(visibility.value);
    specializationArgs.append(writeVisibilityType);
  }

  RankedTensorType writeTrackingType;
  if (clearWrites) {
    ValueType tracking = auxData.writeTracking[(int)memType].at(insertPoint);
    writeTrackingType = cast<RankedTensorType>(tracking.type);
    args.push_back(tracking.value);
    specializationArgs.append(writeTrackingType);
  }

  RankedTensorType readVisibilityType;
  if (clearReads) {
    ValueType visibility = auxData.readVisibility[(int)memType].at(insertPoint);
    readVisibilityType = cast<RankedTensorType>(visibility.type);
    args.push_back(visibility.value);
    specializationArgs.append(readVisibilityType);
  }

  RankedTensorType readTrackingType;
  if (clearReadTracking) {
    ValueType tracking = auxData.readTracking[(int)memType].at(insertPoint);
    readTrackingType = cast<RankedTensorType>(tracking.type);
    args.push_back(tracking.value);
    specializationArgs.append(readTrackingType);
  }

  createCallToCachedFunction(
      b, "publish_write_visibility", args, /*assertInfo=*/std::nullopt,
      specializationArgs,
      [publishWrite, clearWrites, clearReads, clearReadTracking,
       writeVisibilityType, writeTrackingType, readVisibilityType,
       readTrackingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value bufferMask = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value threadMaskVal = entryBlock->getArgument(2);
        Value effectCTAs = entryBlock->getArgument(3);
        unsigned nextArg = 4;

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        if (publishWrite) {
          Value visibilityPtr = entryBlock->getArgument(nextArg++);
          Value visibilityMask =
              convertAndBroadcast(fb, bufferMask, {1}, writeVisibilityType);
          Value relationMask =
              createLeadCTAEffectMask(fb, writeVisibilityType, effectCTAs);
          visibilityMask =
              arith::AndIOp::create(fb, visibilityMask, relationMask);
          auto elemType =
              cast<IntegerType>(writeVisibilityType.getElementType());
          Value threadMaskElem =
              adjustIntegerWidth(fb, threadMaskVal, elemType);
          Value threadMaskTensor =
              triton::SplatOp::create(fb, writeVisibilityType, threadMaskElem);
          tti::createStoreScratchMemory(fb, fb.getLoc(), visibilityPtr,
                                        threadMaskTensor, writeVisibilityType,
                                        /*currentCTAOnly=*/false,
                                        visibilityMask);
        }

        auto clearTable = [&](RankedTensorType tableType) {
          Value tablePtr = entryBlock->getArgument(nextArg++);
          Value tableMask = convertAndBroadcast(fb, bufferMask, {1}, tableType);
          Value ctaMask =
              createCTASetMask(fb, tableType, /*dim=*/0, effectCTAs);
          tableMask = arith::AndIOp::create(fb, tableMask, ctaMask);
          Value zero = tti::createConstIntTensor(fb, fb.getLoc(), 0, tableType);
          tti::createStoreScratchMemory(fb, fb.getLoc(), tablePtr, zero,
                                        tableType, /*currentCTAOnly=*/false,
                                        tableMask);
        };

        if (clearWrites)
          clearTable(writeTrackingType);
        if (clearReads)
          clearTable(readVisibilityType);
        if (clearReadTracking)
          clearTable(readTrackingType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createSetReadVisibilityCall(
    ImplicitLocOpBuilder &b, Value bufferMask, int reader,
    uint64_t observerMask, Value pred, MemType memType, Operation *insertPoint,
    Value effectCTAs, Value bufferIndex) {

  if (auxData.readVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value readerMaskVal = arith::ConstantIntOp::create(b, 1ULL << reader, 64);
  Value observerMaskVal = arith::ConstantIntOp::create(b, observerMask, 64);
  Value readVisibilityVal =
      auxData.readVisibility[(int)memType].at(insertPoint).value;
  auto readVisibilityType = cast<RankedTensorType>(
      auxData.readVisibility[(int)memType].at(insertPoint).type);
  bool hasReadTracking = !auxData.readTracking[(int)memType].empty();
  bool singleBuffer = static_cast<bool>(bufferIndex);
  SmallVector<Value> args = {singleBuffer ? bufferIndex : bufferMask,
                             pred,
                             readerMaskVal,
                             observerMaskVal,
                             readVisibilityVal,
                             effectCTAs};
  RankedTensorType readTrackingType;
  // Keep the full backing types in the cache key: equal one-buffer view shapes
  // can have different physical strides and origin/phase bank offsets.
  ManglingArgs specializationArgs{readVisibilityType, (uint64_t)hasReadTracking,
                                  (uint64_t)singleBuffer};
  if (hasReadTracking) {
    ValueType tracking = auxData.readTracking[(int)memType].at(insertPoint);
    readTrackingType = cast<RankedTensorType>(tracking.type);
    args.push_back(tracking.value);
    specializationArgs.append(readTrackingType);
  }
  createCallToCachedFunction(
      b, "set_read_visibility", args,
      /*assertInfo=*/std::nullopt, specializationArgs,
      [readVisibilityType, readTrackingType, hasReadTracking,
       singleBuffer](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value bufferSelection = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value readerMaskVal = entryBlock->getArgument(2);
        Value observerMaskVal = entryBlock->getArgument(3);
        Value readVisibilityPtr = entryBlock->getArgument(4);
        Value effectCTAs = entryBlock->getArgument(5);
        Value readTrackingPtr;
        if (hasReadTracking)
          readTrackingPtr = entryBlock->getArgument(6);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        SmallVector<int64_t> visibilityStrides, trackingStrides;
        auto createBufferView = [&](RankedTensorType type,
                                    SmallVectorImpl<int64_t> &strides) {
          if (!singleBuffer)
            return type;
          SmallVector<int64_t> shape(type.getShape());
          strides.resize(shape.size(), 1);
          for (unsigned dim = 1; dim < shape.size(); ++dim)
            strides[dim] = strides[dim - 1] * shape[dim - 1];
          shape[1] = 1;
          // Distribute lanes over the narrowed shape to avoid redundant loads.
          return tti::getIntTensorType(
              fb.getInsertionBlock()->getParent(), shape,
              type.getElementType().getIntOrFloatBitWidth());
        };
        RankedTensorType visibilityType =
            createBufferView(readVisibilityType, visibilityStrides);
        RankedTensorType trackingType;
        if (hasReadTracking)
          trackingType = createBufferView(readTrackingType, trackingStrides);

        auto selectBufferPointer = [&](Value ptr,
                                       RankedTensorType type) -> Value {
          if (!singleBuffer)
            return ptr;
          Value stride =
              arith::ConstantIntOp::create(fb, type.getShape()[0], 32);
          Value offset = arith::MulIOp::create(fb, bufferSelection, stride);
          return triton::AddPtrOp::create(fb, ptr.getType(), ptr, offset)
              .getResult();
        };
        readVisibilityPtr =
            selectBufferPointer(readVisibilityPtr, readVisibilityType);
        if (hasReadTracking)
          readTrackingPtr =
              selectBufferPointer(readTrackingPtr, readTrackingType);

        Value currentCTA = createCurrentCTAMask(fb);
        auto createBufferRows = [&](RankedTensorType tableType) -> Value {
          if (singleBuffer) {
            auto maskType =
                cast<RankedTensorType>(tableType.clone(fb.getI1Type()));
            return tti::createConstIntTensor(fb, fb.getLoc(), 1, maskType);
          }
          return convertAndBroadcast(fb, bufferSelection, {1}, tableType);
        };
        auto createReaderRows = [&](RankedTensorType tableType) {
          Value rows = createBufferRows(tableType);
          Value ownerMask =
              createCTASetMask(fb, tableType, /*dim=*/0, effectCTAs);
          Value readerCTAMask =
              createCTASetMask(fb, tableType, /*dim=*/4, currentCTA);
          rows = arith::AndIOp::create(fb, rows, ownerMask);
          return arith::AndIOp::create(fb, rows, readerCTAMask);
        };
        auto clearReader = [&](Value table, RankedTensorType tableType,
                               Value rows) {
          auto elemType = cast<IntegerType>(tableType.getElementType());
          Value readerMaskElem =
              adjustIntegerWidth(fb, readerMaskVal, elemType);
          Value readerBit =
              triton::SplatOp::create(fb, tableType, readerMaskElem);
          Value allOnes =
              tti::createConstIntTensor(fb, fb.getLoc(), -1, tableType,
                                        /*isSigned=*/true);
          Value notReaderBit = arith::XOrIOp::create(fb, readerBit, allOnes);
          Value withoutReader = arith::AndIOp::create(fb, table, notReaderBit);
          return arith::SelectOp::create(fb, rows, withoutReader, table)
              .getResult();
        };

        Value readVisibility =
            tti::createLoadScratchMemory(fb, fb.getLoc(), readVisibilityPtr,
                                         visibilityType, visibilityStrides);
        Value visibilityRows = createReaderRows(visibilityType);
        // A logical reader reuses one bit for every read. Remove the previous
        // generation before publishing the current one so an old observer or
        // barrier snapshot cannot license a later unfinished read.
        Value newVisibility =
            clearReader(readVisibility, visibilityType, visibilityRows);
        auto elemType = cast<IntegerType>(visibilityType.getElementType());
        Value readerMaskElem = adjustIntegerWidth(fb, readerMaskVal, elemType);
        Value readerBit =
            triton::SplatOp::create(fb, visibilityType, readerMaskElem);
        Value observerColumnMask =
            createThreadColumnMask(fb, observerMaskVal, visibilityType,
                                   /*columnDim=*/3);
        Value observerCTAMask =
            createCTASetMask(fb, visibilityType, /*dim=*/2, currentCTA);
        Value observerRows =
            arith::AndIOp::create(fb, visibilityRows, observerCTAMask);
        observerRows =
            arith::AndIOp::create(fb, observerRows, observerColumnMask);
        Value withReader = arith::OrIOp::create(fb, newVisibility, readerBit);
        newVisibility = arith::SelectOp::create(fb, observerRows, withReader,
                                                newVisibility);
        tti::createStoreScratchMemory(fb, fb.getLoc(), readVisibilityPtr,
                                      newVisibility, visibilityType,
                                      /*currentCTAOnly=*/false,
                                      /*storeMask=*/nullptr, visibilityStrides);

        if (hasReadTracking) {
          if (readTrackingType.getShape()[4] == 1) {
            Value readTracking =
                tti::createLoadScratchMemory(fb, fb.getLoc(), readTrackingPtr,
                                             trackingType, trackingStrides);
            Value trackingRows = createReaderRows(trackingType);
            Value newTracking =
                clearReader(readTracking, trackingType, trackingRows);
            tti::createStoreScratchMemory(
                fb, fb.getLoc(), readTrackingPtr, newTracking, trackingType,
                /*currentCTAOnly=*/false,
                /*storeMask=*/nullptr, trackingStrides);
          } else {
            RankedTensorType fullBankType =
                tti::getSlicedTensorType(readTrackingType, {0, 1, 2, 3},
                                         readTrackingType.getElementType());
            RankedTensorType bankType = tti::getSlicedTensorType(
                trackingType, {0, 1, 2, 3}, trackingType.getElementType());
            SmallVector<int64_t> bankStrides;
            if (singleBuffer)
              bankStrides.assign(trackingStrides.begin(),
                                 trackingStrides.begin() + 4);
            Value trackingRows = createBufferRows(bankType);
            Value ownerMask =
                createCTASetMask(fb, bankType, /*dim=*/0, effectCTAs);
            trackingRows = arith::AndIOp::create(fb, trackingRows, ownerMask);
            Value originId =
                tti::ExperimentalClusterCTAIdOp::create(fb, fb.getLoc());
            for (int phase = 0; phase < 2; ++phase) {
              // The pointer is a B=1 view into the original allocation. Origin
              // and phase banks are still separated by the full table stride.
              Value bankPtr = createTrackingOriginPhasePointer(
                  fb, readTrackingPtr, readTrackingType, fullBankType, originId,
                  phase);
              Value tracking = tti::createLoadScratchMemory(
                  fb, fb.getLoc(), bankPtr, bankType, bankStrides);
              Value updated = clearReader(tracking, bankType, trackingRows);
              tti::createStoreScratchMemory(fb, fb.getLoc(), bankPtr, updated,
                                            bankType,
                                            /*currentCTAOnly=*/false,
                                            /*storeMask=*/nullptr, bankStrides);
            }
          }
        }

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createTrackVisibleAccessesCall(
    ImplicitLocOpBuilder &b, Value mbar, int thread, Value pred,
    MemType memType, Operation *insertPoint, Value barrierCTAs,
    Value readBufferMask) {
  if (auxData.barriers.empty() || auxData.barrierStates.empty())
    return;

  const bool trackWrites = !readBufferMask &&
                           !auxData.writeVisibility[(int)memType].empty() &&
                           !auxData.writeTracking[(int)memType].empty();
  const bool trackReads = !auxData.readVisibility[(int)memType].empty() &&
                          !auxData.readTracking[(int)memType].empty();
  if (!trackWrites && !trackReads)
    return;

  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  ValueType barriers = auxData.barriers.at(insertPoint);
  auto barriersType = cast<RankedTensorType>(barriers.type);
  ValueType barrierStates = auxData.barrierStates.at(insertPoint);
  auto barrierStatesType = cast<RankedTensorType>(barrierStates.type);
  uint32_t mbarLength = getMemDescLength(mbar);
  SmallVector<Value> args = {
      tti::ExperimentalMemDescToI32Op::create(b, mbar),
      arith::ConstantIntOp::create(b, mbarLength, 32),
      pred,
      arith::ConstantIntOp::create(b, thread, 32),
      barriers.value,
      barrierStates.value,
      barrierCTAs,
  };
  ManglingArgs specializationArgs;
  specializationArgs.append(barriersType);
  specializationArgs.append(barrierStatesType);
  specializationArgs.append(static_cast<uint64_t>(trackWrites));
  specializationArgs.append(static_cast<uint64_t>(trackReads));

  RankedTensorType writeVisibilityType;
  RankedTensorType writeTrackingType;
  if (trackWrites) {
    ValueType visibility =
        auxData.writeVisibility[(int)memType].at(insertPoint);
    ValueType tracking = auxData.writeTracking[(int)memType].at(insertPoint);
    writeVisibilityType = cast<RankedTensorType>(visibility.type);
    writeTrackingType = cast<RankedTensorType>(tracking.type);
    args.append({visibility.value, tracking.value});
    specializationArgs.append(writeVisibilityType);
    specializationArgs.append(writeTrackingType);
  }

  RankedTensorType readVisibilityType;
  RankedTensorType readTrackingType;
  if (trackReads) {
    ValueType visibility = auxData.readVisibility[(int)memType].at(insertPoint);
    ValueType tracking = auxData.readTracking[(int)memType].at(insertPoint);
    readVisibilityType = cast<RankedTensorType>(visibility.type);
    readTrackingType = cast<RankedTensorType>(tracking.type);
    args.append({visibility.value, tracking.value});
    specializationArgs.append(readVisibilityType);
    specializationArgs.append(readTrackingType);
  }
  if (readBufferMask)
    args.push_back(readBufferMask);

  auto descriptors =
      barriers.value.getDefiningOp<tti::ExperimentalBufferDescriptorsOp>();
  bool sliceBarrierTracking = static_cast<bool>(descriptors);
  if (sliceBarrierTracking) {
    llvm::SmallDenseSet<uint64_t> keys;
    // Match BufferDescriptorsOpConversion's address trimming and length
    // packing. Adding its common base preserves equality modulo this mask.
    uint64_t addressMask = descriptors.getMemType() == MemType::SHARED_MEM
                               ? (1ULL << 24) - 1
                               : 0xffffffffULL;
    for (auto [offset, length] :
         llvm::zip_equal(descriptors.getOffsets(), descriptors.getLengths())) {
      if (static_cast<uint32_t>(length) != mbarLength)
        continue;
      uint64_t key = (static_cast<uint64_t>(mbarLength) << 32) |
                     (static_cast<uint32_t>(offset) & addressMask);
      if (!keys.insert(key).second) {
        sliceBarrierTracking = false;
        break;
      }
    }
  }
  specializationArgs.append(static_cast<uint64_t>(sliceBarrierTracking));

  createCallToCachedFunction(
      b,
      readBufferMask ? "track_barrier_read_for_buffer"
                     : "track_visible_accesses",
      args, /*assertInfo=*/std::nullopt, specializationArgs,
      [trackWrites, trackReads, writeVisibilityType, writeTrackingType,
       readVisibilityType, barrierStatesType,
       filterBuffer = static_cast<bool>(readBufferMask), readTrackingType,
       sliceBarrierTracking](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadVal = entryBlock->getArgument(3);
        Value barriers = entryBlock->getArgument(4);
        Value barrierStatesPtr = entryBlock->getArgument(5);
        Value barrierCTAs = entryBlock->getArgument(6);
        unsigned nextArg = 7;

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);
        Value descriptor = createBufferDescriptor(fb, mbarOffset, lengthVal);
        Value barriersEqBar =
            createCmpIntTensorScalar(fb, barriers, descriptor);
        Value currentCTA = createCurrentCTAMask(fb);
        Value barrierStates = tti::createLoadScratchMemory(
            fb, fb.getLoc(), barrierStatesPtr, barrierStatesType);
        Value currentPhases = arith::AndIOp::create(
            fb, barrierStates,
            tti::createConstIntTensor(fb, fb.getLoc(), 1, barrierStatesType));

        Value barrierIndex, anyMatch;
        Value trackingPhases = currentPhases;
        SmallVector<int> phaseDims{2, 3};
        if (sliceBarrierTracking) {
          auto indexType =
              cast<RankedTensorType>(barriers.getType()).clone(fb.getI32Type());
          Value indices = triton::MakeRangeOp::create(
              fb, indexType, 0, indexType.getNumElements());
          Value selectedIndices = arith::SelectOp::create(
              fb, barriersEqBar, indices,
              tti::createConstIntTensor(fb, fb.getLoc(), 0, indexType));
          barrierIndex = reduceAll<arith::OrIOp>(fb, selectedIndices);
          anyMatch = reduceAll<arith::OrIOp>(fb, barriersEqBar);

          // Each recipient CTA keeps its own phase. Both phase banks remain
          // in the viewport; remove only the uniquely selected barrier axis.
          Value selectedPhases = arith::SelectOp::create(
              fb,
              convertAndBroadcast(fb, barriersEqBar, {1}, barrierStatesType),
              currentPhases,
              tti::createConstIntTensor(fb, fb.getLoc(), 0, barrierStatesType));
          trackingPhases = reduceLastDim<arith::OrIOp>(fb, selectedPhases);
          phaseDims = {2};
        }

        auto createTrackingView = [&](Value ptr, RankedTensorType type) {
          SmallVector<int64_t> strides;
          if (sliceBarrierTracking) {
            ArrayRef<int64_t> shape = type.getShape();
            int64_t barrierStride = shape[0] * shape[1] * shape[2];
            Value offset = arith::MulIOp::create(
                fb, barrierIndex,
                arith::ConstantIntOp::create(fb, barrierStride, 32));
            ptr = triton::AddPtrOp::create(fb, ptr.getType(), ptr, offset);
            SmallVector<int64_t> compactShape;
            int64_t stride = 1;
            for (int dim = 0; dim < type.getRank(); ++dim) {
              if (dim != 3) {
                compactShape.push_back(shape[dim]);
                strides.push_back(stride);
              }
              stride *= shape[dim];
            }
            type = tti::getIntTensorType(
                entryBlock->getParent(), compactShape,
                type.getElementType().getIntOrFloatBitWidth());
          }
          return std::make_tuple(ptr, type, std::move(strides));
        };

        auto createBarrierMask = [&](RankedTensorType type, Value ctaMask) {
          Value mask;
          if (sliceBarrierTracking) {
            mask = triton::SplatOp::create(fb, type.clone(fb.getI1Type()),
                                           anyMatch);
          } else {
            mask = convertAndBroadcast(fb, barriersEqBar, {3}, type);
          }
          mask = arith::AndIOp::create(fb, mask, ctaMask);
          Value currentPhase =
              convertAndBroadcast(fb, trackingPhases, phaseDims, type);
          Value phaseIndices =
              createDimIndices(fb, type, /*dim=*/type.getRank() - 1);
          auto phaseIndicesType =
              cast<RankedTensorType>(phaseIndices.getType());
          currentPhase =
              arith::TruncIOp::create(fb, phaseIndicesType, currentPhase);
          Value phaseMask = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                                  phaseIndices, currentPhase);
          return arith::AndIOp::create(fb, mask, phaseMask).getResult();
        };

        if (trackWrites) {
          Value visibilityPtr = entryBlock->getArgument(nextArg++);
          auto [trackingPtr, trackingType, trackingStrides] =
              createTrackingView(entryBlock->getArgument(nextArg++),
                                 writeTrackingType);
          Value visibility = tti::createLoadScratchMemory(
              fb, fb.getLoc(), visibilityPtr, writeVisibilityType);
          Value barrierCTAMask =
              createCTASetMask(fb, trackingType, /*dim=*/2, barrierCTAs);
          Value barrierMask = createBarrierMask(trackingType, barrierCTAMask);
          auto elemType =
              cast<IntegerType>(writeVisibilityType.getElementType());
          Value threadElem = adjustIntegerWidth(fb, threadVal, elemType);
          Value oneScalar = arith::ConstantOp::create(
              fb, elemType, fb.getIntegerAttr(elemType, 1));
          Value threadBitScalar =
              arith::ShLIOp::create(fb, oneScalar, threadElem);
          Value threadBit =
              triton::SplatOp::create(fb, writeVisibilityType, threadBitScalar);
          Value visibleWrites =
              arith::AndIOp::create(fb, visibility, threadBit);
          visibleWrites = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                                visibleWrites, threadBit);
          Value sourceCTAMask =
              createCTASetMask(fb, writeVisibilityType, /*dim=*/2, currentCTA);
          visibleWrites =
              arith::AndIOp::create(fb, visibleWrites, sourceCTAMask);
          visibleWrites = reduceLastDim<arith::OrIOp>(fb, visibleWrites);
          visibleWrites =
              convertAndBroadcast(fb, visibleWrites, {0, 1}, trackingType);
          Value barAndVisible =
              arith::AndIOp::create(fb, barrierMask, visibleWrites);
          Value one =
              tti::createConstIntTensor(fb, fb.getLoc(), 1, trackingType);
          tti::createStoreScratchMemory(
              fb, fb.getLoc(), trackingPtr, one, trackingType,
              /*currentCTAOnly=*/false, barAndVisible, trackingStrides);
        }

        if (trackReads) {
          Value visibilityPtr = entryBlock->getArgument(nextArg++);
          auto [trackingPtr, trackingType, trackingStrides] =
              createTrackingView(entryBlock->getArgument(nextArg++),
                                 readTrackingType);
          // Select the issuing observer before loading [Cbuf, B, Creader].
          // The retained reader-CTA dimension keeps its full-table stride.
          ArrayRef<int64_t> visibilityShape = readVisibilityType.getShape();
          int64_t observerCTAStride = visibilityShape[0] * visibilityShape[1];
          int64_t observerThreadStride = observerCTAStride * visibilityShape[2];
          Value observerCTA =
              tti::ExperimentalClusterCTAIdOp::create(fb, fb.getLoc());
          Value observerCTAOffset = arith::MulIOp::create(
              fb, observerCTA,
              arith::ConstantIntOp::create(fb, observerCTAStride, 32));
          Value observerThreadOffset = arith::MulIOp::create(
              fb, threadVal,
              arith::ConstantIntOp::create(fb, observerThreadStride, 32));
          Value observerOffset = arith::AddIOp::create(fb, observerCTAOffset,
                                                       observerThreadOffset);
          Value observerPtr = triton::AddPtrOp::create(
              fb, visibilityPtr.getType(), visibilityPtr, observerOffset);
          RankedTensorType observerType = tti::getIntTensorType(
              entryBlock->getParent(),
              {visibilityShape[0], visibilityShape[1], visibilityShape[4]},
              readVisibilityType.getElementType().getIntOrFloatBitWidth());
          Value visibleReads = tti::createLoadScratchMemory(
              fb, fb.getLoc(), observerPtr, observerType,
              {1, visibilityShape[0],
               observerThreadStride * visibilityShape[3]});
          Value tracking = tti::createLoadScratchMemory(
              fb, fb.getLoc(), trackingPtr, trackingType, trackingStrides);
          Value barrierCTAMask =
              createCTASetMask(fb, trackingType, /*dim=*/2, barrierCTAs);
          Value barrierMask = createBarrierMask(trackingType, barrierCTAMask);
          if (filterBuffer) {
            Value bufferMask = convertAndBroadcast(
                fb, entryBlock->getArguments().back(), {1}, trackingType);
            Value bufferCTAMask =
                createCTASetMask(fb, trackingType, /*dim=*/0, barrierCTAs);
            barrierMask = arith::AndIOp::create(fb, barrierMask, bufferMask);
            barrierMask = arith::AndIOp::create(fb, barrierMask, bufferCTAMask);
          }
          SmallVector<int> readerDims{0, 1, sliceBarrierTracking ? 3 : 4};
          visibleReads =
              convertAndBroadcast(fb, visibleReads, readerDims, trackingType);
          Value withVisible = arith::OrIOp::create(fb, tracking, visibleReads);
          tti::createStoreScratchMemory(
              fb, fb.getLoc(), trackingPtr, withVisible, trackingType,
              /*currentCTAOnly=*/false, barrierMask, trackingStrides);
        }

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createTrackBarrierWriteForBufferCall(
    ImplicitLocOpBuilder &b, Value mbar, Value bufferMask, Value pred,
    MemType memType, Operation *insertPoint, Value barrierCTAs,
    Value effectCTAs) {
  if (auxData.barriers.empty() || auxData.barrierStates.empty() ||
      auxData.writeTracking[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value writeTrackingVal =
      auxData.writeTracking[(int)memType].at(insertPoint).value;
  auto writeTrackingType = cast<RankedTensorType>(
      auxData.writeTracking[(int)memType].at(insertPoint).type);
  Value barrierStatesVal = auxData.barrierStates.at(insertPoint).value;
  auto barrierStatesType =
      cast<RankedTensorType>(auxData.barrierStates.at(insertPoint).type);
  uint32_t mbarLength = getMemDescLength(mbar);
  Value mbarOffset = tti::ExperimentalMemDescToI32Op::create(b, mbar);
  Value mbarLengthVal = arith::ConstantIntOp::create(b, mbarLength, 32);
  SmallVector<Value> args = {mbarOffset,       mbarLengthVal, pred,
                             bufferMask,       barriersVal,   writeTrackingVal,
                             barrierStatesVal, barrierCTAs,   effectCTAs};
  createCallToCachedFunction(
      b, "track_barrier_write_for_buffer", args,
      /*assertInfo=*/std::nullopt,
      {barriersType, writeTrackingType, barrierStatesType},
      [writeTrackingType, barrierStatesType](ImplicitLocOpBuilder &fb,
                                             Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value mbarLengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value bufferMask = entryBlock->getArgument(3);
        Value barriers = entryBlock->getArgument(4);
        Value writeTrackingPtr = entryBlock->getArgument(5);
        Value barrierStatesPtr = entryBlock->getArgument(6);
        Value barrierCTAs = entryBlock->getArgument(7);
        Value effectCTAs = entryBlock->getArgument(8);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value writeTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeTrackingPtr, writeTrackingType);
        Value barrierStates = tti::createLoadScratchMemory(
            fb, fb.getLoc(), barrierStatesPtr, barrierStatesType);
        Value barrierDescriptor =
            createBufferDescriptor(fb, mbarOffset, mbarLengthVal);
        Value barriersEqBar =
            createCmpIntTensorScalar(fb, barriers, barrierDescriptor);
        barriersEqBar =
            convertAndBroadcast(fb, barriersEqBar, {3}, writeTrackingType);
        bufferMask =
            convertAndBroadcast(fb, bufferMask, {1}, writeTrackingType);
        Value bufferCTAMask =
            createCTASetMask(fb, writeTrackingType, /*dim=*/0, effectCTAs);
        Value barrierCTAMask =
            createCTASetMask(fb, writeTrackingType, /*dim=*/2, barrierCTAs);
        Value trackMask = arith::AndIOp::create(fb, barriersEqBar, bufferMask);
        trackMask = arith::AndIOp::create(fb, trackMask, bufferCTAMask);
        trackMask = arith::AndIOp::create(fb, trackMask, barrierCTAMask);
        Value phaseOne =
            tti::createConstIntTensor(fb, fb.getLoc(), 1, barrierStatesType);
        Value currentPhase = arith::AndIOp::create(fb, barrierStates, phaseOne);
        currentPhase =
            convertAndBroadcast(fb, currentPhase, {2, 3}, writeTrackingType);
        Value phaseIndices = createDimIndices(
            fb, writeTrackingType, /*dim=*/writeTrackingType.getRank() - 1);
        auto phaseIndicesType = cast<RankedTensorType>(phaseIndices.getType());
        currentPhase =
            arith::TruncIOp::create(fb, phaseIndicesType, currentPhase);
        Value phaseMask = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                                phaseIndices, currentPhase);
        trackMask = arith::AndIOp::create(fb, trackMask, phaseMask);
        Value writeTrackingOne =
            tti::createConstIntTensor(fb, fb.getLoc(), 1, writeTrackingType);
        Value newTracking = arith::SelectOp::create(
            fb, trackMask, writeTrackingOne, writeTracking);
        createMaskedStoreScratchMemory(fb, fb.getLoc(), writeTrackingPtr,
                                       newTracking, writeTrackingType,
                                       trackMask);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

static void createClearBarrierTrackingCall(ImplicitLocOpBuilder &b,
                                           StringRef functionName,
                                           Value selectedBarriers, Value pred,
                                           ValueType tracking) {
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  auto trackingType = cast<RankedTensorType>(tracking.type);
  SmallVector<Value> args = {selectedBarriers, pred, tracking.value};
  ManglingArgs specializationArgs{trackingType};
  createCallToCachedFunction(
      b, functionName.str(), args, /*assertInfo=*/std::nullopt,
      specializationArgs,
      [trackingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value selectedBarriers = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value trackingPtr = entryBlock->getArgument(2);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value barriersEqBar =
            convertAndBroadcast(fb, selectedBarriers, {3}, trackingType);
        Value barrierCTAs = createCurrentCTAMask(fb);
        Value barrierCTAMask =
            createCTASetMask(fb, trackingType, /*dim=*/2, barrierCTAs);
        barriersEqBar =
            arith::AndIOp::create(fb, barriersEqBar, barrierCTAMask);
        Value zero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, trackingType);
        tti::createStoreScratchMemory(fb, fb.getLoc(), trackingPtr, zero,
                                      trackingType, /*currentCTAOnly=*/false,
                                      barriersEqBar);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createClearBarrierStorageTrackingCall(
    ImplicitLocOpBuilder &b, Value selectedBarriers, Value pred,
    Operation *insertPoint) {
  auto clear = [&](StringRef name, AuxDataMap::RegionToValueMap &tracking) {
    if (!tracking.empty())
      createClearBarrierTrackingCall(b, name, selectedBarriers, pred,
                                     tracking.at(insertPoint));
  };
  for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
    clear("clear_barrier_write_tracking", auxData.writeTracking[(int)memType]);
    clear("clear_barrier_read_tracking", auxData.readTracking[(int)memType]);
  }
  clear("clear_barrier_proxy_tracking", auxData.proxyAccessTracking);
}

void FunctionBuilder::createTransferVisibleAccessesCall(
    ImplicitLocOpBuilder &b, Value mbar, Value phase, uint64_t threadMask,
    Value pred, MemType memType, Operation *insertPoint) {
  if (auxData.barriers.empty())
    return;

  const bool transferWrites = !auxData.writeVisibility[(int)memType].empty() &&
                              !auxData.writeTracking[(int)memType].empty();
  const bool transferReads = !auxData.readVisibility[(int)memType].empty() &&
                             !auxData.readTracking[(int)memType].empty();
  if (!transferWrites && !transferReads)
    return;

  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value threadMaskVal = arith::ConstantIntOp::create(b, threadMask, 64);
  ValueType barriers = auxData.barriers.at(insertPoint);
  auto barriersType = cast<RankedTensorType>(barriers.type);
  SmallVector<Value> args = {
      tti::ExperimentalMemDescToI32Op::create(b, mbar),
      arith::ConstantIntOp::create(b, getMemDescLength(mbar), 32),
      phase,
      pred,
      threadMaskVal,
      barriers.value,
  };
  ManglingArgs specializationArgs;
  specializationArgs.append(barriersType);
  specializationArgs.append(static_cast<uint64_t>(transferWrites));
  specializationArgs.append(static_cast<uint64_t>(transferReads));

  RankedTensorType writeVisibilityType;
  RankedTensorType writeTrackingType;
  if (transferWrites) {
    ValueType visibility =
        auxData.writeVisibility[(int)memType].at(insertPoint);
    ValueType tracking = auxData.writeTracking[(int)memType].at(insertPoint);
    writeVisibilityType = cast<RankedTensorType>(visibility.type);
    writeTrackingType = cast<RankedTensorType>(tracking.type);
    args.append({visibility.value, tracking.value});
    specializationArgs.append(writeVisibilityType);
    specializationArgs.append(writeTrackingType);
  }

  RankedTensorType readVisibilityType;
  RankedTensorType readTrackingType;
  if (transferReads) {
    ValueType visibility = auxData.readVisibility[(int)memType].at(insertPoint);
    ValueType tracking = auxData.readTracking[(int)memType].at(insertPoint);
    readVisibilityType = cast<RankedTensorType>(visibility.type);
    readTrackingType = cast<RankedTensorType>(tracking.type);
    args.append({visibility.value, tracking.value});
    specializationArgs.append(readVisibilityType);
    specializationArgs.append(readTrackingType);
  }

  createCallToCachedFunction(
      b, "transfer_visible_accesses", args, /*assertInfo=*/std::nullopt,
      specializationArgs,
      [transferWrites, transferReads, writeVisibilityType, writeTrackingType,
       readVisibilityType,
       readTrackingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value phase = entryBlock->getArgument(2);
        Value pred = entryBlock->getArgument(3);
        Value threadMaskVal = entryBlock->getArgument(4);
        Value barriers = entryBlock->getArgument(5);
        unsigned nextArg = 6;

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);
        Value descriptor = createBufferDescriptor(fb, mbarOffset, lengthVal);
        Value barriersEqBar =
            createCmpIntTensorScalar(fb, barriers, descriptor);
        Value currentCTA = createCurrentCTAMask(fb);

        if (transferWrites) {
          Value visibilityPtr = entryBlock->getArgument(nextArg++);
          Value trackingPtr = entryBlock->getArgument(nextArg++);
          Value visibility = tti::createLoadScratchMemory(
              fb, fb.getLoc(), visibilityPtr, writeVisibilityType);
          Value tracking = createLoadTrackingPhase(fb, trackingPtr,
                                                   writeTrackingType, phase);
          auto phaseTrackingType = cast<RankedTensorType>(tracking.getType());
          Value barrierMask =
              convertAndBroadcast(fb, barriersEqBar, {3}, phaseTrackingType);
          Value barrierCTAMask =
              createCTASetMask(fb, phaseTrackingType, /*dim=*/2, currentCTA);
          barrierMask = arith::AndIOp::create(fb, barrierMask, barrierCTAMask);
          Value zeroTracking =
              tti::createConstIntTensor(fb, fb.getLoc(), 0, phaseTrackingType);
          Value trackingBuffers =
              arith::SelectOp::create(fb, barrierMask, tracking, zeroTracking);
          trackingBuffers = reduce<arith::OrIOp>(fb, trackingBuffers, {2, 3});
          trackingBuffers = convertAndBroadcast(fb, trackingBuffers, {0, 1},
                                                writeVisibilityType);
          auto trackingBuffersType =
              cast<RankedTensorType>(trackingBuffers.getType());
          Value trackingBuffersOne = tti::createConstIntTensor(
              fb, fb.getLoc(), 1, trackingBuffersType);
          trackingBuffers =
              arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                    trackingBuffers, trackingBuffersOne);
          auto elemType =
              cast<IntegerType>(writeVisibilityType.getElementType());
          Value threadMaskElem =
              adjustIntegerWidth(fb, threadMaskVal, elemType);
          Value threadMaskTensor =
              triton::SplatOp::create(fb, writeVisibilityType, threadMaskElem);
          Value zeroVisibility = tti::createConstIntTensor(fb, fb.getLoc(), 0,
                                                           writeVisibilityType);
          Value trackingThreadBit = arith::SelectOp::create(
              fb, trackingBuffers, threadMaskTensor, zeroVisibility);
          Value updated =
              arith::OrIOp::create(fb, visibility, trackingThreadBit);
          Value waitingCTAMask =
              createCTASetMask(fb, writeVisibilityType, /*dim=*/2, currentCTA);
          createMaskedStoreScratchMemory(fb, fb.getLoc(), visibilityPtr,
                                         updated, writeVisibilityType,
                                         waitingCTAMask);
        }

        if (transferReads) {
          Value visibilityPtr = entryBlock->getArgument(nextArg++);
          Value trackingPtr = entryBlock->getArgument(nextArg++);
          Value visibility = tti::createLoadScratchMemory(
              fb, fb.getLoc(), visibilityPtr, readVisibilityType);
          Value tracking =
              createLoadTrackingPhase(fb, trackingPtr, readTrackingType, phase);
          auto phaseTrackingType = cast<RankedTensorType>(tracking.getType());
          Value barrierMask =
              convertAndBroadcast(fb, barriersEqBar, {3}, phaseTrackingType);
          Value barrierCTAMask =
              createCTASetMask(fb, phaseTrackingType, /*dim=*/2, currentCTA);
          barrierMask = arith::AndIOp::create(fb, barrierMask, barrierCTAMask);
          Value zeroTracking =
              tti::createConstIntTensor(fb, fb.getLoc(), 0, phaseTrackingType);
          Value trackingBar =
              arith::SelectOp::create(fb, barrierMask, tracking, zeroTracking);
          trackingBar = reduce<arith::OrIOp>(fb, trackingBar, {2, 3});
          trackingBar = convertAndBroadcast(fb, trackingBar, {0, 1, 4},
                                            readVisibilityType);
          Value visibilityOrTracking =
              arith::OrIOp::create(fb, visibility, trackingBar);
          Value threadColumnMask = createThreadColumnMask(
              fb, threadMaskVal, readVisibilityType, /*columnDim=*/3);
          Value waitingCTAMask =
              createCTASetMask(fb, readVisibilityType, /*dim=*/2, currentCTA);
          threadColumnMask =
              arith::AndIOp::create(fb, threadColumnMask, waitingCTAMask);
          Value updated = arith::SelectOp::create(
              fb, threadColumnMask, visibilityOrTracking, visibility);
          createMaskedStoreScratchMemory(fb, fb.getLoc(), visibilityPtr,
                                         updated, readVisibilityType,
                                         waitingCTAMask);
        }

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createVerifyWriteVisibilityCall(
    ImplicitLocOpBuilder &b, Value bufferMask, int thread,
    StringRef operandName, Value pred, MemType memType, Operation *insertPoint,
    Value effectCTAs) {
  if (auxData.writeVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value writeVisibilityVal =
      auxData.writeVisibility[(int)memType].at(insertPoint).value;
  auto writeVisibilityType = cast<RankedTensorType>(
      auxData.writeVisibility[(int)memType].at(insertPoint).type);
  std::string message = "Buffer being accessed has outstanding writes.";
  if (!operandName.empty())
    message += " Operand: " + operandName.str();
  AssertInfo assertInfo{message, b.getI1Type()};
  SmallVector<Value> args = {bufferMask, pred, threadVal, writeVisibilityVal,
                             effectCTAs};
  createCallToCachedFunction(
      b, "verify_write_visibility", args, assertInfo, {writeVisibilityType},
      [writeVisibilityType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value bufferMask = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value threadVal = entryBlock->getArgument(2);
        Value writeVisibilityPtr = entryBlock->getArgument(3);
        Value effectCTAs = entryBlock->getArgument(4);

        Value writeVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
        bufferMask =
            convertAndBroadcast(fb, bufferMask, {1}, writeVisibilityType);
        Value relationMask =
            createLeadCTAEffectMask(fb, writeVisibilityType, effectCTAs);
        bufferMask = arith::AndIOp::create(fb, bufferMask, relationMask);
        Value writeVisibilityZero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, writeVisibilityType);
        Value noOneIsWriting = arith::CmpIOp::create(
            fb, arith::CmpIPredicate::eq, writeVisibility, writeVisibilityZero);
        auto elemType = cast<IntegerType>(writeVisibilityType.getElementType());
        Value threadElem = adjustIntegerWidth(fb, threadVal, elemType);
        Value threadMask =
            triton::SplatOp::create(fb, writeVisibilityType, threadElem);
        Value bufferMaskExt =
            arith::ExtUIOp::create(fb, writeVisibilityType, bufferMask);
        Value bufferThreadBit =
            arith::ShLIOp::create(fb, bufferMaskExt, threadMask);
        Value bufferHasVisibility =
            arith::AndIOp::create(fb, writeVisibility, bufferThreadBit);
        bufferHasVisibility = arith::CmpIOp::create(
            fb, arith::CmpIPredicate::eq, bufferHasVisibility, bufferThreadBit);
        Value writeVisible =
            arith::OrIOp::create(fb, noOneIsWriting, bufferHasVisibility);
        Value allWritesVisible = reduceAll<arith::AndIOp>(fb, writeVisible);

        Value vTrue =
            arith::ConstantOp::create(fb, allWritesVisible.getType(),
                                      fb.getIntegerAttr(fb.getI1Type(), 1));
        Value predicatedWriteVisible =
            arith::SelectOp::create(fb, pred, allWritesVisible, vTrue);
        triton::ReturnOp::create(fb, predicatedWriteVisible);
      });
}

void FunctionBuilder::createVerifyReadVisibilityCall(
    ImplicitLocOpBuilder &b, Value bufferMask, int thread,
    StringRef operandName, Value pred, MemType memType, Operation *insertPoint,
    Value effectCTAs) {
  if (auxData.readVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value readVisibilityVal =
      auxData.readVisibility[(int)memType].at(insertPoint).value;
  auto readVisibilityType = cast<RankedTensorType>(
      auxData.readVisibility[(int)memType].at(insertPoint).type);
  std::string message = "Buffer being accessed has outstanding reads";
  if (!operandName.empty())
    message += ". Operand: " + operandName.str();
  AssertInfo assertInfo{message, b.getI1Type()};
  SmallVector<Value> args = {bufferMask, pred, threadVal, readVisibilityVal,
                             effectCTAs};
  createCallToCachedFunction(
      b, "verify_read_visibility", args, assertInfo, {readVisibilityType},
      [readVisibilityType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value bufferMask = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value threadVal = entryBlock->getArgument(2);
        Value readVisibilityPtr = entryBlock->getArgument(3);
        Value effectCTAs = entryBlock->getArgument(4);

        Value readVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
        bufferMask =
            convertAndBroadcast(fb, bufferMask, {1}, readVisibilityType);
        Value bufferCTAMask =
            createCTASetMask(fb, readVisibilityType, /*dim=*/0, effectCTAs);
        Value relationMask =
            createLeadCTAEffectMask(fb, readVisibilityType, effectCTAs);
        bufferMask = arith::AndIOp::create(fb, bufferMask, bufferCTAMask);
        Value readVisibilityZero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, readVisibilityType);
        Value bufVisibility = arith::SelectOp::create(
            fb, bufferMask, readVisibility, readVisibilityZero);
        Value totalVisibility = reduce<arith::OrIOp>(fb, bufVisibility, {2, 3});
        Value threadColumnMask =
            createDimMask(fb, threadVal, readVisibilityType, /*dim=*/3);
        Value accessorVisibility = arith::SelectOp::create(
            fb, relationMask, bufVisibility, readVisibilityZero);
        accessorVisibility = arith::SelectOp::create(
            fb, threadColumnMask, accessorVisibility, readVisibilityZero);
        accessorVisibility = reduce<arith::OrIOp>(fb, accessorVisibility, {3});
        auto accessorVisibilityType =
            cast<RankedTensorType>(accessorVisibility.getType());
        totalVisibility = convertAndBroadcast(fb, totalVisibility, {0, 1, 3},
                                              accessorVisibilityType);
        Value threadAndTotalVisibility =
            arith::AndIOp::create(fb, accessorVisibility, totalVisibility);
        Value hasVisibility =
            arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                  threadAndTotalVisibility, totalVisibility);
        Value selectedAccessors =
            arith::AndIOp::create(fb, bufferMask, relationMask);
        selectedAccessors = reduce<arith::OrIOp>(fb, selectedAccessors, {3, 4});
        selectedAccessors = convertAndBroadcast(
            fb, selectedAccessors, {0, 1, 2}, accessorVisibilityType);
        Value one = tti::createConstIntTensor(
            fb, fb.getLoc(), 1,
            cast<RankedTensorType>(selectedAccessors.getType()));
        Value unmatchedAccessors =
            arith::XOrIOp::create(fb, selectedAccessors, one);
        hasVisibility =
            arith::OrIOp::create(fb, hasVisibility, unmatchedAccessors);
        hasVisibility = reduceAll<arith::AndIOp>(fb, hasVisibility);
        Value vTrue = arith::ConstantOp::create(
            fb, hasVisibility.getType(), fb.getIntegerAttr(fb.getI1Type(), 1));
        Value predicatedHasVisibility =
            arith::SelectOp::create(fb, pred, hasVisibility, vTrue);
        triton::ReturnOp::create(fb, predicatedHasVisibility);
      });
}

void FunctionBuilder::createCopyWriteVisibilityCall(ImplicitLocOpBuilder &b,
                                                    int sourceThread,
                                                    uint64_t destMask,
                                                    Value pred, MemType memType,
                                                    Operation *insertPoint) {

  if (auxData.writeVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  auto writeVis = auxData.writeVisibility[(int)memType].at(insertPoint);
  auto writeVisibilityType = cast<RankedTensorType>(writeVis.type);
  Value sourceThreadVal = arith::ConstantIntOp::create(b, sourceThread, 32);
  Value destMaskVal = arith::ConstantIntOp::create(b, destMask, 64);
  SmallVector<Value> args = {sourceThreadVal, destMaskVal, pred,
                             writeVis.value};
  createCallToCachedFunction(
      b, "copy_write_visibility", args,
      /*assertInfo=*/std::nullopt, {writeVisibilityType},
      [writeVisibilityType,
       totalNumThreads = auxData.threadLayout.totalNumThreads](
          ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value sourceThread = entryBlock->getArgument(0);
        Value destMaskVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value writeVisibilityPtr = entryBlock->getArgument(3);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value writeVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
        auto elemType = cast<IntegerType>(writeVisibilityType.getElementType());
        Value zeroTensor =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, writeVisibilityType);

        uint64_t fullMask = totalNumThreads == 64
                                ? std::numeric_limits<uint64_t>::max()
                                : (std::numeric_limits<uint64_t>::max() >>
                                   (64 - totalNumThreads));
        Value fullMaskVal = arith::ConstantIntOp::create(fb, fullMask, 64);
        Value destMaskElem = adjustIntegerWidth(fb, destMaskVal, elemType);
        Value fullMaskElem = adjustIntegerWidth(fb, fullMaskVal, elemType);
        Value clearMaskElem =
            arith::XOrIOp::create(fb, destMaskElem, fullMaskElem);
        Value destMaskTensor =
            triton::SplatOp::create(fb, writeVisibilityType, destMaskElem);
        Value clearMaskTensor =
            triton::SplatOp::create(fb, writeVisibilityType, clearMaskElem);
        Value cleared =
            arith::AndIOp::create(fb, writeVisibility, clearMaskTensor);

        Value sourceThreadElem = adjustIntegerWidth(fb, sourceThread, elemType);
        Value oneScalar = arith::ConstantOp::create(
            fb, elemType, fb.getIntegerAttr(elemType, 1));
        Value sourceMaskElem =
            arith::ShLIOp::create(fb, oneScalar, sourceThreadElem);
        Value sourceMaskTensor =
            triton::SplatOp::create(fb, writeVisibilityType, sourceMaskElem);
        Value sourceBits =
            arith::AndIOp::create(fb, writeVisibility, sourceMaskTensor);
        Value sourceIsSet = arith::CmpIOp::create(fb, arith::CmpIPredicate::ne,
                                                  sourceBits, zeroTensor);
        Value replicated = arith::SelectOp::create(fb, sourceIsSet,
                                                   destMaskTensor, zeroTensor);

        Value updatedCurrent = arith::OrIOp::create(fb, cleared, replicated);
        Value currentCTAMask = createCTASetMask(
            fb, writeVisibilityType, /*dim=*/2, createCurrentCTAMask(fb));
        createMaskedStoreScratchMemory(fb, fb.getLoc(), writeVisibilityPtr,
                                       updatedCurrent, writeVisibilityType,
                                       currentCTAMask);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createCopyReadVisibilityCall(ImplicitLocOpBuilder &b,
                                                   int sourceThread,
                                                   uint64_t destMask,
                                                   Value pred, MemType memType,
                                                   Operation *insertPoint) {

  if (auxData.readVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  auto readVis = auxData.readVisibility[(int)memType].at(insertPoint);
  auto readVisibilityType = cast<RankedTensorType>(readVis.type);
  Value sourceThreadVal = arith::ConstantIntOp::create(b, sourceThread, 32);
  Value destMaskVal = arith::ConstantIntOp::create(b, destMask, 64);
  SmallVector<Value> args = {sourceThreadVal, destMaskVal, pred, readVis.value};
  createCallToCachedFunction(
      b, "copy_read_visibility", args,
      /*assertInfo=*/std::nullopt, {readVisibilityType},
      [readVisibilityType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value sourceThread = entryBlock->getArgument(0);
        Value destMaskVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value readVisibilityPtr = entryBlock->getArgument(3);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value readVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
        Value zeroTensor =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, readVisibilityType);
        Value destMaskTensor = createThreadColumnMask(
            fb, destMaskVal, readVisibilityType, /*columnDim=*/3);
        Value cleared = arith::SelectOp::create(fb, destMaskTensor, zeroTensor,
                                                readVisibility);

        Value sourceColumnMask =
            createDimMask(fb, sourceThread, readVisibilityType, /*dim=*/3);
        Value sourceColumn = arith::SelectOp::create(
            fb, sourceColumnMask, readVisibility, zeroTensor);
        Value sourceVector = reduce<arith::OrIOp>(fb, sourceColumn, {3});
        Value broadcastRow = convertAndBroadcast(fb, sourceVector, {0, 1, 2, 4},
                                                 readVisibilityType);
        Value replicated = arith::SelectOp::create(fb, destMaskTensor,
                                                   broadcastRow, zeroTensor);

        Value updated = arith::OrIOp::create(fb, cleared, replicated);
        Value currentCTAMask = createCTASetMask(
            fb, readVisibilityType, /*dim=*/2, createCurrentCTAMask(fb));
        createMaskedStoreScratchMemory(fb, fb.getLoc(), readVisibilityPtr,
                                       updated, readVisibilityType,
                                       currentCTAMask);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createPublishCTAVisibilityCall(ImplicitLocOpBuilder &b,
                                                     uint64_t sourceMask,
                                                     uint64_t destMask,
                                                     MemType memType,
                                                     Operation *insertPoint) {
  if (auxData.writeVisibility[(int)memType].empty())
    return;

  auto writeVis = auxData.writeVisibility[(int)memType].at(insertPoint);
  auto readVis = auxData.readVisibility[(int)memType].at(insertPoint);
  auto writeVisibilityType = cast<RankedTensorType>(writeVis.type);
  auto readVisibilityType = cast<RankedTensorType>(readVis.type);
  SmallVector<Value> args = {arith::ConstantIntOp::create(b, sourceMask, 64),
                             arith::ConstantIntOp::create(b, destMask, 64),
                             writeVis.value, readVis.value};
  ManglingArgs specializationArgs{writeVisibilityType, readVisibilityType};
  RankedTensorType proxyVisibilityType;
  if (memType == MemType::SHARED_MEM &&
      !auxData.proxyAccessVisibility.empty()) {
    auto proxyVis = auxData.proxyAccessVisibility.at(insertPoint);
    proxyVisibilityType = cast<RankedTensorType>(proxyVis.type);
    args.push_back(proxyVis.value);
    specializationArgs.append(proxyVisibilityType);
  }
  createCallToCachedFunction(
      b, "publish_cta_visibility", args,
      /*assertInfo=*/std::nullopt, specializationArgs,
      [writeVisibilityType, readVisibilityType,
       proxyVisibilityType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value sourceMask = entryBlock->getArgument(0);
        Value destMask = entryBlock->getArgument(1);
        Value writeVisibilityPtr = entryBlock->getArgument(2);
        Value readVisibilityPtr = entryBlock->getArgument(3);
        Value currentCTA = createCurrentCTAMask(fb);

        Value writeVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
        Value zeroWrites =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, writeVisibilityType);
        auto writeElemType =
            cast<IntegerType>(writeVisibilityType.getElementType());
        Value sourceBits = triton::SplatOp::create(
            fb, writeVisibilityType,
            adjustIntegerWidth(fb, sourceMask, writeElemType));
        Value destBits = triton::SplatOp::create(
            fb, writeVisibilityType,
            adjustIntegerWidth(fb, destMask, writeElemType));
        Value sourceVisible = arith::CmpIOp::create(
            fb, arith::CmpIPredicate::ne,
            arith::AndIOp::create(fb, writeVisibility, sourceBits), zeroWrites);
        Value publishedWrites =
            arith::SelectOp::create(fb, sourceVisible, destBits, zeroWrites);
        Value newWriteVisibility =
            arith::OrIOp::create(fb, writeVisibility, publishedWrites);
        Value writeCTAMask =
            createCTASetMask(fb, writeVisibilityType, /*dim=*/2, currentCTA);
        createMaskedStoreScratchMemory(fb, fb.getLoc(), writeVisibilityPtr,
                                       newWriteVisibility, writeVisibilityType,
                                       writeCTAMask);

        auto publishObserverVisibility = [&](Value visibilityPtr,
                                             RankedTensorType visibilityType) {
          Value visibility = tti::createLoadScratchMemory(
              fb, fb.getLoc(), visibilityPtr, visibilityType);
          Value zero =
              tti::createConstIntTensor(fb, fb.getLoc(), 0, visibilityType);
          Value sourceColumns = createThreadColumnMask(
              fb, sourceMask, visibilityType, /*columnDim=*/3);
          Value source =
              arith::SelectOp::create(fb, sourceColumns, visibility, zero);
          source = reduce<arith::OrIOp>(fb, source, {3});
          source =
              convertAndBroadcast(fb, source, {0, 1, 2, 4}, visibilityType);
          Value destColumns = createThreadColumnMask(
              fb, destMask, visibilityType, /*columnDim=*/3);
          Value published =
              arith::SelectOp::create(fb, destColumns, source, zero);
          Value updated = arith::OrIOp::create(fb, visibility, published);
          Value ctaMask =
              createCTASetMask(fb, visibilityType, /*dim=*/2, currentCTA);
          createMaskedStoreScratchMemory(fb, fb.getLoc(), visibilityPtr,
                                         updated, visibilityType, ctaMask);
        };
        publishObserverVisibility(readVisibilityPtr, readVisibilityType);
        if (proxyVisibilityType)
          publishObserverVisibility(entryBlock->getArgument(4),
                                    proxyVisibilityType);

        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createPublishClusterVisibilityCall(
    ImplicitLocOpBuilder &b, Value pred, int thread, uint64_t threadPeersMask,
    bool partitionScoped, MemType memType, Operation *insertPoint) {
  if (auxData.writeVisibility[(int)memType].empty() ||
      auxData.readVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  auto writeVis = auxData.writeVisibility[(int)memType].at(insertPoint);
  auto readVis = auxData.readVisibility[(int)memType].at(insertPoint);
  auto writeVisibilityType = cast<RankedTensorType>(writeVis.type);
  auto readVisibilityType = cast<RankedTensorType>(readVis.type);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value threadPeersMaskVal =
      arith::ConstantIntOp::create(b, threadPeersMask, 64);
  SmallVector<Value> args = {pred, threadVal, threadPeersMaskVal,
                             writeVis.value, readVis.value};
  createCallToCachedFunction(
      b, "publish_cluster_visibility", args,
      /*assertInfo=*/std::nullopt,
      {writeVisibilityType, readVisibilityType, (uint64_t)partitionScoped},
      [writeVisibilityType, readVisibilityType,
       numBaseThreads = auxData.threadLayout.numBaseThreads,
       onlySynchronousThreads = auxData.threadLayout.totalNumThreads ==
                                auxData.threadLayout.numBaseThreads,
       partitionScoped](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value pred = entryBlock->getArgument(0);
        Value threadVal = entryBlock->getArgument(1);
        Value threadPeersMaskVal = entryBlock->getArgument(2);
        Value writeVisibilityPtr = entryBlock->getArgument(3);
        Value readVisibilityPtr = entryBlock->getArgument(4);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value writeVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
        Value readVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);

        Value zeroWrites =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, writeVisibilityType);
        Value syncWrites;
        if (partitionScoped) {
          auto elemType =
              cast<IntegerType>(writeVisibilityType.getElementType());
          Value threadI64 =
              arith::ExtUIOp::create(fb, fb.getI64Type(), threadVal);
          Value threadBitScalar = arith::ShLIOp::create(
              fb, arith::ConstantIntOp::create(fb, 1, 64), threadI64);
          Value threadBit = triton::SplatOp::create(
              fb, writeVisibilityType,
              adjustIntegerWidth(fb, threadBitScalar, elemType));
          Value peersMask = triton::SplatOp::create(
              fb, writeVisibilityType,
              adjustIntegerWidth(fb, threadPeersMaskVal, elemType));
          Value hasThreadWrite = arith::CmpIOp::create(
              fb, arith::CmpIPredicate::ne,
              arith::AndIOp::create(fb, writeVisibility, threadBit),
              zeroWrites);
          syncWrites = arith::SelectOp::create(fb, hasThreadWrite, peersMask,
                                               zeroWrites);
        } else if (onlySynchronousThreads) {
          syncWrites = writeVisibility;
        } else {
          // Top-level cluster barriers represent all synchronous threads. A
          // base-thread bit distinguishes synchronous work from async-only
          // TMA/TC/CLC effects, which use their own completion path.
          uint64_t baseThreadMask = (1ULL << numBaseThreads) - 1;
          Value baseMask = tti::createConstIntTensor(
              fb, fb.getLoc(), baseThreadMask, writeVisibilityType);
          Value hasBaseWrite = arith::CmpIOp::create(
              fb, arith::CmpIPredicate::ne,
              arith::AndIOp::create(fb, writeVisibility, baseMask), zeroWrites);
          syncWrites = arith::SelectOp::create(fb, hasBaseWrite,
                                               writeVisibility, zeroWrites);
        }
        Value writesForCluster = reduce<arith::OrIOp>(fb, syncWrites, {2});
        writesForCluster = convertAndBroadcast(fb, writesForCluster, {0, 1},
                                               writeVisibilityType);
        Value newWriteVisibility =
            arith::OrIOp::create(fb, writeVisibility, writesForCluster);
        tti::createStoreScratchMemory(fb, fb.getLoc(), writeVisibilityPtr,
                                      newWriteVisibility, writeVisibilityType,
                                      /*currentCTAOnly=*/false);

        Value zeroReads =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, readVisibilityType);
        Value readsForCluster;
        if (partitionScoped) {
          Value sourceColumn = arith::SelectOp::create(
              fb, createDimMask(fb, threadVal, readVisibilityType, /*dim=*/3),
              readVisibility, zeroReads);
          Value readsForThread = reduce<arith::OrIOp>(fb, sourceColumn, {2, 3});
          readsForThread = convertAndBroadcast(fb, readsForThread, {0, 1, 4},
                                               readVisibilityType);
          Value peerColumns = createThreadColumnMask(
              fb, threadPeersMaskVal, readVisibilityType, /*columnDim=*/3);
          readsForCluster = arith::SelectOp::create(fb, peerColumns,
                                                    readsForThread, zeroReads);
        } else if (onlySynchronousThreads) {
          readsForCluster = reduce<arith::OrIOp>(fb, readVisibility, {2, 3});
          readsForCluster = convertAndBroadcast(fb, readsForCluster, {0, 1, 4},
                                                readVisibilityType);
        } else {
          uint64_t baseThreadMask = (1ULL << numBaseThreads) - 1;
          Value baseObserverColumns = createThreadColumnMask(
              fb, arith::ConstantIntOp::create(fb, baseThreadMask, 64),
              readVisibilityType, /*columnDim=*/3);
          Value syncReads = arith::SelectOp::create(fb, baseObserverColumns,
                                                    readVisibility, zeroReads);
          readsForCluster = reduce<arith::OrIOp>(fb, syncReads, {2, 3});
          readsForCluster = convertAndBroadcast(fb, readsForCluster, {0, 1, 4},
                                                readVisibilityType);
        }
        Value newReadVisibility =
            arith::OrIOp::create(fb, readVisibility, readsForCluster);
        tti::createStoreScratchMemory(fb, fb.getLoc(), readVisibilityPtr,
                                      newReadVisibility, readVisibilityType,
                                      /*currentCTAOnly=*/false);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createSetProxyAccessCall(ImplicitLocOpBuilder &b,
                                               Value bufferMask, int thread,
                                               Value pred,
                                               Operation *insertPoint,
                                               Value effectCTAs) {
  if (auxData.proxyAccessVisibility.empty())
    return;
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);

  ValueType visibility = auxData.proxyAccessVisibility.at(insertPoint);
  auto visibilityType = cast<RankedTensorType>(visibility.type);
  bool hasTracking = !auxData.proxyAccessTracking.empty();
  RankedTensorType trackingType;
  SmallVector<Value> args = {bufferMask, pred,
                             arith::ConstantIntOp::create(b, thread, 32),
                             visibility.value, effectCTAs};
  ManglingArgs specializationArgs{visibilityType, (uint64_t)hasTracking};
  if (hasTracking) {
    ValueType tracking = auxData.proxyAccessTracking.at(insertPoint);
    trackingType = cast<RankedTensorType>(tracking.type);
    args.push_back(tracking.value);
    specializationArgs.append(trackingType);
  }

  createCallToCachedFunction(
      b, "set_proxy_access", args, /*assertInfo=*/std::nullopt,
      specializationArgs,
      [visibilityType, trackingType, hasTracking](ImplicitLocOpBuilder &fb,
                                                  Block *entryBlock) {
        Value bufferMask = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value threadVal = entryBlock->getArgument(2);
        Value visibilityPtr = entryBlock->getArgument(3);
        Value effectCTAs = entryBlock->getArgument(4);
        Value trackingPtr = hasTracking ? entryBlock->getArgument(5) : Value();

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value visibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), visibilityPtr, visibilityType);
        Value bufferVector = bufferMask;
        bufferMask = convertAndBroadcast(fb, bufferMask, {1}, visibilityType);
        Value bufferCTAMask =
            createCTASetMask(fb, visibilityType, /*dim=*/0, effectCTAs);
        Value currentCTA = createCurrentCTAMask(fb);
        Value originCTAMask =
            createCTASetMask(fb, visibilityType, /*dim=*/4, currentCTA);
        Value selectedBuffers =
            arith::AndIOp::create(fb, bufferMask, bufferCTAMask);
        selectedBuffers =
            arith::AndIOp::create(fb, selectedBuffers, originCTAMask);

        Value threadI64 =
            arith::ExtUIOp::create(fb, fb.getI64Type(), threadVal);
        Value one64 = arith::ConstantIntOp::create(fb, 1, 64);
        Value seenBitScalar = arith::ShLIOp::create(fb, one64, threadI64);
        Value fencedShift =
            arith::AddIOp::create(fb, threadI64,
                                  arith::ConstantIntOp::create(
                                      fb, ProxyAccessBits::fencedOffset, 64));
        Value fencedBitScalar = arith::ShLIOp::create(fb, one64, fencedShift);
        Value allOnes = arith::ConstantIntOp::create(fb, -1, 64);
        Value clearFencedScalar =
            arith::XOrIOp::create(fb, fencedBitScalar, allOnes);
        Value clearFenced =
            triton::SplatOp::create(fb, visibilityType, clearFencedScalar);
        Value clearedVisibility =
            arith::AndIOp::create(fb, visibility, clearFenced);
        clearedVisibility = arith::SelectOp::create(
            fb, selectedBuffers, clearedVisibility, visibility);

        Value consumerCTAMask =
            createCTASetMask(fb, visibilityType, /*dim=*/2, currentCTA);
        Value threadColumnMask =
            createDimMask(fb, threadVal, visibilityType, /*dim=*/3);
        Value ownerMask =
            arith::AndIOp::create(fb, selectedBuffers, consumerCTAMask);
        ownerMask = arith::AndIOp::create(fb, ownerMask, threadColumnMask);
        Value seenBit =
            triton::SplatOp::create(fb, visibilityType, seenBitScalar);
        Value withSeen = arith::OrIOp::create(fb, clearedVisibility, seenBit);
        Value updatedVisibility =
            arith::SelectOp::create(fb, ownerMask, withSeen, clearedVisibility);
        tti::createStoreScratchMemory(fb, fb.getLoc(), visibilityPtr,
                                      updatedVisibility, visibilityType,
                                      /*currentCTAOnly=*/false);

        // A new generic access supersedes fence coverage for an older access
        // from the same source. Clear that source's fence bit in outstanding
        // barrier snapshots from either phase as well, so an old publication
        // cannot mask it.
        if (hasTracking) {
          if (trackingType.getShape()[4] == 1) {
            Value tracking = tti::createLoadScratchMemory(
                fb, fb.getLoc(), trackingPtr, trackingType);
            Value trackingBuffers =
                convertAndBroadcast(fb, bufferVector, {1}, trackingType);
            Value trackingBufferCTAMask =
                createCTASetMask(fb, trackingType, /*dim=*/0, effectCTAs);
            Value trackingOriginCTAMask =
                createCTASetMask(fb, trackingType, /*dim=*/4, currentCTA);
            Value trackingMask = arith::AndIOp::create(fb, trackingBuffers,
                                                       trackingBufferCTAMask);
            trackingMask =
                arith::AndIOp::create(fb, trackingMask, trackingOriginCTAMask);
            Value trackingClear =
                triton::SplatOp::create(fb, trackingType, clearFencedScalar);
            Value clearedTracking =
                arith::AndIOp::create(fb, tracking, trackingClear);
            Value updatedTracking = arith::SelectOp::create(
                fb, trackingMask, clearedTracking, tracking);
            tti::createStoreScratchMemory(fb, fb.getLoc(), trackingPtr,
                                          updatedTracking, trackingType,
                                          /*currentCTAOnly=*/false);
          } else {
            RankedTensorType bankType = tti::getSlicedTensorType(
                trackingType, {0, 1, 2, 3}, trackingType.getElementType());
            Value trackingBuffers =
                convertAndBroadcast(fb, bufferVector, {1}, bankType);
            Value ownerMask =
                createCTASetMask(fb, bankType, /*dim=*/0, effectCTAs);
            Value trackingMask =
                arith::AndIOp::create(fb, trackingBuffers, ownerMask);
            Value trackingClear =
                triton::SplatOp::create(fb, bankType, clearFencedScalar);
            Value originId =
                tti::ExperimentalClusterCTAIdOp::create(fb, fb.getLoc());
            for (int phase = 0; phase < 2; ++phase) {
              Value bankPtr = createTrackingOriginPhasePointer(
                  fb, trackingPtr, trackingType, bankType, originId, phase);
              Value tracking = tti::createLoadScratchMemory(fb, fb.getLoc(),
                                                            bankPtr, bankType);
              Value clearedTracking =
                  arith::AndIOp::create(fb, tracking, trackingClear);
              Value updated = arith::SelectOp::create(
                  fb, trackingMask, clearedTracking, tracking);
              tti::createStoreScratchMemory(fb, fb.getLoc(), bankPtr, updated,
                                            bankType,
                                            /*currentCTAOnly=*/false);
            }
          }
        }

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createFenceProxyAccessesCall(ImplicitLocOpBuilder &b,
                                                   int thread, bool cluster,
                                                   Value pred,
                                                   Operation *insertPoint) {
  if (auxData.proxyAccessVisibility.empty())
    return;
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  ValueType visibility = auxData.proxyAccessVisibility.at(insertPoint);
  auto visibilityType = cast<RankedTensorType>(visibility.type);
  SmallVector<Value> args = {arith::ConstantIntOp::create(b, thread, 32), pred,
                             visibility.value};
  createCallToCachedFunction(
      b, "fence_proxy_accesses", args, /*assertInfo=*/std::nullopt,
      {visibilityType, (uint64_t)cluster},
      [visibilityType, cluster](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value threadVal = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value visibilityPtr = entryBlock->getArgument(2);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);
        Value visibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), visibilityPtr, visibilityType);
        Value currentCTA = createCurrentCTAMask(fb);
        Value mask =
            createCTASetMask(fb, visibilityType, /*dim=*/2, currentCTA);
        mask = arith::AndIOp::create(
            fb, mask, createDimMask(fb, threadVal, visibilityType, /*dim=*/3));
        if (!cluster) {
          mask = arith::AndIOp::create(
              fb, mask,
              createCTASetMask(fb, visibilityType, /*dim=*/0, currentCTA));
        }
        Value seenMask = tti::createConstIntTensor(
            fb, fb.getLoc(), ProxyAccessBits::seenMask, visibilityType);
        Value seen = arith::AndIOp::create(fb, visibility, seenMask);
        Value shift = tti::createConstIntTensor(
            fb, fb.getLoc(), ProxyAccessBits::fencedOffset, visibilityType);
        Value fenced = arith::ShLIOp::create(fb, seen, shift);
        Value covered = arith::OrIOp::create(fb, visibility, fenced);
        Value updated = arith::SelectOp::create(fb, mask, covered, visibility);
        createMaskedStoreScratchMemory(fb, fb.getLoc(), visibilityPtr, updated,
                                       visibilityType, mask);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createTrackProxyAccessesCall(ImplicitLocOpBuilder &b,
                                                   Value mbar, int thread,
                                                   Value pred,
                                                   Operation *insertPoint,
                                                   Value barrierCTAs) {
  createTrackProxyAccessesCallImpl(b, mbar, thread, pred, insertPoint,
                                   barrierCTAs, Value(), Value());
}

void FunctionBuilder::createTrackProxyAccessesForBufferCall(
    ImplicitLocOpBuilder &b, Value mbar, Value bufferMask, int thread,
    Value pred, Operation *insertPoint, Value barrierCTAs, Value effectCTAs) {
  createTrackProxyAccessesCallImpl(b, mbar, thread, pred, insertPoint,
                                   barrierCTAs, bufferMask, effectCTAs);
}

void FunctionBuilder::createTrackProxyAccessesCallImpl(
    ImplicitLocOpBuilder &b, Value mbar, int thread, Value pred,
    Operation *insertPoint, Value barrierCTAs, Value bufferMask,
    Value effectCTAs) {
  bool filterByBuffer = static_cast<bool>(bufferMask);
  if (auxData.barriers.empty() || auxData.barrierStates.empty() ||
      auxData.proxyAccessVisibility.empty() ||
      auxData.proxyAccessTracking.empty())
    return;
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  ValueType barriers = auxData.barriers.at(insertPoint);
  ValueType barrierStates = auxData.barrierStates.at(insertPoint);
  ValueType visibility = auxData.proxyAccessVisibility.at(insertPoint);
  ValueType tracking = auxData.proxyAccessTracking.at(insertPoint);
  auto barriersType = cast<RankedTensorType>(barriers.type);
  auto barrierStatesType = cast<RankedTensorType>(barrierStates.type);
  auto visibilityType = cast<RankedTensorType>(visibility.type);
  auto trackingType = cast<RankedTensorType>(tracking.type);
  SmallVector<Value> args = {
      tti::ExperimentalMemDescToI32Op::create(b, mbar),
      arith::ConstantIntOp::create(b, getMemDescLength(mbar), 32),
      pred,
      arith::ConstantIntOp::create(b, thread, 32),
      barriers.value,
      barrierStates.value,
      visibility.value,
      tracking.value,
      barrierCTAs};
  ManglingArgs specializationArgs{barriersType, barrierStatesType,
                                  visibilityType, trackingType};
  if (filterByBuffer) {
    args.append({bufferMask, effectCTAs});
  }
  createCallToCachedFunction(
      b,
      filterByBuffer ? "track_proxy_accesses_for_buffer"
                     : "track_proxy_accesses",
      args, /*assertInfo=*/std::nullopt, specializationArgs,
      [barrierStatesType, visibilityType, trackingType,
       filterByBuffer](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadVal = entryBlock->getArgument(3);
        Value barriers = entryBlock->getArgument(4);
        Value barrierStatesPtr = entryBlock->getArgument(5);
        Value visibilityPtr = entryBlock->getArgument(6);
        Value trackingPtr = entryBlock->getArgument(7);
        Value barrierCTAs = entryBlock->getArgument(8);
        Value completeMask =
            filterByBuffer ? entryBlock->getArgument(9) : Value();
        Value effectCTAs =
            filterByBuffer ? entryBlock->getArgument(10) : Value();

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);
        Value barrierStates = tti::createLoadScratchMemory(
            fb, fb.getLoc(), barrierStatesPtr, barrierStatesType);
        Value visibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), visibilityPtr, visibilityType);
        Value tracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), trackingPtr, trackingType);
        Value currentCTA = createCurrentCTAMask(fb);
        Value sourceMask =
            createCTASetMask(fb, visibilityType, /*dim=*/2, currentCTA);
        sourceMask = arith::AndIOp::create(
            fb, sourceMask,
            createDimMask(fb, threadVal, visibilityType, /*dim=*/3));
        Value containedBuffers;
        if (filterByBuffer) {
          containedBuffers = completeMask;
          Value visibilityBuffers =
              convertAndBroadcast(fb, containedBuffers, {1}, visibilityType);
          sourceMask = arith::AndIOp::create(fb, sourceMask, visibilityBuffers);
          sourceMask = arith::AndIOp::create(
              fb, sourceMask,
              createCTASetMask(fb, visibilityType, /*dim=*/0, effectCTAs));
        }
        Value zeroVisibility =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, visibilityType);
        Value source =
            arith::SelectOp::create(fb, sourceMask, visibility, zeroVisibility);
        source = reduce<arith::OrIOp>(fb, source, {2, 3});
        source = convertAndBroadcast(fb, source, {0, 1, 4}, trackingType);

        Value descriptor = createBufferDescriptor(fb, mbarOffset, lengthVal);
        Value barriersEqBar =
            createCmpIntTensorScalar(fb, barriers, descriptor);
        barriersEqBar =
            convertAndBroadcast(fb, barriersEqBar, {3}, trackingType);
        Value barrierCTAMask =
            createCTASetMask(fb, trackingType, /*dim=*/2, barrierCTAs);
        Value trackMask =
            arith::AndIOp::create(fb, barriersEqBar, barrierCTAMask);
        Value currentPhases = arith::AndIOp::create(
            fb, barrierStates,
            tti::createConstIntTensor(fb, fb.getLoc(), 1, barrierStatesType));
        currentPhases =
            convertAndBroadcast(fb, currentPhases, {2, 3}, trackingType);
        Value phaseIndices =
            createDimIndices(fb, trackingType, trackingType.getRank() - 1);
        auto phaseIndicesType = cast<RankedTensorType>(phaseIndices.getType());
        currentPhases =
            arith::TruncIOp::create(fb, phaseIndicesType, currentPhases);
        Value phaseMask = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                                currentPhases, phaseIndices);
        trackMask = arith::AndIOp::create(fb, trackMask, phaseMask);
        if (filterByBuffer) {
          Value trackingBuffers =
              convertAndBroadcast(fb, containedBuffers, {1}, trackingType);
          trackMask = arith::AndIOp::create(fb, trackMask, trackingBuffers);
          trackMask = arith::AndIOp::create(
              fb, trackMask,
              createCTASetMask(fb, trackingType, /*dim=*/0, effectCTAs));
        }
        Value withSource = arith::OrIOp::create(fb, tracking, source);
        if (filterByBuffer) {
          Value updated =
              arith::SelectOp::create(fb, trackMask, withSource, tracking);
          createMaskedStoreScratchMemory(fb, fb.getLoc(), trackingPtr, updated,
                                         trackingType, trackMask);
        } else {
          tti::createStoreScratchMemory(fb, fb.getLoc(), trackingPtr,
                                        withSource, trackingType,
                                        /*currentCTAOnly=*/false, trackMask);
        }

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createCompleteBarrierWaitCall(ImplicitLocOpBuilder &b,
                                                    Value mbar, Value phase,
                                                    int thread, Value pred,
                                                    Operation *insertPoint) {
  if (auxData.barriers.empty())
    return;
  if (auxData.proxyAccessVisibility.empty() ||
      auxData.proxyAccessTracking.empty()) {
    createClearWaitingCall(b, mbar, thread, pred, insertPoint);
    return;
  }
  const bool clearWaiting = !auxData.waiting.empty();
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  ValueType barriers = auxData.barriers.at(insertPoint);
  ValueType visibility = auxData.proxyAccessVisibility.at(insertPoint);
  ValueType tracking = auxData.proxyAccessTracking.at(insertPoint);
  auto barriersType = cast<RankedTensorType>(barriers.type);
  auto visibilityType = cast<RankedTensorType>(visibility.type);
  auto trackingType = cast<RankedTensorType>(tracking.type);
  SmallVector<Value> args = {
      tti::ExperimentalMemDescToI32Op::create(b, mbar),
      arith::ConstantIntOp::create(b, getMemDescLength(mbar), 32),
      phase,
      pred,
      arith::ConstantIntOp::create(b, thread, 32),
      barriers.value,
      visibility.value,
      tracking.value};
  ManglingArgs specializationArgs;
  specializationArgs.append(barriersType);
  specializationArgs.append(visibilityType);
  specializationArgs.append(trackingType);
  specializationArgs.append(static_cast<uint64_t>(clearWaiting));

  RankedTensorType waitingType;
  if (clearWaiting) {
    ValueType waiting = auxData.waiting.at(insertPoint);
    waitingType = cast<RankedTensorType>(waiting.type);
    args.push_back(waiting.value);
    specializationArgs.append(waitingType);
  }

  createCallToCachedFunction(
      b, "complete_barrier_wait", args, /*assertInfo=*/std::nullopt,
      specializationArgs,
      [clearWaiting, visibilityType, trackingType,
       waitingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value phase = entryBlock->getArgument(2);
        Value pred = entryBlock->getArgument(3);
        Value threadVal = entryBlock->getArgument(4);
        Value barriers = entryBlock->getArgument(5);
        Value visibilityPtr = entryBlock->getArgument(6);
        Value trackingPtr = entryBlock->getArgument(7);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);
        Value descriptor = createBufferDescriptor(fb, mbarOffset, lengthVal);
        Value barriersEqBar =
            createCmpIntTensorScalar(fb, barriers, descriptor);
        Value currentCTA = createCurrentCTAMask(fb);

        Value visibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), visibilityPtr, visibilityType);
        Value tracking =
            createLoadTrackingPhase(fb, trackingPtr, trackingType, phase);
        auto trackingPhaseType = cast<RankedTensorType>(tracking.getType());
        Value proxyBarrierMask =
            convertAndBroadcast(fb, barriersEqBar, {3}, trackingPhaseType);
        Value barrierCTAMask =
            createCTASetMask(fb, trackingPhaseType, /*dim=*/2, currentCTA);
        Value selected =
            arith::AndIOp::create(fb, proxyBarrierMask, barrierCTAMask);
        Value zeroTracking =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, trackingPhaseType);
        Value frontier =
            arith::SelectOp::create(fb, selected, tracking, zeroTracking);
        frontier = reduce<arith::OrIOp>(fb, frontier, {2, 3});
        frontier = convertAndBroadcast(fb, frontier, {0, 1, 4}, visibilityType);

        Value targetMask =
            createCTASetMask(fb, visibilityType, /*dim=*/2, currentCTA);
        targetMask = arith::AndIOp::create(
            fb, targetMask,
            createDimMask(fb, threadVal, visibilityType, /*dim=*/3));
        Value withFrontier = arith::OrIOp::create(fb, visibility, frontier);
        Value updated =
            arith::SelectOp::create(fb, targetMask, withFrontier, visibility);
        createMaskedStoreScratchMemory(fb, fb.getLoc(), visibilityPtr, updated,
                                       visibilityType, targetMask);

        if (clearWaiting) {
          Value waitingPtr = entryBlock->getArgument(8);
          Value waiting = tti::createLoadScratchMemory(fb, fb.getLoc(),
                                                       waitingPtr, waitingType);
          Value waitingBarrierMask =
              convertAndBroadcast(fb, barriersEqBar, {1}, waitingType);
          Value ctaMask = createLeadCTAEffectMask(fb, waitingType, currentCTA);
          waitingBarrierMask =
              arith::AndIOp::create(fb, waitingBarrierMask, ctaMask);

          Value bitsPerThread =
              arith::ConstantIntOp::create(fb, WaitingBits::bitsPerThread, 32);
          Value flagBit =
              arith::ConstantIntOp::create(fb, WaitingBits::flagBit, 32);
          Value phaseBit =
              arith::ConstantIntOp::create(fb, WaitingBits::phaseBit, 32);
          Value one = arith::ConstantIntOp::create(fb, 1, 32);
          Value minusOne = arith::ConstantIntOp::create(fb, -1, 32);
          Value baseTimesBits =
              arith::MulIOp::create(fb, threadVal, bitsPerThread);
          Value flagShift = arith::AddIOp::create(fb, baseTimesBits, flagBit);
          Value phaseShift = arith::AddIOp::create(fb, baseTimesBits, phaseBit);
          Value flagMask = arith::ShLIOp::create(fb, one, flagShift);
          Value phaseMask = arith::ShLIOp::create(fb, one, phaseShift);
          Value bits = arith::OrIOp::create(fb, flagMask, phaseMask);
          Value clearMask = arith::XOrIOp::create(fb, bits, minusOne);
          Value clearMaskTensor =
              triton::SplatOp::create(fb, waitingType, clearMask);
          Value clearedWaiting =
              arith::AndIOp::create(fb, waiting, clearMaskTensor);
          Value updated = arith::SelectOp::create(fb, waitingBarrierMask,
                                                  clearedWaiting, waiting);
          createMaskedStoreScratchMemory(fb, fb.getLoc(), waitingPtr, updated,
                                         waitingType, ctaMask);
        }

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createVerifyProxyAccessCall(ImplicitLocOpBuilder &b,
                                                  Value bufferMask, int thread,
                                                  StringRef operandName,
                                                  Value pred,
                                                  Operation *insertPoint,
                                                  Value effectCTAs) {
  if (auxData.proxyAccessVisibility.empty())
    return;
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  ValueType visibility = auxData.proxyAccessVisibility.at(insertPoint);
  auto visibilityType = cast<RankedTensorType>(visibility.type);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  std::string message =
      "Async shared-memory access is missing fence_async_shared";
  if (!operandName.empty())
    message += ". Operand: " + operandName.str();
  AssertInfo assertInfo{message, b.getI1Type()};
  SmallVector<Value> args = {bufferMask, pred, threadVal, visibility.value,
                             effectCTAs};
  createCallToCachedFunction(
      b, "verify_proxy_access", args, assertInfo, {visibilityType},
      [visibilityType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value checkMask = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value threadVal = entryBlock->getArgument(2);
        Value visibilityPtr = entryBlock->getArgument(3);
        Value effectCTAs = entryBlock->getArgument(4);

        Value visibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), visibilityPtr, visibilityType);
        checkMask = convertAndBroadcast(fb, checkMask, {1}, visibilityType);
        Value mask = arith::AndIOp::create(
            fb, checkMask,
            createCTASetMask(fb, visibilityType, /*dim=*/0, effectCTAs));
        Value currentCTA = createCurrentCTAMask(fb);
        mask = arith::AndIOp::create(
            fb, mask,
            createCTASetMask(fb, visibilityType, /*dim=*/2, currentCTA));
        mask = arith::AndIOp::create(
            fb, mask, createDimMask(fb, threadVal, visibilityType, /*dim=*/3));
        Value zero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, visibilityType);
        Value selected = arith::SelectOp::create(fb, mask, visibility, zero);
        Value seenMask = tti::createConstIntTensor(
            fb, fb.getLoc(), ProxyAccessBits::seenMask, visibilityType);
        Value seen = arith::AndIOp::create(fb, selected, seenMask);
        Value shift = tti::createConstIntTensor(
            fb, fb.getLoc(), ProxyAccessBits::fencedOffset, visibilityType);
        Value fenced = arith::ShRUIOp::create(fb, selected, shift);
        fenced = arith::AndIOp::create(fb, fenced, seenMask);
        Value notFenced = arith::XOrIOp::create(fb, fenced, seenMask);
        Value missing = arith::AndIOp::create(fb, seen, notFenced);
        Value missingBits =
            arith::CmpIOp::create(fb, arith::CmpIPredicate::ne, missing, zero);
        Value missingAny = reduceAll<arith::OrIOp>(fb, missingBits);
        Value zeroScalar = arith::ConstantOp::create(
            fb, missingAny.getType(),
            fb.getIntegerAttr(missingAny.getType(), 0));
        Value ok = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                         missingAny, zeroScalar);
        Value vTrue = arith::ConstantOp::create(
            fb, ok.getType(), fb.getIntegerAttr(fb.getI1Type(), 1));
        Value predicatedOk = arith::SelectOp::create(fb, pred, ok, vTrue);
        triton::ReturnOp::create(fb, predicatedOk);
      });
}

void FunctionBuilder::createCopyProxyAccessesCall(ImplicitLocOpBuilder &b,
                                                  int sourceThread,
                                                  uint64_t destMask, Value pred,
                                                  Operation *insertPoint) {
  if (auxData.proxyAccessVisibility.empty())
    return;
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  ValueType visibility = auxData.proxyAccessVisibility.at(insertPoint);
  auto visibilityType = cast<RankedTensorType>(visibility.type);
  SmallVector<Value> args = {arith::ConstantIntOp::create(b, sourceThread, 32),
                             arith::ConstantIntOp::create(b, destMask, 64),
                             pred, visibility.value};
  createCallToCachedFunction(
      b, "copy_proxy_accesses", args, /*assertInfo=*/std::nullopt,
      {visibilityType},
      [visibilityType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value sourceThread = entryBlock->getArgument(0);
        Value destMaskVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value visibilityPtr = entryBlock->getArgument(3);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);
        Value visibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), visibilityPtr, visibilityType);
        Value zero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, visibilityType);
        Value destColumns = createThreadColumnMask(
            fb, destMaskVal, visibilityType, /*columnDim=*/3);
        Value cleared =
            arith::SelectOp::create(fb, destColumns, zero, visibility);
        Value sourceColumn = arith::SelectOp::create(
            fb, createDimMask(fb, sourceThread, visibilityType, /*dim=*/3),
            visibility, zero);
        sourceColumn = reduce<arith::OrIOp>(fb, sourceColumn, {3});
        sourceColumn =
            convertAndBroadcast(fb, sourceColumn, {0, 1, 2, 4}, visibilityType);
        Value replicated =
            arith::SelectOp::create(fb, destColumns, sourceColumn, zero);
        Value updated = arith::OrIOp::create(fb, cleared, replicated);
        Value currentCTAMask = createCTASetMask(fb, visibilityType, /*dim=*/2,
                                                createCurrentCTAMask(fb));
        createMaskedStoreScratchMemory(fb, fb.getLoc(), visibilityPtr, updated,
                                       visibilityType, currentCTAMask);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createPublishClusterProxyAccessesCall(
    ImplicitLocOpBuilder &b, Value pred, int thread, bool partitionScoped,
    Operation *insertPoint) {
  if (auxData.proxyAccessVisibility.empty())
    return;
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  ValueType visibility = auxData.proxyAccessVisibility.at(insertPoint);
  auto visibilityType = cast<RankedTensorType>(visibility.type);
  SmallVector<Value> args = {pred, arith::ConstantIntOp::create(b, thread, 32),
                             visibility.value};
  createCallToCachedFunction(
      b, "publish_cluster_proxy_accesses", args,
      /*assertInfo=*/std::nullopt, {visibilityType, (uint64_t)partitionScoped},
      [visibilityType, partitionScoped](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
        Value pred = entryBlock->getArgument(0);
        Value threadVal = entryBlock->getArgument(1);
        Value visibilityPtr = entryBlock->getArgument(2);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);
        Value visibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), visibilityPtr, visibilityType);
        Value source = visibility;
        Value zero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, visibilityType);
        if (partitionScoped) {
          Value sourceColumn =
              createDimMask(fb, threadVal, visibilityType, /*dim=*/3);
          source = arith::SelectOp::create(fb, sourceColumn, visibility, zero);
        }
        Value frontier = reduce<arith::OrIOp>(fb, source, {2, 3});
        frontier = convertAndBroadcast(fb, frontier, {0, 1, 4}, visibilityType);
        if (partitionScoped) {
          Value destinationColumn =
              createDimMask(fb, threadVal, visibilityType, /*dim=*/3);
          frontier =
              arith::SelectOp::create(fb, destinationColumn, frontier, zero);
        }
        Value updated = arith::OrIOp::create(fb, visibility, frontier);
        tti::createStoreScratchMemory(fb, fb.getLoc(), visibilityPtr, updated,
                                      visibilityType,
                                      /*currentCTAOnly=*/false);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createStageAccessForCommitCall(
    ImplicitLocOpBuilder &b, Value bufferMask, int thread, Value pred,
    MemType /*memType*/, CommitKind::Kind commitKind, Operation *insertPoint) {
  if (auxData.commits[commitKind].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  ValueType outstandingCommits = auxData.commits[commitKind].at(insertPoint);
  auto commitsType = cast<RankedTensorType>(outstandingCommits.type);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  SmallVector<Value> args = {bufferMask, pred, threadVal,
                             outstandingCommits.value};
  createCallToCachedFunction(
      b, "stage_access_for_commit", args,
      /*assertInfo=*/std::nullopt, {commitsType},
      [commitsType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value bufferMask = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value threadVal = entryBlock->getArgument(2);
        Value outstandingCommitsPtr = entryBlock->getArgument(3);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value commits = tti::createLoadScratchMemory(
            fb, fb.getLoc(), outstandingCommitsPtr, commitsType);
        bufferMask = convertAndBroadcast(fb, bufferMask, {1}, commitsType);
        Value ctaMask = createCTASetMask(fb, commitsType, /*dim=*/0,
                                         createCurrentCTAMask(fb));
        bufferMask = arith::AndIOp::create(fb, bufferMask, ctaMask);
        Value threadColumnMask =
            createDimMask(fb, threadVal, commitsType, /*dim=*/2);
        Value bufAndThread =
            arith::AndIOp::create(fb, bufferMask, threadColumnMask);
        Value minusOne =
            tti::createConstIntTensor(fb, fb.getLoc(), -1, commitsType, true);
        Value updated =
            arith::SelectOp::create(fb, bufAndThread, minusOne, commits);
        createMaskedStoreScratchMemory(fb, fb.getLoc(), outstandingCommitsPtr,
                                       updated, commitsType, ctaMask);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createCommitAccessesCall(ImplicitLocOpBuilder &b,
                                               int thread, Value pred,
                                               CommitKind::Kind commitKind,
                                               Operation *insertPoint) {
  if (auxData.commits[commitKind].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  ValueType outstandingCommits = auxData.commits[commitKind].at(insertPoint);
  auto commitsType = cast<RankedTensorType>(outstandingCommits.type);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  SmallVector<Value> args = {threadVal, pred, outstandingCommits.value};
  createCallToCachedFunction(
      b, "commit_accesses", args,
      /*assertInfo=*/std::nullopt, {commitsType},
      [commitsType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value threadVal = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value outstandingCommitsPtr = entryBlock->getArgument(2);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value commits = tti::createLoadScratchMemory(
            fb, fb.getLoc(), outstandingCommitsPtr, commitsType);
        Type elementType = commitsType.getElementType();
        Value zero = arith::ConstantOp::create(
            fb, elementType, fb.getIntegerAttr(elementType, 0));
        Value minusOne = arith::ConstantOp::create(
            fb, elementType, fb.getIntegerAttr(elementType, -1));
        Value ones = tti::createConstIntTensor(fb, fb.getLoc(), 1, commitsType);

        Value threadMask = createDimMask(fb, threadVal, commitsType, /*dim=*/2);
        Value ctaMask = createCTASetMask(fb, commitsType, /*dim=*/0,
                                         createCurrentCTAMask(fb));
        threadMask = arith::AndIOp::create(fb, threadMask, ctaMask);
        auto commitsGtZero = createCmpIntTensorScalar(
            fb, commits, zero, arith::CmpIPredicate::sgt);
        commitsGtZero = arith::AndIOp::create(fb, commitsGtZero, threadMask);
        Value commitsPlusOne = arith::AddIOp::create(fb, commits, ones);
        commits =
            arith::SelectOp::create(fb, commitsGtZero, commitsPlusOne, commits);

        auto commitsEqMinusOne = createCmpIntTensorScalar(
            fb, commits, minusOne, arith::CmpIPredicate::eq);
        commitsEqMinusOne =
            arith::AndIOp::create(fb, commitsEqMinusOne, threadMask);
        commits = arith::SelectOp::create(fb, commitsEqMinusOne, ones, commits);

        createMaskedStoreScratchMemory(fb, fb.getLoc(), outstandingCommitsPtr,
                                       commits, commitsType, ctaMask);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createClearOutstandingCommitsTransferWritesCall(
    ImplicitLocOpBuilder &b, int thread, uint64_t transferThreadMask,
    int outstandingNum, Value pred, CommitKind::Kind commitKind,
    MemType memType, Operation *insertPoint) {
  createClearOutstandingCommitsTransferCall(
      b, thread, transferThreadMask, outstandingNum, pred, commitKind, memType,
      insertPoint, /*transferWrites=*/true, /*transferReads=*/false);
}

void FunctionBuilder::createClearOutstandingCommitsTransferReadsCall(
    ImplicitLocOpBuilder &b, int thread, uint64_t transferThreadMask,
    int outstandingNum, Value pred, CommitKind::Kind commitKind,
    MemType memType, Operation *insertPoint) {
  createClearOutstandingCommitsTransferCall(
      b, thread, transferThreadMask, outstandingNum, pred, commitKind, memType,
      insertPoint, /*transferWrites=*/false, /*transferReads=*/true);
}

void FunctionBuilder::createClearOutstandingCommitsTransferBothCall(
    ImplicitLocOpBuilder &b, int thread, uint64_t transferThreadMask,
    int outstandingNum, Value pred, CommitKind::Kind commitKind,
    MemType memType, Operation *insertPoint) {
  createClearOutstandingCommitsTransferCall(
      b, thread, transferThreadMask, outstandingNum, pred, commitKind, memType,
      insertPoint, /*transferWrites=*/true, /*transferReads=*/true);
}

void FunctionBuilder::createClearOutstandingCommitsTransferCall(
    ImplicitLocOpBuilder &b, int thread, uint64_t transferThreadMask,
    int outstandingNum, Value pred, CommitKind::Kind commitKind,
    MemType memType, Operation *insertPoint, bool transferWrites,
    bool transferReads) {
  if (auxData.commits[commitKind].empty())
    return;
  bool hasWriteVisibility =
      transferWrites && !auxData.writeVisibility[(int)memType].empty();
  bool hasReadVisibility =
      transferReads && !auxData.readVisibility[(int)memType].empty();
  if (!hasWriteVisibility && !hasReadVisibility)
    return;
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);

  ValueType outstandingCommits = auxData.commits[commitKind].at(insertPoint);
  auto commitsType = cast<RankedTensorType>(outstandingCommits.type);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value transferMaskVal =
      arith::ConstantIntOp::create(b, transferThreadMask, 64);
  Value outstandingNumVal = arith::ConstantIntOp::create(b, outstandingNum, 32);
  SmallVector<Value> args = {threadVal, transferMaskVal, outstandingNumVal,
                             pred, outstandingCommits.value};
  ManglingArgs specializationArgs{commitsType};

  RankedTensorType writeVisibilityType;
  if (hasWriteVisibility) {
    ValueType writeVisibility =
        auxData.writeVisibility[(int)memType].at(insertPoint);
    writeVisibilityType = cast<RankedTensorType>(writeVisibility.type);
    args.push_back(writeVisibility.value);
    specializationArgs.append(Type(writeVisibilityType));
  }
  RankedTensorType readVisibilityType;
  if (hasReadVisibility) {
    ValueType readVisibility =
        auxData.readVisibility[(int)memType].at(insertPoint);
    readVisibilityType = cast<RankedTensorType>(readVisibility.type);
    args.push_back(readVisibility.value);
    specializationArgs.append(Type(readVisibilityType));
  }

  std::string functionName = hasWriteVisibility && hasReadVisibility
                                 ? "clear_outstanding_commits_transfer_both"
                             : hasWriteVisibility
                                 ? "clear_outstanding_commits_transfer_writes"
                                 : "clear_outstanding_commits_transfer_reads";
  createCallToCachedFunction(
      b, functionName, args, /*assertInfo=*/std::nullopt, specializationArgs,
      [commitsType, writeVisibilityType, readVisibilityType, hasWriteVisibility,
       hasReadVisibility](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value threadVal = entryBlock->getArgument(0);
        Value transferMaskVal = entryBlock->getArgument(1);
        Value outstandingNumVal = entryBlock->getArgument(2);
        Value pred = entryBlock->getArgument(3);
        Value outstandingCommitsPtr = entryBlock->getArgument(4);
        unsigned nextArg = 5;
        Value writeVisibilityPtr;
        if (hasWriteVisibility)
          writeVisibilityPtr = entryBlock->getArgument(nextArg++);
        Value readVisibilityPtr;
        if (hasReadVisibility)
          readVisibilityPtr = entryBlock->getArgument(nextArg++);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value outstandingCommits = tti::createLoadScratchMemory(
            fb, fb.getLoc(), outstandingCommitsPtr, commitsType);
        auto elemIntType = cast<IntegerType>(commitsType.getElementType());
        Value outstandingNumElem =
            adjustIntegerWidth(fb, outstandingNumVal, elemIntType);
        Value threadColumnMask =
            createDimMask(fb, threadVal, commitsType, /*dim=*/2);
        Value commitCTAMask = createCTASetMask(fb, commitsType, /*dim=*/0,
                                               createCurrentCTAMask(fb));
        threadColumnMask =
            arith::AndIOp::create(fb, threadColumnMask, commitCTAMask);
        Value outstandingCommitsGtOutstandingNum =
            createCmpIntTensorScalar(fb, outstandingCommits, outstandingNumElem,
                                     arith::CmpIPredicate::sgt);
        outstandingCommitsGtOutstandingNum = arith::AndIOp::create(
            fb, outstandingCommitsGtOutstandingNum, threadColumnMask);
        Value rowMask =
            reduceLastDim<arith::OrIOp>(fb, outstandingCommitsGtOutstandingNum);

        if (hasWriteVisibility) {
          Value writeVisibility = tti::createLoadScratchMemory(
              fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
          Value writeRowMask =
              convertAndBroadcast(fb, rowMask, {0, 1}, writeVisibilityType);
          Value transferMaskElem = adjustIntegerWidth(
              fb, transferMaskVal,
              cast<IntegerType>(writeVisibilityType.getElementType()));
          Value transferMaskTensor = triton::SplatOp::create(
              fb, writeVisibilityType, transferMaskElem);
          Value withTransfer =
              arith::OrIOp::create(fb, writeVisibility, transferMaskTensor);
          Value updated = arith::SelectOp::create(
              fb, writeRowMask, withTransfer, writeVisibility);
          Value writeMask = createCTASetMask(fb, writeVisibilityType, /*dim=*/2,
                                             createCurrentCTAMask(fb));
          createMaskedStoreScratchMemory(fb, fb.getLoc(), writeVisibilityPtr,
                                         updated, writeVisibilityType,
                                         writeMask);
        }

        if (hasReadVisibility) {
          Value readVisibility = tti::createLoadScratchMemory(
              fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
          Value readRowMask =
              convertAndBroadcast(fb, rowMask, {0, 1}, readVisibilityType);
          Value currentCTAMask =
              createCTASetMask(fb, readVisibilityType,
                               /*dim=*/4, createCurrentCTAMask(fb));
          readRowMask = arith::AndIOp::create(fb, readRowMask, currentCTAMask);
          Value transferMaskElem = adjustIntegerWidth(
              fb, transferMaskVal,
              cast<IntegerType>(readVisibilityType.getElementType()));
          Value transferMaskTensor =
              triton::SplatOp::create(fb, readVisibilityType, transferMaskElem);
          Value withTransfer =
              arith::OrIOp::create(fb, readVisibility, transferMaskTensor);
          Value updated = arith::SelectOp::create(fb, readRowMask, withTransfer,
                                                  readVisibility);
          Value readMask = createCTASetMask(fb, readVisibilityType, /*dim=*/2,
                                            createCurrentCTAMask(fb));
          createMaskedStoreScratchMemory(fb, fb.getLoc(), readVisibilityPtr,
                                         updated, readVisibilityType, readMask);
        }

        Value zero = tti::createConstIntTensor(fb, fb.getLoc(), 0, commitsType);
        outstandingCommits = arith::SelectOp::create(
            fb, outstandingCommitsGtOutstandingNum, zero, outstandingCommits);
        createMaskedStoreScratchMemory(fb, fb.getLoc(), outstandingCommitsPtr,
                                       outstandingCommits, commitsType,
                                       commitCTAMask);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createCheckOutstandingCommitsCall(
    ImplicitLocOpBuilder &b, Value bufferMask, int thread,
    StringRef pendingAccessType, Value pred, MemType /*memType*/,
    CommitKind::Kind commitKind, Operation *insertPoint, Value effectCTAs,
    bool excludeSelf) {
  if (auxData.commits[commitKind].empty()) {
    return;
  }
  ValueType outstandingCommits = auxData.commits[commitKind].at(insertPoint);
  assert(thread < auxData.threadLayout.numBaseThreads &&
         "Commit-count tracking must operate on base threads");
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  auto commitsType = cast<RankedTensorType>(outstandingCommits.type);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  std::string message =
      "Accessing buffer with pending access. Pending access type: " +
      pendingAccessType.str();
  AssertInfo assertInfo{message, b.getI1Type()};
  SmallVector<Value> args = {bufferMask, pred, threadVal,
                             outstandingCommits.value, effectCTAs};
  std::string funcName = excludeSelf ? "check_outstanding_commits_excl_self"
                                     : "check_outstanding_commits";
  createCallToCachedFunction(
      b, funcName, args, assertInfo, {commitsType, (uint64_t)thread},
      [commitsType, excludeSelf](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value checkMask = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value threadVal = entryBlock->getArgument(2);
        Value outstandingCommitsPtr = entryBlock->getArgument(3);
        Value effectCTAs = entryBlock->getArgument(4);

        Value outstandingCommits = tti::createLoadScratchMemory(
            fb, fb.getLoc(), outstandingCommitsPtr, commitsType);
        checkMask = convertAndBroadcast(fb, checkMask, {1}, commitsType);
        Value ctaMask =
            createCTASetMask(fb, commitsType, /*dim=*/0, effectCTAs);
        checkMask = arith::AndIOp::create(fb, checkMask, ctaMask);
        Value zeroTensor =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, commitsType);
        Value selectedRows = arith::SelectOp::create(
            fb, checkMask, outstandingCommits, zeroTensor);
        if (excludeSelf) {
          Value threadColumnMask =
              createDimMask(fb, threadVal, commitsType, /*dim=*/2);
          selectedRows = arith::SelectOp::create(fb, threadColumnMask,
                                                 zeroTensor, selectedRows);
        }
        Value selectedEqZero = arith::CmpIOp::create(
            fb, arith::CmpIPredicate::eq, selectedRows, zeroTensor);
        Value allSelectedEqZero = reduceAll<arith::AndIOp>(fb, selectedEqZero);
        Value vTrue =
            arith::ConstantOp::create(fb, allSelectedEqZero.getType(),
                                      fb.getIntegerAttr(fb.getI1Type(), 1));
        Value predicatedSelectedEqZero =
            arith::SelectOp::create(fb, pred, allSelectedEqZero, vTrue);

        triton::ReturnOp::create(fb, predicatedSelectedEqZero);
      });
}

} // namespace mlir::triton::instrument
