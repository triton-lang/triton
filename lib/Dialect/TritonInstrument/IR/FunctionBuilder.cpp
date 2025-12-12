#include "triton/Dialect/TritonInstrument/IR/FunctionBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton::instrument {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tti = mlir::triton::instrument;

namespace {

namespace BarrierBits {
constexpr unsigned phaseBit = 0;
constexpr unsigned initCountLsb = 1;
constexpr unsigned currentCountLsb = 9;
constexpr unsigned countBitWidth = 8;
constexpr unsigned countMask = (1u << countBitWidth) - 1;
} // namespace BarrierBits

namespace WaitingBits {
constexpr unsigned bitsPerThread = 2;
constexpr unsigned flagBit = 0;
constexpr unsigned phaseBit = 1;

constexpr uint32_t makeInterleavedMask(unsigned bit) {
  uint32_t mask = 0;
  for (unsigned i = 0; i < tti::NUM_THREADS; ++i)
    mask |= 1u << (bitsPerThread * i + bit);
  return mask;
}

constexpr uint32_t flagMask = makeInterleavedMask(flagBit);
constexpr uint32_t phaseMask = makeInterleavedMask(phaseBit);
} // namespace WaitingBits

// Information about the optional assert message and tensor type to check.
struct AssertInfo {
  StringRef message;
  Type type;
};

static uint64_t expandActiveMask(uint64_t activeMask) {
  uint64_t expanded = 0;
  for (unsigned i = 0; i < tti::NUM_THREADS; ++i) {
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

Value createBitwiseOrReduce(ImplicitLocOpBuilder &b, Value tensor, int axis) {
  OpBuilder::InsertionGuard guard(b);
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto reduceOp = triton::ReduceOp::create(b, std::vector<Value>{tensor}, axis);
  auto &region = reduceOp.getRegion();
  auto &block = region.emplaceBlock();
  block.addArguments({tensorType.getElementType(), tensorType.getElementType()},
                     {b.getLoc(), b.getLoc()});
  b.setInsertionPointToStart(&block);
  auto result =
      arith::OrIOp::create(b, block.getArgument(0), block.getArgument(1));
  triton::ReduceReturnOp::create(b, std::vector<Value>{result});
  return reduceOp->getResult(0);
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
  Block *entryBlock = func.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entryBlock);
  ImplicitLocOpBuilder fb(loc, bodyBuilder);
  buildBody(fb, entryBlock);
  return func;
}

// Create a call to a function with body given by `buildBody`.
// If the function does not exist, it will be created, otherwise the
// existing function will be used.
// If `assertInfo` is provided, the function should return a tensor of
// the given type and the result of the function will be asserted.
void createCallToCachedFunction(
    ImplicitLocOpBuilder &b, const std::string &name, ArrayRef<Value> args,
    std::optional<AssertInfo> assertInfo, ManglingArgs specializationArgs,
    std::function<void(ImplicitLocOpBuilder &b, Block *entryBlock)> buildBody) {
  ModuleOp module = b.getInsertionPoint()->getParentOfType<ModuleOp>();
  int numWarps = ttg::lookupNumWarps(b.getInsertionPoint()->getParentRegion());
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
    StringRef message = b.getStringAttr(assertInfo->message);
    tti::ExperimentalAssertInThreadOp::create(b, result, message, false);
  }
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

uint32_t getMemDescLength(Value buf) {
  auto memDescType = cast<ttg::MemDescType>(buf.getType());
  if (isa<ttg::SharedEncodingTrait>(memDescType.getEncoding())) {
    unsigned elSize = memDescType.getElementType().getIntOrFloatBitWidth() / 8;
    return static_cast<uint32_t>(product(memDescType.getShape()) * elSize);
  }
  if (isa<ttng::TensorMemorySpaceAttr>(memDescType.getMemorySpace())) {
    return ttng::getTmemAllocSizes(memDescType).numCols;
  }
  llvm_unreachable("Unsupported memory space for memdesc");
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

Value convertAndBroadcast(ImplicitLocOpBuilder &b, Value tensor, int dim,
                          RankedTensorType dstType) {
  auto loc = b.getLoc();
  ArrayRef<int64_t> shape = dstType.getShape();
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto encoding = cast<ttg::BlockedEncodingAttr>(dstType.getEncoding());
  RankedTensorType resultType =
      RankedTensorType::get(shape, tensorType.getElementType(), encoding);
  auto slicedEncoding =
      ttg::SliceEncodingAttr::get(b.getContext(), dim, encoding);
  tensor = ttg::ConvertLayoutOp::create(
      b, tensorType.cloneWithEncoding(slicedEncoding), tensor);
  tensor = tti::expandOuterSlicedDim(b, loc, tensor);
  tensor = triton::BroadcastOp::create(b, resultType, tensor);
  return tensor;
}

Value createConvertLayout(ImplicitLocOpBuilder &b, Value tensor,
                          Attribute encoding) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto dstType = tensorType.cloneWithEncoding(encoding);
  return ttg::ConvertLayoutOp::create(b, dstType, tensor);
}

Value expandAliases(ImplicitLocOpBuilder &b, Value bufferMask,
                    Value aliasMatrix, RankedTensorType aliasMatrixType) {
  assert(aliasMatrixType.getRank() == 2 &&
         "Alias matrix expected to be rank-2");
  auto bufferMaskType = cast<RankedTensorType>(bufferMask.getType());
  Value bufMaskMatrix =
      convertAndBroadcast(b, bufferMask, /*dim=*/1, aliasMatrixType);
  Value aliasingMask = arith::AndIOp::create(b, aliasMatrix, bufMaskMatrix);
  Value aliasVector = createBitwiseOrReduce(b, aliasingMask, /*axis=*/0);
  return createConvertLayout(b, aliasVector, bufferMaskType.getEncoding());
}

Value createOneHot(ImplicitLocOpBuilder &b, int size, int index,
                   Attribute encoding) {
  auto loc = b.getLoc();
  auto type = RankedTensorType::get({size}, b.getI32Type(), encoding);
  Value arange =
      triton::MakeRangeOp::create(b, type, /*start=*/0, /*end=*/size);
  Value indexTensor =
      tti::createConstIntTensor(b, loc, index, type, /*isSigned=*/false);
  return arith::CmpIOp::create(b, arith::CmpIPredicate::eq, arange,
                               indexTensor);
}

Value createColumnMask(ImplicitLocOpBuilder &b, int column,
                       RankedTensorType tensorType) {
  auto encoding = cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding());
  auto columnEncoding = tti::getSingleDimSliceEncoding(encoding, /*dim=*/1);
  Value oneHot =
      createOneHot(b, tensorType.getShape()[1], column, columnEncoding);
  return convertAndBroadcast(b, oneHot, /*dim=*/0, tensorType);
}

Value createMultiColumnMask(ImplicitLocOpBuilder &b, uint64_t columnMask,
                            RankedTensorType tensorType) {
  auto loc = b.getLoc();
  auto i1TensorType =
      cast<RankedTensorType>(tensorType.cloneWith(std::nullopt, b.getI1Type()));
  Value maskTensor = tti::createConstIntTensor(b, loc, 0, i1TensorType);
  for (int i = 0; i < 64; ++i) {
    if (columnMask & (1ULL << i)) {
      Value columnMaskTensor = createColumnMask(b, i, tensorType);
      maskTensor = arith::OrIOp::create(b, maskTensor, columnMaskTensor);
    }
  }
  return maskTensor;
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
                             RankedTensorType tensorType) {
  auto loc = b.getLoc();
  auto encoding = cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding());
  auto sliceEncoding = tti::getSingleDimSliceEncoding(encoding, /*dim=*/1);
  int columns = tensorType.getShape()[1];

  RankedTensorType rangeType =
      RankedTensorType::get({columns}, b.getI32Type(), sliceEncoding);
  Value range = triton::MakeRangeOp::create(b, rangeType, 0, columns);

  auto elemType = cast<IntegerType>(tensorType.getElementType());
  RankedTensorType rangeElemType =
      RankedTensorType::get({columns}, elemType, sliceEncoding);
  Value rangeElem = range;
  if (elemType.getWidth() != 32)
    rangeElem = arith::ExtUIOp::create(b, rangeElemType, range);

  Value indices = convertAndBroadcast(b, rangeElem, /*dim=*/0, tensorType);

  Value threadMaskElem = adjustIntegerWidth(b, threadMask, elemType);
  Value maskTensor = triton::SplatOp::create(b, tensorType, threadMaskElem);

  Value shifted = arith::ShRUIOp::create(b, maskTensor, indices);
  Value one = tti::createConstIntTensor(b, loc, 1, tensorType);
  Value bits = arith::AndIOp::create(b, shifted, one);
  Value zero = tti::createConstIntTensor(b, loc, 0, tensorType);
  return arith::CmpIOp::create(b, arith::CmpIPredicate::ne, bits, zero);
}

Value createColumnMask(ImplicitLocOpBuilder &b, Value column,
                       RankedTensorType tensorType) {
  auto loc = b.getLoc();
  auto encoding = cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding());
  auto sliceEncoding = tti::getSingleDimSliceEncoding(encoding, /*dim=*/1);
  auto colType = RankedTensorType::get({tensorType.getShape()[1]},
                                       b.getI32Type(), sliceEncoding);
  Value range = triton::MakeRangeOp::create(b, colType, /*start=*/0,
                                            /*end=*/tensorType.getShape()[1]);
  Value columnTensor = triton::SplatOp::create(b, colType, column);
  Value mask1D =
      arith::CmpIOp::create(b, arith::CmpIPredicate::eq, range, columnTensor);
  return convertAndBroadcast(b, mask1D, /*dim=*/0, tensorType);
}

} // namespace

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
      [barriersType, waitingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
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
        tti::createStoreScratchMemory(fb, fb.getLoc(), waitingPtr, newWaiting,
                                      waitingType);

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
      [barriersType, waitingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
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

        tti::createStoreScratchMemory(fb, fb.getLoc(), waitingPtr, newWaiting,
                                      waitingType);
        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createCheckAllActiveWaitingCall(ImplicitLocOpBuilder &b,
                                                      int activeMask,
                                                      Value pred,
                                                      Operation *insertPoint) {
  if (auxData.waiting.empty() || auxData.barrierStates.empty()) {
    return;
  }
  if (!pred) {
    pred = arith::ConstantIntOp::create(b, 1, 1);
  }
  int64_t expandedActiveMask = expandActiveMask(activeMask);
  Value expandedActiveMaskVal =
      arith::ConstantIntOp::create(b, expandedActiveMask, 32);
  Value waitingVal = auxData.waiting.at(insertPoint).value;
  auto waitingType =
      cast<RankedTensorType>(auxData.waiting.at(insertPoint).type);
  Value barrierStatesVal = auxData.barrierStates.at(insertPoint).value;
  auto barrierStatesType =
      cast<RankedTensorType>(auxData.barrierStates.at(insertPoint).type);
  SmallVector<Value> args = {expandedActiveMaskVal, pred, waitingVal,
                             barrierStatesVal};
  AssertInfo assertInfo{
      "Deadlock detected: all active threads are waiting on mbarriers",
      b.getI1Type()};
  createCallToCachedFunction(
      b, "check_all_active_waiting", args, assertInfo,
      {waitingType, barrierStatesType},
      [waitingType, barrierStatesType](ImplicitLocOpBuilder &fb,
                                       Block *entryBlock) {
        Value expandedActiveMaskVal = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);

        Value waitingPtr = entryBlock->getArgument(2);
        Value barrierStatesPtr = entryBlock->getArgument(3);

        Value waiting = tti::createLoadScratchMemory(fb, fb.getLoc(),
                                                     waitingPtr, waitingType);
        Value barrierStates = tti::createLoadScratchMemory(
            fb, fb.getLoc(), barrierStatesPtr, barrierStatesType);

        Value flagMaskTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), WaitingBits::flagMask, waitingType);
        Value phaseMaskTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), WaitingBits::phaseMask, waitingType);

        Value flags = arith::AndIOp::create(fb, waiting, flagMaskTensor);
        Value phases = arith::AndIOp::create(fb, waiting, phaseMaskTensor);
        Value shiftOneTensor =
            tti::createConstIntTensor(fb, fb.getLoc(), 1, waitingType);
        Value phasesAligned =
            arith::ShRUIOp::create(fb, phases, shiftOneTensor);

        Value phasesComplement =
            arith::XOrIOp::create(fb, phasesAligned, flagMaskTensor);
        Value waitingPhase0 =
            arith::AndIOp::create(fb, flags, phasesComplement);
        Value waitingPhase1 = arith::AndIOp::create(fb, flags, phasesAligned);

        Value oneState =
            tti::createConstIntTensor(fb, fb.getLoc(), 1, barrierStatesType);
        Value barrierPhase = arith::AndIOp::create(fb, barrierStates, oneState);
        Value phaseIsOne = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                                 barrierPhase, oneState);

        Value effectiveWaiting = arith::SelectOp::create(
            fb, phaseIsOne, waitingPhase1, waitingPhase0);
        Value waitingOr =
            createBitwiseOrReduce(fb, effectiveWaiting, /*axis=*/0);

        auto waitingOrTy = waitingOr.getType();
        Value waitingMasked =
            arith::AndIOp::create(fb, waitingOr, expandedActiveMaskVal);
        Value eq = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                         waitingMasked, expandedActiveMaskVal);

        Value vTrue = arith::ConstantOp::create(
            fb, eq.getType(), fb.getIntegerAttr(fb.getI1Type(), 1));
        Value ok = arith::XOrIOp::create(fb, eq, vTrue);
        Value predicatedOk = arith::SelectOp::create(fb, pred, ok, vTrue);
        triton::ReturnOp::create(fb, predicatedOk);
      });
}

void FunctionBuilder::createInitBarrierStateCall(ImplicitLocOpBuilder &b,
                                                 Value mbar, int count,
                                                 Operation *insertPoint) {

  if (auxData.barriers.empty() || auxData.barrierStates.empty()) {
    return;
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
  SmallVector<Value> args = {mbarOffset, lengthVal, countVal, barriersVal,
                             barrierStatesVal};
  createCallToCachedFunction(
      b, "init_barrier_state", args,
      /*assertInfo=*/std::nullopt, {barriersType, barrierStatesType},
      [barriersType, barrierStatesType](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value count = entryBlock->getArgument(2);

        Value barriers = entryBlock->getArgument(3);
        Value statesPtr = entryBlock->getArgument(4);

        Value states = tti::createLoadScratchMemory(fb, fb.getLoc(), statesPtr,
                                                    barrierStatesType);
        Value descriptor = createBufferDescriptor(fb, mbarOffset, lengthVal);
        Value mask = createCmpIntTensorScalar(fb, barriers, descriptor);

        Value countMask =
            arith::ConstantIntOp::create(fb, BarrierBits::countMask, 32);
        Value maskedCount = arith::AndIOp::create(fb, count, countMask);
        Value countTensor =
            triton::SplatOp::create(fb, barrierStatesType, maskedCount);

        Value shiftOneTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::initCountLsb, barrierStatesType);
        Value shiftNineTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::currentCountLsb, barrierStatesType);

        Value initField =
            arith::ShLIOp::create(fb, countTensor, shiftOneTensor);
        Value currentField =
            arith::ShLIOp::create(fb, countTensor, shiftNineTensor);
        Value newState = arith::OrIOp::create(fb, initField, currentField);

        Value updated = arith::SelectOp::create(fb, mask, newState, states);
        tti::createStoreScratchMemory(fb, fb.getLoc(), statesPtr, updated,
                                      barrierStatesType);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createVerifyBarrierArriveCall(ImplicitLocOpBuilder &b,
                                                    Value mbar, int count,
                                                    Value pred,
                                                    Operation *insertPoint) {

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
  AssertInfo assertInfo{
      "Barrier arrive underflow: current count would become negative",
      barrierStatesType.cloneWith(std::nullopt, b.getI1Type())};
  createCallToCachedFunction(
      b, "verify_barrier_arrive", args, assertInfo,
      {barriersType, barrierStatesType},
      [barriersType, barrierStatesType](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
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

        Value zero32 =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, barrierStatesType);
        Value maskFF = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::countMask, barrierStatesType);
        Value shiftNineTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::currentCountLsb, barrierStatesType);

        Value currentCount =
            arith::ShRUIOp::create(fb, states, shiftNineTensor);
        currentCount = arith::AndIOp::create(fb, currentCount, maskFF);

        Value countMask =
            arith::ConstantIntOp::create(fb, BarrierBits::countMask, 32);
        Value maskedCount = arith::AndIOp::create(fb, count, countMask);
        Value arriveCount =
            triton::SplatOp::create(fb, barrierStatesType, maskedCount);

        Value newCurrent = arith::SubIOp::create(fb, currentCount, arriveCount);
        Value newCurrentMasked =
            arith::SelectOp::create(fb, mask, newCurrent, zero32);
        Value nonNegative = arith::CmpIOp::create(fb, arith::CmpIPredicate::sge,
                                                  newCurrentMasked, zero32);
        Value vTrue = tti::createConstIntTensor(
            fb, fb.getLoc(), 1, cast<RankedTensorType>(nonNegative.getType()));
        Value predicatedNonNegative =
            arith::SelectOp::create(fb, pred, nonNegative, vTrue);

        triton::ReturnOp::create(fb, predicatedNonNegative);
      });
}

void FunctionBuilder::createUpdateBarrierStateCall(ImplicitLocOpBuilder &b,
                                                   Value mbar, int count,
                                                   Value pred,
                                                   Operation *insertPoint) {

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
      b, "update_barrier_state", args,
      /*assertInfo=*/std::nullopt, {barriersType, barrierStatesType},
      [barriersType, barrierStatesType](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value count = entryBlock->getArgument(2);
        Value pred = entryBlock->getArgument(3);

        Value barriers = entryBlock->getArgument(4);
        Value statesPtr = entryBlock->getArgument(5);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value states = tti::createLoadScratchMemory(fb, fb.getLoc(), statesPtr,
                                                    barrierStatesType);
        Value descriptor = createBufferDescriptor(fb, mbarOffset, lengthVal);
        Value mask = createCmpIntTensorScalar(fb, barriers, descriptor);

        Value zero32 =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, barrierStatesType);
        Value one32 =
            tti::createConstIntTensor(fb, fb.getLoc(), 1, barrierStatesType);
        Value maskFF = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::countMask, barrierStatesType);
        Value shiftOneTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::initCountLsb, barrierStatesType);
        Value shiftNineTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::currentCountLsb, barrierStatesType);

        Value phase = arith::AndIOp::create(fb, states, one32);
        Value initCount = arith::ShRUIOp::create(fb, states, shiftOneTensor);
        initCount = arith::AndIOp::create(fb, initCount, maskFF);
        Value currentCount =
            arith::ShRUIOp::create(fb, states, shiftNineTensor);
        currentCount = arith::AndIOp::create(fb, currentCount, maskFF);

        Value countMask =
            arith::ConstantIntOp::create(fb, BarrierBits::countMask, 32);
        Value maskedCount = arith::AndIOp::create(fb, count, countMask);
        Value arriveCount =
            triton::SplatOp::create(fb, barrierStatesType, maskedCount);

        Value newCurrent = arith::SubIOp::create(fb, currentCount, arriveCount);
        Value newCurrentMasked =
            arith::SelectOp::create(fb, mask, newCurrent, currentCount);

        Value zeroCond = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                               newCurrentMasked, zero32);
        zeroCond = arith::AndIOp::create(fb, zeroCond, mask);
        Value zeroCondI32 =
            arith::ExtUIOp::create(fb, barrierStatesType, zeroCond);
        Value newPhase = arith::XOrIOp::create(fb, phase, zeroCondI32);
        Value newCurrentValue =
            arith::SelectOp::create(fb, zeroCond, initCount, newCurrentMasked);

        Value initField = arith::ShLIOp::create(fb, initCount, shiftOneTensor);
        Value currentField =
            arith::ShLIOp::create(fb, newCurrentValue, shiftNineTensor);
        Value newState = arith::OrIOp::create(fb, newPhase, initField);
        newState = arith::OrIOp::create(fb, newState, currentField);

        Value updated = arith::SelectOp::create(fb, mask, newState, states);
        tti::createStoreScratchMemory(fb, fb.getLoc(), statesPtr, updated,
                                      barrierStatesType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createSetWriteVisibilityCall(ImplicitLocOpBuilder &b,
                                                   Value buf, uint32_t length,
                                                   uint64_t threadMask,
                                                   Value pred, MemType memType,
                                                   Operation *insertPoint) {

  if (auxData.buffers[(int)memType].empty() ||
      auxData.writeVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value threadMaskVal = arith::ConstantIntOp::create(b, threadMask, 64);
  Value buffersVal = auxData.buffers[(int)memType].at(insertPoint).value;
  auto buffersType = cast<RankedTensorType>(
      auxData.buffers[(int)memType].at(insertPoint).type);
  Value writeVisibilityVal =
      auxData.writeVisibility[(int)memType].at(insertPoint).value;
  auto writeVisibilityType = cast<RankedTensorType>(
      auxData.writeVisibility[(int)memType].at(insertPoint).type);
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {bufOffset,     lengthVal,  pred,
                             threadMaskVal, buffersVal, writeVisibilityVal};
  createCallToCachedFunction(
      b, "set_write_visibility", args,
      /*assertInfo=*/std::nullopt,
      {buffersType, writeVisibilityType, (int)memType},
      [buffersType, writeVisibilityType](ImplicitLocOpBuilder &fb,
                                         Block *entryBlock) {
        Value bufOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadMaskVal = entryBlock->getArgument(3);
        Value buffers = entryBlock->getArgument(4);
        Value writeVisibilityPtr = entryBlock->getArgument(5);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value writeVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
        Value descriptor = createBufferDescriptor(fb, bufOffset, lengthVal);
        Value buffersEqBuf = createCmpIntTensorScalar(fb, buffers, descriptor);
        auto elemType = cast<IntegerType>(writeVisibilityType.getElementType());
        Value threadMaskElem = adjustIntegerWidth(fb, threadMaskVal, elemType);
        Value threadMaskTensor =
            triton::SplatOp::create(fb, writeVisibilityType, threadMaskElem);
        Value newVisibility = arith::SelectOp::create(
            fb, buffersEqBuf, threadMaskTensor, writeVisibility);
        tti::createStoreScratchMemory(fb, fb.getLoc(), writeVisibilityPtr,
                                      newVisibility, writeVisibilityType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createSetReadVisibilityCall(ImplicitLocOpBuilder &b,
                                                  Value buf, uint32_t length,
                                                  uint64_t threadMask,
                                                  Value pred, MemType memType,
                                                  Operation *insertPoint) {

  if (auxData.buffers[(int)memType].empty() ||
      auxData.readVisibility[(int)memType].empty() ||
      auxData.aliasMatrices[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value threadMaskVal = arith::ConstantIntOp::create(b, threadMask, 64);
  Value buffersVal = auxData.buffers[(int)memType].at(insertPoint).value;
  auto buffersType = cast<RankedTensorType>(
      auxData.buffers[(int)memType].at(insertPoint).type);
  Value readVisibilityVal =
      auxData.readVisibility[(int)memType].at(insertPoint).value;
  auto readVisibilityType = cast<RankedTensorType>(
      auxData.readVisibility[(int)memType].at(insertPoint).type);
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {bufOffset,     lengthVal,  pred,
                             threadMaskVal, buffersVal, readVisibilityVal};
  createCallToCachedFunction(
      b, "set_read_visibility", args,
      /*assertInfo=*/std::nullopt,
      {buffersType, readVisibilityType, (int)memType},
      [buffersType, readVisibilityType](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
        Value bufOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadMaskVal = entryBlock->getArgument(3);
        Value buffers = entryBlock->getArgument(4);
        Value readVisibilityPtr = entryBlock->getArgument(5);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value readVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
        Value descriptor = createBufferDescriptor(fb, bufOffset, lengthVal);
        Value buffersEqBuf = createCmpIntTensorScalar(fb, buffers, descriptor);
        buffersEqBuf = convertAndBroadcast(fb, buffersEqBuf, /*dim=*/1,
                                           readVisibilityType);
        auto elemType = cast<IntegerType>(readVisibilityType.getElementType());
        Value threadMaskElem = adjustIntegerWidth(fb, threadMaskVal, elemType);
        Value threadBit =
            triton::SplatOp::create(fb, readVisibilityType, threadMaskElem);
        Value threadColumnMask =
            createThreadColumnMask(fb, threadMaskVal, readVisibilityType);
        Value readVisibilityOrThreadBit =
            arith::OrIOp::create(fb, readVisibility, threadBit);
        Value bufAndThread =
            arith::AndIOp::create(fb, buffersEqBuf, threadColumnMask);
        Value newVisibility = arith::SelectOp::create(
            fb, bufAndThread, readVisibilityOrThreadBit, readVisibility);
        tti::createStoreScratchMemory(fb, fb.getLoc(), readVisibilityPtr,
                                      newVisibility, readVisibilityType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createClearWriteTrackingCall(ImplicitLocOpBuilder &b,
                                                   Value buf, uint32_t length,
                                                   Value pred, MemType memType,
                                                   Operation *insertPoint) {
  if (auxData.buffers[(int)memType].empty() ||
      auxData.writeTracking[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value buffersVal = auxData.buffers[(int)memType].at(insertPoint).value;
  auto buffersType = cast<RankedTensorType>(
      auxData.buffers[(int)memType].at(insertPoint).type);
  Value writeTrackingVal =
      auxData.writeTracking[(int)memType].at(insertPoint).value;
  auto writeTrackingType = cast<RankedTensorType>(
      auxData.writeTracking[(int)memType].at(insertPoint).type);
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {bufOffset, lengthVal, pred, buffersVal,
                             writeTrackingVal};
  createCallToCachedFunction(
      b, "clear_write_tracking", args,
      /*assertInfo=*/std::nullopt,
      {buffersType, writeTrackingType, (int)memType},
      [buffersType, writeTrackingType](ImplicitLocOpBuilder &fb,
                                       Block *entryBlock) {
        Value bufOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value buffers = entryBlock->getArgument(3);
        Value writeTrackingPtr = entryBlock->getArgument(4);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value writeTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeTrackingPtr, writeTrackingType);
        Value descriptor = createBufferDescriptor(fb, bufOffset, lengthVal);
        Value buffersEqBuf = createCmpIntTensorScalar(fb, buffers, descriptor);
        buffersEqBuf =
            convertAndBroadcast(fb, buffersEqBuf, /*dim=*/1, writeTrackingType);
        Value zero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, writeTrackingType);
        Value newTracking =
            arith::SelectOp::create(fb, buffersEqBuf, zero, writeTracking);
        tti::createStoreScratchMemory(fb, fb.getLoc(), writeTrackingPtr,
                                      newTracking, writeTrackingType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createClearReadVisibilityCall(ImplicitLocOpBuilder &b,
                                                    Value buf, uint32_t length,
                                                    Value pred, MemType memType,
                                                    Operation *insertPoint) {
  if (auxData.buffers[(int)memType].empty() ||
      auxData.readVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value buffersVal = auxData.buffers[(int)memType].at(insertPoint).value;
  auto buffersType = cast<RankedTensorType>(
      auxData.buffers[(int)memType].at(insertPoint).type);
  Value readVisibilityVal =
      auxData.readVisibility[(int)memType].at(insertPoint).value;
  auto readVisibilityType = cast<RankedTensorType>(
      auxData.readVisibility[(int)memType].at(insertPoint).type);
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {bufOffset, lengthVal, pred, buffersVal,
                             readVisibilityVal};
  createCallToCachedFunction(
      b, "clear_read_visibility", args,
      /*assertInfo=*/std::nullopt,
      {buffersType, readVisibilityType, (int)memType},
      [buffersType, readVisibilityType](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
        Value bufOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value buffers = entryBlock->getArgument(3);
        Value readVisibilityPtr = entryBlock->getArgument(4);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value readVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
        Value descriptor = createBufferDescriptor(fb, bufOffset, lengthVal);
        Value buffersEqBuf = createCmpIntTensorScalar(fb, buffers, descriptor);
        buffersEqBuf = convertAndBroadcast(fb, buffersEqBuf, /*dim=*/1,
                                           readVisibilityType);
        Value zero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, readVisibilityType);
        Value newVisibility =
            arith::SelectOp::create(fb, buffersEqBuf, zero, readVisibility);
        tti::createStoreScratchMemory(fb, fb.getLoc(), readVisibilityPtr,
                                      newVisibility, readVisibilityType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createClearReadTrackingCall(ImplicitLocOpBuilder &b,
                                                  Value buf, uint32_t length,
                                                  Value pred, MemType memType,
                                                  Operation *insertPoint) {

  if (auxData.buffers[(int)memType].empty() ||
      auxData.readTracking[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value buffersVal = auxData.buffers[(int)memType].at(insertPoint).value;
  auto buffersType = cast<RankedTensorType>(
      auxData.buffers[(int)memType].at(insertPoint).type);
  Value readTrackingVal =
      auxData.readTracking[(int)memType].at(insertPoint).value;
  auto readTrackingType = cast<RankedTensorType>(
      auxData.readTracking[(int)memType].at(insertPoint).type);
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {bufOffset, lengthVal, pred, buffersVal,
                             readTrackingVal};
  createCallToCachedFunction(
      b, "clear_read_tracking", args,
      /*assertInfo=*/std::nullopt,
      {buffersType, readTrackingType, (int)memType},
      [buffersType, readTrackingType](ImplicitLocOpBuilder &fb,
                                      Block *entryBlock) {
        Value bufOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value buffers = entryBlock->getArgument(3);
        Value readTrackingPtr = entryBlock->getArgument(4);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value readTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readTrackingPtr, readTrackingType);
        Value descriptor = createBufferDescriptor(fb, bufOffset, lengthVal);
        Value buffersEqBuf = createCmpIntTensorScalar(fb, buffers, descriptor);
        buffersEqBuf =
            convertAndBroadcast(fb, buffersEqBuf, /*dim=*/1, readTrackingType);
        Value zero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, readTrackingType);
        Value newTracking =
            arith::SelectOp::create(fb, buffersEqBuf, zero, readTracking);
        tti::createStoreScratchMemory(fb, fb.getLoc(), readTrackingPtr,
                                      newTracking, readTrackingType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createTrackVisibleWritesCall(ImplicitLocOpBuilder &b,
                                                   Value mbar, int thread,
                                                   Value pred, MemType memType,
                                                   Operation *insertPoint) {
  if (auxData.barriers.empty() ||
      auxData.writeVisibility[(int)memType].empty() ||
      auxData.writeTracking[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value writeVisibilityVal =
      auxData.writeVisibility[(int)memType].at(insertPoint).value;
  auto writeVisibilityType = cast<RankedTensorType>(
      auxData.writeVisibility[(int)memType].at(insertPoint).type);
  Value writeTrackingVal =
      auxData.writeTracking[(int)memType].at(insertPoint).value;
  auto writeTrackingType = cast<RankedTensorType>(
      auxData.writeTracking[(int)memType].at(insertPoint).type);
  uint32_t length = getMemDescLength(mbar);
  Value mbarOffset = tti::ExperimentalMemDescToI32Op::create(b, mbar);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {mbarOffset,      lengthVal,   pred,
                             threadVal,       barriersVal, writeVisibilityVal,
                             writeTrackingVal};
  createCallToCachedFunction(
      b, "track_visible_writes", args,
      /*assertInfo=*/std::nullopt,
      {barriersType, writeVisibilityType, writeTrackingType, (int)memType},
      [barriersType, writeVisibilityType,
       writeTrackingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadVal = entryBlock->getArgument(3);
        Value barriers = entryBlock->getArgument(4);
        Value writeVisibilityPtr = entryBlock->getArgument(5);
        Value writeTrackingPtr = entryBlock->getArgument(6);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value writeVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
        Value writeTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeTrackingPtr, writeTrackingType);
        Value descriptor = createBufferDescriptor(fb, mbarOffset, lengthVal);
        Value barriersEqBar =
            createCmpIntTensorScalar(fb, barriers, descriptor);
        barriersEqBar = convertAndBroadcast(fb, barriersEqBar, /*dim=*/0,
                                            writeTrackingType);
        Value threadI64 =
            arith::ExtUIOp::create(fb, fb.getI64Type(), threadVal);
        Value one64 = arith::ConstantIntOp::create(fb, 1, 64);
        Value threadBitScalar = arith::ShLIOp::create(fb, one64, threadI64);
        Value threadBit =
            triton::SplatOp::create(fb, writeVisibilityType, threadBitScalar);
        Value visibleWrites =
            arith::AndIOp::create(fb, writeVisibility, threadBit);
        visibleWrites = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                              visibleWrites, threadBit);
        visibleWrites = convertAndBroadcast(fb, visibleWrites, /*dim=*/1,
                                            writeTrackingType);
        Value barAndVisible =
            arith::AndIOp::create(fb, barriersEqBar, visibleWrites);
        Value writeTrackingOne =
            tti::createConstIntTensor(fb, fb.getLoc(), 1, writeTrackingType);
        Value newTracking = arith::SelectOp::create(
            fb, barAndVisible, writeTrackingOne, writeTracking);
        tti::createStoreScratchMemory(fb, fb.getLoc(), writeTrackingPtr,
                                      newTracking, writeTrackingType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createTrackVisibleReadsCall(ImplicitLocOpBuilder &b,
                                                  Value mbar, int thread,
                                                  Value pred, MemType memType,
                                                  Operation *insertPoint) {

  if (auxData.barriers.empty() ||
      auxData.readVisibility[(int)memType].empty() ||
      auxData.readTracking[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value readVisibilityVal =
      auxData.readVisibility[(int)memType].at(insertPoint).value;
  auto readVisibilityType = cast<RankedTensorType>(
      auxData.readVisibility[(int)memType].at(insertPoint).type);
  Value readTrackingVal =
      auxData.readTracking[(int)memType].at(insertPoint).value;
  auto readTrackingType = cast<RankedTensorType>(
      auxData.readTracking[(int)memType].at(insertPoint).type);
  uint32_t length = getMemDescLength(mbar);
  Value mbarOffset = tti::ExperimentalMemDescToI32Op::create(b, mbar);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {mbarOffset,     lengthVal,   pred,
                             threadVal,      barriersVal, readVisibilityVal,
                             readTrackingVal};
  createCallToCachedFunction(
      b, "track_visible_reads", args,
      /*assertInfo=*/std::nullopt,
      {barriersType, readVisibilityType, readTrackingType, (int)memType},
      [barriersType, readVisibilityType,
       readTrackingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadVal = entryBlock->getArgument(3);
        Value barriers = entryBlock->getArgument(4);
        Value readVisibilityPtr = entryBlock->getArgument(5);
        Value readTrackingPtr = entryBlock->getArgument(6);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value readVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
        Value readTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readTrackingPtr, readTrackingType);
        Value descriptor = createBufferDescriptor(fb, mbarOffset, lengthVal);
        Value barriersEqBar =
            createCmpIntTensorScalar(fb, barriers, descriptor);
        barriersEqBar =
            convertAndBroadcast(fb, barriersEqBar, /*dim=*/0, readTrackingType);
        Value threadColumnMask =
            createColumnMask(fb, threadVal, readVisibilityType);
        Value readVisibilityZero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, readVisibilityType);
        Value visibleReads = arith::SelectOp::create(
            fb, threadColumnMask, readVisibility, readVisibilityZero);
        visibleReads = createBitwiseOrReduce(fb, visibleReads, /*axis=*/1);
        visibleReads =
            convertAndBroadcast(fb, visibleReads, /*dim=*/1, readTrackingType);
        Value readTrackingOrVisible =
            arith::OrIOp::create(fb, readTracking, visibleReads);
        Value newTracking = arith::SelectOp::create(
            fb, barriersEqBar, readTrackingOrVisible, readTracking);
        tti::createStoreScratchMemory(fb, fb.getLoc(), readTrackingPtr,
                                      newTracking, readTrackingType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createTransferVisibleWritesCall(
    ImplicitLocOpBuilder &b, Value mbar, uint64_t threadMask, Value pred,
    MemType memType, Operation *insertPoint) {

  if (auxData.barriers.empty() ||
      auxData.writeVisibility[(int)memType].empty() ||
      auxData.writeTracking[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value threadMaskVal = arith::ConstantIntOp::create(b, threadMask, 64);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value writeVisibilityVal =
      auxData.writeVisibility[(int)memType].at(insertPoint).value;
  auto writeVisibilityType = cast<RankedTensorType>(
      auxData.writeVisibility[(int)memType].at(insertPoint).type);
  Value writeTrackingVal =
      auxData.writeTracking[(int)memType].at(insertPoint).value;
  auto writeTrackingType = cast<RankedTensorType>(
      auxData.writeTracking[(int)memType].at(insertPoint).type);
  uint32_t length = getMemDescLength(mbar);
  Value mbarOffset = tti::ExperimentalMemDescToI32Op::create(b, mbar);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {mbarOffset,      lengthVal,   pred,
                             threadMaskVal,   barriersVal, writeVisibilityVal,
                             writeTrackingVal};
  createCallToCachedFunction(
      b, "transfer_visible_writes", args,
      /*assertInfo=*/std::nullopt,
      {barriersType, writeVisibilityType, writeTrackingType, (int)memType},
      [barriersType, writeVisibilityType,
       writeTrackingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadMaskVal = entryBlock->getArgument(3);
        Value barriers = entryBlock->getArgument(4);
        Value writeVisibilityPtr = entryBlock->getArgument(5);
        Value writeTrackingPtr = entryBlock->getArgument(6);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value writeVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
        Value writeTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeTrackingPtr, writeTrackingType);
        Value descriptor = createBufferDescriptor(fb, mbarOffset, lengthVal);
        Value barriersEqBar =
            createCmpIntTensorScalar(fb, barriers, descriptor);
        barriersEqBar = convertAndBroadcast(fb, barriersEqBar, /*dim=*/0,
                                            writeTrackingType);
        Value zeroTracking =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, writeTrackingType);
        Value trackingBuffers = arith::SelectOp::create(
            fb, barriersEqBar, writeTracking, zeroTracking);
        trackingBuffers =
            createBitwiseOrReduce(fb, trackingBuffers, /*axis=*/1);
        trackingBuffers = createConvertLayout(
            fb, trackingBuffers, writeVisibilityType.getEncoding());
        auto trackingBuffersType =
            cast<RankedTensorType>(trackingBuffers.getType());
        Value trackingBuffersOne =
            tti::createConstIntTensor(fb, fb.getLoc(), 1, trackingBuffersType);
        trackingBuffers = arith::CmpIOp::create(
            fb, arith::CmpIPredicate::eq, trackingBuffers, trackingBuffersOne);
        auto elemType = cast<IntegerType>(writeVisibilityType.getElementType());
        Value threadMaskElem = adjustIntegerWidth(fb, threadMaskVal, elemType);
        Value threadMaskTensor =
            triton::SplatOp::create(fb, writeVisibilityType, threadMaskElem);
        Value zeroVisibility =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, writeVisibilityType);
        Value trackingThreadBit = arith::SelectOp::create(
            fb, trackingBuffers, threadMaskTensor, zeroVisibility);
        Value newVisibility =
            arith::OrIOp::create(fb, writeVisibility, trackingThreadBit);
        tti::createStoreScratchMemory(fb, fb.getLoc(), writeVisibilityPtr,
                                      newVisibility, writeVisibilityType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createTransferVisibleReadsCall(
    ImplicitLocOpBuilder &b, Value mbar, uint64_t threadMask, Value pred,
    MemType memType, Operation *insertPoint) {

  if (auxData.barriers.empty() ||
      auxData.readVisibility[(int)memType].empty() ||
      auxData.readTracking[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value threadMaskVal = arith::ConstantIntOp::create(b, threadMask, 64);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value readVisibilityVal =
      auxData.readVisibility[(int)memType].at(insertPoint).value;
  auto readVisibilityType = cast<RankedTensorType>(
      auxData.readVisibility[(int)memType].at(insertPoint).type);
  Value readTrackingVal =
      auxData.readTracking[(int)memType].at(insertPoint).value;
  auto readTrackingType = cast<RankedTensorType>(
      auxData.readTracking[(int)memType].at(insertPoint).type);
  uint32_t length = getMemDescLength(mbar);
  Value mbarOffset = tti::ExperimentalMemDescToI32Op::create(b, mbar);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {mbarOffset,     lengthVal,   pred,
                             threadMaskVal,  barriersVal, readVisibilityVal,
                             readTrackingVal};
  createCallToCachedFunction(
      b, "transfer_visible_reads", args,
      /*assertInfo=*/std::nullopt,
      {barriersType, readVisibilityType, readTrackingType, (int)memType},
      [barriersType, readVisibilityType,
       readTrackingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadMaskVal = entryBlock->getArgument(3);
        Value barriers = entryBlock->getArgument(4);
        Value readVisibilityPtr = entryBlock->getArgument(5);
        Value readTrackingPtr = entryBlock->getArgument(6);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value readVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
        Value readTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readTrackingPtr, readTrackingType);
        Value descriptor = createBufferDescriptor(fb, mbarOffset, lengthVal);
        Value barriersEqBar =
            createCmpIntTensorScalar(fb, barriers, descriptor);
        barriersEqBar =
            convertAndBroadcast(fb, barriersEqBar, /*dim=*/0, readTrackingType);
        Value readTrackingZero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, readTrackingType);
        Value trackingBar = arith::SelectOp::create(
            fb, barriersEqBar, readTracking, readTrackingZero);
        trackingBar = createBitwiseOrReduce(fb, trackingBar, /*axis=*/1);
        trackingBar =
            convertAndBroadcast(fb, trackingBar, /*dim=*/1, readVisibilityType);
        Value readVisibilityOrTracking =
            arith::OrIOp::create(fb, readVisibility, trackingBar);
        Value threadColumnMask =
            createThreadColumnMask(fb, threadMaskVal, readVisibilityType);
        Value newVisibility = arith::SelectOp::create(
            fb, threadColumnMask, readVisibilityOrTracking, readVisibility);
        tti::createStoreScratchMemory(fb, fb.getLoc(), readVisibilityPtr,
                                      newVisibility, readVisibilityType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createVerifyWriteVisibilityCall(
    ImplicitLocOpBuilder &b, Value buf, uint32_t length, int thread,
    StringRef operandName, Value pred, MemType memType,
    Operation *insertPoint) {
  if (auxData.buffers[(int)memType].empty() ||
      auxData.writeVisibility[(int)memType].empty() ||
      auxData.aliasMatrices[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value buffersVal = auxData.buffers[(int)memType].at(insertPoint).value;
  auto buffersType = cast<RankedTensorType>(
      auxData.buffers[(int)memType].at(insertPoint).type);
  Value writeVisibilityVal =
      auxData.writeVisibility[(int)memType].at(insertPoint).value;
  auto writeVisibilityType = cast<RankedTensorType>(
      auxData.writeVisibility[(int)memType].at(insertPoint).type);
  Value aliasMatrixVal =
      auxData.aliasMatrices[(int)memType].at(insertPoint).value;
  auto aliasMatrixType = cast<RankedTensorType>(
      auxData.aliasMatrices[(int)memType].at(insertPoint).type);
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {bufOffset,     lengthVal,  pred,
                             threadVal,     buffersVal, writeVisibilityVal,
                             aliasMatrixVal};
  std::string message = "Buffer being accessed has outstanding writes.";
  if (!operandName.empty())
    message += " Operand: " + operandName.str();
  AssertInfo assertInfo{message,
                        buffersType.cloneWith(std::nullopt, b.getI1Type())};
  createCallToCachedFunction(
      b, "verify_write_visibility", args, assertInfo,
      {buffersType, writeVisibilityType, aliasMatrixType, (int)memType},
      [buffersType, writeVisibilityType,
       aliasMatrixType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value bufOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadVal = entryBlock->getArgument(3);
        Value buffers = entryBlock->getArgument(4);
        Value writeVisibilityPtr = entryBlock->getArgument(5);
        Value aliasMatrix = entryBlock->getArgument(6);

        Value writeVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
        Value descriptor = createBufferDescriptor(fb, bufOffset, lengthVal);
        Value buffersEqBuf = createCmpIntTensorScalar(fb, buffers, descriptor);
        buffersEqBuf =
            expandAliases(fb, buffersEqBuf, aliasMatrix, aliasMatrixType);
        Value writeVisibilityZero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, writeVisibilityType);
        Value bufVisibility = arith::SelectOp::create(
            fb, buffersEqBuf, writeVisibility, writeVisibilityZero);
        Value noOneIsWriting = arith::CmpIOp::create(
            fb, arith::CmpIPredicate::eq, bufVisibility, writeVisibilityZero);
        Value threadI64 =
            arith::ExtUIOp::create(fb, fb.getI64Type(), threadVal);
        Value threadMask =
            triton::SplatOp::create(fb, writeVisibilityType, threadI64);
        Value buffersEqBufExt =
            arith::ExtUIOp::create(fb, writeVisibilityType, buffersEqBuf);
        Value bufferThreadBit =
            arith::ShLIOp::create(fb, buffersEqBufExt, threadMask);
        Value bufferHasVisibility =
            arith::AndIOp::create(fb, bufVisibility, bufferThreadBit);
        bufferHasVisibility = arith::CmpIOp::create(
            fb, arith::CmpIPredicate::eq, bufferHasVisibility, bufferThreadBit);
        Value writeVisible =
            arith::OrIOp::create(fb, noOneIsWriting, bufferHasVisibility);

        Value vTrue = tti::createConstIntTensor(
            fb, fb.getLoc(), 1, cast<RankedTensorType>(writeVisible.getType()));
        Value predicatedWriteVisible =
            arith::SelectOp::create(fb, pred, writeVisible, vTrue);
        triton::ReturnOp::create(fb, predicatedWriteVisible);
      });
}

void FunctionBuilder::createVerifyReadVisibilityCall(
    ImplicitLocOpBuilder &b, Value buf, uint32_t length, int thread,
    StringRef operandName, Value pred, MemType memType,
    Operation *insertPoint) {
  if (auxData.buffers[(int)memType].empty() ||
      auxData.readVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value buffersVal = auxData.buffers[(int)memType].at(insertPoint).value;
  auto buffersType = cast<RankedTensorType>(
      auxData.buffers[(int)memType].at(insertPoint).type);
  Value readVisibilityVal =
      auxData.readVisibility[(int)memType].at(insertPoint).value;
  auto readVisibilityType = cast<RankedTensorType>(
      auxData.readVisibility[(int)memType].at(insertPoint).type);
  Value aliasMatrixVal =
      auxData.aliasMatrices[(int)memType].at(insertPoint).value;
  auto aliasMatrixType = cast<RankedTensorType>(
      auxData.aliasMatrices[(int)memType].at(insertPoint).type);
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {bufOffset,     lengthVal,  pred,
                             threadVal,     buffersVal, readVisibilityVal,
                             aliasMatrixVal};
  std::string message = "Buffer being accessed has outstanding reads";
  if (!operandName.empty())
    message += ". Operand: " + operandName.str();
  AssertInfo assertInfo{message,
                        buffersType.cloneWith(std::nullopt, b.getI1Type())};
  createCallToCachedFunction(
      b, "verify_read_visibility", args, assertInfo,
      {buffersType, readVisibilityType, aliasMatrixType, (int)memType},
      [buffersType, readVisibilityType,
       aliasMatrixType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value bufOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadVal = entryBlock->getArgument(3);
        Value buffers = entryBlock->getArgument(4);
        Value readVisibilityPtr = entryBlock->getArgument(5);
        Value aliasMatrix = entryBlock->getArgument(6);

        Value readVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
        Value descriptor = createBufferDescriptor(fb, bufOffset, lengthVal);
        Value buffersEqBuf = createCmpIntTensorScalar(fb, buffers, descriptor);
        buffersEqBuf =
            expandAliases(fb, buffersEqBuf, aliasMatrix, aliasMatrixType);
        buffersEqBuf = convertAndBroadcast(fb, buffersEqBuf, /*dim=*/1,
                                           readVisibilityType);
        Value readVisibilityZero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, readVisibilityType);
        Value bufVisibility = arith::SelectOp::create(
            fb, buffersEqBuf, readVisibility, readVisibilityZero);
        Value totalVisibility =
            createBitwiseOrReduce(fb, bufVisibility, /*axis=*/1);
        Value threadColumnMask =
            createColumnMask(fb, threadVal, readVisibilityType);
        Value bufThreadVisibility = arith::SelectOp::create(
            fb, threadColumnMask, bufVisibility, readVisibilityZero);
        bufThreadVisibility =
            createBitwiseOrReduce(fb, bufThreadVisibility, /*axis=*/1);
        Value threadAndTotalVisibility =
            arith::AndIOp::create(fb, bufThreadVisibility, totalVisibility);
        Value hasVisibility =
            arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                  threadAndTotalVisibility, totalVisibility);
        Value vTrue = tti::createConstIntTensor(
            fb, fb.getLoc(), 1,
            cast<RankedTensorType>(hasVisibility.getType()));
        Value predicatedHasVisibility =
            arith::SelectOp::create(fb, pred, hasVisibility, vTrue);
        predicatedHasVisibility = createConvertLayout(
            fb, predicatedHasVisibility, buffersType.getEncoding());
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
      /*assertInfo=*/std::nullopt, {writeVisibilityType, (int)memType},
      [writeVisibilityType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
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

        constexpr uint64_t fullMask =
            tti::THREADS_BITMASK_SIZE >= 64
                ? std::numeric_limits<uint64_t>::max()
                : ((1ull << tti::THREADS_BITMASK_SIZE) - 1);
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

        Value updated = arith::OrIOp::create(fb, cleared, replicated);
        tti::createStoreScratchMemory(fb, fb.getLoc(), writeVisibilityPtr,
                                      updated, writeVisibilityType);

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
  SmallVector<Value> args = {sourceThreadVal,
                             arith::ConstantIntOp::create(b, destMask, 64),
                             pred, readVis.value};
  createCallToCachedFunction(
      b, "copy_read_visibility", args,
      /*assertInfo=*/std::nullopt, {readVisibilityType, (int)memType},
      [readVisibilityType, destMask](ImplicitLocOpBuilder &fb,
                                     Block *entryBlock) {
        Value sourceThread = entryBlock->getArgument(0);
        /*Value destMaskVal = entryBlock->getArgument(1);*/
        Value pred = entryBlock->getArgument(2);
        Value readVisibilityPtr = entryBlock->getArgument(3);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value readVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
        Value zeroTensor =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, readVisibilityType);
        Value destMaskTensor =
            createMultiColumnMask(fb, destMask, readVisibilityType);
        Value cleared = arith::SelectOp::create(fb, destMaskTensor, zeroTensor,
                                                readVisibility);

        Value sourceColumnMask =
            createColumnMask(fb, sourceThread, readVisibilityType);
        Value sourceColumn = arith::SelectOp::create(
            fb, sourceColumnMask, readVisibility, zeroTensor);
        Value sourceVector =
            createBitwiseOrReduce(fb, sourceColumn, /*axis=*/1);
        Value broadcastRow = convertAndBroadcast(fb, sourceVector, /*dim=*/1,
                                                 readVisibilityType);
        Value replicated = arith::SelectOp::create(fb, destMaskTensor,
                                                   broadcastRow, zeroTensor);

        Value updated = arith::OrIOp::create(fb, cleared, replicated);
        tti::createStoreScratchMemory(fb, fb.getLoc(), readVisibilityPtr,
                                      updated, readVisibilityType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createStageAccessForCommitCall(
    ImplicitLocOpBuilder &b, Value buf, uint32_t length, int thread, Value pred,
    MemType memType, CommitKind::Kind commitKind, Operation *insertPoint) {
  if (auxData.buffers[(int)memType].empty() ||
      auxData.commits[commitKind].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  ValueType buffers = auxData.buffers[(int)memType].at(insertPoint);
  ValueType outstandingCommits = auxData.commits[commitKind].at(insertPoint);
  auto buffersType = cast<RankedTensorType>(buffers.type);
  auto commitsType = cast<RankedTensorType>(outstandingCommits.type);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {bufOffset,     lengthVal,
                             pred,          threadVal,
                             buffers.value, outstandingCommits.value};
  createCallToCachedFunction(
      b, "stage_access_for_commit", args,
      /*assertInfo=*/std::nullopt, {buffersType, commitsType},
      [buffersType, commitsType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value bufOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadVal = entryBlock->getArgument(3);
        Value buffers = entryBlock->getArgument(4);
        Value outstandingCommitsPtr = entryBlock->getArgument(5);

        (void)threadVal;

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value commits = tti::createLoadScratchMemory(
            fb, fb.getLoc(), outstandingCommitsPtr, commitsType);
        Value descriptor = createBufferDescriptor(fb, bufOffset, lengthVal);
        Value buffersEqBuf = createCmpIntTensorScalar(fb, buffers, descriptor);
        buffersEqBuf =
            convertAndBroadcast(fb, buffersEqBuf, /*dim=*/1, commitsType);
        Value threadColumnMask = createColumnMask(fb, threadVal, commitsType);
        Value bufAndThread =
            arith::AndIOp::create(fb, buffersEqBuf, threadColumnMask);
        Value minusOne =
            tti::createConstIntTensor(fb, fb.getLoc(), -1, commitsType, true);
        Value updated =
            arith::SelectOp::create(fb, bufAndThread, minusOne, commits);
        tti::createStoreScratchMemory(fb, fb.getLoc(), outstandingCommitsPtr,
                                      updated, commitsType);

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

        Value threadMask = createColumnMask(fb, threadVal, commitsType);
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

        tti::createStoreScratchMemory(fb, fb.getLoc(), outstandingCommitsPtr,
                                      commits, commitsType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createClearOutstandingCommitsTransferWritesCall(
    ImplicitLocOpBuilder &b, int thread, uint64_t transferThreadMask,
    int outstandingNum, Value pred, CommitKind::Kind commitKind,
    MemType memType, Operation *insertPoint) {
  if (auxData.commits[commitKind].empty() ||
      auxData.writeVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  ValueType outstandingCommits = auxData.commits[commitKind].at(insertPoint);
  ValueType writeVisibility =
      auxData.writeVisibility[(int)memType].at(insertPoint);
  auto commitsType = cast<RankedTensorType>(outstandingCommits.type);
  auto writeVisibilityType = cast<RankedTensorType>(writeVisibility.type);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value transferMaskVal =
      arith::ConstantIntOp::create(b, transferThreadMask, 64);
  Value outstandingNumVal = arith::ConstantIntOp::create(b, outstandingNum, 32);
  SmallVector<Value> args = {
      threadVal, transferMaskVal,          outstandingNumVal,
      pred,      outstandingCommits.value, writeVisibility.value};
  createCallToCachedFunction(
      b, "clear_outstanding_commits_transfer_writes", args,
      /*assertInfo=*/std::nullopt, {commitsType, writeVisibilityType},
      [commitsType, writeVisibilityType](ImplicitLocOpBuilder &fb,
                                         Block *entryBlock) {
        Value threadVal = entryBlock->getArgument(0);
        Value transferMaskVal = entryBlock->getArgument(1);
        Value outstandingNumVal = entryBlock->getArgument(2);
        Value pred = entryBlock->getArgument(3);
        Value outstandingCommitsPtr = entryBlock->getArgument(4);
        Value writeVisibilityPtr = entryBlock->getArgument(5);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value outstandingCommits = tti::createLoadScratchMemory(
            fb, fb.getLoc(), outstandingCommitsPtr, commitsType);
        Value writeVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);

        auto elemIntType = cast<IntegerType>(commitsType.getElementType());
        Value outstandingNumElem =
            adjustIntegerWidth(fb, outstandingNumVal, elemIntType);
        Value threadColumnMask = createColumnMask(fb, threadVal, commitsType);
        auto outstandingCommitsGtOutstandingNum =
            createCmpIntTensorScalar(fb, outstandingCommits, outstandingNumElem,
                                     arith::CmpIPredicate::sgt);
        outstandingCommitsGtOutstandingNum = arith::AndIOp::create(
            fb, outstandingCommitsGtOutstandingNum, threadColumnMask);

        Value rowMask =
            createBitwiseOrReduce(fb, outstandingCommitsGtOutstandingNum,
                                  /*axis=*/1);
        rowMask =
            createConvertLayout(fb, rowMask, writeVisibilityType.getEncoding());
        Value transferMaskElem = adjustIntegerWidth(
            fb, transferMaskVal,
            cast<IntegerType>(writeVisibilityType.getElementType()));
        Value transferMaskTensor =
            triton::SplatOp::create(fb, writeVisibilityType, transferMaskElem);
        Value writeVisibilityOrThreadBit =
            arith::OrIOp::create(fb, writeVisibility, transferMaskTensor);
        Value writeVisibilityUpdated = arith::SelectOp::create(
            fb, rowMask, writeVisibilityOrThreadBit, writeVisibility);
        tti::createStoreScratchMemory(fb, fb.getLoc(), writeVisibilityPtr,
                                      writeVisibilityUpdated,
                                      writeVisibilityType);

        Value outstandingCommitsZero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, commitsType);
        outstandingCommits =
            arith::SelectOp::create(fb, outstandingCommitsGtOutstandingNum,
                                    outstandingCommitsZero, outstandingCommits);
        tti::createStoreScratchMemory(fb, fb.getLoc(), outstandingCommitsPtr,
                                      outstandingCommits, commitsType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createClearOutstandingCommitsTransferReadsCall(
    ImplicitLocOpBuilder &b, int thread, uint64_t transferThreadMask,
    int outstandingNum, Value pred, CommitKind::Kind commitKind,
    MemType memType, Operation *insertPoint) {
  if (auxData.commits[commitKind].empty() ||
      auxData.readVisibility[(int)memType].empty()) {
    return;
  }
  ValueType outstandingCommits = auxData.commits[commitKind].at(insertPoint);
  ValueType readVisibility =
      auxData.readVisibility[(int)memType].at(insertPoint);
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  auto commitsType = cast<RankedTensorType>(outstandingCommits.type);
  auto readVisibilityType = cast<RankedTensorType>(readVisibility.type);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value transferMaskVal =
      arith::ConstantIntOp::create(b, transferThreadMask, 64);
  Value outstandingNumVal = arith::ConstantIntOp::create(b, outstandingNum, 32);
  SmallVector<Value> args = {
      threadVal, transferMaskVal,          outstandingNumVal,
      pred,      outstandingCommits.value, readVisibility.value};
  createCallToCachedFunction(
      b, "clear_outstanding_commits_transfer_reads", args,
      /*assertInfo=*/std::nullopt, {commitsType, readVisibilityType},
      [commitsType, readVisibilityType](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
        Value threadVal = entryBlock->getArgument(0);
        Value transferMaskVal = entryBlock->getArgument(1);
        Value outstandingNumVal = entryBlock->getArgument(2);
        Value pred = entryBlock->getArgument(3);
        Value outstandingCommitsPtr = entryBlock->getArgument(4);
        Value readVisibilityPtr = entryBlock->getArgument(5);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value outstandingCommits = tti::createLoadScratchMemory(
            fb, fb.getLoc(), outstandingCommitsPtr, commitsType);
        Value readVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);

        auto elemIntType = cast<IntegerType>(commitsType.getElementType());
        Value outstandingNumElem =
            adjustIntegerWidth(fb, outstandingNumVal, elemIntType);
        Value threadColumnMask = createColumnMask(fb, threadVal, commitsType);
        auto outstandingCommitsGtOutstandingNum =
            createCmpIntTensorScalar(fb, outstandingCommits, outstandingNumElem,
                                     arith::CmpIPredicate::sgt);
        outstandingCommitsGtOutstandingNum = arith::AndIOp::create(
            fb, outstandingCommitsGtOutstandingNum, threadColumnMask);

        Value rowMask =
            createBitwiseOrReduce(fb, outstandingCommitsGtOutstandingNum,
                                  /*axis=*/1);
        rowMask =
            convertAndBroadcast(fb, rowMask, /*dim=*/1, readVisibilityType);
        Value transferMaskElem = adjustIntegerWidth(
            fb, transferMaskVal,
            cast<IntegerType>(readVisibilityType.getElementType()));
        Value transferMaskTensor =
            triton::SplatOp::create(fb, readVisibilityType, transferMaskElem);
        Value readVisibilityOrThreadBit =
            arith::OrIOp::create(fb, readVisibility, transferMaskTensor);
        Value readVisibilityUpdated = arith::SelectOp::create(
            fb, rowMask, readVisibilityOrThreadBit, readVisibility);
        tti::createStoreScratchMemory(fb, fb.getLoc(), readVisibilityPtr,
                                      readVisibilityUpdated,
                                      readVisibilityType);

        Value outstandingCommitsZero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, commitsType);
        outstandingCommits =
            arith::SelectOp::create(fb, outstandingCommitsGtOutstandingNum,
                                    outstandingCommitsZero, outstandingCommits);
        tti::createStoreScratchMemory(fb, fb.getLoc(), outstandingCommitsPtr,
                                      outstandingCommits, commitsType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createCheckOutstandingCommitsCall(
    ImplicitLocOpBuilder &b, Value buf, uint32_t length, int thread,
    StringRef pendingAccessType, Value pred, MemType memType,
    CommitKind::Kind commitKind, Operation *insertPoint) {
  if (auxData.buffers[(int)memType].empty() ||
      auxData.commits[commitKind].empty() ||
      auxData.aliasMatrices[(int)memType].empty()) {
    return;
  }
  ValueType buffers = auxData.buffers[(int)memType].at(insertPoint);
  ValueType outstandingCommits = auxData.commits[commitKind].at(insertPoint);
  ValueType aliasMatrix = auxData.aliasMatrices[(int)memType].at(insertPoint);
  assert(thread < NUM_THREADS &&
         "Commit-count tracking must operate on base threads");
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  auto buffersType = cast<RankedTensorType>(buffers.type);
  auto commitsType = cast<RankedTensorType>(outstandingCommits.type);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  auto aliasMatrixType = cast<RankedTensorType>(aliasMatrix.type);
  SmallVector<Value> args = {
      bufOffset,        lengthVal,     pred,
      threadVal,        buffers.value, outstandingCommits.value,
      aliasMatrix.value};
  std::string message =
      "Accessing buffer with pending access. Pending access type: " +
      pendingAccessType.str();
  AssertInfo assertInfo{message,
                        commitsType.cloneWith(std::nullopt, b.getI1Type())};
  createCallToCachedFunction(
      b, "check_outstanding_commits", args, assertInfo,
      {buffersType, commitsType, aliasMatrixType, (int)thread},
      [buffersType, commitsType, aliasMatrixType](ImplicitLocOpBuilder &fb,
                                                  Block *entryBlock) {
        Value bufOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadVal = entryBlock->getArgument(3);
        Value buffers = entryBlock->getArgument(4);
        Value outstandingCommitsPtr = entryBlock->getArgument(5);
        Value aliasMatrix = entryBlock->getArgument(6);

        Value outstandingCommits = tti::createLoadScratchMemory(
            fb, fb.getLoc(), outstandingCommitsPtr, commitsType);
        Value descriptor = createBufferDescriptor(fb, bufOffset, lengthVal);
        Value buffersEqBuf = createCmpIntTensorScalar(fb, buffers, descriptor);
        buffersEqBuf =
            expandAliases(fb, buffersEqBuf, aliasMatrix, aliasMatrixType);
        buffersEqBuf =
            convertAndBroadcast(fb, buffersEqBuf, /*dim=*/1, commitsType);
        Value zeroTensor =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, commitsType);
        Value selectedRows = arith::SelectOp::create(
            fb, buffersEqBuf, outstandingCommits, zeroTensor);
        Value selectedEqZero = arith::CmpIOp::create(
            fb, arith::CmpIPredicate::eq, selectedRows, zeroTensor);
        Value vTrue = tti::createConstIntTensor(
            fb, fb.getLoc(), 1,
            cast<RankedTensorType>(selectedEqZero.getType()));
        Value predicatedSelectedEqZero =
            arith::SelectOp::create(fb, pred, selectedEqZero, vTrue);

        triton::ReturnOp::create(fb, predicatedSelectedEqZero);
      });
}

} // namespace mlir::triton::instrument
