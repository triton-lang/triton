#include "triton/Dialect/TritonInstrument/IR/FunctionBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/DebugStringHelper.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include <limits>

namespace mlir::triton::instrument {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace tti = mlir::triton::instrument;
namespace ttn = mlir::triton::nvgpu;

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

constexpr uint64_t getFullThreadBitmask() {
  if constexpr (tti::THREADS_BITMASK_SIZE >= 64)
    return std::numeric_limits<uint64_t>::max();
  return (uint64_t{1} << tti::THREADS_BITMASK_SIZE) - 1;
}

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

template <typename BinaryOp>
Value reduceDim(ImplicitLocOpBuilder &b, Value tensor, int axis) {
  OpBuilder::InsertionGuard guard(b);
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto reduceOp = triton::ReduceOp::create(b, std::vector<Value>{tensor}, axis);
  auto &region = reduceOp.getRegion();
  auto &block = region.emplaceBlock();
  block.addArguments({tensorType.getElementType(), tensorType.getElementType()},
                     {b.getLoc(), b.getLoc()});
  b.setInsertionPointToStart(&block);
  auto result = BinaryOp::create(b, block.getArgument(0), block.getArgument(1));
  triton::ReduceReturnOp::create(b, std::vector<Value>{result});
  return reduceOp->getResult(0);
}

template <typename BinaryOp>
Value reduceLastDim(ImplicitLocOpBuilder &b, Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  return reduceDim<BinaryOp>(b, tensor, tensorType.getRank() - 1);
}

template <typename BinaryOp>
Value reduceAllDims(ImplicitLocOpBuilder &b, Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  Value flattened = triton::ReshapeOp::create(
      b, ArrayRef<int64_t>{tensorType.getNumElements()}, tensor,
      /*allowReorder=*/true);
  return reduceLastDim<BinaryOp>(b, flattened);
}

Value createMaskedAllOf(ImplicitLocOpBuilder &b, Value mask, Value predicate) {
  auto predicateType = cast<RankedTensorType>(predicate.getType());
  Value selected = arith::SelectOp::create(
      b, mask, predicate,
      tti::createConstIntTensor(b, b.getLoc(), 1, predicateType));
  return reduceAllDims<arith::AndIOp>(b, selected);
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
  func->setAttr("always_use_warp_shuffle", moduleBuilder.getUnitAttr());
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
    createAssertInThread(b, result, message);
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

Value createConvertLayout(ImplicitLocOpBuilder &b, Value tensor,
                          Attribute encoding) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto dstType = tensorType.cloneWithEncoding(encoding);
  return ttg::ConvertLayoutOp::create(b, dstType, tensor);
}

Value convertAndBroadcast(ImplicitLocOpBuilder &b, Value tensor,
                          ArrayRef<int> keptDims, RankedTensorType dstType) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto encoding = cast<ttg::DistributedEncodingTrait>(dstType.getEncoding());
  assert(static_cast<size_t>(tensorType.getRank()) == keptDims.size() &&
         "expected one kept dimension per source tensor rank");
  llvm::SmallDenseSet<int> keptDimsSet(keptDims.begin(), keptDims.end());
  Attribute sliceEncoding = encoding;
  for (int dim = encoding.getRepOrder().size() - 1; dim >= 0; --dim) {
    if (!keptDimsSet.contains(dim))
      sliceEncoding = ttg::SliceEncodingAttr::get(
          b.getContext(), dim,
          cast<ttg::DistributedEncodingTrait>(sliceEncoding));
  }
  tensor = createConvertLayout(b, tensor, sliceEncoding);
  while (cast<RankedTensorType>(tensor.getType()).getRank() < dstType.getRank())
    tensor = tti::expandOuterSlicedDim(b, b.getLoc(), tensor);
  auto resultType = RankedTensorType::get(
      dstType.getShape(), tensorType.getElementType(), encoding);
  return triton::BroadcastOp::create(b, resultType, tensor);
}

Value expandAliases(ImplicitLocOpBuilder &b, Value bufferMask,
                    Value aliasMatrix, RankedTensorType aliasMatrixType) {
  assert(aliasMatrixType.getRank() == 2 &&
         "Alias matrix expected to be rank-2");
  Value baseMaskMatrix =
      convertAndBroadcast(b, bufferMask, {1}, aliasMatrixType);
  Value aliasingMask = arith::AndIOp::create(b, aliasMatrix, baseMaskMatrix);
  return reduceDim<arith::OrIOp>(b, aliasingMask, /*axis=*/0);
}

// Given a constexpr integer mask `columnMask` and `tensor<...xNxiD>`, codegen
// a mask `tensor<Nxi1>` for the last tensor dimension and broadcast it to the
// full tensor shape.
Value createMultiColumnMask(ImplicitLocOpBuilder &b, uint64_t columnMask,
                            RankedTensorType tensorType) {
  int columnDim = tensorType.getRank() - 1;
  unsigned size = tensorType.getShape()[columnDim];
  SmallVector<bool> maskBits(size);
  for (unsigned i : llvm::seq(size))
    maskBits[i] = (columnMask >> i) & 1;

  auto encoding = cast<ttg::DistributedEncodingTrait>(tensorType.getEncoding());
  auto sliceEncoding = tti::getSingleDimSliceEncoding(encoding, columnDim);
  auto mask1DType = RankedTensorType::get({tensorType.getShape()[columnDim]},
                                          b.getI1Type(), sliceEncoding);
  auto valueAttr = DenseElementsAttr::get(mask1DType, maskBits);
  Value mask1D = arith::ConstantOp::create(b, valueAttr);
  auto maskType =
      cast<RankedTensorType>(tensorType.cloneWith(std::nullopt, b.getI1Type()));
  return convertAndBroadcast(b, mask1D, {columnDim}, maskType);
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
  int columnDim = tensorType.getRank() - 1;
  auto encoding = cast<ttg::DistributedEncodingTrait>(tensorType.getEncoding());
  auto sliceEncoding = tti::getSingleDimSliceEncoding(encoding, columnDim);
  int columns = tensorType.getShape()[columnDim];

  RankedTensorType rangeType =
      RankedTensorType::get({columns}, b.getI32Type(), sliceEncoding);
  Value range = triton::MakeRangeOp::create(b, rangeType, 0, columns);

  auto elemType = cast<IntegerType>(tensorType.getElementType());
  RankedTensorType rangeElemType =
      RankedTensorType::get({columns}, elemType, sliceEncoding);
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

Value createColumnMask(ImplicitLocOpBuilder &b, Value column,
                       RankedTensorType tensorType) {
  int columnDim = tensorType.getRank() - 1;
  auto encoding = cast<ttg::DistributedEncodingTrait>(tensorType.getEncoding());
  auto sliceEncoding = tti::getSingleDimSliceEncoding(encoding, columnDim);
  auto colType = RankedTensorType::get({tensorType.getShape()[columnDim]},
                                       b.getI32Type(), sliceEncoding);
  Value range =
      triton::MakeRangeOp::create(b, colType, /*start=*/0,
                                  /*end=*/tensorType.getShape()[columnDim]);
  Value columnTensor = triton::SplatOp::create(b, colType, column);
  Value mask1D =
      arith::CmpIOp::create(b, arith::CmpIPredicate::eq, range, columnTensor);
  auto maskType =
      cast<RankedTensorType>(tensorType.cloneWith(std::nullopt, b.getI1Type()));
  return convertAndBroadcast(b, mask1D, {columnDim}, maskType);
}

Value createCurrentCTAId(ImplicitLocOpBuilder &b) {
  if (ttg::lookupNumCTAs(b.getInsertionBlock()->getParentOp()) == 1)
    return arith::ConstantIntOp::create(b, 0, 32);
  OperationState state(b.getLoc(), ttn::ClusterCTAIdOp::getOperationName());
  state.addTypes(b.getI32Type());
  return b.create(state)->getResult(0);
}

Value createTrueTensorLike(ImplicitLocOpBuilder &b,
                           RankedTensorType tensorType) {
  return tti::createConstIntTensor(b, b.getLoc(), 1, tensorType);
}

Value createCurrentCTAMask(ImplicitLocOpBuilder &b) {
  Value one = arith::ConstantIntOp::create(b, 1, 32);
  return arith::ShLIOp::create(b, one, createCurrentCTAId(b));
}

Value createCTAMaskMatch1D(ImplicitLocOpBuilder &b, RankedTensorType tensorType,
                           Value ctaMask) {
  assert(tensorType.getRank() == 1 && "expected rank-1 CTA tensor type");
  auto maskType = RankedTensorType::get(tensorType.getShape(), b.getI1Type(),
                                        tensorType.getEncoding());
  int numCTAs = ttg::lookupNumCTAs(b.getInsertionBlock()->getParentOp());
  assert(tensorType.getShape()[0] == numCTAs &&
         "CTA dimension must match the kernel CTA count");
  if (numCTAs == 1)
    return createTrueTensorLike(b, maskType);

  auto ctaType = RankedTensorType::get(tensorType.getShape(), b.getI32Type(),
                                       tensorType.getEncoding());
  Value ctaIds =
      triton::MakeRangeOp::create(b, ctaType, /*start=*/0, /*end=*/numCTAs);
  Value maskTensor = triton::SplatOp::create(b, ctaType, ctaMask);
  Value shifted = arith::ShRUIOp::create(b, maskTensor, ctaIds);
  Value one = tti::createConstIntTensor(b, b.getLoc(), 1, ctaType);
  Value bits = arith::AndIOp::create(b, shifted, one);
  Value zero = tti::createConstIntTensor(b, b.getLoc(), 0, ctaType);
  return arith::CmpIOp::create(b, arith::CmpIPredicate::ne, bits, zero);
}

Value createCTAMaskMatch(ImplicitLocOpBuilder &b, RankedTensorType tensorType,
                         Value ctaMask, int dim) {
  assert(dim >= 0 && dim < tensorType.getRank() &&
         "expected valid CTA mask dimension");
  auto encoding = cast<ttg::DistributedEncodingTrait>(tensorType.getEncoding());
  auto sliceEncoding = tti::getSingleDimSliceEncoding(encoding, dim);
  auto sliceType = RankedTensorType::get({tensorType.getShape()[dim]},
                                         b.getI1Type(), sliceEncoding);
  Value mask1D = createCTAMaskMatch1D(b, sliceType, ctaMask);
  auto maskType =
      cast<RankedTensorType>(tensorType.cloneWith(std::nullopt, b.getI1Type()));
  return convertAndBroadcast(b, mask1D, {dim}, maskType);
}

Value createBufferMask1D(ImplicitLocOpBuilder &b, Value bufferOffset,
                         Value bufferLength, Value buffers) {
  return createCmpIntTensorScalar(
      b, buffers, createBufferDescriptor(b, bufferOffset, bufferLength));
}

Value createBroadcastBufferMatch(ImplicitLocOpBuilder &b, Value bufferOffset,
                                 Value bufferLength, Value buffers,
                                 RankedTensorType targetType, Value ctaMask) {
  Value bufferMask = convertAndBroadcast(
      b, createBufferMask1D(b, bufferOffset, bufferLength, buffers), {1},
      cast<RankedTensorType>(
          targetType.cloneWith(std::nullopt, b.getI1Type())));
  return arith::AndIOp::create(
      b, bufferMask, createCTAMaskMatch(b, targetType, ctaMask, /*dim=*/0));
}

Value createBarrierMask1D(ImplicitLocOpBuilder &b, Value barrierOffset,
                          Value barrierLength, Value barriers) {
  Value descriptor = createBufferDescriptor(b, barrierOffset, barrierLength);
  return createCmpIntTensorScalar(b, barriers, descriptor);
}

Value createIndexMask1D(ImplicitLocOpBuilder &b, RankedTensorType tensorType,
                        int index) {
  auto rangeType = RankedTensorType::get(tensorType.getShape(), b.getI32Type(),
                                         tensorType.getEncoding());
  Value range =
      triton::MakeRangeOp::create(b, rangeType, /*start=*/0,
                                  /*end=*/tensorType.getShape().front());
  Value indexVal = arith::ConstantIntOp::create(b, index, 32);
  return createCmpIntTensorScalar(b, range, indexVal);
}

Value createBarrierRowMask(ImplicitLocOpBuilder &b, Value barrierOffset,
                           Value barrierLength, Value barriers,
                           RankedTensorType barriersType) {
  return createBarrierMask1D(b, barrierOffset, barrierLength, barriers);
}

Value createBroadcastBarrierRows(ImplicitLocOpBuilder &b, Value barrierOffset,
                                 Value barrierLength, Value barriers,
                                 RankedTensorType barriersType,
                                 RankedTensorType targetType, int barrierDim) {
  assert(barrierDim >= 0 && barrierDim < targetType.getRank() &&
         "expected valid barrier dimension");
  Value barrierRows = createBarrierRowMask(b, barrierOffset, barrierLength,
                                           barriers, barriersType);
  return convertAndBroadcast(b, barrierRows, {barrierDim},
                             cast<RankedTensorType>(targetType.cloneWith(
                                 std::nullopt, b.getI1Type())));
}

Value createMaskedBroadcastBarrierRows(ImplicitLocOpBuilder &b,
                                       Value barrierOffset, Value barrierLength,
                                       Value barriers,
                                       RankedTensorType barriersType,
                                       RankedTensorType targetType,
                                       Value ctaMask) {
  int barrierDim = targetType.getRank() - 1;
  return arith::AndIOp::create(
      b,
      createBroadcastBarrierRows(b, barrierOffset, barrierLength, barriers,
                                 barriersType, targetType, barrierDim),
      createCTAMaskMatch(b, targetType, ctaMask, /*dim=*/0));
}

Value createLeaderCTAPredicate(ImplicitLocOpBuilder &b, Value ctaMask) {
  Value zero = arith::ConstantIntOp::create(b, 0, 32);
  Value leaderMask = arith::AndIOp::create(
      b, ctaMask, arith::SubIOp::create(b, zero, ctaMask));
  return arith::CmpIOp::create(b, arith::CmpIPredicate::eq,
                               createCurrentCTAMask(b), leaderMask);
}

Value createTruePred(ImplicitLocOpBuilder &b) {
  return arith::ConstantIntOp::create(b, 1, 1);
}

Value createWaitingThreadShift(ImplicitLocOpBuilder &b, Value baseThread,
                               unsigned bit) {
  Value bitsPerThread =
      arith::ConstantIntOp::create(b, WaitingBits::bitsPerThread, 32);
  Value bitVal = arith::ConstantIntOp::create(b, bit, 32);
  Value baseTimesBits = arith::MulIOp::create(b, baseThread, bitsPerThread);
  return arith::AddIOp::create(b, baseTimesBits, bitVal);
}

Value createWaitingThreadBitMask(ImplicitLocOpBuilder &b, Value baseThread,
                                 unsigned bit) {
  Value one = arith::ConstantIntOp::create(b, 1, 32);
  return arith::ShLIOp::create(b, one,
                               createWaitingThreadShift(b, baseThread, bit));
}

Value createWaitingThreadClearMask(ImplicitLocOpBuilder &b, Value baseThread) {
  Value flagMask =
      createWaitingThreadBitMask(b, baseThread, WaitingBits::flagBit);
  Value phaseMask =
      createWaitingThreadBitMask(b, baseThread, WaitingBits::phaseBit);
  Value combinedMask = arith::OrIOp::create(b, flagMask, phaseMask);
  Value minusOne = arith::ConstantIntOp::create(b, -1, 32);
  return arith::XOrIOp::create(b, combinedMask, minusOne);
}

Value createWaitingThreadStateBits(ImplicitLocOpBuilder &b, Value baseThread,
                                   Value phase) {
  Value one = arith::ConstantIntOp::create(b, 1, 32);
  Value flagMask =
      createWaitingThreadBitMask(b, baseThread, WaitingBits::flagBit);
  Value phaseScalar = arith::AndIOp::create(b, phase, one);
  Value phaseBits = arith::ShLIOp::create(
      b, phaseScalar,
      createWaitingThreadShift(b, baseThread, WaitingBits::phaseBit));
  return arith::OrIOp::create(b, flagMask, phaseBits);
}

Value createIntConstantLike(ImplicitLocOpBuilder &b, Type type, int64_t value) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return tti::createConstIntTensor(b, b.getLoc(), value, tensorType);
  return arith::ConstantIntOp::create(b, value,
                                      cast<IntegerType>(type).getWidth());
}

Value createEffectiveWaitingBits(ImplicitLocOpBuilder &b, Value waiting,
                                 Value barrierState) {
  Type waitingType = waiting.getType();
  Value one = createIntConstantLike(b, waitingType, 1);
  Value flagMask = createIntConstantLike(b, waitingType, WaitingBits::flagMask);
  Value phaseMask =
      createIntConstantLike(b, waitingType, WaitingBits::phaseMask);
  Value flags = arith::AndIOp::create(b, waiting, flagMask);
  Value phases = arith::AndIOp::create(b, waiting, phaseMask);
  Value phasesAligned = arith::ShRUIOp::create(b, phases, one);
  Value phasesComplement = arith::XOrIOp::create(b, phasesAligned, flagMask);
  Value waitingPhase0 = arith::AndIOp::create(b, flags, phasesComplement);
  Value waitingPhase1 = arith::AndIOp::create(b, flags, phasesAligned);
  Value barrierOne = createIntConstantLike(b, barrierState.getType(), 1);
  Value barrierPhase = arith::AndIOp::create(b, barrierState, barrierOne);
  Value phaseIsOne = arith::CmpIOp::create(b, arith::CmpIPredicate::eq,
                                           barrierPhase, barrierOne);
  return arith::SelectOp::create(b, phaseIsOne, waitingPhase1, waitingPhase0);
}

} // namespace

void FunctionBuilder::createFillGlobalTensorCall(ImplicitLocOpBuilder &b,
                                                 Value ptr,
                                                 RankedTensorType type,
                                                 Value scalar) {
  createCallToCachedFunction(
      b, "fill_global_tensor", {ptr, scalar}, /*assertInfo=*/std::nullopt,
      {type}, [type](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value ptr = entryBlock->getArgument(0);
        Value scalar = entryBlock->getArgument(1);
        Value tensor = triton::SplatOp::create(fb, type, scalar);
        createStoreScratchMemory(fb, fb.getLoc(), ptr, tensor, type);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createSetWaitingCall(ImplicitLocOpBuilder &b,
                                           Value barrier, int thread,
                                           Value phase, Value pred,
                                           Operation *insertPoint) {
  if (auxData.waiting.empty()) {
    return;
  }
  if (!pred)
    pred = createTruePred(b);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value barrierOffset = tti::ExperimentalMemDescToI32Op::create(b, barrier);
  Value barrierLength =
      arith::ConstantIntOp::create(b, getMemDescLength(barrier), 32);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value waitingVal = auxData.waiting.at(insertPoint).value;
  auto waitingType =
      cast<RankedTensorType>(auxData.waiting.at(insertPoint).type);
  SmallVector<Value> args = {threadVal,     phase,         pred,
                             barrierOffset, barrierLength, barriersVal,
                             waitingVal};
  createCallToCachedFunction(
      b, "set_waiting", args,
      /*assertInfo=*/std::nullopt, {barriersType, waitingType},
      [barriersType, waitingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value baseThread = entryBlock->getArgument(0);
        Value phase = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value barrierOffset = entryBlock->getArgument(3);
        Value barrierLength = entryBlock->getArgument(4);
        Value barriers = entryBlock->getArgument(5);
        Value waitingPtr = entryBlock->getArgument(6);

        Value clearMaskScalar = createWaitingThreadClearMask(fb, baseThread);
        Value stateBits = createWaitingThreadStateBits(fb, baseThread, phase);
        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);
        Value waiting = tti::createLoadScratchMemory(fb, fb.getLoc(),
                                                     waitingPtr, waitingType);
        Value clearMask =
            triton::SplatOp::create(fb, waitingType, clearMaskScalar);
        Value stateBitsTensor =
            triton::SplatOp::create(fb, waitingType, stateBits);
        Value barriersEqBar = createMaskedBroadcastBarrierRows(
            fb, barrierOffset, barrierLength, barriers, barriersType,
            waitingType, createCurrentCTAMask(fb));
        Value clearedWaiting = arith::AndIOp::create(fb, waiting, clearMask);
        Value updatedWaiting =
            arith::OrIOp::create(fb, clearedWaiting, stateBitsTensor);
        Value newWaiting =
            arith::SelectOp::create(fb, barriersEqBar, updatedWaiting, waiting);
        tti::createStoreScratchMemory(fb, fb.getLoc(), waitingPtr, newWaiting,
                                      waitingType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createClearWaitingCall(ImplicitLocOpBuilder &b,
                                             Value barrier, int thread,
                                             Value pred,
                                             Operation *insertPoint) {
  if (auxData.waiting.empty()) {
    return;
  }
  if (!pred)
    pred = createTruePred(b);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value barrierOffset = tti::ExperimentalMemDescToI32Op::create(b, barrier);
  Value barrierLength =
      arith::ConstantIntOp::create(b, getMemDescLength(barrier), 32);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value waitingVal = auxData.waiting.at(insertPoint).value;
  auto waitingType =
      cast<RankedTensorType>(auxData.waiting.at(insertPoint).type);
  SmallVector<Value> args = {threadVal,     pred,        barrierOffset,
                             barrierLength, barriersVal, waitingVal};
  createCallToCachedFunction(
      b, "clear_waiting", args,
      /*assertInfo=*/std::nullopt, {barriersType, waitingType},
      [barriersType, waitingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value baseThread = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value barrierOffset = entryBlock->getArgument(2);
        Value barrierLength = entryBlock->getArgument(3);
        Value barriers = entryBlock->getArgument(4);
        Value waitingPtr = entryBlock->getArgument(5);

        Value clearMaskScalar = createWaitingThreadClearMask(fb, baseThread);
        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);
        Value waiting = tti::createLoadScratchMemory(fb, fb.getLoc(),
                                                     waitingPtr, waitingType);
        Value clearMask =
            triton::SplatOp::create(fb, waitingType, clearMaskScalar);
        Value barriersEqBar = createMaskedBroadcastBarrierRows(
            fb, barrierOffset, barrierLength, barriers, barriersType,
            waitingType, createCurrentCTAMask(fb));
        Value clearedWaiting = arith::AndIOp::create(fb, waiting, clearMask);
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
        Value effectiveWaiting =
            createEffectiveWaitingBits(fb, waiting, barrierStates);
        Value waitingOr = reduceLastDim<arith::OrIOp>(fb, effectiveWaiting);
        auto waitingOrType = cast<RankedTensorType>(waitingOr.getType());
        Value expandedActiveMaskTensor =
            triton::SplatOp::create(fb, waitingOrType, expandedActiveMaskVal);
        Value waitingMasked =
            arith::AndIOp::create(fb, waitingOr, expandedActiveMaskTensor);
        Value allActiveWaiting =
            arith::CmpIOp::create(fb, arith::CmpIPredicate::eq, waitingMasked,
                                  expandedActiveMaskTensor);
        allActiveWaiting = reduceAllDims<arith::AndIOp>(fb, allActiveWaiting);
        Value ok =
            arith::XOrIOp::create(fb, allActiveWaiting, createTruePred(fb));
        Value predicatedOk =
            arith::SelectOp::create(fb, pred, ok, createTruePred(fb));
        triton::ReturnOp::create(fb, predicatedOk);
      });
}

void FunctionBuilder::createVerifyBarrierCanInitCall(ImplicitLocOpBuilder &b,
                                                     Value barrier,
                                                     Operation *insertPoint,
                                                     Value ctaMask) {
  assert(!auxData.barrierStates.empty() &&
         "barrier states must exist when verifying barrier init");
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  Value barrierOffset = tti::ExperimentalMemDescToI32Op::create(b, barrier);
  Value barrierLength =
      arith::ConstantIntOp::create(b, getMemDescLength(barrier), 32);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value barrierStatesVal = auxData.barrierStates.at(insertPoint).value;
  auto barrierStatesType =
      cast<RankedTensorType>(auxData.barrierStates.at(insertPoint).type);
  SmallVector<Value> args = {barrierOffset, barrierLength, barriersVal,
                             barrierStatesVal, ctaMask};
  AssertInfo assertInfo{"Barrier re-initialized without prior invalidation",
                        b.getI1Type()};
  createCallToCachedFunction(
      b, "verify_barrier_can_init", args, assertInfo,
      {barriersType, barrierStatesType},
      [barriersType, barrierStatesType](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
        Value barrierOffset = entryBlock->getArgument(0);
        Value barrierLength = entryBlock->getArgument(1);
        Value barriers = entryBlock->getArgument(2);
        Value statesPtr = entryBlock->getArgument(3);
        Value ctaMask = entryBlock->getArgument(4);
        Value leaderPred = createLeaderCTAPredicate(fb, ctaMask);
        Value states = tti::createLoadScratchMemory(fb, fb.getLoc(), statesPtr,
                                                    barrierStatesType);
        Value barriersEqBar = createMaskedBroadcastBarrierRows(
            fb, barrierOffset, barrierLength, barriers, barriersType,
            barrierStatesType, ctaMask);
        Value zero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, barrierStatesType);
        Value canInit =
            arith::CmpIOp::create(fb, arith::CmpIPredicate::eq, states, zero);
        Value ok = createMaskedAllOf(fb, barriersEqBar, canInit);
        ok = arith::SelectOp::create(fb, leaderPred, ok, createTruePred(fb));
        triton::ReturnOp::create(fb, ok);
      });
}

void FunctionBuilder::createVerifyBarrierInitializedCall(
    ImplicitLocOpBuilder &b, Value barrier, Value pred, Operation *insertPoint,
    Value ctaMask) {
  assert(!auxData.barrierStates.empty() &&
         "barrier states must exist when verifying barrier use");
  if (!pred)
    pred = createTruePred(b);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  Value barrierOffset = tti::ExperimentalMemDescToI32Op::create(b, barrier);
  Value barrierLength =
      arith::ConstantIntOp::create(b, getMemDescLength(barrier), 32);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value barrierStatesVal = auxData.barrierStates.at(insertPoint).value;
  auto barrierStatesType =
      cast<RankedTensorType>(auxData.barrierStates.at(insertPoint).type);
  SmallVector<Value> args = {pred,        barrierOffset,    barrierLength,
                             barriersVal, barrierStatesVal, ctaMask};
  AssertInfo assertInfo{"Barrier used before initialization or after "
                        "invalidation",
                        b.getI1Type()};
  createCallToCachedFunction(
      b, "verify_barrier_initialized", args, assertInfo,
      {barriersType, barrierStatesType},
      [barriersType, barrierStatesType](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
        Value pred = entryBlock->getArgument(0);
        Value barrierOffset = entryBlock->getArgument(1);
        Value barrierLength = entryBlock->getArgument(2);
        Value barriers = entryBlock->getArgument(3);
        Value statesPtr = entryBlock->getArgument(4);
        Value ctaMask = entryBlock->getArgument(5);
        Value states = tti::createLoadScratchMemory(fb, fb.getLoc(), statesPtr,
                                                    barrierStatesType);
        Value barriersEqBar = createMaskedBroadcastBarrierRows(
            fb, barrierOffset, barrierLength, barriers, barriersType,
            barrierStatesType, ctaMask);
        Value zero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, barrierStatesType);
        Value initialized =
            arith::CmpIOp::create(fb, arith::CmpIPredicate::ne, states, zero);
        Value ok = createMaskedAllOf(fb, barriersEqBar, initialized);
        Value predicatedOk =
            arith::SelectOp::create(fb, pred, ok, createTruePred(fb));
        triton::ReturnOp::create(fb, predicatedOk);
      });
}

void FunctionBuilder::createInitBarrierStateCall(ImplicitLocOpBuilder &b,
                                                 Value barrier, int count,
                                                 Operation *insertPoint,
                                                 Value ctaMask) {
  if (auxData.barrierStates.empty()) {
    return;
  }
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  Value countVal = arith::ConstantIntOp::create(b, count, 32);
  Value barrierOffset = tti::ExperimentalMemDescToI32Op::create(b, barrier);
  Value barrierLength =
      arith::ConstantIntOp::create(b, getMemDescLength(barrier), 32);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value barrierStatesVal = auxData.barrierStates.at(insertPoint).value;
  auto barrierStatesType =
      cast<RankedTensorType>(auxData.barrierStates.at(insertPoint).type);
  SmallVector<Value> args = {countVal,    barrierOffset,    barrierLength,
                             barriersVal, barrierStatesVal, ctaMask};
  createCallToCachedFunction(
      b, "init_barrier_state", args,
      /*assertInfo=*/std::nullopt, {barriersType, barrierStatesType},
      [barriersType, barrierStatesType](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
        Value count = entryBlock->getArgument(0);
        Value barrierOffset = entryBlock->getArgument(1);
        Value barrierLength = entryBlock->getArgument(2);
        Value barriers = entryBlock->getArgument(3);
        Value statesPtr = entryBlock->getArgument(4);
        Value ctaMask = entryBlock->getArgument(5);
        Value countMask =
            arith::ConstantIntOp::create(fb, BarrierBits::countMask, 32);
        Value maskedCount = arith::AndIOp::create(fb, count, countMask);
        Value initField = arith::ShLIOp::create(
            fb, maskedCount,
            arith::ConstantIntOp::create(fb, BarrierBits::initCountLsb, 32));
        Value currentField = arith::ShLIOp::create(
            fb, maskedCount,
            arith::ConstantIntOp::create(fb, BarrierBits::currentCountLsb, 32));
        Value newState = arith::OrIOp::create(fb, initField, currentField);
        Value leaderPred = createLeaderCTAPredicate(fb, ctaMask);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, leaderPred);
        fb.setInsertionPointToStart(ifBlock);
        Value states = tti::createLoadScratchMemory(fb, fb.getLoc(), statesPtr,
                                                    barrierStatesType);
        Value barriersEqBar = createMaskedBroadcastBarrierRows(
            fb, barrierOffset, barrierLength, barriers, barriersType,
            barrierStatesType, ctaMask);
        Value stateTensor =
            triton::SplatOp::create(fb, barrierStatesType, newState);
        Value updated =
            arith::SelectOp::create(fb, barriersEqBar, stateTensor, states);
        tti::createStoreScratchMemory(fb, fb.getLoc(), statesPtr, updated,
                                      barrierStatesType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createInvalidateBarrierStateCall(ImplicitLocOpBuilder &b,
                                                       Value barrier,
                                                       Operation *insertPoint,
                                                       Value ctaMask) {
  assert(!auxData.barrierStates.empty() &&
         "barrier states must exist when invalidating a barrier");
  assert(!auxData.waiting.empty() &&
         "waiting state must exist when invalidating a barrier");
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  Value barrierOffset = tti::ExperimentalMemDescToI32Op::create(b, barrier);
  Value barrierLength =
      arith::ConstantIntOp::create(b, getMemDescLength(barrier), 32);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value barrierStatesVal = auxData.barrierStates.at(insertPoint).value;
  auto barrierStatesType =
      cast<RankedTensorType>(auxData.barrierStates.at(insertPoint).type);
  Value waitingVal = auxData.waiting.at(insertPoint).value;
  auto waitingType =
      cast<RankedTensorType>(auxData.waiting.at(insertPoint).type);
  SmallVector<Value> args = {barrierOffset,    barrierLength, barriersVal,
                             barrierStatesVal, waitingVal,    ctaMask};
  createCallToCachedFunction(
      b, "invalidate_barrier_state", args,
      /*assertInfo=*/std::nullopt,
      {barriersType, barrierStatesType, waitingType},
      [barriersType, barrierStatesType, waitingType](ImplicitLocOpBuilder &fb,
                                                     Block *entryBlock) {
        Value barrierOffset = entryBlock->getArgument(0);
        Value barrierLength = entryBlock->getArgument(1);
        Value barriers = entryBlock->getArgument(2);
        Value statesPtr = entryBlock->getArgument(3);
        Value waitingPtr = entryBlock->getArgument(4);
        Value ctaMask = entryBlock->getArgument(5);
        Value leaderPred = createLeaderCTAPredicate(fb, ctaMask);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, leaderPred);
        fb.setInsertionPointToStart(ifBlock);
        Value states = tti::createLoadScratchMemory(fb, fb.getLoc(), statesPtr,
                                                    barrierStatesType);
        Value waiting = tti::createLoadScratchMemory(fb, fb.getLoc(),
                                                     waitingPtr, waitingType);
        Value barriersEqBar = createMaskedBroadcastBarrierRows(
            fb, barrierOffset, barrierLength, barriers, barriersType,
            barrierStatesType, ctaMask);
        Value zeroState =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, barrierStatesType);
        Value zeroWaiting =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, waitingType);
        Value updatedStates =
            arith::SelectOp::create(fb, barriersEqBar, zeroState, states);
        Value updatedWaiting =
            arith::SelectOp::create(fb, barriersEqBar, zeroWaiting, waiting);
        tti::createStoreScratchMemory(fb, fb.getLoc(), statesPtr, updatedStates,
                                      barrierStatesType);
        tti::createStoreScratchMemory(fb, fb.getLoc(), waitingPtr,
                                      updatedWaiting, waitingType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createVerifyBarrierArriveCall(ImplicitLocOpBuilder &b,
                                                    Value barrier, int count,
                                                    Value pred,
                                                    Operation *insertPoint,
                                                    Value ctaMask) {
  if (auxData.barrierStates.empty()) {
    return;
  }
  if (!pred)
    pred = createTruePred(b);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  Value countVal = arith::ConstantIntOp::create(b, count, 32);
  Value barrierOffset = tti::ExperimentalMemDescToI32Op::create(b, barrier);
  Value barrierLength =
      arith::ConstantIntOp::create(b, getMemDescLength(barrier), 32);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value barrierStatesVal = auxData.barrierStates.at(insertPoint).value;
  auto barrierStatesType =
      cast<RankedTensorType>(auxData.barrierStates.at(insertPoint).type);
  SmallVector<Value> args = {countVal,      pred,        barrierOffset,
                             barrierLength, barriersVal, barrierStatesVal,
                             ctaMask};
  AssertInfo assertInfo{"Barrier arrive underflow: current count would become "
                        "negative",
                        b.getI1Type()};
  createCallToCachedFunction(
      b, "verify_barrier_arrive", args, assertInfo,
      {barriersType, barrierStatesType},
      [barriersType, barrierStatesType](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
        Value count = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value barrierOffset = entryBlock->getArgument(2);
        Value barrierLength = entryBlock->getArgument(3);
        Value barriers = entryBlock->getArgument(4);
        Value statesPtr = entryBlock->getArgument(5);
        Value ctaMask = entryBlock->getArgument(6);
        Value countMask =
            arith::ConstantIntOp::create(fb, BarrierBits::countMask, 32);
        Value maskedCount = arith::AndIOp::create(fb, count, countMask);
        Value states = tti::createLoadScratchMemory(fb, fb.getLoc(), statesPtr,
                                                    barrierStatesType);
        Value barriersEqBar = createMaskedBroadcastBarrierRows(
            fb, barrierOffset, barrierLength, barriers, barriersType,
            barrierStatesType, ctaMask);
        Value countMaskTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::countMask, barrierStatesType);
        Value shiftNineTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::currentCountLsb, barrierStatesType);
        Value currentCount =
            arith::ShRUIOp::create(fb, states, shiftNineTensor);
        currentCount = arith::AndIOp::create(fb, currentCount, countMaskTensor);
        Value arriveCount =
            triton::SplatOp::create(fb, barrierStatesType, maskedCount);
        Value newCurrent = arith::SubIOp::create(fb, currentCount, arriveCount);
        Value zero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, barrierStatesType);
        Value nonNegative = arith::CmpIOp::create(fb, arith::CmpIPredicate::sge,
                                                  newCurrent, zero);
        Value ok = createMaskedAllOf(fb, barriersEqBar, nonNegative);
        Value predicatedOk =
            arith::SelectOp::create(fb, pred, ok, createTruePred(fb));
        triton::ReturnOp::create(fb, predicatedOk);
      });
}

void FunctionBuilder::createUpdateBarrierStateCall(ImplicitLocOpBuilder &b,
                                                   Value barrier, int count,
                                                   Value pred,
                                                   Operation *insertPoint,
                                                   Value ctaMask) {
  if (auxData.barrierStates.empty()) {
    return;
  }
  if (!pred)
    pred = createTruePred(b);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  Value countVal = arith::ConstantIntOp::create(b, count, 32);
  Value barrierOffset = tti::ExperimentalMemDescToI32Op::create(b, barrier);
  Value barrierLength =
      arith::ConstantIntOp::create(b, getMemDescLength(barrier), 32);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value barrierStatesVal = auxData.barrierStates.at(insertPoint).value;
  auto barrierStatesType =
      cast<RankedTensorType>(auxData.barrierStates.at(insertPoint).type);
  SmallVector<Value> args = {countVal,      pred,        barrierOffset,
                             barrierLength, barriersVal, barrierStatesVal,
                             ctaMask};
  createCallToCachedFunction(
      b, "update_barrier_state", args,
      /*assertInfo=*/std::nullopt, {barriersType, barrierStatesType},
      [barriersType, barrierStatesType](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
        Value count = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value barrierOffset = entryBlock->getArgument(2);
        Value barrierLength = entryBlock->getArgument(3);
        Value barriers = entryBlock->getArgument(4);
        Value statesPtr = entryBlock->getArgument(5);
        Value ctaMask = entryBlock->getArgument(6);
        Value countMask =
            arith::ConstantIntOp::create(fb, BarrierBits::countMask, 32);
        Value maskedCount = arith::AndIOp::create(fb, count, countMask);
        Value shiftOne =
            arith::ConstantIntOp::create(fb, BarrierBits::initCountLsb, 32);
        Value shiftNine =
            arith::ConstantIntOp::create(fb, BarrierBits::currentCountLsb, 32);
        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);
        Value states = tti::createLoadScratchMemory(fb, fb.getLoc(), statesPtr,
                                                    barrierStatesType);
        Value barriersEqBar = createMaskedBroadcastBarrierRows(
            fb, barrierOffset, barrierLength, barriers, barriersType,
            barrierStatesType, ctaMask);
        Value one =
            tti::createConstIntTensor(fb, fb.getLoc(), 1, barrierStatesType);
        Value countMaskTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::countMask, barrierStatesType);
        Value shiftOneTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::initCountLsb, barrierStatesType);
        Value shiftNineTensor = tti::createConstIntTensor(
            fb, fb.getLoc(), BarrierBits::currentCountLsb, barrierStatesType);
        Value phase = arith::AndIOp::create(fb, states, one);
        Value initCount = arith::ShRUIOp::create(fb, states, shiftOneTensor);
        initCount = arith::AndIOp::create(fb, initCount, countMaskTensor);
        Value currentCount =
            arith::ShRUIOp::create(fb, states, shiftNineTensor);
        currentCount = arith::AndIOp::create(fb, currentCount, countMaskTensor);
        Value arriveCount =
            triton::SplatOp::create(fb, barrierStatesType, maskedCount);
        Value newCurrent = arith::SubIOp::create(fb, currentCount, arriveCount);
        Value zero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, barrierStatesType);
        Value zeroCond = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                               newCurrent, zero);
        Value phaseDelta = arith::SelectOp::create(fb, zeroCond, one, zero);
        Value newPhase = arith::XOrIOp::create(fb, phase, phaseDelta);
        Value newCurrentValue =
            arith::SelectOp::create(fb, zeroCond, initCount, newCurrent);
        Value initField = arith::ShLIOp::create(fb, initCount, shiftOneTensor);
        Value currentField =
            arith::ShLIOp::create(fb, newCurrentValue, shiftNineTensor);
        Value newState = arith::OrIOp::create(fb, newPhase, initField);
        newState = arith::OrIOp::create(fb, newState, currentField);
        Value updated =
            arith::SelectOp::create(fb, barriersEqBar, newState, states);
        tti::createStoreScratchMemory(fb, fb.getLoc(), statesPtr, updated,
                                      barrierStatesType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createSetWriteVisibilityCall(
    ImplicitLocOpBuilder &b, Value buf, uint32_t length, uint64_t threadMask,
    Value pred, MemType memType, Operation *insertPoint, Value ctaMask) {

  if (auxData.buffers[(int)memType].empty() ||
      auxData.writeVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
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
                             threadMaskVal, buffersVal, writeVisibilityVal,
                             ctaMask};
  createCallToCachedFunction(
      b, "set_write_visibility", args,
      /*assertInfo=*/std::nullopt,
      {buffersType, writeVisibilityType, (uint64_t)memType},
      [buffersType, writeVisibilityType](ImplicitLocOpBuilder &fb,
                                         Block *entryBlock) {
        Value bufOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadMaskVal = entryBlock->getArgument(3);
        Value buffers = entryBlock->getArgument(4);
        Value writeVisibilityPtr = entryBlock->getArgument(5);
        Value ctaMask = entryBlock->getArgument(6);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value writeVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
        Value buffersEqBuf = createBroadcastBufferMatch(
            fb, bufOffset, lengthVal, buffers, writeVisibilityType, ctaMask);
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

void FunctionBuilder::createSetReadVisibilityCall(
    ImplicitLocOpBuilder &b, Value buf, uint32_t length, uint64_t threadMask,
    Value pred, MemType memType, Operation *insertPoint, Value ctaMask) {

  if (auxData.buffers[(int)memType].empty() ||
      auxData.readVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
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
                             threadMaskVal, buffersVal, readVisibilityVal,
                             ctaMask};
  createCallToCachedFunction(
      b, "set_read_visibility", args,
      /*assertInfo=*/std::nullopt,
      {buffersType, readVisibilityType, (uint64_t)memType},
      [buffersType, readVisibilityType](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
        Value bufOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadMaskVal = entryBlock->getArgument(3);
        Value buffers = entryBlock->getArgument(4);
        Value readVisibilityPtr = entryBlock->getArgument(5);
        Value ctaMask = entryBlock->getArgument(6);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value readVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
        Value buffersEqBuf = createBroadcastBufferMatch(
            fb, bufOffset, lengthVal, buffers, readVisibilityType, ctaMask);
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
                                                   Operation *insertPoint,
                                                   Value ctaMask) {
  if (auxData.buffers[(int)memType].empty() ||
      auxData.writeTracking[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  Value buffersVal = auxData.buffers[(int)memType].at(insertPoint).value;
  auto buffersType = cast<RankedTensorType>(
      auxData.buffers[(int)memType].at(insertPoint).type);
  Value writeTrackingVal =
      auxData.writeTracking[(int)memType].at(insertPoint).value;
  auto writeTrackingType = cast<RankedTensorType>(
      auxData.writeTracking[(int)memType].at(insertPoint).type);
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {bufOffset,  lengthVal,        pred,
                             buffersVal, writeTrackingVal, ctaMask};
  createCallToCachedFunction(
      b, "clear_write_tracking", args,
      /*assertInfo=*/std::nullopt,
      {buffersType, writeTrackingType, (uint64_t)memType},
      [buffersType, writeTrackingType](ImplicitLocOpBuilder &fb,
                                       Block *entryBlock) {
        Value bufOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value buffers = entryBlock->getArgument(3);
        Value writeTrackingPtr = entryBlock->getArgument(4);
        Value ctaMask = entryBlock->getArgument(5);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value writeTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeTrackingPtr, writeTrackingType);
        Value buffersEqBuf = createBroadcastBufferMatch(
            fb, bufOffset, lengthVal, buffers, writeTrackingType, ctaMask);
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
                                                    Operation *insertPoint,
                                                    Value ctaMask) {
  if (auxData.buffers[(int)memType].empty() ||
      auxData.readVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  Value buffersVal = auxData.buffers[(int)memType].at(insertPoint).value;
  auto buffersType = cast<RankedTensorType>(
      auxData.buffers[(int)memType].at(insertPoint).type);
  Value readVisibilityVal =
      auxData.readVisibility[(int)memType].at(insertPoint).value;
  auto readVisibilityType = cast<RankedTensorType>(
      auxData.readVisibility[(int)memType].at(insertPoint).type);
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {bufOffset,  lengthVal,         pred,
                             buffersVal, readVisibilityVal, ctaMask};
  createCallToCachedFunction(
      b, "clear_read_visibility", args,
      /*assertInfo=*/std::nullopt,
      {buffersType, readVisibilityType, (uint64_t)memType},
      [buffersType, readVisibilityType](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
        Value bufOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value buffers = entryBlock->getArgument(3);
        Value readVisibilityPtr = entryBlock->getArgument(4);
        Value ctaMask = entryBlock->getArgument(5);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value readVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
        Value buffersEqBuf = createBroadcastBufferMatch(
            fb, bufOffset, lengthVal, buffers, readVisibilityType, ctaMask);
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
                                                  Operation *insertPoint,
                                                  Value ctaMask) {

  if (auxData.buffers[(int)memType].empty() ||
      auxData.readTracking[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  Value buffersVal = auxData.buffers[(int)memType].at(insertPoint).value;
  auto buffersType = cast<RankedTensorType>(
      auxData.buffers[(int)memType].at(insertPoint).type);
  Value readTrackingVal =
      auxData.readTracking[(int)memType].at(insertPoint).value;
  auto readTrackingType = cast<RankedTensorType>(
      auxData.readTracking[(int)memType].at(insertPoint).type);
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {bufOffset,  lengthVal,       pred,
                             buffersVal, readTrackingVal, ctaMask};
  createCallToCachedFunction(
      b, "clear_read_tracking", args,
      /*assertInfo=*/std::nullopt,
      {buffersType, readTrackingType, (uint64_t)memType},
      [buffersType, readTrackingType](ImplicitLocOpBuilder &fb,
                                      Block *entryBlock) {
        Value bufOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value buffers = entryBlock->getArgument(3);
        Value readTrackingPtr = entryBlock->getArgument(4);
        Value ctaMask = entryBlock->getArgument(5);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value readTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readTrackingPtr, readTrackingType);
        Value buffersEqBuf = createBroadcastBufferMatch(
            fb, bufOffset, lengthVal, buffers, readTrackingType, ctaMask);
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
                                                   Operation *insertPoint,
                                                   Value ctaMask) {
  if (auxData.barriers.empty() ||
      auxData.writeVisibility[(int)memType].empty() ||
      auxData.writeTracking[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
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
  SmallVector<Value> args = {mbarOffset,       lengthVal,   pred,
                             threadVal,        barriersVal, writeVisibilityVal,
                             writeTrackingVal, ctaMask};
  createCallToCachedFunction(
      b, "track_visible_writes", args,
      /*assertInfo=*/std::nullopt,
      {barriersType, writeVisibilityType, writeTrackingType, (uint64_t)memType},
      [barriersType, writeVisibilityType,
       writeTrackingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadVal = entryBlock->getArgument(3);
        Value barriers = entryBlock->getArgument(4);
        Value writeVisibilityPtr = entryBlock->getArgument(5);
        Value writeTrackingPtr = entryBlock->getArgument(6);
        Value ctaMask = entryBlock->getArgument(7);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value writeVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
        Value writeTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeTrackingPtr, writeTrackingType);
        Value barriersEqBar = createBroadcastBarrierRows(
            fb, mbarOffset, lengthVal, barriers, barriersType,
            writeTrackingType, /*barrierDim=*/2);
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
        Value ctaGroupMask =
            createCTAMaskMatch(fb, writeVisibilityType, ctaMask, /*dim=*/0);
        visibleWrites = arith::AndIOp::create(fb, visibleWrites, ctaGroupMask);
        visibleWrites =
            convertAndBroadcast(fb, visibleWrites, {0, 1}, writeTrackingType);
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
                                                  Operation *insertPoint,
                                                  Value ctaMask) {

  if (auxData.barriers.empty() ||
      auxData.readVisibility[(int)memType].empty() ||
      auxData.readTracking[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
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
  SmallVector<Value> args = {mbarOffset,      lengthVal,   pred,
                             threadVal,       barriersVal, readVisibilityVal,
                             readTrackingVal, ctaMask};
  createCallToCachedFunction(
      b, "track_visible_reads", args,
      /*assertInfo=*/std::nullopt,
      {barriersType, readVisibilityType, readTrackingType, (uint64_t)memType},
      [barriersType, readVisibilityType,
       readTrackingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadVal = entryBlock->getArgument(3);
        Value barriers = entryBlock->getArgument(4);
        Value readVisibilityPtr = entryBlock->getArgument(5);
        Value readTrackingPtr = entryBlock->getArgument(6);
        Value ctaMask = entryBlock->getArgument(7);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value readVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
        Value readTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readTrackingPtr, readTrackingType);
        Value barriersEqBar = createBroadcastBarrierRows(
            fb, mbarOffset, lengthVal, barriers, barriersType, readTrackingType,
            /*barrierDim=*/2);
        Value threadColumnMask =
            createColumnMask(fb, threadVal, readVisibilityType);
        Value readVisibilityZero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, readVisibilityType);
        Value ctaGroupMask =
            createCTAMaskMatch(fb, readVisibilityType, ctaMask, /*dim=*/0);
        Value visibleReads = arith::SelectOp::create(
            fb, threadColumnMask, readVisibility, readVisibilityZero);
        visibleReads = arith::SelectOp::create(fb, ctaGroupMask, visibleReads,
                                               readVisibilityZero);
        visibleReads = reduceLastDim<arith::OrIOp>(fb, visibleReads);
        visibleReads =
            convertAndBroadcast(fb, visibleReads, {0, 1}, readTrackingType);
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

void FunctionBuilder::createClearBarrierWriteTrackingCall(
    ImplicitLocOpBuilder &b, Value mbar, Value pred, MemType memType,
    Operation *insertPoint, Value ctaMask) {
  if (auxData.writeTracking[(int)memType].empty()) {
    return;
  }
  assert(!auxData.barriers.empty() &&
         "barrier descriptors must exist when clearing barrier write tracking");
  if (!pred) {
    pred = arith::ConstantIntOp::create(b, 1, 1);
  }
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value writeTrackingVal =
      auxData.writeTracking[(int)memType].at(insertPoint).value;
  auto writeTrackingType = cast<RankedTensorType>(
      auxData.writeTracking[(int)memType].at(insertPoint).type);
  uint32_t length = getMemDescLength(mbar);
  Value mbarOffset = tti::ExperimentalMemDescToI32Op::create(b, mbar);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {mbarOffset,  lengthVal,        pred,
                             barriersVal, writeTrackingVal, ctaMask};
  createCallToCachedFunction(
      b, "clear_barrier_write_tracking", args,
      /*assertInfo=*/std::nullopt,
      {barriersType, writeTrackingType, (uint64_t)memType},
      [barriersType, writeTrackingType](ImplicitLocOpBuilder &fb,
                                        Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value barriers = entryBlock->getArgument(3);
        Value writeTrackingPtr = entryBlock->getArgument(4);
        Value ctaMask = entryBlock->getArgument(5);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value writeTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeTrackingPtr, writeTrackingType);
        Value barriersEqBar = createMaskedBroadcastBarrierRows(
            fb, mbarOffset, lengthVal, barriers, barriersType,
            writeTrackingType, ctaMask);
        Value zero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, writeTrackingType);
        Value updated =
            arith::SelectOp::create(fb, barriersEqBar, zero, writeTracking);
        tti::createStoreScratchMemory(fb, fb.getLoc(), writeTrackingPtr,
                                      updated, writeTrackingType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createClearBarrierReadTrackingCall(
    ImplicitLocOpBuilder &b, Value mbar, Value pred, MemType memType,
    Operation *insertPoint, Value ctaMask) {
  if (auxData.readTracking[(int)memType].empty()) {
    return;
  }
  assert(!auxData.barriers.empty() &&
         "barrier descriptors must exist when clearing barrier read tracking");
  if (!pred) {
    pred = arith::ConstantIntOp::create(b, 1, 1);
  }
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  Value barriersVal = auxData.barriers.at(insertPoint).value;
  auto barriersType =
      cast<RankedTensorType>(auxData.barriers.at(insertPoint).type);
  Value readTrackingVal =
      auxData.readTracking[(int)memType].at(insertPoint).value;
  auto readTrackingType = cast<RankedTensorType>(
      auxData.readTracking[(int)memType].at(insertPoint).type);
  uint32_t length = getMemDescLength(mbar);
  Value mbarOffset = tti::ExperimentalMemDescToI32Op::create(b, mbar);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {mbarOffset,  lengthVal,       pred,
                             barriersVal, readTrackingVal, ctaMask};
  createCallToCachedFunction(
      b, "clear_barrier_read_tracking", args,
      /*assertInfo=*/std::nullopt,
      {barriersType, readTrackingType, (uint64_t)memType},
      [barriersType, readTrackingType](ImplicitLocOpBuilder &fb,
                                       Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value barriers = entryBlock->getArgument(3);
        Value readTrackingPtr = entryBlock->getArgument(4);
        Value ctaMask = entryBlock->getArgument(5);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value readTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readTrackingPtr, readTrackingType);
        Value barriersEqBar = createMaskedBroadcastBarrierRows(
            fb, mbarOffset, lengthVal, barriers, barriersType, readTrackingType,
            ctaMask);
        Value zero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, readTrackingType);
        Value updated =
            arith::SelectOp::create(fb, barriersEqBar, zero, readTracking);
        tti::createStoreScratchMemory(fb, fb.getLoc(), readTrackingPtr, updated,
                                      readTrackingType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createTransferVisibleWritesCall(
    ImplicitLocOpBuilder &b, Value mbar, uint64_t threadMask, Value pred,
    MemType memType, Operation *insertPoint, Value ctaMask) {

  if (auxData.barriers.empty() ||
      auxData.writeVisibility[(int)memType].empty() ||
      auxData.writeTracking[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
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
  SmallVector<Value> args = {mbarOffset,       lengthVal,   pred,
                             threadMaskVal,    barriersVal, writeVisibilityVal,
                             writeTrackingVal, ctaMask};
  createCallToCachedFunction(
      b, "transfer_visible_writes", args,
      /*assertInfo=*/std::nullopt,
      {barriersType, writeVisibilityType, writeTrackingType, (uint64_t)memType},
      [barriersType, writeVisibilityType,
       writeTrackingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadMaskVal = entryBlock->getArgument(3);
        Value barriers = entryBlock->getArgument(4);
        Value writeVisibilityPtr = entryBlock->getArgument(5);
        Value writeTrackingPtr = entryBlock->getArgument(6);
        Value ctaMask = entryBlock->getArgument(7);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value writeVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
        Value writeTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeTrackingPtr, writeTrackingType);
        Value barriersEqBar = createBroadcastBarrierRows(
            fb, mbarOffset, lengthVal, barriers, barriersType,
            writeTrackingType, /*barrierDim=*/2);
        Value zeroTracking =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, writeTrackingType);
        Value trackingBuffers = arith::SelectOp::create(
            fb, barriersEqBar, writeTracking, zeroTracking);
        trackingBuffers = reduceLastDim<arith::OrIOp>(fb, trackingBuffers);
        trackingBuffers = createConvertLayout(
            fb, trackingBuffers, writeVisibilityType.getEncoding());
        auto trackingBuffersType =
            cast<RankedTensorType>(trackingBuffers.getType());
        Value ctaGroupMask =
            createCTAMaskMatch(fb, trackingBuffersType, ctaMask, /*dim=*/0);
        Value zeroTrackingBuffers =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, trackingBuffersType);
        trackingBuffers = arith::SelectOp::create(
            fb, ctaGroupMask, trackingBuffers, zeroTrackingBuffers);
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
    MemType memType, Operation *insertPoint, Value ctaMask) {

  if (auxData.barriers.empty() ||
      auxData.readVisibility[(int)memType].empty() ||
      auxData.readTracking[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
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
  SmallVector<Value> args = {mbarOffset,      lengthVal,   pred,
                             threadMaskVal,   barriersVal, readVisibilityVal,
                             readTrackingVal, ctaMask};
  createCallToCachedFunction(
      b, "transfer_visible_reads", args,
      /*assertInfo=*/std::nullopt,
      {barriersType, readVisibilityType, readTrackingType, (uint64_t)memType},
      [barriersType, readVisibilityType,
       readTrackingType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value mbarOffset = entryBlock->getArgument(0);
        Value lengthVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value threadMaskVal = entryBlock->getArgument(3);
        Value barriers = entryBlock->getArgument(4);
        Value readVisibilityPtr = entryBlock->getArgument(5);
        Value readTrackingPtr = entryBlock->getArgument(6);
        Value ctaMask = entryBlock->getArgument(7);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value readVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
        Value readTracking = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readTrackingPtr, readTrackingType);
        Value barriersEqBar = createBroadcastBarrierRows(
            fb, mbarOffset, lengthVal, barriers, barriersType, readTrackingType,
            /*barrierDim=*/2);
        Value readTrackingZero =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, readTrackingType);
        Value trackingBar = arith::SelectOp::create(
            fb, barriersEqBar, readTracking, readTrackingZero);
        trackingBar = reduceLastDim<arith::OrIOp>(fb, trackingBar);
        auto trackingBarType = cast<RankedTensorType>(trackingBar.getType());
        Value ctaGroupMask =
            createCTAMaskMatch(fb, trackingBarType, ctaMask, /*dim=*/0);
        Value zeroTrackingBar =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, trackingBarType);
        trackingBar = arith::SelectOp::create(fb, ctaGroupMask, trackingBar,
                                              zeroTrackingBar);
        trackingBar =
            convertAndBroadcast(fb, trackingBar, {0, 1}, readVisibilityType);
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
    StringRef operandName, Value pred, MemType memType, Operation *insertPoint,
    Value ctaMask) {
  if (auxData.buffers[(int)memType].empty() ||
      auxData.writeVisibility[(int)memType].empty() ||
      (auxData.hasNonTrivialAliasing[(int)memType] &&
       auxData.aliasMatrices[(int)memType].empty())) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value buffersVal = auxData.buffers[(int)memType].at(insertPoint).value;
  auto buffersType = cast<RankedTensorType>(
      auxData.buffers[(int)memType].at(insertPoint).type);
  Value writeVisibilityVal =
      auxData.writeVisibility[(int)memType].at(insertPoint).value;
  auto writeVisibilityType = cast<RankedTensorType>(
      auxData.writeVisibility[(int)memType].at(insertPoint).type);
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  std::string message = "Buffer being accessed has outstanding writes.";
  if (!operandName.empty())
    message += " Operand: " + operandName.str();
  AssertInfo assertInfo{
      message, writeVisibilityType.cloneWith(std::nullopt, b.getI1Type())};
  Type aliasMatrixTypeBase;
  auto buildVerifyWriteBody = [&buffersType, &writeVisibilityType,
                               &aliasMatrixTypeBase](bool useAlias) {
    return [=](ImplicitLocOpBuilder &fb, Block *entryBlock) {
      Value bufOffset = entryBlock->getArgument(0);
      Value lengthVal = entryBlock->getArgument(1);
      Value pred = entryBlock->getArgument(2);
      Value threadVal = entryBlock->getArgument(3);
      Value buffers = entryBlock->getArgument(4);
      Value writeVisibilityPtr = entryBlock->getArgument(5);
      Value ctaMask = entryBlock->getArgument(6);
      Value aliasMatrix = useAlias ? entryBlock->getArgument(7) : Value();

      Value writeVisibility = tti::createLoadScratchMemory(
          fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
      Value buffersEqBuf =
          createBufferMask1D(fb, bufOffset, lengthVal, buffers);
      if (useAlias) {
        buffersEqBuf =
            expandAliases(fb, buffersEqBuf, aliasMatrix,
                          cast<RankedTensorType>(aliasMatrixTypeBase));
      }
      buffersEqBuf = convertAndBroadcast(
          fb, buffersEqBuf, {1},
          cast<RankedTensorType>(
              writeVisibilityType.cloneWith(std::nullopt, fb.getI1Type())));
      buffersEqBuf = arith::AndIOp::create(
          fb, buffersEqBuf,
          createCTAMaskMatch(fb, writeVisibilityType, ctaMask, /*dim=*/0));
      Value writeVisibilityZero =
          tti::createConstIntTensor(fb, fb.getLoc(), 0, writeVisibilityType);
      Value bufVisibility = arith::SelectOp::create(
          fb, buffersEqBuf, writeVisibility, writeVisibilityZero);
      Value noOneIsWriting = arith::CmpIOp::create(
          fb, arith::CmpIPredicate::eq, bufVisibility, writeVisibilityZero);
      Value threadI64 = arith::ExtUIOp::create(fb, fb.getI64Type(), threadVal);
      Value bufferThreadBitScalar = arith::ShLIOp::create(
          fb, arith::ConstantIntOp::create(fb, 1, 64), threadI64);
      Value bufferThreadBit = triton::SplatOp::create(fb, writeVisibilityType,
                                                      bufferThreadBitScalar);
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
    };
  };
  if (auxData.hasNonTrivialAliasing[(int)memType]) {
    Value aliasMatrixVal =
        auxData.aliasMatrices[(int)memType].at(insertPoint).value;
    aliasMatrixTypeBase =
        auxData.aliasMatrices[(int)memType].at(insertPoint).type;
    auto aliasMatrixType = cast<RankedTensorType>(aliasMatrixTypeBase);
    SmallVector<Value> args = {bufOffset, lengthVal,     pred,
                               threadVal, buffersVal,    writeVisibilityVal,
                               ctaMask,   aliasMatrixVal};
    createCallToCachedFunction(
        b, "verify_write_visibility", args, assertInfo,
        {buffersType, writeVisibilityType, aliasMatrixType, (uint64_t)memType},
        buildVerifyWriteBody(/*useAlias=*/true));
  } else {
    SmallVector<Value> args = {bufOffset, lengthVal,  pred,
                               threadVal, buffersVal, writeVisibilityVal,
                               ctaMask};
    createCallToCachedFunction(
        b, "verify_write_visibility_noalias", args, assertInfo,
        {buffersType, writeVisibilityType, (uint64_t)memType},
        buildVerifyWriteBody(/*useAlias=*/false));
  }
}

void FunctionBuilder::createVerifyReadVisibilityCall(
    ImplicitLocOpBuilder &b, Value buf, uint32_t length, int thread,
    StringRef operandName, Value pred, MemType memType, Operation *insertPoint,
    Value ctaMask) {
  if (auxData.buffers[(int)memType].empty() ||
      auxData.readVisibility[(int)memType].empty() ||
      (auxData.hasNonTrivialAliasing[(int)memType] &&
       auxData.aliasMatrices[(int)memType].empty())) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value buffersVal = auxData.buffers[(int)memType].at(insertPoint).value;
  auto buffersType = cast<RankedTensorType>(
      auxData.buffers[(int)memType].at(insertPoint).type);
  Value readVisibilityVal =
      auxData.readVisibility[(int)memType].at(insertPoint).value;
  auto readVisibilityType = cast<RankedTensorType>(
      auxData.readVisibility[(int)memType].at(insertPoint).type);
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  std::string message = "Buffer being accessed has outstanding reads";
  if (!operandName.empty())
    message += ". Operand: " + operandName.str();
  AssertInfo assertInfo{
      message, readVisibilityType.cloneWith(std::nullopt, b.getI1Type())};
  Type aliasMatrixTypeBase;
  auto buildVerifyReadBody = [&readVisibilityType,
                              &aliasMatrixTypeBase](bool useAlias) {
    return [=](ImplicitLocOpBuilder &fb, Block *entryBlock) {
      Value bufOffset = entryBlock->getArgument(0);
      Value lengthVal = entryBlock->getArgument(1);
      Value pred = entryBlock->getArgument(2);
      Value threadVal = entryBlock->getArgument(3);
      Value buffers = entryBlock->getArgument(4);
      Value readVisibilityPtr = entryBlock->getArgument(5);
      Value ctaMask = entryBlock->getArgument(6);
      Value aliasMatrix = useAlias ? entryBlock->getArgument(7) : Value();

      Value readVisibility = tti::createLoadScratchMemory(
          fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
      Value buffersEqBuf =
          createBufferMask1D(fb, bufOffset, lengthVal, buffers);
      if (useAlias) {
        buffersEqBuf =
            expandAliases(fb, buffersEqBuf, aliasMatrix,
                          cast<RankedTensorType>(aliasMatrixTypeBase));
      }
      buffersEqBuf = convertAndBroadcast(
          fb, buffersEqBuf, {1},
          cast<RankedTensorType>(
              readVisibilityType.cloneWith(std::nullopt, fb.getI1Type())));
      buffersEqBuf = arith::AndIOp::create(
          fb, buffersEqBuf,
          createCTAMaskMatch(fb, readVisibilityType, ctaMask, /*dim=*/0));
      Value readVisibilityZero =
          tti::createConstIntTensor(fb, fb.getLoc(), 0, readVisibilityType);
      Value bufVisibility = arith::SelectOp::create(
          fb, buffersEqBuf, readVisibility, readVisibilityZero);
      Value totalVisibility = reduceLastDim<arith::OrIOp>(fb, bufVisibility);
      Value threadColumnMask =
          createColumnMask(fb, threadVal, readVisibilityType);
      Value bufThreadVisibility = arith::SelectOp::create(
          fb, threadColumnMask, bufVisibility, readVisibilityZero);
      bufThreadVisibility =
          reduceLastDim<arith::OrIOp>(fb, bufThreadVisibility);
      Value threadAndTotalVisibility =
          arith::AndIOp::create(fb, bufThreadVisibility, totalVisibility);
      Value hasVisibility =
          arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                threadAndTotalVisibility, totalVisibility);
      auto resultType = cast<RankedTensorType>(
          readVisibilityType.cloneWith(std::nullopt, fb.getI1Type()));
      hasVisibility =
          convertAndBroadcast(fb, hasVisibility, {0, 1}, resultType);
      Value vTrue = tti::createConstIntTensor(
          fb, fb.getLoc(), 1, cast<RankedTensorType>(hasVisibility.getType()));
      Value predicatedHasVisibility =
          arith::SelectOp::create(fb, pred, hasVisibility, vTrue);
      triton::ReturnOp::create(fb, predicatedHasVisibility);
    };
  };
  if (auxData.hasNonTrivialAliasing[(int)memType]) {
    Value aliasMatrixVal =
        auxData.aliasMatrices[(int)memType].at(insertPoint).value;
    aliasMatrixTypeBase =
        auxData.aliasMatrices[(int)memType].at(insertPoint).type;
    auto aliasMatrixType = cast<RankedTensorType>(aliasMatrixTypeBase);
    SmallVector<Value> args = {bufOffset, lengthVal,     pred,
                               threadVal, buffersVal,    readVisibilityVal,
                               ctaMask,   aliasMatrixVal};
    createCallToCachedFunction(
        b, "verify_read_visibility", args, assertInfo,
        {buffersType, readVisibilityType, aliasMatrixType, (uint64_t)memType},
        buildVerifyReadBody(/*useAlias=*/true));
  } else {
    SmallVector<Value> args = {bufOffset,  lengthVal,         pred,   threadVal,
                               buffersVal, readVisibilityVal, ctaMask};
    createCallToCachedFunction(
        b, "verify_read_visibility_noalias", args, assertInfo,
        {buffersType, readVisibilityType, (uint64_t)memType},
        buildVerifyReadBody(/*useAlias=*/false));
  }
}

void FunctionBuilder::createCopyWriteVisibilityCall(
    ImplicitLocOpBuilder &b, int sourceThread, uint64_t destMask, Value pred,
    MemType memType, Operation *insertPoint, Value ctaMask) {

  if (auxData.writeVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  auto writeVis = auxData.writeVisibility[(int)memType].at(insertPoint);
  auto writeVisibilityType = cast<RankedTensorType>(writeVis.type);
  Value sourceThreadVal = arith::ConstantIntOp::create(b, sourceThread, 32);
  Value destMaskVal = arith::ConstantIntOp::create(b, destMask, 64);
  SmallVector<Value> args = {sourceThreadVal, destMaskVal, pred, writeVis.value,
                             ctaMask};
  createCallToCachedFunction(
      b, "copy_write_visibility", args,
      /*assertInfo=*/std::nullopt, {writeVisibilityType, (uint64_t)memType},
      [writeVisibilityType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value sourceThread = entryBlock->getArgument(0);
        Value destMaskVal = entryBlock->getArgument(1);
        Value pred = entryBlock->getArgument(2);
        Value writeVisibilityPtr = entryBlock->getArgument(3);
        Value ctaMask = entryBlock->getArgument(4);

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value writeVisibility = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
        auto elemType = cast<IntegerType>(writeVisibilityType.getElementType());
        Value zeroTensor =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, writeVisibilityType);

        constexpr uint64_t fullMask = getFullThreadBitmask();
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
        Value rowMask =
            createCTAMaskMatch(fb, writeVisibilityType, ctaMask, /*dim=*/0);
        updated =
            arith::SelectOp::create(fb, rowMask, updated, writeVisibility);
        tti::createStoreScratchMemory(fb, fb.getLoc(), writeVisibilityPtr,
                                      updated, writeVisibilityType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createCopyReadVisibilityCall(
    ImplicitLocOpBuilder &b, int sourceThread, uint64_t destMask, Value pred,
    MemType memType, Operation *insertPoint, Value ctaMask) {

  if (auxData.readVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  auto readVis = auxData.readVisibility[(int)memType].at(insertPoint);
  auto readVisibilityType = cast<RankedTensorType>(readVis.type);
  Value sourceThreadVal = arith::ConstantIntOp::create(b, sourceThread, 32);
  SmallVector<Value> args = {sourceThreadVal,
                             arith::ConstantIntOp::create(b, destMask, 64),
                             pred, readVis.value, ctaMask};
  createCallToCachedFunction(
      b, "copy_read_visibility", args,
      /*assertInfo=*/std::nullopt,
      {readVisibilityType, destMask, (uint64_t)memType},
      [readVisibilityType, destMask](ImplicitLocOpBuilder &fb,
                                     Block *entryBlock) {
        Value sourceThread = entryBlock->getArgument(0);
        /*Value destMaskVal = entryBlock->getArgument(1);*/
        Value pred = entryBlock->getArgument(2);
        Value readVisibilityPtr = entryBlock->getArgument(3);
        Value ctaMask = entryBlock->getArgument(4);

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
        Value sourceVector = reduceLastDim<arith::OrIOp>(fb, sourceColumn);
        Value broadcastRow =
            convertAndBroadcast(fb, sourceVector, {0, 1}, readVisibilityType);
        Value replicated = arith::SelectOp::create(fb, destMaskTensor,
                                                   broadcastRow, zeroTensor);

        Value updated = arith::OrIOp::create(fb, cleared, replicated);
        Value rowMask =
            createCTAMaskMatch(fb, readVisibilityType, ctaMask, /*dim=*/0);
        updated = arith::SelectOp::create(fb, rowMask, updated, readVisibility);
        tti::createStoreScratchMemory(fb, fb.getLoc(), readVisibilityPtr,
                                      updated, readVisibilityType);

        fb.setInsertionPointToEnd(thenBlock);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createStageAccessForCommitCall(
    ImplicitLocOpBuilder &b, Value buf, uint32_t length, int thread, Value pred,
    MemType memType, CommitKind::Kind commitKind, Operation *insertPoint,
    Value ctaMask) {
  if (auxData.buffers[(int)memType].empty() ||
      auxData.commits[commitKind].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  ValueType buffers = auxData.buffers[(int)memType].at(insertPoint);
  ValueType outstandingCommits = auxData.commits[commitKind].at(insertPoint);
  auto buffersType = cast<RankedTensorType>(buffers.type);
  auto commitsType = cast<RankedTensorType>(outstandingCommits.type);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  SmallVector<Value> args = {bufOffset, lengthVal,     pred,
                             threadVal, buffers.value, outstandingCommits.value,
                             ctaMask};
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
        Value ctaMask = entryBlock->getArgument(6);

        (void)threadVal;

        auto [prevBlock, ifBlock, thenBlock] = createIfBlock(fb, pred);
        fb.setInsertionPointToStart(ifBlock);

        Value commits = tti::createLoadScratchMemory(
            fb, fb.getLoc(), outstandingCommitsPtr, commitsType);
        Value buffersEqBuf = createBroadcastBufferMatch(
            fb, bufOffset, lengthVal, buffers, commitsType, ctaMask);
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
                                               Operation *insertPoint,
                                               Value ctaMask) {
  if (auxData.commits[commitKind].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  ValueType outstandingCommits = auxData.commits[commitKind].at(insertPoint);
  auto commitsType = cast<RankedTensorType>(outstandingCommits.type);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  SmallVector<Value> args = {threadVal, pred, outstandingCommits.value,
                             ctaMask};
  createCallToCachedFunction(
      b, "commit_accesses", args,
      /*assertInfo=*/std::nullopt, {commitsType},
      [commitsType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value threadVal = entryBlock->getArgument(0);
        Value pred = entryBlock->getArgument(1);
        Value outstandingCommitsPtr = entryBlock->getArgument(2);
        Value ctaMask = entryBlock->getArgument(3);

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
        Value rowMask = createCTAMaskMatch(fb, commitsType, ctaMask, /*dim=*/0);
        auto commitsGtZero = createCmpIntTensorScalar(
            fb, commits, zero, arith::CmpIPredicate::sgt);
        commitsGtZero = arith::AndIOp::create(fb, commitsGtZero, threadMask);
        commitsGtZero = arith::AndIOp::create(fb, commitsGtZero, rowMask);
        Value commitsPlusOne = arith::AddIOp::create(fb, commits, ones);
        commits =
            arith::SelectOp::create(fb, commitsGtZero, commitsPlusOne, commits);

        auto commitsEqMinusOne = createCmpIntTensorScalar(
            fb, commits, minusOne, arith::CmpIPredicate::eq);
        commitsEqMinusOne =
            arith::AndIOp::create(fb, commitsEqMinusOne, threadMask);
        commitsEqMinusOne =
            arith::AndIOp::create(fb, commitsEqMinusOne, rowMask);
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
    MemType memType, Operation *insertPoint, Value ctaMask) {
  if (auxData.commits[commitKind].empty() ||
      auxData.writeVisibility[(int)memType].empty()) {
    return;
  }
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
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
      pred,      outstandingCommits.value, writeVisibility.value,
      ctaMask};
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
        Value ctaMask = entryBlock->getArgument(6);

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
        Value rowMask = createCTAMaskMatch(fb, commitsType, ctaMask, /*dim=*/0);
        auto outstandingCommitsGtOutstandingNum =
            createCmpIntTensorScalar(fb, outstandingCommits, outstandingNumElem,
                                     arith::CmpIPredicate::sgt);
        outstandingCommitsGtOutstandingNum = arith::AndIOp::create(
            fb, outstandingCommitsGtOutstandingNum, threadColumnMask);
        outstandingCommitsGtOutstandingNum = arith::AndIOp::create(
            fb, outstandingCommitsGtOutstandingNum, rowMask);

        Value selectedRows =
            reduceLastDim<arith::OrIOp>(fb, outstandingCommitsGtOutstandingNum);
        selectedRows = createConvertLayout(fb, selectedRows,
                                           writeVisibilityType.getEncoding());
        Value transferMaskElem = adjustIntegerWidth(
            fb, transferMaskVal,
            cast<IntegerType>(writeVisibilityType.getElementType()));
        Value transferMaskTensor =
            triton::SplatOp::create(fb, writeVisibilityType, transferMaskElem);
        Value writeVisibilityOrThreadBit =
            arith::OrIOp::create(fb, writeVisibility, transferMaskTensor);
        Value writeVisibilityUpdated = arith::SelectOp::create(
            fb, selectedRows, writeVisibilityOrThreadBit, writeVisibility);
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
    MemType memType, Operation *insertPoint, Value ctaMask) {
  if (auxData.commits[commitKind].empty() ||
      auxData.readVisibility[(int)memType].empty()) {
    return;
  }
  ValueType outstandingCommits = auxData.commits[commitKind].at(insertPoint);
  ValueType readVisibility =
      auxData.readVisibility[(int)memType].at(insertPoint);
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  auto commitsType = cast<RankedTensorType>(outstandingCommits.type);
  auto readVisibilityType = cast<RankedTensorType>(readVisibility.type);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value transferMaskVal =
      arith::ConstantIntOp::create(b, transferThreadMask, 64);
  Value outstandingNumVal = arith::ConstantIntOp::create(b, outstandingNum, 32);
  SmallVector<Value> args = {
      threadVal, transferMaskVal,          outstandingNumVal,
      pred,      outstandingCommits.value, readVisibility.value,
      ctaMask};
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
        Value ctaMask = entryBlock->getArgument(6);

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
        Value rowMask = createCTAMaskMatch(fb, commitsType, ctaMask, /*dim=*/0);
        auto outstandingCommitsGtOutstandingNum =
            createCmpIntTensorScalar(fb, outstandingCommits, outstandingNumElem,
                                     arith::CmpIPredicate::sgt);
        outstandingCommitsGtOutstandingNum = arith::AndIOp::create(
            fb, outstandingCommitsGtOutstandingNum, threadColumnMask);
        outstandingCommitsGtOutstandingNum = arith::AndIOp::create(
            fb, outstandingCommitsGtOutstandingNum, rowMask);

        Value selectedRows =
            reduceLastDim<arith::OrIOp>(fb, outstandingCommitsGtOutstandingNum);
        selectedRows =
            convertAndBroadcast(fb, selectedRows, {0, 1}, readVisibilityType);
        Value transferMaskElem = adjustIntegerWidth(
            fb, transferMaskVal,
            cast<IntegerType>(readVisibilityType.getElementType()));
        Value transferMaskTensor =
            triton::SplatOp::create(fb, readVisibilityType, transferMaskElem);
        Value readVisibilityOrThreadBit =
            arith::OrIOp::create(fb, readVisibility, transferMaskTensor);
        Value readVisibilityUpdated = arith::SelectOp::create(
            fb, selectedRows, readVisibilityOrThreadBit, readVisibility);
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
    CommitKind::Kind commitKind, Operation *insertPoint, Value ctaMask) {
  if (auxData.buffers[(int)memType].empty() ||
      auxData.commits[commitKind].empty() ||
      (auxData.hasNonTrivialAliasing[(int)memType] &&
       auxData.aliasMatrices[(int)memType].empty())) {
    return;
  }
  ValueType buffers = auxData.buffers[(int)memType].at(insertPoint);
  ValueType outstandingCommits = auxData.commits[commitKind].at(insertPoint);
  assert(thread < NUM_THREADS &&
         "Commit-count tracking must operate on base threads");
  Value bufOffset = tti::ExperimentalMemDescToI32Op::create(b, buf);
  if (!pred)
    pred = arith::ConstantIntOp::create(b, 1, 1);
  if (!ctaMask)
    ctaMask = createCurrentCTAMask(b);
  auto buffersType = cast<RankedTensorType>(buffers.type);
  auto commitsType = cast<RankedTensorType>(outstandingCommits.type);
  Value threadVal = arith::ConstantIntOp::create(b, thread, 32);
  Value lengthVal = arith::ConstantIntOp::create(b, length, 32);
  std::string message =
      "Accessing buffer with pending access. Pending access type: " +
      pendingAccessType.str();
  AssertInfo assertInfo{message,
                        commitsType.cloneWith(std::nullopt, b.getI1Type())};
  Type aliasMatrixTypeBase;
  auto buildCheckOutstandingCommitsBody = [&buffersType, &commitsType,
                                           &aliasMatrixTypeBase](
                                              bool useAlias) {
    return [=](ImplicitLocOpBuilder &fb, Block *entryBlock) {
      Value bufOffset = entryBlock->getArgument(0);
      Value lengthVal = entryBlock->getArgument(1);
      Value pred = entryBlock->getArgument(2);
      Value threadVal = entryBlock->getArgument(3);
      Value buffers = entryBlock->getArgument(4);
      Value outstandingCommitsPtr = entryBlock->getArgument(5);
      Value ctaMask = entryBlock->getArgument(6);
      Value aliasMatrix = useAlias ? entryBlock->getArgument(7) : Value();

      Value outstandingCommits = tti::createLoadScratchMemory(
          fb, fb.getLoc(), outstandingCommitsPtr, commitsType);
      Value buffersEqBuf =
          createBufferMask1D(fb, bufOffset, lengthVal, buffers);
      if (useAlias) {
        buffersEqBuf =
            expandAliases(fb, buffersEqBuf, aliasMatrix,
                          cast<RankedTensorType>(aliasMatrixTypeBase));
      }
      buffersEqBuf =
          convertAndBroadcast(fb, buffersEqBuf, {1},
                              cast<RankedTensorType>(commitsType.cloneWith(
                                  std::nullopt, fb.getI1Type())));
      buffersEqBuf = arith::AndIOp::create(
          fb, buffersEqBuf,
          createCTAMaskMatch(fb, commitsType, ctaMask, /*dim=*/0));
      Value zeroTensor =
          tti::createConstIntTensor(fb, fb.getLoc(), 0, commitsType);
      Value selectedRows = arith::SelectOp::create(
          fb, buffersEqBuf, outstandingCommits, zeroTensor);
      Value selectedEqZero = arith::CmpIOp::create(fb, arith::CmpIPredicate::eq,
                                                   selectedRows, zeroTensor);
      Value vTrue = tti::createConstIntTensor(
          fb, fb.getLoc(), 1, cast<RankedTensorType>(selectedEqZero.getType()));
      Value predicatedSelectedEqZero =
          arith::SelectOp::create(fb, pred, selectedEqZero, vTrue);

      triton::ReturnOp::create(fb, predicatedSelectedEqZero);
    };
  };
  if (auxData.hasNonTrivialAliasing[(int)memType]) {
    ValueType aliasMatrix = auxData.aliasMatrices[(int)memType].at(insertPoint);
    aliasMatrixTypeBase = aliasMatrix.type;
    auto aliasMatrixType = cast<RankedTensorType>(aliasMatrixTypeBase);
    SmallVector<Value> args = {bufOffset,     lengthVal,
                               pred,          threadVal,
                               buffers.value, outstandingCommits.value,
                               ctaMask,       aliasMatrix.value};
    createCallToCachedFunction(
        b, "check_outstanding_commits", args, assertInfo,
        {buffersType, commitsType, aliasMatrixType, (uint64_t)thread},
        buildCheckOutstandingCommitsBody(/*useAlias=*/true));
  } else {
    SmallVector<Value> args = {bufOffset,     lengthVal,
                               pred,          threadVal,
                               buffers.value, outstandingCommits.value,
                               ctaMask};
    createCallToCachedFunction(
        b, "check_outstanding_commits_noalias", args, assertInfo,
        {buffersType, commitsType, (uint64_t)thread},
        buildCheckOutstandingCommitsBody(/*useAlias=*/false));
  }
}

void FunctionBuilder::createClusterSyncWritesCall(ImplicitLocOpBuilder &b,
                                                  MemType memType,
                                                  Operation *insertPoint) {
  if (auxData.writeVisibility[(int)memType].empty())
    return;

  ValueType writeVisibility =
      auxData.writeVisibility[(int)memType].at(insertPoint);
  auto writeVisibilityType = cast<RankedTensorType>(writeVisibility.type);
  createCallToCachedFunction(
      b, "cluster_sync_writes", {writeVisibility.value},
      /*assertInfo=*/std::nullopt, {writeVisibilityType, (uint64_t)memType},
      [writeVisibilityType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value writeVisibilityPtr = entryBlock->getArgument(0);
        Value writeVisibilityTensor = tti::createLoadScratchMemory(
            fb, fb.getLoc(), writeVisibilityPtr, writeVisibilityType);
        auto elemType = cast<IntegerType>(writeVisibilityType.getElementType());
        constexpr uint64_t fullMask = getFullThreadBitmask();
        Value fullMaskVal = arith::ConstantIntOp::create(fb, fullMask, 64);
        Value fullMaskElem = adjustIntegerWidth(fb, fullMaskVal, elemType);
        Value fullMaskTensor =
            triton::SplatOp::create(fb, writeVisibilityType, fullMaskElem);
        Value zeroTensor =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, writeVisibilityType);
        Value hasWrites = arith::CmpIOp::create(
            fb, arith::CmpIPredicate::ne, writeVisibilityTensor, zeroTensor);
        Value updated = arith::SelectOp::create(fb, hasWrites, fullMaskTensor,
                                                writeVisibilityTensor);
        tti::createStoreScratchMemory(fb, fb.getLoc(), writeVisibilityPtr,
                                      updated, writeVisibilityType);
        triton::ReturnOp::create(fb);
      });
}

void FunctionBuilder::createClusterSyncReadsCall(ImplicitLocOpBuilder &b,
                                                 MemType memType,
                                                 Operation *insertPoint) {
  if (auxData.readVisibility[(int)memType].empty())
    return;

  ValueType readVisibility =
      auxData.readVisibility[(int)memType].at(insertPoint);
  auto readVisibilityType = cast<RankedTensorType>(readVisibility.type);
  createCallToCachedFunction(
      b, "cluster_sync_reads", {readVisibility.value},
      /*assertInfo=*/std::nullopt, {readVisibilityType, (uint64_t)memType},
      [readVisibilityType](ImplicitLocOpBuilder &fb, Block *entryBlock) {
        Value readVisibilityPtr = entryBlock->getArgument(0);
        Value readVisibilityTensor = tti::createLoadScratchMemory(
            fb, fb.getLoc(), readVisibilityPtr, readVisibilityType);
        auto elemType = cast<IntegerType>(readVisibilityType.getElementType());
        constexpr uint64_t fullMask = getFullThreadBitmask();
        Value fullMaskVal = arith::ConstantIntOp::create(fb, fullMask, 64);
        Value fullMaskElem = adjustIntegerWidth(fb, fullMaskVal, elemType);
        Value fullMaskTensor =
            triton::SplatOp::create(fb, readVisibilityType, fullMaskElem);
        Value zeroTensor =
            tti::createConstIntTensor(fb, fb.getLoc(), 0, readVisibilityType);
        Value hasReads = arith::CmpIOp::create(
            fb, arith::CmpIPredicate::ne, readVisibilityTensor, zeroTensor);
        Value rowMask = reduceLastDim<arith::OrIOp>(fb, hasReads);
        rowMask = convertAndBroadcast(fb, rowMask, {0, 1}, readVisibilityType);
        Value updated = arith::SelectOp::create(fb, rowMask, fullMaskTensor,
                                                readVisibilityTensor);
        tti::createStoreScratchMemory(fb, fb.getLoc(), readVisibilityPtr,
                                      updated, readVisibilityType);
        triton::ReturnOp::create(fb);
      });
}

} // namespace mlir::triton::instrument
