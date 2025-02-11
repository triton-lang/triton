#include "Dialect/NVGPU/IR/Dialect.h"
#include "DotOpToLLVM/MMAHelpers.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// The maximum number of registers that can be accessed by a single message
// regardless of shape or repetitions
static constexpr int largestTmemLoadStore = 128;
// Factor used to intentionally narrow contiguous tmem access down from
// the maximum message width to avoid register pressure
static constexpr int narrowingFactor = 4;

namespace {

// Tensor memory access traits, currently only 32b is used
template <bool IsStrided> struct TMemAccess32b {
  static constexpr int opBitWidth = 32;
  static constexpr int colsPerThread = 1;
  static constexpr int rowsPerThread = 1;
  static constexpr const char *opShape = IsStrided ? "16x32bx2" : "32x32b";
  static constexpr bool usesSecondHalfOffset = IsStrided;
};

struct TMemAccess256b {
  static constexpr int opBitWidth = 256;
  static constexpr int colsPerThread = 2;
  static constexpr int rowsPerThread = 2;
  static constexpr char opShape[] = "16x256b";
};

template <typename TMemAccess> struct TMemMessage {
  using traits = TMemAccess;
  static constexpr int opBitWidth = TMemAccess::opBitWidth;
  static constexpr int colsPerThread = TMemAccess::colsPerThread;
  static constexpr int rowsPerThread = TMemAccess::rowsPerThread;
  static constexpr int numThreadsPerWarp = 32;
  static constexpr int maxNumRepeats =
      largestTmemLoadStore / (colsPerThread * rowsPerThread);
  static constexpr int maxColsPerMessage = (opBitWidth / 32) * maxNumRepeats;
  static constexpr int rowsPerMessage = numThreadsPerWarp / rowsPerThread;
  static constexpr int colsPerMessage = maxColsPerMessage / narrowingFactor;
};

struct TMemRuntimeInfo {
  static constexpr int numRowsPerWarp = 32;
  int numWarps;
  int numWarpGroups;
  int numElementsPer32B;
  int numElements;
  int numCols;
  int blockM;
  int blockN;
  bool unpackedb16;
  bool useStridedMessage;
  int numBlocks;
  int numWarpGroupsPerBlock;
  bool blocksInterleaved;
  int numColsPerBlock;
  int colsPerWarpGroup;
};

SmallVector<Value> packToI32(const SmallVector<Value> &values, Location loc,
                             ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> packedValues;
  Type elType = values[0].getType();
  int numElementsPer32B = 32 / elType.getIntOrFloatBitWidth();
  if (numElementsPer32B == 1)
    return values;
  Value packed = b.undef(vec_ty(elType, numElementsPer32B));
  for (int i = 0; i < values.size(); i++) {
    Value val = values[i];
    packed = b.insert_element(packed.getType(), packed, val,
                              b.i32_val(i % numElementsPer32B));
    if (i % numElementsPer32B == numElementsPer32B - 1 ||
        i == values.size() - 1) {
      packed = b.bitcast(packed, i32_ty);
      packedValues.push_back(packed);
      packed = b.undef(vec_ty(elType, numElementsPer32B));
    }
  }
  return packedValues;
}

int getNum32BRegs(bool unpackedb16, int numElementsPer32B, int numCols) {
  int numRegPerMessage = numCols;
  if (unpackedb16)
    numRegPerMessage = numRegPerMessage / numElementsPer32B;
  return numRegPerMessage;
}

TMemRuntimeInfo getTMemRuntimeInfo(Operation *op, RankedTensorType tensorType,
                                   MemDescType memType) {
  TMemRuntimeInfo info;
  static_assert(TMemRuntimeInfo::numRowsPerWarp == 32,
                "A single warp must access exactly 32 rows of tmem");
  assert(
      nvidia_gpu::isDistributedLayoutTMemCompatible(op, tensorType, memType) &&
      "unsupported distributed layout for tensor memory");

  info.numWarps = triton::gpu::lookupNumWarps(op);
  assert(info.numWarps % 4 == 0 && "Unexpected number of warps");
  info.numWarpGroups = info.numWarps / 4;
  info.numElementsPer32B = 32 / tensorType.getElementTypeBitWidth();
  auto shapePerCTA = mlir::triton::gpu::getShapePerCTA(tensorType);
  info.numElements = product(shapePerCTA);

  triton::nvidia_gpu::TMemAllocation tmemAlloc =
      triton::nvidia_gpu::getTmemAllocSizes(memType);
  info.numCols = tmemAlloc.numCols;

  info.blockM = 0;
  info.blockN = 0;
  info.unpackedb16 = false;
  if (auto attr = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
          memType.getEncoding())) {
    info.blockM = attr.getBlockM();
    info.blockN = attr.getBlockN();
    assert((!attr.getUnpacked() || info.numElementsPer32B <= 2) &&
           "unsupported unpacked layout");
    info.unpackedb16 = attr.getUnpacked() && (info.numElementsPer32B == 2);
  } else {
    assert(isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
               memType.getEncoding()) &&
           "Expecting a tensor memory encoding attribute");
    info.blockM = 128;
    info.blockN = 32;
  }

  info.useStridedMessage = (info.blockM == 64);

  info.numBlocks = ceil<int>(info.numElements, info.blockM * info.blockN);
  info.numWarpGroupsPerBlock = ceil<int>(info.numWarpGroups, info.numBlocks);
  info.blocksInterleaved = (info.numBlocks > 1 && info.useStridedMessage);
  info.numColsPerBlock = info.numCols / info.numBlocks;
  info.colsPerWarpGroup = info.numCols / info.numWarpGroups;
  // If more than one warp group processes the same block,
  // then fewer columns must be processed per message per warp group
  info.numColsPerBlock /= info.numWarpGroupsPerBlock;

  if (info.blocksInterleaved) {
    info.numColsPerBlock *= 2;
  }

  return info;
}

template <typename TMemMsgT>
void calculateAddressAndEmitTmemMessage(
    Location loc, Value baseAddress, TMemRuntimeInfo info,
    ConversionPatternRewriter &rewriter,
    const std::function<void(Value, int, bool, int, bool, int)>
        &createMemoryOp) {

  TritonLLVMOpBuilder b(loc, rewriter);
  Value warpId = rewriter.create<nvgpu::WarpIdOp>(loc);
  Value warpIdInGroup = b.urem(warpId, b.i32_val(4));
  Value warpGroupId = b.udiv(warpId, b.i32_val(4));

  int numRegs = getNum32BRegs(info.unpackedb16, info.numElementsPer32B,
                              info.colsPerWarpGroup);
  if (info.useStridedMessage)
    numRegs /= 2;

  // If the workload runtime requires fewer registers than the default message
  // width, use a message width that matches the workload
  int colsPerMessage = TMemMsgT::colsPerMessage;
  int numRepeats = colsPerMessage / (TMemMsgT::opBitWidth / 32);
  int numRegsPerMsg =
      TMemMsgT::colsPerThread * TMemMsgT::rowsPerThread * numRepeats;
  numRegsPerMsg = std::min(numRegsPerMsg, numRegs);
  // Invert the above formulas to calculate the effective runtime message width
  colsPerMessage = (numRegsPerMsg * (TMemMsgT::opBitWidth / 32)) /
                   (TMemMsgT::colsPerThread * TMemMsgT::rowsPerThread);

  // Half as many registers are needed for 16-bit packed elements,
  // so twice as many columns are accessed per message.
  colsPerMessage *= info.numElementsPer32B;
  for (int block = 0; block < info.numBlocks; block += info.numWarpGroups) {
    Value address = b.ptrtoint(i32_ty, baseAddress);
    Value blockId =
        b.add(b.i32_val(block),
              b.udiv(warpGroupId, b.i32_val(info.numWarpGroupsPerBlock)));
    Value warpGroupIdInBlock =
        b.urem(warpGroupId, b.i32_val(info.numWarpGroupsPerBlock));
    Value startColumnId =
        b.mul(warpGroupIdInBlock, b.i32_val(info.colsPerWarpGroup));
    Value blockRowId =
        b.mul(warpIdInGroup, b.i32_val(TMemRuntimeInfo::numRowsPerWarp));

    if (info.blocksInterleaved) {
      Value blockIdIsOdd = b.urem(blockId, b.i32_val(2));
      Value blockIdPrevEven = b.sub(blockId, blockIdIsOdd);
      blockRowId = b.add(blockRowId, b.mul(blockIdIsOdd, b.i32_val(16)));
      startColumnId =
          b.add(startColumnId,
                b.mul(blockIdPrevEven, b.i32_val(info.numColsPerBlock / 2)));
    } else {
      startColumnId =
          b.add(startColumnId, b.mul(blockId, b.i32_val(info.numColsPerBlock)));
    }

    // A strided message accesses twice as many columns per message,
    // thus half as many messages are required
    int numColumns = info.useStridedMessage ? info.numColsPerBlock / 2
                                            : info.numColsPerBlock;
    for (int colStart = 0; colStart < numColumns; colStart += colsPerMessage) {
      // For messages that span only 16 rows (e.g. 16x256b), multiple messages
      // are required to cover the entire set of rows per warp.
      for (int rowStart = 0; rowStart < TMemRuntimeInfo::numRowsPerWarp;
           rowStart += TMemMsgT::rowsPerMessage) {
        Value rowOffset = b.add(blockRowId, b.i32_val(rowStart));
        Value warpGroupAddress =
            b.add(address, b.shl(rowOffset, b.i32_val(16)));
        warpGroupAddress = b.add(warpGroupAddress, startColumnId);

        Value msgAddress = b.add(warpGroupAddress, b.i32_val(colStart));
        int secondHalfColOffset = 0;
        if (info.useStridedMessage) {
          // Offset to half way through the set of columns for this warpgroup.
          secondHalfColOffset = numColumns;
        }
        createMemoryOp(msgAddress, secondHalfColOffset, info.unpackedb16,
                       numRegsPerMsg, info.useStridedMessage,
                       TMemMsgT::opBitWidth);
      }
    }
  }
}

template <typename TMemMsgT>
static void
createTensorMemoryStore(Location loc, Value address, SmallVector<Value> &srcs,
                        int secondHalfOffset, Value pred, bool unpacked,
                        ConversionPatternRewriter &rewriter) {
  PTXBuilder ptxBuilder;
  std::string opcode;
  std::string packedStr = unpacked ? ".unpack::16b" : "";
  unsigned numRepeats = srcs.size() / (TMemMsgT::traits::rowsPerThread *
                                       TMemMsgT::traits::colsPerThread);
  opcode = "@$0 tcgen05.st.sync.aligned." +
           std::string(TMemMsgT::traits::opShape) + ".x" +
           std::to_string(numRepeats) + packedStr;
  if (secondHalfOffset)
    opcode += ".b32 [$1], " + std::to_string(secondHalfOffset) + ", {";
  else
    opcode += ".b32 [$1], {";

  SmallVector<PTXInstr::Operand *> operands;
  operands.push_back(ptxBuilder.newOperand(pred, "b"));
  operands.push_back(ptxBuilder.newOperand(address, "r"));
  for (int i = 0; i < srcs.size(); i++) {
    opcode += "$" + std::to_string(i + 2);
    auto *resultOp = ptxBuilder.newOperand(srcs[i], "r");
    operands.push_back(resultOp);
    if (i < srcs.size() - 1)
      opcode += ", ";
  }
  opcode += "};";

  auto &st = *ptxBuilder.create<PTXInstr>(opcode);
  st(operands, /*onlyAttachMLIRArgs=*/true);
  Type voidTy = void_ty(rewriter.getContext());
  ptxBuilder.launch(rewriter, loc, voidTy);
}

static void createWaitOpSt(Location loc, ConversionPatternRewriter &rewriter) {
  PTXBuilder ptxBuilder;
  std::string opcode = "tcgen05.wait::st.sync.aligned;";
  auto &wait = *ptxBuilder.create<PTXInstr>(opcode);
  wait({}, /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

// Helper to reorder the packed scales before storing them. Each block of 4
// scales along K are separated in their own block.
static void reorderScales(SmallVector<Value> &srcValues, int64_t k) {
  if (k <= 4)
    return;
  assert(k % 4 == 0);
  int64_t numKBlocks = k / 4;
  int64_t blockSize = srcValues.size() / numKBlocks;
  SmallVector<Value> reorderedValues(srcValues.size());
  for (int i = 0; i < srcValues.size(); i++) {
    int64_t row = i / numKBlocks;
    int64_t col = i % numKBlocks;
    int64_t newIdx = col * blockSize + row;
    reorderedValues[newIdx] = srcValues[i];
  }
  srcValues = std::move(reorderedValues);
}

template <typename Fn> void emitMemoryOp(bool isStrided, Fn &&emitMemoryOpFn) {
  if (isStrided)
    emitMemoryOpFn(std::integral_constant<bool, true>{});
  else
    emitMemoryOpFn(std::integral_constant<bool, false>{});
}

static void lowerStoreToTensorMemory(Location loc, Operation *op, Value src,
                                     Value dest, Value llSrc, Value pred,
                                     Value tmemBase,
                                     ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> srcValues = unpackLLElements(loc, llSrc, rewriter);
  srcValues = packToI32(srcValues, loc, rewriter);
  auto dstType = cast<MemDescType>(dest.getType());
  if (isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
          dstType.getEncoding())) {
    // Scales are stored in a different layout where each block of 4 scales
    // along K are stored separately.
    reorderScales(srcValues, dstType.getShape().back());
  }

  auto tmemInfo = getTMemRuntimeInfo(op, cast<RankedTensorType>(src.getType()),
                                     cast<MemDescType>(dest.getType()));
  int regIdx = 0;
  emitMemoryOp(tmemInfo.useStridedMessage, [&](auto isStrided) {
    using MsgT = TMemMessage<TMemAccess32b<decltype(isStrided)::value>>;
    calculateAddressAndEmitTmemMessage<MsgT>(
        loc, tmemBase, tmemInfo, rewriter,
        [&](Value startAddress, int secondHalfColOffset, bool unpackedb16,
            int regsPerMsg, bool useStridedMessage, int opBitWidth) {
          SmallVector<Value> srcValuesSlice(srcValues.begin() + regIdx,
                                            srcValues.begin() + regIdx +
                                                regsPerMsg);
          regIdx += regsPerMsg;
          createTensorMemoryStore<MsgT>(loc, startAddress, srcValuesSlice,
                                        secondHalfColOffset, pred, unpackedb16,
                                        rewriter);
        });
  });
  createWaitOpSt(loc, rewriter);

  // Emit a barrier to ensure all threads have finished writing to tensor memory
  // before any use of the tensor memory.
  b.barrier();
}

struct TensorMemoryAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMAllocOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value base = rewriter.create<nvgpu::TensorMemoryBaseAddress>(loc);
    Value baseInt = b.ptrtoint(i32_ty, base);
    int colOffset = cast<IntegerAttr>(op->getAttr("tensor_memory_col_offset"))
                        .getValue()
                        .getZExtValue();
    int rowOffset = cast<IntegerAttr>(op->getAttr("tensor_memory_row_offset"))
                        .getValue()
                        .getZExtValue();
    Value allocAddress = b.add(baseInt, b.i32_val(colOffset | rowOffset << 16));
    // Cast to address space 3 as the shared memory object uses 3.
    // TODO: clean this up and use either a int or ptr address space 6
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    Value ptr = b.inttoptr(ptrTy, allocAddress);
    SmallVector<unsigned> order(op.getType().getRank());
    std::iota(order.begin(), order.end(), 0);
    std::reverse(order.begin(), order.end());
    auto shape = op.getType().getShape();

    if (op.getSrc()) {
      lowerStoreToTensorMemory(loc, op, op.getSrc(), op.getResult(),
                               adaptor.getSrc(), b.i1_val(true), ptr, rewriter);
    }

    rewriter.replaceOp(op, ptr);
    return success();
  }
};

template <typename TMemMsgT>
static Value createTensorMemoryLoad(Location loc,
                                    triton::nvidia_gpu::TMEMLoadOp op,
                                    Value address, int secondHalfOffset,
                                    bool unpacked, int numRegPerMessage,
                                    ConversionPatternRewriter &rewriter) {
  PTXBuilder ptxBuilder;
  std::string opcode;
  // If the memory is unpacked we need to pack on the fly when loading.
  std::string packedStr = unpacked ? ".pack::16b" : "";
  unsigned numRepeats = numRegPerMessage / (TMemMsgT::traits::rowsPerThread *
                                            TMemMsgT::traits::colsPerThread);
  opcode = "tcgen05.ld.sync.aligned." + std::string(TMemMsgT::traits::opShape) +
           ".x" + std::to_string(numRepeats) + packedStr + ".b32 {";

  SmallVector<PTXInstr::Operand *> operands;
  for (int i = 0; i < numRegPerMessage; i++) {
    opcode += "$" + std::to_string(i);
    auto *resultOp = ptxBuilder.newOperand("=r");
    operands.push_back(resultOp);
    if (i < numRegPerMessage - 1)
      opcode += ", ";
  }
  opcode += "}, [$" + std::to_string(numRegPerMessage) + "]";
  if (secondHalfOffset)
    opcode += ", " + std::to_string(secondHalfOffset);
  opcode += ";";
  operands.push_back(ptxBuilder.newOperand(address, "r"));
  auto &ld = *ptxBuilder.create<PTXInstr>(opcode);
  ld(operands, /*onlyAttachMLIRArgs=*/true);
  SmallVector<Type> elemTypes(numRegPerMessage, i32_ty);
  MLIRContext *ctx = op.getContext();
  Type structTy = struct_ty(elemTypes);
  Value ret = ptxBuilder.launch(rewriter, loc, structTy);
  return ret;
}

static SmallVector<Value> unpackResults(Value packedValues, Type elemTy,
                                        int numCols, Location loc,
                                        ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> resultVals;
  int numElementsPer32B = 32 / elemTy.getIntOrFloatBitWidth();
  Type packedType = elemTy;
  if (numElementsPer32B > 1)
    packedType = vec_ty(elemTy, numElementsPer32B);
  for (int i = 0; i < numCols; i++) {
    Value result = b.extract_val(i32_ty, packedValues, i);
    result = b.bitcast(result, packedType);
    if (numElementsPer32B > 1) {
      for (int j = 0; j < numElementsPer32B; j++) {
        Value elem = b.extract_element(elemTy, result, b.i32_val(j));
        resultVals.push_back(elem);
      }
    } else {
      resultVals.push_back(result);
    }
  }
  return resultVals;
}

static void createWaitOpLd(Location loc, ConversionPatternRewriter &rewriter) {
  PTXBuilder ptxBuilder;
  std::string opcode = "tcgen05.wait::ld.sync.aligned;";
  auto &wait = *ptxBuilder.create<PTXInstr>(opcode);
  wait({}, /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

struct TensorMemoryLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMLoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto llvmElemTy =
        getTypeConverter()->convertType(op.getSrc().getType().getElementType());
    auto tmemBase = adaptor.getSrc();

    auto tmemInfo =
        getTMemRuntimeInfo(op, cast<RankedTensorType>(op.getType()),
                           cast<MemDescType>(op.getSrc().getType()));
    SmallVector<Value> resultVals;
    emitMemoryOp(tmemInfo.useStridedMessage, [&](auto isStrided) {
      using MsgT = TMemMessage<TMemAccess32b<decltype(isStrided)::value>>;
      calculateAddressAndEmitTmemMessage<MsgT>(
          loc, tmemBase, tmemInfo, rewriter,
          [&](Value startAddress, int secondHalfColOffset, bool unpackedb16,
              int regsPerMessage, bool useStridedMessage, int opBitWidth) {
            Value packedValues = createTensorMemoryLoad<MsgT>(
                loc, op, startAddress, secondHalfColOffset, unpackedb16,
                regsPerMessage, rewriter);
            auto results =
                unpackResults(packedValues, op.getType().getElementType(),
                              regsPerMessage, loc, rewriter);
            resultVals.append(results.begin(), results.end());
          });
    });
    Type structTy = getTypeConverter()->convertType(op.getType());
    Value resultStruct =
        packLLElements(loc, getTypeConverter(), resultVals, rewriter, structTy);
    // Wait insertion could be moved to the TTGIR level if needed.
    createWaitOpLd(loc, rewriter);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct TensorMemoryStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMStoreOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto llvmElemTy =
        getTypeConverter()->convertType(op.getDst().getType().getElementType());
    auto tmemBase = adaptor.getDst();
    Value pred = adaptor.getPred();
    lowerStoreToTensorMemory(loc, op, op.getSrc(), op.getDst(),
                             adaptor.getSrc(), pred, tmemBase, rewriter);

    rewriter.eraseOp(op);
    return success();
  }
};

static Value
createBlockedScalesSMEMDescriptor(ConversionPatternRewriter &rewriter,
                                  Location loc, Value baseSrc) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  static_assert(sizeof(NVIDIA::SMEMDescriptor) == 8,
                "Descriptor size should be 64 bits.");
  NVIDIA::SMEMDescriptor desc;
  desc.descriptor = 0;
  desc.swizzlingMode = 0;                    // No swizzling for now
  desc.leadDimensionBaseOffset = 16 >> 4;    // 16 bytes
  desc.strideDimensionBaseOffset = 128 >> 4; // 8 x 16 bytes
  // See matrix-descriptor-encode(x) function in the ptx doc.
  // matrix-descriptor-encode(addr) = (addr & 0x3FFFF) >> 4
  auto smemAddr = b.ptrtoint(i64_ty, baseSrc);
  return b.add(b.int_val(64, desc.descriptor),
               b.lshr(b.shl(smemAddr, b.int_val(64, 46)), b.int_val(64, 50)));
}

static void createCommit(ConversionPatternRewriter &rewriter, Location loc,
                         Value barrier, Value pred) {
  PTXBuilder ptxBuilder;
  auto *barrierOperand = ptxBuilder.newAddrOperand(barrier, "r");
  std::string opcode = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64";
  auto &barrierOp = *ptxBuilder.create<PTXInstr>(opcode);
  barrierOp(barrierOperand).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static void createTcgen05Cp(ConversionPatternRewriter &rewriter, Location loc,
                            Value tmem_address, Value src_desc, Value pred) {
  PTXBuilder ptxBuilder;
  auto dst = ptxBuilder.newAddrOperand(tmem_address, "r");
  auto src = ptxBuilder.newOperand(src_desc, "l");
  std::string opcode = "tcgen05.cp.cta_group::1.warpx4.32x128b";
  auto &op = *ptxBuilder.create<PTXInstr>(opcode);
  op({dst, src}).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

struct TensorMemoryCopyOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMCopyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    assert(isa<triton::gpu::SharedMemorySpaceAttr>(srcTy.getMemorySpace()));
    assert(isa<triton::gpu::SwizzledSharedEncodingAttr>(srcTy.getEncoding()));

    auto sharedEnc =
        cast<triton::gpu::SwizzledSharedEncodingAttr>(srcTy.getEncoding());
    assert(
        sharedEnc.getMaxPhase() == 1 && sharedEnc.getPerPhase() == 1 &&
        sharedEnc.getVec() == 1 &&
        "The src SMEM of tmem_copy should not have swizzling applied for now");

    Value baseSrc =
        LLVM::getSharedMemoryObjectFromStruct(
            loc, adaptor.getSrc(),
            typeConverter->convertType(srcTy.getElementType()), rewriter)
            .getBase();

    Value baseDst = adaptor.getDst();

    // The following codegen assumes that we use tcgen05.cp only with
    // the warpx4.32x128b mode, to load blocked scales from MXFP.
    // We will expand the support as we find more use cases for the instruction.

    Value smemDesc = createBlockedScalesSMEMDescriptor(rewriter, loc, baseSrc);
    Value pred = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);

    auto createCopy = [&](int repMorN, int repK) {
      for (int i = 0; i < repMorN; ++i) {
        for (int j = 0; j < repK; ++j) {
          // Multiple copies of 32x128b blocks are laid out along M/N first then
          // K
          auto colOffset = b.int_val(32, (j * repMorN + i) * 4);
          auto tmemAddr = b.add(b.ptrtoint(i32_ty, baseDst), colOffset);
          createTcgen05Cp(rewriter, loc, tmemAddr, smemDesc, pred);
          smemDesc =
              b.add(smemDesc, b.int_val(64, 512 >> 4)); // one chunk = 32x16B
        }
      }
    };

    // Break up src axes into rep_m x rep_k x 32x128b, where rep_m = BLOCK_M /
    // 128 and rep_k = BLOCK_K / 128 32x128b blockes are contiguously laid out
    // in SMEM. rep_m * rep_k copies of such blocks are consumed by one
    // dot_scaled op for given BLOCK_M / BLOCK_K. Some axes of the scale shape
    // can be flattened into one, to reduce the rank of the load. Since rep_m
    // blocks are not contiguous in SMEM, we need to identify the original rep_m
    // axis from the given input shape.

    // The SMEM shapes are expected to be one of the followings. As long as
    // rep_m and rep_k can be identified correctly, other patterns are allowed.
    // * (rep_m x 32, 16B), meant only for TMEMCopy unit tests
    // * (rep_m, rep_k * 32 x 4 x 4B), 2D scale load with cp.async
    // * (rep_m, rep_k, 32, 16B), 4D scale load with TMA
    // * (rep_m, rep_k, 32, 4, 4B), 5D scale load with cp.async
    auto elemBits = srcTy.getElementType().getIntOrFloatBitWidth();
    int prodInner = 1;
    int repMorN = 1;
    int repK = 1;

    for (int i = srcTy.getRank() - 1; i >= 0; --i) {
      prodInner *= srcTy.getDimSize(i);
      if (prodInner * elemBits >= 32 * 128) {
        if (i == 0) {
          repMorN = prodInner * elemBits / (32 * 128);
          repK = 1;
        } else if (i == 1) {
          repMorN = srcTy.getDimSize(0);
          repK = prodInner * elemBits / (32 * 128);
        } else {
          repMorN = srcTy.getDimSize(0);
          repK = srcTy.getDimSize(1);
        }
        break;
      }
    }

    createCopy(repMorN, repK);

    if (op.getBarrier()) {
      auto barrier = LLVM::getSharedMemoryObjectFromStruct(
          op.getLoc(), adaptor.getBarrier(), i64_ty, rewriter);
      createCommit(rewriter, loc, barrier.getBase(), pred);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct MemDescSubviewOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::MemDescSubviewOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::MemDescSubviewOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescSubviewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());

    if (!isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
            srcTy.getEncoding())) {
      return failure();
    }

    // newBase = base + offset
    auto tmemBase = adaptor.getSrc();
    SmallVector<Value> opOffsetVals = op.getOffsets();
    size_t destRank = op.getResult().getType().getRank();
    SmallVector<Value> offsetVals;
    int rankReduced = srcTy.getRank() - destRank;
    for (int i = rankReduced; i < opOffsetVals.size(); i++) {
      offsetVals.push_back(opOffsetVals[i]);
    }

    triton::nvidia_gpu::TMemAllocation tmemAlloc =
        triton::nvidia_gpu::getTmemAllocSizes(cast<MemDescType>(dstTy));
    int numColOffset = tmemAlloc.numCols;
    Value newBase = b.ptrtoint(rewriter.getI32Type(), tmemBase);
    newBase = rewriter.create<LLVM::AddOp>(
        loc, newBase,
        rewriter.create<LLVM::MulOp>(loc, opOffsetVals[0],
                                     b.i32_val(numColOffset)));
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    rewriter.replaceOp(op, b.inttoptr(elemPtrTy, newBase));
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateTensorMemoryOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<TensorMemoryAllocOpConversion, TensorMemoryLoadOpConversion,
               TensorMemoryStoreOpConversion, TensorMemoryCopyOpConversion>(
      typeConverter, benefit);
  return;
}

void mlir::triton::NVIDIA::populateTensorMemorySubviewOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MemDescSubviewOpConversion>(typeConverter, benefit);
  return;
}
