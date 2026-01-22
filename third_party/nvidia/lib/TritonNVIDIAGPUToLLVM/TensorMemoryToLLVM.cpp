#include "Dialect/NVGPU/IR/Dialect.h"
#include "DotOpToLLVM/MMAHelpers.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::NVIDIA;

// The maximum number of tensor memory registers that can be accessed
// by a single message regardless of shape or repetitions
static constexpr int largestTmemLoadStore = 128;
// The maximum number of thread registers that can be populated by
// multiple messages
static constexpr int maxRegisters = 256;

namespace {

struct TMemCopyAtom {
  int nRow;
  int bCol;
  // a multicast of n represents that warps with (warpId & n) != 0 are
  // broadcasted
  int multicast;
};

// .shape     = { .128x256b, .128x128b, .64x128b, .32x128b }
// .multicast = { .warpx2::02_13 , .warpx2::01_23, .warpx4}
// .shape = .4x256b NYI
constexpr TMemCopyAtom TMemCopyAtomNone128{128 /*nRow*/, 128 /*bCol*/,
                                           0 /*multicast*/};

constexpr TMemCopyAtom TMemCopyAtomNone256{128 /*nRow*/, 256 /*bCol*/,
                                           0 /*multicast*/};

constexpr TMemCopyAtom TMemCopyAtomWarp02_13{64 /*nRow*/, 128 /*bCol*/,
                                             1 /*multicast*/};

constexpr TMemCopyAtom TMemCopyAtomWarp01_23{64 /*nRow*/, 128 /*bCol*/,
                                             2 /*multicast*/};

constexpr TMemCopyAtom TMemCopyAtomWarp4{32 /*nRow*/, 128 /*bCol*/,
                                         3 /*multicast*/};

TMemCopyAtom getTMemCopyAtom(const LinearLayout &cvt, int bitwidth) {
  auto *ctx = cvt.getInDimNames().begin()->getContext();
  auto S = [&](StringRef str) { return StringAttr::get(ctx, str); };
  auto kRow = S("row");
  auto kCol = S("col");
  auto kOffset = S("offset");
  assert(cvt.getInDimSize(kRow) == 128);
  auto multicastBit = [&](int i) {
    assert(i == 0 || i == 1);
    return cvt.getBasis(kRow, llvm::Log2_32(32) + i, kOffset) == 0;
  };
  auto multicast = multicastBit(0) | multicastBit(1) << 1;
  if (multicast == 0) {
    // TODO we will assert this in the verifier
    if (cvt.getInDimSize(kCol) * bitwidth == 128) {
      return TMemCopyAtomNone128;
    } else {
      assert(cvt.getInDimSize(kCol) * bitwidth >= 256);
      return TMemCopyAtomNone256;
    }
  } else if (multicast == 1) {
    return TMemCopyAtomWarp02_13;
  } else if (multicast == 2) {
    return TMemCopyAtomWarp01_23;
  } else if (multicast == 3) {
    return TMemCopyAtomWarp4;
  } else {
    llvm_unreachable("invalid multicast");
  }
}

SmallVector<Value> pack(ArrayRef<Value> values, Type outType, Location loc,
                        ConversionPatternRewriter &rewriter, bool pad = false) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type inType = values[0].getType();
  if (inType == outType) {
    return to_vector(values);
  }

  auto inbitwidth = inType.getIntOrFloatBitWidth();
  auto outbitwidth = outType.getIntOrFloatBitWidth();
  assert(inbitwidth <= outbitwidth);
  SmallVector<Value> packedValues;
  if (inbitwidth == outbitwidth) {
    for (auto &val : values) {
      packedValues.push_back(b.bitcast(val, outType));
    }
    return packedValues;
  }

  auto vecSize = outbitwidth / inbitwidth;
  auto vecTy = vec_ty(inType, vecSize);

  auto elemsPerVec = pad ? 1 : vecSize;
  assert(values.size() % elemsPerVec == 0);
  for (int i = 0; i < values.size(); i += elemsPerVec) {
    Value packed = b.undef(vecTy);
    for (int j = 0; j < elemsPerVec; j++) {
      Value val = values[i + j];
      packed = b.insert_element(vecTy, packed, val, b.i32_val(j));
    }
    packed = b.bitcast(packed, outType);
    packedValues.emplace_back(std::move(packed));
  }
  return packedValues;
}

SmallVector<Value> unpack(ArrayRef<Value> packedValues, Type outType,
                          Location loc, ConversionPatternRewriter &rewriter,
                          bool pad = false) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type inType = packedValues[0].getType();
  if (inType == outType) {
    return to_vector(packedValues);
  }

  auto inbitwidth = inType.getIntOrFloatBitWidth();
  auto outbitwidth = outType.getIntOrFloatBitWidth();
  assert(inbitwidth >= outbitwidth);
  SmallVector<Value> unpackedValues;
  if (inbitwidth == outbitwidth) {
    for (auto val : packedValues) {
      unpackedValues.push_back(b.bitcast(val, outType));
    }
    return unpackedValues;
  }
  auto vecSize = inbitwidth / outbitwidth;
  auto vecTy = vec_ty(outType, vecSize);

  auto elemsPerVec = pad ? 1 : vecSize;
  for (auto val : packedValues) {
    Value packed = b.bitcast(val, vecTy);
    for (int j = 0; j < elemsPerVec; j++) {
      unpackedValues.push_back(
          b.extract_element(outType, packed, b.i32_val(j)));
    }
  }
  return unpackedValues;
}

void createTensorMemoryStore(Location loc, Value address, int colOffset,
                             SmallVector<Value> &srcs,
                             std::optional<int> secondHalfOffset, Value pred,
                             bool unpacked, TMemAccessAtom atom,
                             ConversionPatternRewriter &rewriter) {
  PTXBuilder ptxBuilder;
  std::string packedStr = unpacked ? ".unpack::16b" : "";
  unsigned numRepeats = srcs.size() / getElementsPerThread(atom);
  std::string opcode = "@$0 tcgen05.st.sync.aligned.";
  opcode += getOpShape(atom);
  opcode += ".x" + std::to_string(numRepeats) + packedStr;
  opcode += ".b32 [$1 + " + std::to_string(colOffset) + "], ";
  if (secondHalfOffset)
    opcode += std::to_string(*secondHalfOffset) + ", {";
  else
    opcode += "{";

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

  auto &st = *ptxBuilder.create(opcode);
  st(operands, /*onlyAttachMLIRArgs=*/true);
  Type voidTy = void_ty(rewriter.getContext());
  ptxBuilder.launch(rewriter, loc, voidTy);
}

// Returns {loadResult, redvalResult} where redvalResult is null if no reduction
std::pair<Value, Value>
createTensorMemoryLoad(Location loc, MLIRContext *ctx, Value address,
                       int colOffset, std::optional<int> secondHalfOffset,
                       bool unpacked, int numRegPerMessage, TMemAccessAtom atom,
                       std::optional<TMEMLoadReduceModifier> redOp, bool useAbs,
                       bool useNaN, Type elemTy,
                       ConversionPatternRewriter &rewriter) {
  PTXBuilder ptxBuilder;
  // If the memory is unpacked we need to pack on the fly when loading.
  std::string packedStr = unpacked ? ".pack::16b" : "";
  unsigned numRepeats = numRegPerMessage / getElementsPerThread(atom);

  std::string opcode = std::string("tcgen05.ld.") + (redOp ? "red." : "");
  opcode += "sync.aligned.";
  opcode += getOpShape(atom);
  opcode += ".x" + std::to_string(numRepeats);

  if (redOp) {
    if (unpacked) {
      llvm_unreachable("Unpacked is unsupported with TMEM reduction");
    }
    // Add reduction modifier: .min or .max
    switch (*redOp) {
    case TMEMLoadReduceModifier::MIN:
      opcode += ".min";
      break;
    case TMEMLoadReduceModifier::MAX:
      opcode += ".max";
      break;
    default:
      llvm_unreachable("Unsupported reduction modifier");
    }
    if (useAbs)
      opcode += ".abs";
    if (useNaN)
      opcode += ".NaN";

    std::string redStr;
    if (elemTy.isF32()) {
      redStr = ".f32";
    } else {
      llvm_unreachable("Unsupported type for TMEM reduction");
    }
    opcode += redStr;
  } else {
    opcode += packedStr + ".b32";
  }

  opcode += " {";

  SmallVector<PTXInstr::Operand *> operands;
  for (int i = 0; i < numRegPerMessage; i++) {
    opcode += "$" + std::to_string(i);
    auto *resultOp = ptxBuilder.newOperand("=r");
    operands.push_back(resultOp);
    if (i < numRegPerMessage - 1)
      opcode += ", ";
  }
  opcode += "}";

  int nextOperandIdx = numRegPerMessage;

  // Add redval output operand if reduction is enabled
  if (redOp) {
    opcode += ", {$" + std::to_string(nextOperandIdx) + "}";
    auto *redvalOp = ptxBuilder.newOperand("=r");
    operands.push_back(redvalOp);
    nextOperandIdx++;
  }

  opcode += ", [$" + std::to_string(nextOperandIdx) + " + " +
            std::to_string(colOffset) + "]";
  if (secondHalfOffset)
    opcode += ", " + std::to_string(*secondHalfOffset);
  opcode += ";";
  operands.push_back(ptxBuilder.newOperand(address, "r"));
  auto &ld = *ptxBuilder.create(opcode);
  ld(operands, /*onlyAttachMLIRArgs=*/true);

  // Build return type: data registers + optional redval register
  int totalResults = numRegPerMessage + (redOp ? 1 : 0);
  Type retTy;
  if (totalResults == 1) {
    retTy = i32_ty;
  } else {
    SmallVector<Type> elemTypes(totalResults, i32_ty);
    retTy = struct_ty(elemTypes);
  }
  Value ret = ptxBuilder.launch(rewriter, loc, retTy);

  // Extract load result and redval if needed
  Value loadResult = ret;
  Value redvalResult = nullptr;

  if (redOp) {
    // Per PTX spec: .num must be at least .x2 when .red is specified,
    // so numRegPerMessage >= 2 * getElementsPerThread(atom) >= 2.
    // ret is a struct with numRegPerMessage + 1 elements: {loadVals..., redval}
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Type> loadElemTypes(numRegPerMessage, i32_ty);
    Type loadStructTy = struct_ty(loadElemTypes);
    Value loadStruct = b.undef(loadStructTy);
    for (int i = 0; i < numRegPerMessage; i++) {
      Value elem = b.extract_val(i32_ty, ret, i);
      loadStruct = b.insert_val(loadStructTy, loadStruct, elem, i);
    }
    loadResult = loadStruct;
    redvalResult = b.extract_val(i32_ty, ret, numRegPerMessage);
    // Bitcast redval from i32 to the target element type
    if (redvalResult && elemTy != i32_ty) {
      redvalResult = b.bitcast(redvalResult, elemTy);
    }
  }

  return {loadResult, redvalResult};
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

  auto unpackElement = [&](Value result) {
    result = b.bitcast(result, packedType);
    if (numElementsPer32B > 1) {
      for (int j = 0; j < numElementsPer32B; j++) {
        Value elem = b.extract_element(elemTy, result, b.i32_val(j));
        resultVals.push_back(elem);
      }
    } else {
      resultVals.push_back(result);
    }
  };

  if (isa<LLVM::LLVMStructType>(packedValues.getType())) {
    for (int i = 0; i < numCols; i++) {
      Value result = b.extract_val(i32_ty, packedValues, i);
      unpackElement(result);
    }
  } else {
    unpackElement(packedValues);
  }
  return resultVals;
}

// Returns {resultVals, redvalVals} where redvalVals is empty if no reduction.
// Reduction produces exactly one value per thread; if multiple messages
// contribute partial reductions, they are combined into one.
std::pair<SmallVector<Value>, SmallVector<Value>> lowerTMemLdSt(
    Location loc, ConversionPatternRewriter &rewriter, const LinearLayout &reps,
    ArrayRef<Value> vals, TMemAccessAtom atom, Type llvmElemTy, Value tmemBase,
    Value pred, int valsPerMessage, bool unpacked,
    std::optional<uint32_t> secondHalfOffset,
    std::optional<TMEMLoadReduceModifier> redOp, bool useAbs, bool useNaN) {
  auto *ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  auto kCol = str_attr("col");
  auto kRow = str_attr("row");
  bool isStore = !vals.empty();

  tmemBase = b.ptrtoint(i32_ty, tmemBase);

  assert(to_vector(reps.getOutDimNames()) ==
         SmallVector<StringAttr>({kRow, kCol}));
  auto getRowCol = [kRow, kCol](const auto &rowCol) {
    assert(rowCol.size() == 2);
    assert(std::get<0>(rowCol[0]) == kRow);
    assert(std::get<0>(rowCol[1]) == kCol);
    return std::make_pair(std::get<1>(rowCol[0]), std::get<1>(rowCol[1]));
  };

  Value warpId = WarpIdOp::create(rewriter, loc);
  // Map warpId to rows 32 and 64
  auto warpIdInGroup = b.and_(warpId, b.i32_val(3));
  tmemBase = b.add(tmemBase, b.shl(warpIdInGroup, b.i32_val(5 + 16)));
  // The block offset is already added to the tmemBase
  // Add warp groups to tmemBase
  if (reps.getInDimSize(kWarp) > 4) {
    auto rowCol = applyLinearLayout(loc, rewriter, reps,
                                    {{kReg, b.i32_val(0)},
                                     {kLane, b.i32_val(0)},
                                     {kWarp, warpId},
                                     {kBlock, b.i32_val(0)}});
    auto [row, col] = getRowCol(rowCol);
    tmemBase = b.add(tmemBase,
                     b.or_(b.shl(row, b.i32_val(16)), col, /*disjoint*/ true));
  }

  SmallVector<Value> resultVals, redvalVals;
  for (int i = 0; i < reps.getInDimSize(kReg); i += valsPerMessage) {
    auto [row, col] =
        getRowCol(reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}}));
    // Encode row into the base address and pass col as an immediate colOffset.
    int staticOffset = col | (row << 16);
    if (isStore) {
      auto chunk = to_vector(vals.slice(i, valsPerMessage));
      createTensorMemoryStore(loc, tmemBase, /*colOffset=*/staticOffset, chunk,
                              /*secondHalfOffset=*/secondHalfOffset, pred,
                              /*unpacked=*/unpacked, atom, rewriter);
    } else {
      auto [outVals, redval] =
          createTensorMemoryLoad(loc, ctx, tmemBase, /*colOffset=*/staticOffset,
                                 /*secondHalfOffset=*/secondHalfOffset,
                                 /*unpacked=*/unpacked,
                                 /*numRegPerMessage=*/valsPerMessage, atom,
                                 redOp, useAbs, useNaN, llvmElemTy, rewriter);
      resultVals.append(
          unpackResults(outVals, llvmElemTy, valsPerMessage, loc, rewriter));
      if (redval)
        redvalVals.push_back(redval);
    }
  }

  // Combine partial reductions into one value per thread
  if (redvalVals.size() > 1) {
    auto isMin = *redOp == TMEMLoadReduceModifier::MIN;
    auto applyMinMax = [&](Value lhs, Value rhs) {
      return useNaN ? (isMin ? LLVM::MinimumOp::create(rewriter, loc, lhs, rhs)
                             : LLVM::MaximumOp::create(rewriter, loc, lhs, rhs))
                          ->getResult(0)
                    : (isMin ? LLVM::MinNumOp::create(rewriter, loc, lhs, rhs)
                             : LLVM::MaxNumOp::create(rewriter, loc, lhs, rhs))
                          ->getResult(0);
    };
    // Use tree reduction: pair up elements at each level
    while (redvalVals.size() > 1) {
      SmallVector<Value> reduced;
      assert(redvalVals.size() % 2 == 0 &&
             "redvalVals must be a multiple of 2");
      for (size_t i = 0; i < redvalVals.size(); i += 2) {
        reduced.push_back(applyMinMax(redvalVals[i], redvalVals[i + 1]));
      }
      redvalVals = std::move(reduced);
    }
  }

  return {resultVals, redvalVals};
}

// Returns {resultVals, redvalVals} where redvalVals is empty if no reduction
static std::pair<SmallVector<Value>, SmallVector<Value>>
lowerTMemLdStFromInfo(Location loc, ConversionPatternRewriter &rewriter,
                      TMemLdStEncodingInfo &info, Value pred, Type llvmElemTy,
                      ArrayRef<Value> vals, Value tmemBase,
                      std::optional<TMEMLoadReduceModifier> redOp, bool useAbs,
                      bool useNaN) {
  bool isStore = !vals.empty();
  if (info.broadcast) {
    auto removeBroadcast = std::move(info.broadcast.value());
    info.broadcast = std::nullopt;

    auto inVals = to_vector(vals);
    if (isStore) {
      inVals = removeBroadcast.apply(inVals);
    }
    auto [outVals, redvalVals] =
        lowerTMemLdStFromInfo(loc, rewriter, info, pred, llvmElemTy, inVals,
                              tmemBase, redOp, useAbs, useNaN);
    if (!isStore) {
      outVals = broadcastAs(outVals, info.reps);
    }
    return {outVals, redvalVals};
  }
  if (llvmElemTy.getIntOrFloatBitWidth() < 32) {
    unsigned bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    bool padding = false;
    Type packedElemTy;
    if (info.vec > 1) {
      // There are contiguous elements along kCol, so we can pack them into a
      // larger dtype
      packedElemTy = int_ty(bitwidth * info.vec);
      info.vec = 1;
    } else {
      padding = info.padding;
      assert(info.unpacked || info.padding);
      packedElemTy = i32_ty;
    }
    SmallVector<Value> inVals = to_vector(vals);
    if (isStore) {
      inVals = pack(inVals, packedElemTy, loc, rewriter, padding);
    }
    auto [outVals, redvalVals] =
        lowerTMemLdStFromInfo(loc, rewriter, info, pred, packedElemTy, inVals,
                              tmemBase, redOp, useAbs, useNaN);
    if (!isStore) {
      outVals = unpack(outVals, llvmElemTy, loc, rewriter, padding);
    }
    return {outVals, redvalVals};
  }

  SmallVector<Value> inVals = to_vector(vals);
  if (isStore) {
    inVals = info.perm.apply(inVals);
  }
  auto [outVals, redvalVals] =
      lowerTMemLdSt(loc, rewriter, info.reps, inVals, info.atom, llvmElemTy,
                    tmemBase, pred, info.numRegsPerMessage, info.unpacked,
                    info.secondHalfOffset, redOp, useAbs, useNaN);
  if (!isStore) {
    outVals = info.perm.inverse().apply(outVals);
  }
  return {outVals, redvalVals};
}

// Returns {resultVals, redvalVals} where redvalVals is empty if no reduction
static std::pair<SmallVector<Value>, SmallVector<Value>> lowerTMemLdStFromTypes(
    Location loc, ConversionPatternRewriter &rewriter, RankedTensorType regTy,
    MemDescType memTy, Value tmemBase, int maxnreg, Value pred, Type llvmElemTy,
    ArrayRef<Value> vals,
    std::optional<TMEMLoadReduceModifier> redOp = std::nullopt,
    bool useAbs = false, bool useNaN = false) {
  auto diag = [loc]() { return emitError(loc); };
  auto encodingInfoOr =
      computeTMemLdStEncodingInfo(regTy, memTy, maxnreg, diag);
  assert(succeeded(encodingInfoOr) &&
         "TMEM layout verification should catch invalid layouts");
  return lowerTMemLdStFromInfo(loc, rewriter, *encodingInfoOr, pred, llvmElemTy,
                               vals, tmemBase, redOp, useAbs, useNaN);
}

struct TensorMemoryLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMLoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto ctx = op.getContext();
    auto llvmElemTy =
        getTypeConverter()->convertType(op.getSrc().getType().getElementType());
    auto tmemBase = adaptor.getSrc();
    auto regTy = cast<RankedTensorType>(op.getType());
    auto memTy = cast<MemDescType>(op.getSrc().getType());

    // Extract reduction attributes
    auto redOp = op.getRedOp();
    auto useAbs = op.getAbs().value_or(false);
    auto useNaN = op.getNaN().value_or(false);
    if (redOp) {
      auto redTy = cast<RankedTensorType>(op.getRed().getType());
      assert(getTotalElemsPerThread(redTy) == 1 &&
             "reduction layout must produce exactly one value per thread");
    }

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto maxnreg = getContextualMaxNReg(op);
    auto [resultVals, redvalVals] = lowerTMemLdStFromTypes(
        loc, rewriter, regTy, memTy, tmemBase, maxnreg, b.i1_val(true),
        llvmElemTy, {}, redOp, useAbs, useNaN);

    Type structTy = getTypeConverter()->convertType(op.getType());
    Value resultStruct =
        packLLElements(loc, getTypeConverter(), resultVals, rewriter, structTy);
    // Wait insertion could be moved to the TTGIR level if needed.
    NVVM::Tcgen05WaitOp::create(rewriter, loc, NVVM::Tcgen05WaitKind::LOAD);

    // Handle reduction output if present
    SmallVector<Value> results = {resultStruct};
    if (redOp) {
      // Pack redval values into the red tensor result
      Type redStructTy = getTypeConverter()->convertType(op.getRed().getType());
      Value redStruct = packLLElements(loc, getTypeConverter(), redvalVals,
                                       rewriter, redStructTy);
      results.push_back(redStruct);
    }

    rewriter.replaceOp(op, results);
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
    auto ctx = op.getContext();
    auto llvmElemTy =
        getTypeConverter()->convertType(op.getDst().getType().getElementType());

    auto tmemBase = adaptor.getDst();
    Value pred = adaptor.getPred();
    auto memTy = cast<MemDescType>(op.getDst().getType());
    auto regTy = cast<RankedTensorType>(op.getSrc().getType());
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    SmallVector<Value> srcValues =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto maxnreg = getContextualMaxNReg(op);
    lowerTMemLdStFromTypes(loc, rewriter, regTy, memTy, tmemBase, maxnreg, pred,
                           llvmElemTy, srcValues);
    NVVM::Tcgen05WaitOp::create(rewriter, loc, NVVM::Tcgen05WaitKind::STORE);

    // Emit a barrier to ensure all threads have finished writing to tensor
    // memory before any use of the tensor memory.
    // Can be AddrSpace::TensorWrite if we emit
    // NVVM::Tcgen05WaitKind::STORE during barrier lowering
    b.barrier(triton::gpu::AddrSpace::None);

    rewriter.eraseOp(op);
    return success();
  }
};

struct TensorMemoryAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMAllocOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto ctx = op.getContext();
    Value base = nvgpu::TensorMemoryBaseAddress::create(rewriter, loc);
    Value baseInt = b.ptrtoint(i32_ty, base);
    int colOffset = cast<IntegerAttr>(op->getAttr("tensor_memory_col_offset"))
                        .getValue()
                        .getZExtValue();
    int rowOffset = cast<IntegerAttr>(op->getAttr("tensor_memory_row_offset"))
                        .getValue()
                        .getZExtValue();
    Value allocAddress = b.add(baseInt, b.i32_val(colOffset | rowOffset << 16));
    SmallVector<unsigned> order(op.getType().getRank());
    std::iota(order.begin(), order.end(), 0);
    std::reverse(order.begin(), order.end());
    auto shape = op.getType().getShape();

    if (op.getSrc()) {
      auto regTy = cast<RankedTensorType>(op.getSrc().getType());
      auto memTy = cast<MemDescType>(op.getResult().getType());
      auto llvmElemTy = getTypeConverter()->convertType(regTy.getElementType());
      auto maxnreg = getContextualMaxNReg(op);
      SmallVector<Value> srcValues =
          unpackLLElements(loc, adaptor.getSrc(), rewriter);
      Value ptr = b.inttoptr(base.getType(), allocAddress);
      lowerTMemLdStFromTypes(loc, rewriter, regTy, memTy, ptr, maxnreg,
                             b.i1_val(true), llvmElemTy, srcValues);
      NVVM::Tcgen05WaitOp::create(rewriter, loc, NVVM::Tcgen05WaitKind::STORE);
      // Emit a barrier to ensure all threads have finished writing to tensor
      // memory before any use of the tensor memory.
      // Can be AddrSpace::TensorWrite if we emit
      // NVVM::Tcgen05WaitKind::STORE during barrier lowering
      b.barrier(triton::gpu::AddrSpace::None);
    }
    // Cast to address space 3 as the shared memory object uses 3.
    // TODO: clean this up and use either a int or ptr address space 6
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    Value ptr = b.inttoptr(ptrTy, allocAddress);
    rewriter.replaceOp(op, ptr);
    return success();
  }
};

static void createCommit(ConversionPatternRewriter &rewriter, Location loc,
                         Value barrier, Value pred, bool twoCTAs) {
  PTXBuilder ptxBuilder;
  auto *barrierOperand = ptxBuilder.newAddrOperand(barrier, "r");
  std::string opcode =
      "tcgen05.commit.cta_group::" + std::to_string(twoCTAs ? 2 : 1) +
      ".mbarrier::arrive::one.shared::cluster.b64";
  auto &barrierOp = *ptxBuilder.create(opcode);
  barrierOp(barrierOperand).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static void createTcgen05Cp(ConversionPatternRewriter &rewriter, Location loc,
                            Value tmem_address, Value src_desc, Value pred,
                            TMemCopyAtom atom, bool twoCTAs) {
  PTXBuilder ptxBuilder;
  auto dst = ptxBuilder.newAddrOperand(tmem_address, "r");
  auto src = ptxBuilder.newOperand(src_desc, "l");
  std::string warp;
  if (atom.multicast == 1) {
    warp = ".warpx2::02_13";
  } else if (atom.multicast == 2) {
    warp = ".warpx2::01_23";
  } else if (atom.multicast == 3) {
    warp = ".warpx4";
  }
  std::string opcode =
      "tcgen05.cp.cta_group::" + std::to_string(twoCTAs ? 2 : 1) + warp + "." +
      std::to_string(atom.nRow) + "x" + std::to_string(atom.bCol) + "b";
  auto &op = *ptxBuilder.create(opcode);
  op({dst, src}).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static LogicalResult copySharedToTmem(ConversionPatternRewriter &rewriter,
                                      Location loc,
                                      const TypeConverter *typeConverter,
                                      triton::nvidia_gpu::TMEMCopyOp op,
                                      Value src, Value baseDst, Value pred) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = op.getContext();
  auto kOffset = str_attr("offset");
  auto kRow = str_attr("row");
  auto kCol = str_attr("col");

  MemDescType srcTy = op.getSrc().getType();
  MemDescType dstTy = op.getDst().getType();
  auto shmemLl = toLinearLayout(srcTy);
  auto tmemLl = toLinearLayout(dstTy);

  // This subtlely handles subviews
  auto cvt = tmemLl.invertAndCompose(shmemLl);

  auto bitwidth = srcTy.getElementType().getIntOrFloatBitWidth();
  auto atom = getTMemCopyAtom(cvt, bitwidth);
  // Get shmem ptr
  Type elemTy = typeConverter->convertType(srcTy.getElementType());
  auto smemObj =
      LLVM::getSharedMemoryObjectFromStruct(loc, src, elemTy, rewriter);
  auto smemBase = smemObj.getShmemAffineBase(loc, rewriter, srcTy);

  // We handle the multicast (the last 2 bits) after the descriptor
  // once we have access to the lbo/sbo
  const SmallVector<unsigned> instrShape = {32, atom.bCol / bitwidth};
  auto kWarp = str_attr("warp");
  auto cvtWarp =
      cvt.reshapeIns({{kRow, 32}, {kWarp, 4}, {kCol, cvt.getInDimSize(kCol)}})
          .sublayout({kRow, kCol}, to_vector(cvt.getOutDimNames()));

  auto loader = DotOpMmaSmemLoader::build(loc, rewriter, cvtWarp, bitwidth,
                                          smemBase, instrShape, 0, 5);
  if (failed(loader)) {
    return op->emitOpError("failed to find valid tcgen05.copy layout from "
                           "shared memory descriptor ")
           << srcTy << " to tensor memory descriptor " << dstTy;
  }
  if (loader->getDescriptor().transposed)
    return op->emitOpError("does not support transposed shared memory layout");

  bool twoCTAs = getModuleTwoCTAs(op);
  // Check correct lbo/sbo along the multicast
  auto strideRow = cvt.getBasis(kRow, llvm::Log2_32(8), kOffset);
  if ((atom.multicast & 1) == 0) {
    assert(cvt.getBasis(kRow, llvm::Log2_32(32), kOffset) ==
           strideRow * (32 / 8));
  }
  if ((atom.multicast & 2) == 0) {
    assert(cvt.getBasis(kRow, llvm::Log2_32(64), kOffset) ==
           strideRow * (64 / 8));
  }

  for (int col = 0; col < cvt.getInDimSize(kCol); col += instrShape[1]) {
    auto desc = loader->smemLoad(0, col, rewriter, loc);
    auto tmemAddr =
        b.add(b.ptrtoint(i32_ty, baseDst), b.i32_val(col * bitwidth / 32));
    createTcgen05Cp(rewriter, loc, tmemAddr, desc, pred, atom, twoCTAs);
  }
  return success();
}

struct TensorMemoryCopyOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMCopyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(lookupNumCTAs(rewriter) == 1 && "NYI");
    Location loc = op->getLoc();
    Value pred = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);
    bool twoCTAs = getModuleTwoCTAs(op);
    if (failed(copySharedToTmem(rewriter, loc, typeConverter, op,
                                adaptor.getSrc(), adaptor.getDst(), pred)))
      return failure();

    if (op.getBarrier()) {
      auto barrier = LLVM::getSharedMemoryObjectFromStruct(
          op.getLoc(), adaptor.getBarrier(), i64_ty, rewriter);
      createCommit(rewriter, loc, barrier.getBase(), pred, twoCTAs);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct MemDescIndexOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::MemDescIndexOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::MemDescIndexOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescIndexOp op, OpAdaptor adaptor,
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
    auto idx = op.getIndex();
    triton::nvidia_gpu::TMemAllocation tmemAlloc =
        triton::nvidia_gpu::getTmemAllocSizes(cast<MemDescType>(dstTy));
    int numColOffset = tmemAlloc.numCols;
    Value newBase = b.ptrtoint(rewriter.getI32Type(), tmemBase);
    newBase = LLVM::AddOp::create(
        rewriter, loc, newBase,
        LLVM::MulOp::create(rewriter, loc, idx, b.i32_val(numColOffset)));
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    rewriter.replaceOp(op, b.inttoptr(elemPtrTy, newBase));
    return success();
  }
};

class MemDescReinterpretOpConversion
    : public ConvertOpToLLVMPattern<MemDescReinterpretOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MemDescReinterpretOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSrc().getType();
    auto tmem =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(srcTy.getContext());
    if (srcTy.getMemorySpace() != tmem) {
      return failure();
    }
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

struct TMEMSubSliceOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMSubSliceOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::TMEMSubSliceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMSubSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());

    auto encoding = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
        srcTy.getEncoding());
    auto shapePerCTA = getShapePerCTA(srcTy);
    int blockN = encoding.getBlockN();
    int blockM = encoding.getBlockM();
    int offsetCol = 0;
    int offsetRow = 0;
    assert(llvm::is_contained({64, 128}, blockM) && "checked by the verifier");
    offsetCol = op.getN();

    if (blockM == 64) {
      // The layout interleaves blocks along the N dimension with the rows, such
      // that the odd numbered blocks are in lanes [16, 32), below the previous
      // even-numbered block.
      int blockOffset = op.getN() / blockN;
      if (blockOffset % 2) {
        // Offset into rows [16, 32).
        offsetRow = 16;
        // Normalize column offset to the even block.
        offsetCol -= blockN;
      }
      offsetCol -= blockN * (blockOffset / 2);
    }

    unsigned elementBitWidth = srcTy.getElementTypeBitWidth();
    if (encoding.getColStride() * elementBitWidth != 32) {
      // Adjust the column offset based on the element size.
      int numElementsPer32B = 32 / (encoding.getColStride() * elementBitWidth);
      if (offsetCol % numElementsPer32B != 0) {
        return failure();
      }
      offsetCol /= numElementsPer32B;
    }

    Value tmemBase = adaptor.getSrc();
    Value offsetVal = b.i32_val(offsetCol | offsetRow << 16);
    Value newBase = b.add(b.ptrtoint(i32_ty, tmemBase), offsetVal);
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    rewriter.replaceOp(op, b.inttoptr(elemPtrTy, newBase));
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateTensorMemoryOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<TensorMemoryCopyOpConversion, TMEMSubSliceOpConversion,
               TensorMemoryLoadOpConversion, TensorMemoryStoreOpConversion,
               TensorMemoryAllocOpConversion>(typeConverter, benefit);
}

void mlir::triton::NVIDIA::populateTensorMemorySubviewOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MemDescIndexOpConversion>(typeConverter, benefit);
  patterns.add<MemDescReinterpretOpConversion>(typeConverter, benefit);
  return;
}
