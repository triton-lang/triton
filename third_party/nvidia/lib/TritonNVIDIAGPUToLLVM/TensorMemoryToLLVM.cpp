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
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::NVIDIA;

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
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto shape = [&] {
    switch (atom) {
    case TMemAccessAtom::I32x32b:
      return NVVM::Tcgen05LdStShape::SHAPE_32X32B;
    case TMemAccessAtom::I16x64b:
      return NVVM::Tcgen05LdStShape::SHAPE_16X64B;
    case TMemAccessAtom::I16x128b:
      return NVVM::Tcgen05LdStShape::SHAPE_16X128B;
    case TMemAccessAtom::I16x256b:
      return NVVM::Tcgen05LdStShape::SHAPE_16X256B;
    case TMemAccessAtom::I16x32bx2:
      return NVVM::Tcgen05LdStShape::SHAPE_16X32BX2;
    }
    llvm_unreachable("Unknown TMemAccessAtom");
  }();
  auto shapeAttr = NVVM::Tcgen05LdStShapeAttr::get(ctx, shape);

  auto tmemPtrTy = ptr_ty(ctx, 6);
  Value tmemPtr = b.inttoptr(tmemPtrTy, address);
  tmemPtr = b.gep(tmemPtrTy, i8_ty, tmemPtr, LLVM::GEPArg(colOffset));
  Value offset = secondHalfOffset ? b.i64_val(*secondHalfOffset) : Value();

  if (!redOp) {
    // Pack adjacent 16-bit columns when the memory uses an unpacked layout.
    Type resultTy = vec_ty(i32_ty, numRegPerMessage);
    Value result = NVVM::Tcgen05LdOp::create(
        rewriter, loc, resultTy, unpacked ? rewriter.getUnitAttr() : UnitAttr(),
        shapeAttr, tmemPtr, offset);
    return {result, Value()};
  }

  assert(!unpacked && "Unpacked is unsupported with TMEM reduction");
  assert(elemTy.isF32() && "Unsupported type for TMEM reduction");
  Type dataTy = vec_ty(elemTy, numRegPerMessage);
  auto reduction = *redOp == TMEMLoadReduceModifier::MIN
                       ? NVVM::ReductionKind::MIN
                       : NVVM::ReductionKind::MAX;
  auto result = NVVM::Tcgen05LdRedOp::create(
      rewriter, loc, dataTy, elemTy, shapeAttr,
      NVVM::ReductionKindAttr::get(ctx, reduction),
      useAbs ? rewriter.getUnitAttr() : UnitAttr(),
      useNaN ? rewriter.getUnitAttr() : UnitAttr(), tmemPtr, offset);
  return {result.getData(), result.getRedVal()};
}

struct TensorMemorySparsifyLoadResult {
  SmallVector<Value> cdataVals;
  SmallVector<Value> mdataVals;
  Value redVal;
};

TensorMemorySparsifyLoadResult
createTensorMemoryLoadSparsify(Location loc, Value address, int colOffset,
                               int numTmemCols, TMEMRowOp rowOp, bool useAbs,
                               bool useNaN, bool hasRed, Type elemTy,
                               ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  assert(numTmemCols >= 4 && numTmemCols % 4 == 0 &&
         "spcompress requires a multiple of four TMEM columns");
  int numCDataRegs = numTmemCols / 2;
  int numMDataRegs = (numTmemCols + 31) / 32;

  PTXBuilder ptxBuilder;
  std::string opcode = std::string("tcgen05.ld.") + (hasRed ? "red." : "");
  opcode += "spcompress.sync.aligned.32x32b.x" + std::to_string(numTmemCols);
  opcode += rowOp == TMEMRowOp::MAX ? ".max" : ".min";
  opcode += ".sp::2:4";
  if (useAbs)
    opcode += ".abs";
  if (hasRed && useNaN)
    opcode += ".NaN";
  opcode += ".f32.b2 {";

  SmallVector<PTXInstr::Operand *> operands;
  for (int i = 0; i < numMDataRegs; ++i) {
    if (i)
      opcode += ", ";
    opcode += "$" + std::to_string(operands.size());
    operands.push_back(ptxBuilder.newOperand("=r"));
  }
  opcode += "}, {";
  for (int i = 0; i < numCDataRegs; ++i) {
    if (i)
      opcode += ", ";
    opcode += "$" + std::to_string(operands.size());
    operands.push_back(ptxBuilder.newOperand("=r"));
  }
  opcode += "}";
  if (hasRed) {
    opcode += ", $" + std::to_string(operands.size());
    operands.push_back(ptxBuilder.newOperand("=r"));
  }
  opcode += ", [$" + std::to_string(operands.size()) + " + " +
            std::to_string(colOffset) + "];";
  operands.push_back(ptxBuilder.newOperand(address, "r"));

  auto &ld = *ptxBuilder.create(opcode);
  ld(operands, /*onlyAttachMLIRArgs=*/true);

  int totalResults = numMDataRegs + numCDataRegs + (hasRed ? 1 : 0);
  Type retTy = i32_ty;
  if (totalResults > 1)
    retTy = struct_ty(SmallVector<Type>(totalResults, i32_ty));
  Value ret = ptxBuilder.launch(rewriter, loc, retTy);

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto extractI32 = [&](int idx) {
    return totalResults == 1 ? ret : b.extract_val(i32_ty, ret, idx);
  };

  TensorMemorySparsifyLoadResult result;
  for (int i = 0; i < numMDataRegs; ++i)
    result.mdataVals.push_back(extractI32(i));
  for (int i = 0; i < numCDataRegs; ++i) {
    Value raw = extractI32(numMDataRegs + i);
    result.cdataVals.push_back(elemTy == i32_ty ? raw : b.bitcast(raw, elemTy));
  }
  if (hasRed) {
    Value raw = extractI32(numMDataRegs + numCDataRegs);
    result.redVal = elemTy == i32_ty ? raw : b.bitcast(raw, elemTy);
  }
  return result;
}

struct TMemSparsifyMessage {
  int regBase;
  int row;
  int col;
};

struct TMemSparsifySchedule {
  Value tmemBase;
  SmallVector<TMemSparsifyMessage> messages;
};

static TMemSparsifySchedule
getTMemSparsifySchedule(Location loc, ConversionPatternRewriter &rewriter,
                        const LinearLayout &reps, int regsPerMessage,
                        Value tmemBase) {
  auto *ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kRow = str_attr("row");
  auto kCol = str_attr("col");

  assert(to_vector(reps.getOutDimNames()) ==
         SmallVector<StringAttr>({kRow, kCol}));
  TMemSparsifySchedule schedule;
  schedule.tmemBase = b.ptrtoint(i32_ty, tmemBase);

  Value warpId = WarpIdOp::create(rewriter, loc);
  Value warpIdInGroup = b.and_(warpId, b.i32_val(3));
  schedule.tmemBase =
      b.add(schedule.tmemBase, b.shl(warpIdInGroup, b.i32_val(5 + 16)));
  if (reps.getInDimSize(kWarp) > 4) {
    auto rowCol = applyLinearLayout(
        loc, rewriter, reps,
        {{kReg, b.i32_val(0)}, {kLane, b.i32_val(0)}, {kWarp, warpId}});
    assert(rowCol.size() == 2 && rowCol[0].first == kRow &&
           rowCol[1].first == kCol);
    Value row = rowCol[0].second;
    Value col = rowCol[1].second;
    schedule.tmemBase =
        b.add(schedule.tmemBase,
              b.or_(b.shl(row, b.i32_val(16)), col, /*disjoint=*/true));
  }

  for (int i = 0; i < reps.getInDimSize(kReg); i += regsPerMessage) {
    auto rowCol = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}});
    assert(rowCol.size() == 2 && rowCol[0].first == kRow &&
           rowCol[1].first == kCol);
    schedule.messages.push_back({i, rowCol[0].second, rowCol[1].second});
  }
  return schedule;
}

static bool tmemSparsifyMessageHasSingleRow(const LinearLayout &reps,
                                            int regBase, int regsPerMessage,
                                            int expectedRow) {
  auto *ctx = reps.getInDimNames().begin()->getContext();
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kRow = str_attr("row");
  for (int i = 0; i < regsPerMessage; ++i) {
    auto rowCol = reps.apply({{kReg, regBase + i}, {kLane, 0}, {kWarp, 0}});
    assert(rowCol.size() == 2 && rowCol[0].first == kRow);
    if (rowCol[0].second != expectedRow)
      return false;
  }
  return true;
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
    auto rowCol = applyLinearLayout(
        loc, rewriter, reps,
        {{kReg, b.i32_val(0)}, {kLane, b.i32_val(0)}, {kWarp, warpId}});
    auto [row, col] = getRowCol(rowCol);
    tmemBase = b.add(tmemBase,
                     b.or_(b.shl(row, b.i32_val(16)), col, /*disjoint*/ true));
  }

  SmallVector<Value> resultVals, redvalVals;
  for (int i = 0; i < reps.getInDimSize(kReg); i += valsPerMessage) {
    auto [row, col] =
        getRowCol(reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}}));
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
      auto loadedValues = unpackLLVector(loc, outVals, rewriter);
      resultVals.append(unpack(loadedValues, llvmElemTy, loc, rewriter));
      if (redval)
        redvalVals.push_back(redval);
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

// Combine partial reductions into one value per thread via tree reduction.
static void combinePartialReductions(Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     SmallVector<Value> &redvalVals,
                                     TMEMLoadReduceModifier redOp,
                                     bool useNaN) {
  if (redvalVals.size() <= 1)
    return;
  auto isMin = redOp == TMEMLoadReduceModifier::MIN;
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
    assert(redvalVals.size() % 2 == 0 && "redvalVals must be a multiple of 2");
    for (size_t i = 0; i < redvalVals.size(); i += 2) {
      reduced.push_back(applyMinMax(redvalVals[i], redvalVals[i + 1]));
    }
    redvalVals = std::move(reduced);
  }
}

struct TensorMemoryWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMWaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto kind = op.getKind() == TMEMWaitKind::LOAD
                    ? NVVM::Tcgen05WaitKind::LOAD
                    : NVVM::Tcgen05WaitKind::STORE;
    rewriter.replaceOpWithNewOp<NVVM::Tcgen05WaitOp>(op, kind);
    return success();
  }
};

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

    Value resultStruct = packUniqueTensorElements(
        loc, getTypeConverter(), resultVals, rewriter, op.getType());

    // tcgen05.ld.red honors true register dependencies without a wait.
    if (redOp)
      combinePartialReductions(loc, rewriter, redvalVals, *redOp, useNaN);

    // Handle reduction output if present
    SmallVector<Value> results = {resultStruct};
    if (redOp) {
      // Pack redval values into the red tensor result
      Value redStruct = packUniqueTensorElements(
          loc, getTypeConverter(), redvalVals, rewriter, op.getRed().getType());
      results.push_back(redStruct);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

static void combineSparsifyReductions(Location loc,
                                      ConversionPatternRewriter &rewriter,
                                      SmallVector<Value> &values,
                                      TMEMRowOp rowOp, bool useNaN) {
  bool isMin = rowOp == TMEMRowOp::MIN;
  auto combine = [&](Value lhs, Value rhs) {
    return useNaN ? (isMin ? LLVM::MinimumOp::create(rewriter, loc, lhs, rhs)
                           : LLVM::MaximumOp::create(rewriter, loc, lhs, rhs))
                        ->getResult(0)
                  : (isMin ? LLVM::MinNumOp::create(rewriter, loc, lhs, rhs)
                           : LLVM::MaxNumOp::create(rewriter, loc, lhs, rhs))
                        ->getResult(0);
  };
  while (values.size() > 1) {
    SmallVector<Value> reduced;
    reduced.reserve((values.size() + 1) / 2);
    for (size_t i = 0; i + 1 < values.size(); i += 2)
      reduced.push_back(combine(values[i], values[i + 1]));
    if (values.size() % 2)
      reduced.push_back(values.back());
    values = std::move(reduced);
  }
}

struct TensorMemoryLoadSparsifyOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMLoadSparsifyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMLoadSparsifyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getDep() || op.getToken())
      return rewriter.notifyMatchFailure(op, "token form is unsupported");

    Location loc = op.getLoc();
    auto *ctx = op.getContext();
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto cdataTy = cast<RankedTensorType>(op.getCdata().getType());
    auto mdataTy = cast<RankedTensorType>(op.getMdata().getType());
    auto srcEnc = cast<TensorMemoryEncodingAttr>(srcTy.getEncoding());
    Type llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    int maxnreg = getContextualMaxNReg(op);
    int sparsifyMaxnreg = std::min(maxnreg, 128);

    int64_t srcM = srcTy.getShape()[0];
    int64_t cdataN = cdataTy.getShape()[1];
    unsigned cdataBlockN = std::max<unsigned>(
        1,
        std::min<unsigned>(srcEnc.getBlockN(), static_cast<unsigned>(cdataN)));
    auto cdataMemEnc = TensorMemoryEncodingAttr::get(
        ctx, srcEnc.getBlockM(), cdataBlockN, srcEnc.getColStride(),
        srcEnc.getCGALayout(), srcEnc.getTwoCTAs());
    auto cdataMemTy =
        MemDescType::get({srcM, cdataN}, srcTy.getElementType(), cdataMemEnc,
                         srcTy.getMemorySpace(), srcTy.getMutableMemory(),
                         srcTy.getAllocShape());

    auto diag = [loc]() { return emitError(loc); };
    auto cdataInfoOr =
        computeTMemLdStEncodingInfo(cdataTy, cdataMemTy, sparsifyMaxnreg, diag);
    if (failed(cdataInfoOr))
      return rewriter.notifyMatchFailure(
          op, "failed to compute cdata TMEM encoding info");
    TMemLdStEncodingInfo cdataInfo = *cdataInfoOr;
    if (cdataInfo.unpacked || cdataInfo.secondHalfOffset ||
        cdataInfo.atom != TMemAccessAtom::I32x32b) {
      return rewriter.notifyMatchFailure(
          op, "unsupported cdata layout for spcompress lowering");
    }

    int64_t mdataN = mdataTy.getShape()[1];
    unsigned mdataBlockN = std::max<unsigned>(
        1,
        std::min<unsigned>(srcEnc.getBlockN(), static_cast<unsigned>(mdataN)));
    auto mdataMemEnc = TensorMemoryEncodingAttr::get(
        ctx, srcEnc.getBlockM(), mdataBlockN, srcEnc.getColStride(),
        srcEnc.getCGALayout(), srcEnc.getTwoCTAs());
    auto mdataMemTy =
        MemDescType::get({srcM, mdataN}, rewriter.getI32Type(), mdataMemEnc,
                         srcTy.getMemorySpace(), srcTy.getMutableMemory(),
                         srcTy.getAllocShape());
    auto mdataInfoOr =
        computeTMemLdStEncodingInfo(mdataTy, mdataMemTy, maxnreg, diag);
    if (failed(mdataInfoOr))
      return rewriter.notifyMatchFailure(
          op, "failed to compute mdata TMEM encoding info");
    TMemLdStEncodingInfo mdataInfo = *mdataInfoOr;
    if (mdataInfo.unpacked || mdataInfo.secondHalfOffset ||
        mdataInfo.atom != TMemAccessAtom::I32x32b) {
      return rewriter.notifyMatchFailure(
          op, "unsupported mdata layout for spcompress lowering");
    }

    bool hasRed = static_cast<bool>(op.getRed());
    bool useNaN = op.getNaN().value_or(false);
    auto kReg = str_attr("register");
    int expectedCDataRegs = cdataInfo.reps.getInDimSize(kReg);
    int expectedMDataRegs = mdataInfo.reps.getInDimSize(kReg);
    SmallVector<Value> cdataVals, mdataVals, redVals;
    auto schedule =
        getTMemSparsifySchedule(loc, rewriter, cdataInfo.reps,
                                cdataInfo.numRegsPerMessage, adaptor.getSrc());

    for (const TMemSparsifyMessage &message : schedule.messages) {
      int srcCol = message.col * 2;
      if (srcCol % 32)
        return rewriter.notifyMatchFailure(
            op, "unsupported sparsify source-column alignment");
      if (!tmemSparsifyMessageHasSingleRow(cdataInfo.reps, message.regBase,
                                           cdataInfo.numRegsPerMessage,
                                           message.row)) {
        return rewriter.notifyMatchFailure(
            op, "unsupported mixed-row sparsify message");
      }

      int numTmemCols = cdataInfo.numRegsPerMessage * 2;
      int staticOffset = srcCol | (message.row << 16);
      auto loaded = createTensorMemoryLoadSparsify(
          loc, schedule.tmemBase, staticOffset, numTmemCols, op.getRowOp(),
          op.getAbs().value_or(false), useNaN, hasRed, llvmElemTy, rewriter);
      if (loaded.cdataVals.size() !=
              static_cast<size_t>(cdataInfo.numRegsPerMessage) ||
          loaded.mdataVals.size() != static_cast<size_t>(numTmemCols / 32)) {
        return rewriter.notifyMatchFailure(
            op, "unexpected spcompress register count");
      }
      cdataVals.append(loaded.cdataVals.begin(), loaded.cdataVals.end());
      mdataVals.append(loaded.mdataVals.begin(), loaded.mdataVals.end());
      if (loaded.redVal)
        redVals.push_back(loaded.redVal);
    }

    if (cdataVals.size() != static_cast<size_t>(expectedCDataRegs) ||
        mdataVals.size() != static_cast<size_t>(expectedMDataRegs)) {
      return rewriter.notifyMatchFailure(
          op, "unexpected total spcompress register count");
    }
    if (!cdataInfo.perm.isIdentity())
      cdataVals = cdataInfo.perm.inverse().apply(cdataVals);
    if (!mdataInfo.perm.isIdentity())
      mdataVals = mdataInfo.perm.inverse().apply(mdataVals);

    SmallVector<Value> replacement;
    replacement.push_back(packUniqueTensorElements(
        loc, getTypeConverter(), cdataVals, rewriter, op.getCdata().getType()));
    replacement.push_back(packUniqueTensorElements(
        loc, getTypeConverter(), mdataVals, rewriter, op.getMdata().getType()));
    if (hasRed) {
      if (redVals.empty())
        return rewriter.notifyMatchFailure(op, "missing reduction values");
      combineSparsifyReductions(loc, rewriter, redVals, op.getRowOp(), useNaN);
      replacement.push_back(packUniqueTensorElements(
          loc, getTypeConverter(), redVals, rewriter, op.getRed().getType()));
    }

    // TMEM waits are inserted at TTGIR level for memory ordering.
    rewriter.replaceOp(op, replacement);
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
    auto memTy = cast<MemDescType>(op.getDst().getType());
    auto regTy = cast<RankedTensorType>(op.getSrc().getType());

    SmallVector<Value> srcValues =
        unpackUniqueTensorElements(loc, adaptor.getSrc(), rewriter);
    auto maxnreg = getContextualMaxNReg(op);
    lowerTMemLdStFromTypes(loc, rewriter, regTy, memTy, tmemBase, maxnreg, pred,
                           llvmElemTy, srcValues);
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

    if (op.getSrc()) {
      auto regTy = cast<RankedTensorType>(op.getSrc().getType());
      auto memTy = cast<MemDescType>(op.getResult().getType());
      auto llvmElemTy = getTypeConverter()->convertType(regTy.getElementType());
      auto maxnreg = getContextualMaxNReg(op);
      SmallVector<Value> srcValues =
          unpackUniqueTensorElements(loc, adaptor.getSrc(), rewriter);
      Value ptr = b.inttoptr(base.getType(), allocAddress);
      lowerTMemLdStFromTypes(loc, rewriter, regTy, memTy, ptr, maxnreg,
                             b.i1_val(true), llvmElemTy, srcValues);
    }
    // Cast to address space 3 as the shared memory object uses 3.
    // TODO: clean this up and use either a int or ptr address space 6
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    Value ptr = b.inttoptr(ptrTy, allocAddress);
    rewriter.replaceOp(op, ptr);
    return success();
  }
};

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
  auto kBlock = str_attr("block");

  MemDescType srcTy = op.getSrc().getType();
  MemDescType dstTy = op.getDst().getType();
  auto shmemLl = toLinearLayout(srcTy);
  auto tmemLl = toLinearLayout(dstTy);

  // This subtlely handles subviews
  auto cvt = tmemLl.invertAndCompose(shmemLl);

  unsigned bitwidth = getIntOrFloatOrPtrBitWidth(srcTy.getElementType());
  uint64_t holeMask = 0;
  // Zero col bases are physical holes. Compact them for the SMEM descriptor,
  // then skip the corresponding physical TMEM ranges when issuing copies.
  if (isa<TensorMemoryEncodingAttr>(dstTy.getEncoding())) {
    unsigned colBits = tmemLl.getInDimSizeLog2(kCol);
    auto outDims = llvm::to_vector(tmemLl.getOutDimNames());
    holeMask = ~getInputBasisMask(tmemLl, kCol, outDims) &
               llvm::maskTrailingOnes<uint64_t>(colBits);
  }
  auto compact = holeMask ? cvt.removeZeroBasesAlongDim(kCol) : cvt;
  unsigned contiguousCols = holeMask
                                ? uint64_t{1} << llvm::countr_zero(holeMask)
                                : compact.getInDimSize(kCol);
  auto atom = getTMemCopyAtom(
      compact.resizeInDim(
          kCol, std::min<unsigned>(contiguousCols, compact.getInDimSize(kCol))),
      bitwidth);
  // Get shmem ptr
  Type elemTy = typeConverter->convertType(srcTy.getElementType());
  auto smemObj =
      LLVM::getSharedMemoryObjectFromStruct(loc, src, elemTy, rewriter);
  auto smemBase = smemObj.getShmemAffineBase(loc, rewriter, srcTy);

  // We handle the multicast (the last 2 bits) after the descriptor
  // once we have access to the lbo/sbo
  const SmallVector<unsigned> instrShape = {32, atom.bCol / bitwidth};
  auto kWarp = str_attr("warp");
  auto cvtWarp = compact
                     .reshapeIns({{kRow, 32},
                                  {kWarp, 4},
                                  {kCol, compact.getInDimSize(kCol)},
                                  {kBlock, compact.getInDimSize(kBlock)}})
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

  int logicalCol = 0;
  for (int col = 0; col < cvt.getInDimSize(kCol); col += instrShape[1]) {
    if (col & holeMask)
      continue;
    auto desc = loader->smemLoad(0, logicalCol, rewriter, loc);
    auto tmemAddr =
        b.add(b.ptrtoint(i32_ty, baseDst), b.i32_val(col * bitwidth / 32));
    createTcgen05Cp(rewriter, loc, tmemAddr, desc, pred, atom, twoCTAs);
    logicalCol += instrShape[1];
  }
  return success();
}

struct TensorMemoryCopyOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMCopyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value pred = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);
    bool twoCTAs = getModuleTwoCTAs(op);
    // Similar to twoCTA tcgen05.mma, the 2CTA version of this op should only be
    // emitted from the lead CTA.
    if (twoCTAs) {
      Value cluster0 = LLVM::NVIDIA::createLeadCTAPredicate(loc, rewriter);
      pred = TritonLLVMOpBuilder(loc, rewriter).and_(pred, cluster0);
    }

    if (failed(copySharedToTmem(rewriter, loc, typeConverter, op,
                                adaptor.getSrc(), adaptor.getDst(), pred)))
      return failure();

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
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    uint32_t offset = getTMemSubSliceOffset(srcTy, op.getOffset(), op.getDim());

    Value tmemBase = adaptor.getSrc();
    Value offsetVal = b.i32_val(offset);
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
  patterns
      .add<TensorMemoryCopyOpConversion, TensorMemoryLoadOpConversion,
           TensorMemoryLoadSparsifyOpConversion, TensorMemoryStoreOpConversion,
           TensorMemoryAllocOpConversion, TensorMemoryWaitOpConversion>(
          typeConverter, benefit);
}

void mlir::triton::NVIDIA::populateTensorMemorySubviewOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MemDescReinterpretOpConversion, MemDescIndexOpConversion,
               TMEMSubSliceOpConversion>(typeConverter, benefit);
}
