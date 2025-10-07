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

struct TMemAccessAtom {
  int colsPerThread;
  int rowsPerThread;
  const char *opShape;
};

constexpr TMemAccessAtom TMemAccess32x32b{
    1 /*colsPerThread*/, 1 /*rowsPerThread*/, "32x32b" /*opShape*/};

constexpr TMemAccessAtom TMemAccess16x256b{
    2 /*colsPerThread*/, 2 /*rowsPerThread*/, "16x256b" /*opShape*/};

constexpr TMemAccessAtom TMemAccess16x32bx2{
    1 /*colsPerThread*/, 1 /*rowsPerThread*/, "16x32bx2" /*opShape*/};

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

// Similar to largestVectorisation in TritonGPUToLLVM/Utility.cpp
std::optional<std::tuple<LinearLayout, ColumnAction, int>>
getVec(const LinearLayout &cvt, const LinearLayout &tile, int maxnreg) {
  auto *ctx = cvt.getInDimNames().begin()->getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kCol = StringAttr::get(ctx, "col");
  LinearLayout reps, vec;
  ColumnAction perm;
  // Heuristic:
  // Do not use more than half the registers as otherwise it's prone to spilling
  assert(maxnreg / 2 <= largestTmemLoadStore);
  auto maxReg = maxnreg / 2;
  // Heuristic:
  // If maxnreg is 256 and we need more than one message, we don't use max
  // vectorisation as ptxas' scheduler breaks...
  if (maxnreg == 256 && cvt.getInDimSize(kReg) > maxReg) {
    maxReg /= 2;
  }
  auto maxVec = maxReg / tile.getInDimSize(kReg);
  int i = 1;
  for (; i <= maxVec; i *= 2) {
    vec = LinearLayout::identity1D(i, kReg, kCol);
    auto vecTile = tile * vec;
    auto maybePerm = regPermForDivide(cvt, vecTile, /*left=*/true);
    if (!maybePerm) {
      if (i == 1) {
        // Couldn't lower the tile
        return std::nullopt;
      }
      break;
    }
    // nb. We could remove this part once we are confident the algo works
    perm = *maybePerm;
    auto newCvt = maybePerm->apply(cvt);
    auto maybeReps = getReps(newCvt, vecTile);
    if (!maybeReps.has_value()) {
      if (i == 1) {
        // Couldn't lower the tile
        return std::nullopt;
      }
      break;
    }
    reps = *maybeReps;
  }
  // i is the smallest power of 2 that *cannot* be used to lower the tile
  // so we return i / 2.
  assert(i > 1);
  return std::make_tuple(std::move(reps), std::move(perm),
                         (i / 2) * tile.getInDimSize(kReg));
}

LinearLayout getTileLayout(MLIRContext *ctx, TMemAccessAtom atom,
                           bool unpacked) {
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kRow = str_attr("row");
  auto kCol = str_attr("col");
  // Set the output order to be kRow, kCol and the input order to be kReg first
  LinearLayout tile = LinearLayout::identity1D(1, kReg, kRow) *
                      LinearLayout::identity1D(1, kReg, kCol);
  if (atom.opShape == std::string("32x32b")) {
    tile *= LinearLayout::identity1D(32, kLane, kRow);
  } else if (atom.opShape == std::string("16x32bx2")) {
    tile *= LinearLayout::identity1D(16, kLane, kRow);
  } else if (atom.opShape == std::string("16x256b")) {
    tile *= LinearLayout::identity1D(2, kReg, kCol) *
            LinearLayout::identity1D(4, kLane, kCol) *
            LinearLayout::identity1D(8, kLane, kRow) *
            LinearLayout::identity1D(2, kReg, kRow);
  } else {
    llvm_unreachable("Unsupported TMEM access atom");
  }
  // Each register moves 32/bitwidth (= 2) columns when unpacked
  if (unpacked) {
    tile = LinearLayout::zeros1D(1, kReg, kCol, 2) * tile;
  }
  auto nCol = tile.getOutDimSize(kCol);
  auto bases = tile.getBases();
  bases[kWarp].push_back({32, 0});
  bases[kWarp].push_back({64, 0});
  auto ret = LinearLayout(bases, {{kRow, 128}, {kCol, nCol}}, false);
  return ret;
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
                             bool unpacked, const TMemAccessAtom &atom,
                             ConversionPatternRewriter &rewriter) {
  PTXBuilder ptxBuilder;
  std::string packedStr = unpacked ? ".unpack::16b" : "";
  unsigned numRepeats = srcs.size() / (atom.rowsPerThread * atom.colsPerThread);
  std::string opcode = "@$0 tcgen05.st.sync.aligned." +
                       std::string(atom.opShape) + ".x" +
                       std::to_string(numRepeats) + packedStr;
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

  auto &st = *ptxBuilder.create<PTXInstr>(opcode);
  st(operands, /*onlyAttachMLIRArgs=*/true);
  Type voidTy = void_ty(rewriter.getContext());
  ptxBuilder.launch(rewriter, loc, voidTy);
}

// Get the maximum number of registers per thread based on the context. This is
// by default 256, but it can be overridden by `ttg.maxnreg` set on the module
// or a contextual register limit set by the compiler on partitions.
int getContextualMaxNReg(Operation *op) {
  // Check the immediate parent op to see if it places a register constraint.
  auto getFromParent = [](Operation *op) -> std::optional<int> {
    Operation *parent = op->getParentOp();
    if (auto mod = dyn_cast<ModuleOp>(parent)) {
      if (auto attr = mod->getAttrOfType<IntegerAttr>(AttrMaxRegistersName))
        return attr.getInt();
      return {};
    }

    if (auto partitions = dyn_cast<WarpSpecializePartitionsOp>(parent)) {
      // Check if the partition has reduced registers.
      unsigned idx = op->getParentRegion()->getRegionNumber();
      if (auto actRegisters = partitions.getParentOp().getActualRegisters())
        return (*actRegisters)[1 + idx];
      return {};
    }

    if (auto wsOp = dyn_cast<WarpSpecializeOp>(op->getParentOp())) {
      // Check the register usage of the default warpgroup.
      if (auto actRegisters = wsOp.getActualRegisters())
        return actRegisters->front();
      return {};
    }

    return {};
  };

  // PTXAS validates the register usage of `tcgen05.ld` and `tcgen05.st`
  // instructions based on the static number of registers set on the module, not
  // the dynamic allocation. This just means the register limit used for the
  // purpose of subtiling TMEM messages cannot be higher than the module's.
  auto mod = op->getParentOfType<ModuleOp>();
  int maxnreg = maxRegisters;

  for (; op != mod; op = op->getParentOp()) {
    if (std::optional<int> limit = getFromParent(op)) {
      maxnreg = std::min(maxnreg, *limit);
      break;
    }
  }

  if (auto maxnregAttr = mod->getAttrOfType<IntegerAttr>(AttrMaxRegistersName))
    maxnreg = std::min<int>(maxnreg, maxnregAttr.getInt());

  return maxnreg;
}

Value createTensorMemoryLoad(Location loc, MLIRContext *ctx, Value address,
                             int colOffset, std::optional<int> secondHalfOffset,
                             bool unpacked, int numRegPerMessage,
                             const TMemAccessAtom &atom,
                             ConversionPatternRewriter &rewriter) {
  PTXBuilder ptxBuilder;
  // If the memory is unpacked we need to pack on the fly when loading.
  std::string packedStr = unpacked ? ".pack::16b" : "";
  unsigned numRepeats =
      numRegPerMessage / (atom.rowsPerThread * atom.colsPerThread);
  std::string opcode = "tcgen05.ld.sync.aligned." + std::string(atom.opShape) +
                       ".x" + std::to_string(numRepeats) + packedStr + ".b32 {";

  SmallVector<PTXInstr::Operand *> operands;
  for (int i = 0; i < numRegPerMessage; i++) {
    opcode += "$" + std::to_string(i);
    auto *resultOp = ptxBuilder.newOperand("=r");
    operands.push_back(resultOp);
    if (i < numRegPerMessage - 1)
      opcode += ", ";
  }
  opcode += "}, [$" + std::to_string(numRegPerMessage) + " + " +
            std::to_string(colOffset) + "]";
  if (secondHalfOffset)
    opcode += ", " + std::to_string(*secondHalfOffset);
  opcode += ";";
  operands.push_back(ptxBuilder.newOperand(address, "r"));
  auto &ld = *ptxBuilder.create<PTXInstr>(opcode);
  ld(operands, /*onlyAttachMLIRArgs=*/true);

  // LLVM inline_asm with 1 result cannot return a struct.
  Type retTy;
  if (numRegPerMessage == 1) {
    retTy = i32_ty;
  } else {
    SmallVector<Type> elemTypes(numRegPerMessage, i32_ty);
    retTy = struct_ty(elemTypes);
  }
  Value ret = ptxBuilder.launch(rewriter, loc, retTy);
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

FailureOr<SmallVector<Value>>
lowerTMemLdSt(Location loc, MLIRContext *ctx,
              ConversionPatternRewriter &rewriter, const LinearLayout &reps,
              ArrayRef<Value> vals, TMemAccessAtom atom, Type llvmElemTy,
              Value tmemBase, Value pred, int valsPerMessage, bool unpacked,
              std::optional<uint32_t> secondHalfOffset) {
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

  Value warpId = rewriter.create<nvgpu::WarpIdOp>(loc);
  // Map warpId to rows 32 and 64
  auto warpIdInGroup = b.and_(warpId, b.i32_val(3));
  tmemBase = b.add(tmemBase, b.shl(warpIdInGroup, b.i32_val(5 + 16)));
  // Add warp groups to tmemBase
  if (reps.getInDimSize(kWarp) > 4) {
    auto rowCol = applyLinearLayout(
        loc, rewriter, reps,
        {{kReg, b.i32_val(0)}, {kLane, b.i32_val(0)}, {kWarp, warpId}});
    auto [row, col] = getRowCol(rowCol);
    tmemBase = b.add(tmemBase,
                     b.or_(b.shl(row, b.i32_val(16)), col, /*disjoint*/ true));
  }

  SmallVector<Value> resultVals;
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
      Value outVals = createTensorMemoryLoad(
          loc, ctx, tmemBase, /*colOffset=*/staticOffset,
          /*secondHalfOffset=*/secondHalfOffset,
          /*unpacked=*/unpacked,
          /*numRegPerMessage=*/valsPerMessage, atom, rewriter);
      resultVals.append(
          unpackResults(outVals, llvmElemTy, valsPerMessage, loc, rewriter));
    }
  }

  return resultVals;
}

FailureOr<SmallVector<Value>>
lowerTMemLdSt(Location loc, MLIRContext *ctx,
              ConversionPatternRewriter &rewriter, const LinearLayout &cvt,
              ArrayRef<Value> vals, Type llvmElemTy, Value tmemBase,
              int maxnreg, Value pred, bool isScales = false,
              bool unpacked = false) {
  assert(cvt.getNumOutDims() == 2);
  bool isStore = !vals.empty();
  // Remove broadcasting in the registers
  auto removeBroadcastSrc = actionRemoveBroadcastedRegs(cvt);
  if (!removeBroadcastSrc.isIdentity()) {
    auto prmtCvt = removeBroadcastSrc.apply(cvt);
    auto inVals = to_vector(vals);
    if (isStore) {
      inVals = removeBroadcastSrc.apply(inVals);
    }
    auto outValsOr =
        lowerTMemLdSt(loc, ctx, rewriter, prmtCvt, inVals, llvmElemTy, tmemBase,
                      maxnreg, pred, isScales, unpacked);
    if (failed(outValsOr))
      return failure();
    auto outVals = std::move(*outValsOr);
    if (!isStore) {
      outVals = broadcastAs(outVals, cvt);
    }
    return outVals;
  }
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kRow = str_attr("row");
  auto kCol = str_attr("col");

  // Default to unpacked=false for bitwidth == 32
  if (llvmElemTy.getIntOrFloatBitWidth() < 32) {
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    LinearLayout quot;
    Type packedElemTy;
    int bestContig = 1;
    for (int contig = 1; bitwidth * contig <= 32; contig *= 2) {
      auto maybeQuot =
          divideLeft(cvt, LinearLayout::identity1D(contig, kReg, kCol));
      if (!maybeQuot)
        break;
      quot = *maybeQuot;
      bestContig = contig;
    }
    bool padding = false;
    if (bestContig > 1) {
      // There are contiguous elements along kCol, so we can pack them into a
      // larger dtype
      unpacked = false;
      packedElemTy = int_ty(bitwidth * bestContig);
    } else if (auto maybeQuot = divideLeft(
                   cvt, LinearLayout::zeros1D(1, kReg, kCol, 32 / bitwidth) *
                            LinearLayout::identity1D(2, kReg, kCol));
               bitwidth == 16 && maybeQuot) {
      // Unpacked just supported for bitwidth 16
      unpacked = true;
      quot = *maybeQuot;
      packedElemTy = i32_ty;
    } else if (auto maybeQuot = divideLeft(
                   cvt, LinearLayout::zeros1D(1, kReg, kCol, 32 / bitwidth))) {
      // We software-pad the elements when we either do not have enough elements
      // to fill a full 32b register, e.g., colN = 1 and colStride != 1 or when
      // bitwidth == 8 (this happens with scales with K=1).
      // These two cases are mostly supported for testing purposes.
      unpacked = bitwidth == 16;
      quot = *maybeQuot;
      packedElemTy = i32_ty;
      padding = true;
    } else {
      emitError(loc, "Failed to lower TMEM load/store: TMEM layout is not "
                     "packed or unpacked");
      return failure();
    }
    // When unpacked each register moves 32/bitwidth (= 2) columns
    if (unpacked) {
      quot = LinearLayout::zeros1D(1, kReg, kCol, 32 / bitwidth) * quot;
    }
    SmallVector<Value> inVals;
    if (isStore) {
      inVals = pack(vals, packedElemTy, loc, rewriter, padding);
    }
    auto outValsOr =
        lowerTMemLdSt(loc, ctx, rewriter, quot, inVals, packedElemTy, tmemBase,
                      maxnreg, pred, isScales, unpacked);
    if (failed(outValsOr))
      return failure();
    auto outVals = std::move(*outValsOr);
    if (!isStore) {
      outVals = unpack(outVals, llvmElemTy, loc, rewriter, padding);
    }
    return outVals;
  }

  assert(!isStore || cvt.getInDimSize(kReg) == vals.size());
  assert(llvmElemTy.getIntOrFloatBitWidth() == 32);

  // The algorithm goes as:
  // - Try to match the tile with one of the standard messages
  // - If it doesn't match, we use the 16x32bx2 message
  // Note that it can match one and only one of the layouts, even after register
  // reordering, as the layouts yield predetermined positions for the lanes
  // We store the instruction, the resulting reps layout, the permutation and
  // the number of registers per message
  std::optional<std::tuple<TMemAccessAtom, LinearLayout, ColumnAction, int>>
      msgInfo;
  for (auto atom : {TMemAccess32x32b, TMemAccess16x256b}) {
    auto tile = getTileLayout(ctx, atom, unpacked);
    auto maybeReps = getVec(cvt, tile, maxnreg);
    if (maybeReps) {
      // Cannot match more than one
      msgInfo = {atom, std::get<0>(*maybeReps), std::get<1>(*maybeReps),
                 std::get<2>(*maybeReps)};
      break;
    }
  }
  std::optional<uint32_t> secondHalfOffset = std::nullopt;
  if (!msgInfo) {
    // Quotient by the smaller tile and then, if possible, we set the
    // secondHalfOffset to the last kLane basis
    auto tile = getTileLayout(ctx, TMemAccess16x32bx2, unpacked);
    auto maybeReps = getVec(cvt, tile, maxnreg);
    if (maybeReps) {
      auto [reps, perm, numRegsPerMessage] = std::move(*maybeReps);
      // Find the last kLane basis and use it as secondHalfOffset
      auto row = reps.getBasis(kLane, 4, kRow);
      auto col = reps.getBasis(kLane, 4, kCol);
      secondHalfOffset = (row << 16) | col;
      if (*secondHalfOffset == 0) {
        // Workaround for ptxas bug, we cannot use secondHalfOffset = 0 to write
        // only 16 elements. We use secondHalfOffset = 1 instead and we pad the
        // allocation.
        assert(isScales &&
               "Only supported for scales as we pad the allocation.");
        secondHalfOffset = 1;
      }
      // We "quotient it out", meaning we remove the last basis from reps
      auto basis = reps.getBases();
      basis[kLane][4] = {0, 0};
      reps = LinearLayout(basis, reps.getOutDims(), /*isSurjective=*/false);
      msgInfo = {TMemAccess16x32bx2, reps, perm, numRegsPerMessage};
    }
  }

  if (!msgInfo) {
    emitError(loc, "Failed to lower TMEM load/store: unsupported dst layout\n" +
                       cvt.toString());
    return failure();
  }
  auto [atom, reps, perm, numRegsPerMessage] = std::move(msgInfo.value());

  SmallVector<Value> inVals;
  if (isStore) {
    inVals = to_vector(vals);
    inVals = perm.apply(inVals);
  }
  auto outValsOr = lowerTMemLdSt(loc, ctx, rewriter, reps, inVals, atom,
                                 llvmElemTy, tmemBase, pred, numRegsPerMessage,
                                 unpacked, secondHalfOffset);
  if (failed(outValsOr))
    return failure();
  auto outVals = std::move(*outValsOr);
  assert(isStore || outVals.size() == cvt.getInDimSize(kReg));
  if (!isStore) {
    outVals = perm.inverse().apply(outVals);
  }
  return outVals;
}

static FailureOr<SmallVector<Value>> lowerTMemLdStFromTypes(
    Location loc, MLIRContext *ctx, ConversionPatternRewriter &rewriter,
    RankedTensorType regTy, MemDescType memTy, Value tmemBase, int maxnreg,
    Value pred, Type llvmElemTy, ArrayRef<Value> vals) {
  auto memLayout = toLinearLayout(memTy);
  auto regLayout = toLinearLayout(regTy);
  auto cvt = regLayout.invertAndCompose(memLayout);
  auto kWarp = str_attr("warp");
  auto kRow = str_attr("row");
  // Warps 0-3 must map to row=32 and row=64 whether with broadcasting or not
  if (!(regLayout.getBasis(kWarp, 0) == memLayout.getBasis(kRow, 5) &&
        regLayout.getBasis(kWarp, 1) == memLayout.getBasis(kRow, 6))) {
    emitError(
        loc,
        "Failed to lower TMEM load/store: unsupported src/dst combination\n" +
            regLayout.toString() + "\n" + memLayout.toString());
    return failure();
  }
  // Map warp bases to row=32 and row=64 in the cvt. This would be done
  // automatically in `invertAndCompose` if we had a different dimension name
  // for these rows. We can do this in the future if needed.
  auto bases = cvt.getBases();
  bases[kWarp][0] = {32, 0};
  bases[kWarp][1] = {64, 0};
  cvt = LinearLayout(bases, cvt.getOutDims(),
                     /*isSurjective=*/cvt.isSurjective());

  // tmemBase already encodes CTA/block offsets so we just remove them from the
  // cvt
  auto kBlock = str_attr("block");
  auto kCol = str_attr("col");
  auto nCTAs = cvt.getInDimSize(kBlock);
  auto maybeQuot =
      divideRight(cvt, LinearLayout::identity1D(nCTAs, kBlock, kCol));
  assert(maybeQuot.has_value());
  auto quot = maybeQuot->unsqueezeIn(kBlock);

  bool isScales = isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding());
  return lowerTMemLdSt(loc, ctx, rewriter, quot, vals, llvmElemTy, tmemBase,
                       maxnreg, pred, isScales);
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

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto maxnreg = getContextualMaxNReg(op);
    auto resultValsOr =
        lowerTMemLdStFromTypes(loc, ctx, rewriter, regTy, memTy, tmemBase,
                               maxnreg, b.i1_val(true), llvmElemTy, {});
    if (failed(resultValsOr))
      return failure();

    Type structTy = getTypeConverter()->convertType(op.getType());
    Value resultStruct = packLLElements(loc, getTypeConverter(), *resultValsOr,
                                        rewriter, structTy);
    // Wait insertion could be moved to the TTGIR level if needed.
    rewriter.create<NVVM::Tcgen05WaitOp>(loc, NVVM::Tcgen05WaitKind::LOAD);
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
    auto lowered =
        lowerTMemLdStFromTypes(loc, ctx, rewriter, regTy, memTy, tmemBase,
                               maxnreg, pred, llvmElemTy, srcValues);
    if (failed(lowered))
      return failure();
    rewriter.create<NVVM::Tcgen05WaitOp>(loc, NVVM::Tcgen05WaitKind::STORE);

    // Emit a barrier to ensure all threads have finished writing to tensor
    // memory before any use of the tensor memory.
    b.barrier();

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
    Value base = rewriter.create<nvgpu::TensorMemoryBaseAddress>(loc);
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
      auto lowered =
          lowerTMemLdStFromTypes(loc, ctx, rewriter, regTy, memTy, ptr, maxnreg,
                                 b.i1_val(true), llvmElemTy, srcValues);
      if (failed(lowered))
        return failure();
      rewriter.create<NVVM::Tcgen05WaitOp>(loc, NVVM::Tcgen05WaitKind::STORE);
      // Emit a barrier to ensure all threads have finished writing to tensor
      // memory before any use of the tensor memory.
      b.barrier();
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
                         Value barrier, Value pred) {
  PTXBuilder ptxBuilder;
  auto *barrierOperand = ptxBuilder.newAddrOperand(barrier, "r");
  std::string opcode = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64";
  auto &barrierOp = *ptxBuilder.create<PTXInstr>(opcode);
  barrierOp(barrierOperand).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static void createTcgen05Cp(ConversionPatternRewriter &rewriter, Location loc,
                            Value tmem_address, Value src_desc, Value pred,
                            TMemCopyAtom atom) {
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
  std::string opcode = "tcgen05.cp.cta_group::1" + warp + "." +
                       std::to_string(atom.nRow) + "x" +
                       std::to_string(atom.bCol) + "b";
  auto &op = *ptxBuilder.create<PTXInstr>(opcode);
  op({dst, src}).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static void copySharedToTmem(ConversionPatternRewriter &rewriter, Location loc,
                             const TypeConverter *typeConverter,
                             triton::nvidia_gpu::TMEMCopyOp op, Value src,
                             Value baseDst, Value pred) {
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
                                          smemBase, instrShape, 5);
  assert(!loader.getDescriptor().transposed);

  // The lbo/sbo are swapped for swizzling == 0 when passing a descriptor to
  // tcgen05.cp vs passing it to wgmma/tcgen05.mma!!
  auto &descData = loader.getDescriptor();
  if (descData.swizzlingByteWidth == 0) {
    auto lbo = descData.descriptor.leadDimensionBaseOffset;
    auto sbo = descData.descriptor.strideDimensionBaseOffset;
    descData.descriptor.leadDimensionBaseOffset = sbo;
    descData.descriptor.strideDimensionBaseOffset = lbo;
  }
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
    // smemLoad takes the colRep. It'd be nice to change this but we would need
    // to change the wgmma and mmav5 lowering
    auto desc = loader.smemLoad(0, col / instrShape[1], rewriter, loc);
    auto tmemAddr =
        b.or_(b.ptrtoint(i32_ty, baseDst), b.i32_val(col * bitwidth / 32),
              /*disjoint=*/true);
    createTcgen05Cp(rewriter, loc, tmemAddr, desc, pred, atom);
  }
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
    copySharedToTmem(rewriter, loc, typeConverter, op, adaptor.getSrc(),
                     adaptor.getDst(), pred);

    if (op.getBarrier()) {
      auto barrier = LLVM::getSharedMemoryObjectFromStruct(
          op.getLoc(), adaptor.getBarrier(), i64_ty, rewriter);
      createCommit(rewriter, loc, barrier.getBase(), pred);
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
    newBase = rewriter.create<LLVM::AddOp>(
        loc, newBase,
        rewriter.create<LLVM::MulOp>(loc, idx, b.i32_val(numColOffset)));
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
