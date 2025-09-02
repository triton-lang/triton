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
    .colsPerThread = 1, .rowsPerThread = 1, .opShape = "32x32b"};

constexpr TMemAccessAtom TMemAccess16x256b{
    .colsPerThread = 2, .rowsPerThread = 2, .opShape = "16x256b"};

constexpr TMemAccessAtom TMemAccess16x32bx2{
    .colsPerThread = 1, .rowsPerThread = 1, .opShape = "16x32bx2"};

std::optional<LinearLayout> getReps(const LinearLayout &cvt,
                                    const LinearLayout &tile) {
  // Close cousin of doing zerosLike(tile) * divideLeft(cvt, tile)
  // This one is a tad more general in the sense that it allows to divide
  //  cvt:
  // - register=1 -> (0, 1)
  //   register=2 -> (8, 0)
  //   register=4 -> (0, 8)
  //   register=8 -> (0, 16)
  //   register=16 -> (0, 32)
  //   register=32 -> (0, 64)
  //   register=64 -> (16, 0)
  // - lane=1 -> (0, 2)
  //   lane=2 -> (0, 4)
  //   lane=4 -> (1, 0)
  //   lane=8 -> (2, 0)
  //   lane=16 -> (4, 0)
  // - warp=1 -> (32, 0)
  //   warp=2 -> (64, 0)
  // - block is a size 1 dimension
  // where out dims are: [row (size 128), col (size 128)]
  // tile:
  //  - register=1 -> (0, 1)
  //    register=2 -> (8, 0)
  //  - lane=1 -> (0, 2)
  //    lane=2 -> (0, 4)
  //    lane=4 -> (1, 0)
  //    lane=8 -> (2, 0)
  //    lane=16 -> (4, 0)
  //  - warp=1 -> (32, 0)
  //    warp=2 -> (64, 0)
  // where out dims are: [row (size 128), col (size 8)]
  // which would not be possible to lower via the divideLeft approach as we
  // cannot divide by the tile given the `register=64 -> (16, 0)` basis.

  // Ensure tile out-dims are subset of cvt out-dims.
  for (auto od : tile.getOutDimNames())
    assert(cvt.hasOutDim(od) && "tile out-dims must be contained in cvt");

  // Precompute tile out-dim bit-widths.
  llvm::SmallDenseMap<StringAttr, int> outBLog2;
  for (StringAttr od : cvt.getOutDimNames())
    outBLog2[od] = tile.hasOutDim(od) ? tile.getOutDimSizeLog2(od) : 0;

  // Build a per-out-dimension mask by OR-ing all tile bases that touch it.
  llvm::SmallDenseMap<StringAttr, int32_t> tileMaskPerOutDim;
  for (StringAttr od : cvt.getOutDimNames())
    tileMaskPerOutDim[od] = 0;
  for (auto &[inDim, inBases] : tile.getBases()) {
    (void)inDim;
    for (auto &basis : inBases) {
      int idx = 0;
      for (StringAttr od : tile.getOutDimNames()) {
        tileMaskPerOutDim[od] |= basis[idx++];
      }
    }
  }

  // Build reps with the same in/out dims as cvt, but zeroing out the leading
  // inB bases (per in-dim) and keeping the remainder bases unchanged from cvt.
  LinearLayout::BasesT repsBases;
  for (StringAttr id : cvt.getInDimNames()) {
    int inA = cvt.getInDimSizeLog2(id);
    int inB = tile.hasInDim(id) ? tile.getInDimSizeLog2(id) : 0;
    assert(inB <= inA && "tile has more in-bits than cvt for a given in-dim");

    std::vector<std::vector<int32_t>> basesForDim;
    basesForDim.reserve(inA);

    // 1) Validate the starting bases match exactly.
    for (int i = 0; i < inB; ++i) {
      for (StringAttr od : cvt.getOutDimNames()) {
        int a = cvt.getBasis(id, i, od);
        int b = tile.getBasis(id, i, od);
        if (a != b) {
          return std::nullopt;
        }
      }
    }

    // 2) Validate no overlap: the remaining cvt bases must have zeros in all
    //    tile-bit positions (computed as OR of all tile bases) for each
    //    out-dim.
    for (int i = inB; i < inA; ++i) {
      for (StringAttr od : cvt.getOutDimNames()) {
        int32_t mask = tileMaskPerOutDim.lookup(od);
        if (mask == 0)
          continue;
        int v = cvt.getBasis(id, i, od);
        if ((v & mask) != 0) {
          return std::nullopt;
        }
      }
    }

    // 3) Emit reps bases: first inB as all-zeros; remainder copied from cvt.
    for (int i = 0; i < inB; ++i) {
      std::vector<int32_t> zero(cvt.getNumOutDims(), 0);
      basesForDim.push_back(std::move(zero));
    }
    for (int i = inB; i < inA; ++i) {
      std::vector<int32_t> keep;
      keep.reserve(cvt.getNumOutDims());
      for (StringAttr od : cvt.getOutDimNames())
        keep.push_back(cvt.getBasis(id, i, od));
      basesForDim.push_back(std::move(keep));
    }

    repsBases[id] = std::move(basesForDim);
  }

  return LinearLayout(std::move(repsBases), cvt.getOutDims(),
                      /*requireSurjective=*/false);
}

// Similar to largestVectorisation in TritonGPUToLLVM/Utility.cpp
std::optional<std::tuple<LinearLayout, ColumnAction, int>>
getVec(const LinearLayout &cvt, const LinearLayout &tile, int maxnreg,
       int bitwidth) {
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
  if (maxnreg == 256 && cvt.getInDimSize(kReg) / (32 / bitwidth) > maxReg) {
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

LinearLayout getTileLayout(MLIRContext *ctx, TMemAccessAtom atom, int bitwidth,
                           bool unpacked, int nRow) {
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
  auto nCol = tile.getOutDimSize(kCol);
  auto bases = tile.getBases();
  bases[kWarp].push_back({32, 0});
  bases[kWarp].push_back({64, 0});
  auto ret = LinearLayout(bases, {{kRow, 128}, {kCol, nCol}}, false);
  // Broadcast the row dimension if it's smaller than 128
  ret = ensureLayoutNotLargerThan(ret, {{kRow, nRow}, {kCol, nCol}}, true);
  // For unpacked, the tile above is for 32-bit elements, so we have to multiply
  // by identity1D(32 / bitwidth, kReg, kCol) to get the correct tile to allow
  // us to divide the cvt layout by it
  if (unpacked) {
    ret = LinearLayout::identity1D(32 / bitwidth, kReg, kCol) * ret;
  }
  return ret;
}

SmallVector<Value> pack(ArrayRef<Value> values, Type outType, Location loc,
                        ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> packedValues;
  Type inType = values[0].getType();
  auto inbitwidth = inType.getIntOrFloatBitWidth();
  auto outbitwidth = outType.getIntOrFloatBitWidth();
  assert(inbitwidth <= outbitwidth);
  if (inbitwidth == outbitwidth) {
    for (auto &val : values) {
      packedValues.push_back(b.bitcast(val, outType));
    }
    return packedValues;
  }

  auto vecSize = outbitwidth / inbitwidth;
  auto vecTy = vec_ty(inType, vecSize);
  for (int i = 0; i < values.size(); i += vecSize) {
    Value packed = b.undef(vecTy);
    for (int j = 0; j < vecSize; j++) {
      Value val = values[i + j];
      packed = b.insert_element(vecTy, packed, val, b.i32_val(j));
    }
    packed = b.bitcast(packed, outType);
    packedValues.emplace_back(std::move(packed));
  }
  return packedValues;
}

SmallVector<Value> unpack(ArrayRef<Value> packedValues, Type outType,
                          Location loc, ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type inType = packedValues[0].getType();
  auto inbitwidth = inType.getIntOrFloatBitWidth();
  auto outbitwidth = outType.getIntOrFloatBitWidth();
  assert(inbitwidth >= outbitwidth);
  SmallVector<Value> unpackedValues;
  if (inbitwidth == outbitwidth) {
    for (auto &val : packedValues) {
      unpackedValues.push_back(b.bitcast(val, outType));
    }
    return unpackedValues;
  }
  auto vecSize = inbitwidth / outbitwidth;
  auto vecTy = vec_ty(outType, vecSize);
  for (int i = 0; i < packedValues.size(); i++) {
    Value packed = b.bitcast(packedValues[i], vecTy);
    for (int j = 0; j < vecSize; j++) {
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

  // Pack into bitwidth=32 if it was not packed already
  if (llvmElemTy.getIntOrFloatBitWidth() < 32) {
    assert(unpacked);
    SmallVector<Value> inVals;
    if (isStore) {
      inVals = pack(vals, i32_ty, loc, rewriter);
    }
    // kill the first logValsPerReg bases of kReg as they are now packed
    // again, super hacky, we should probably do this when building the
    // instruction
    auto bases = reps.getBases();
    auto valsPerReg = 32 / llvmElemTy.getIntOrFloatBitWidth();
    auto logValsPerReg = llvm::Log2_32(valsPerReg);
    assert(reps.getInDimSizeLog2(kReg) >= logValsPerReg);
    auto &reg = bases[kReg];
    reg.erase(reg.begin(), reg.begin() + logValsPerReg);
    auto quot = LinearLayout(bases, reps.getOutDims(), /*isSurjective=*/false);
    auto outValsOr = lowerTMemLdSt(
        loc, ctx, rewriter, quot, inVals, atom, i32_ty, tmemBase, pred,
        valsPerMessage / valsPerReg, unpacked, secondHalfOffset);
    if (failed(outValsOr))
      return failure();
    auto outVals = std::move(*outValsOr);
    if (!isStore) {
      outVals = unpack(outVals, llvmElemTy, loc, rewriter);
    }
    return outVals;
  }

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
              int maxnreg, bool unpacked, Value pred) {
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
                      maxnreg, unpacked, pred);
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

  // There must be a better way to do this. We should follow something similar
  // to what we do in shmem, where we pack/unpack just before the PTX instr
  // creation
  if (!unpacked && llvmElemTy.getIntOrFloatBitWidth() < 32) {
    SmallVector<Value> inVals;
    if (isStore) {
      inVals = pack(vals, i32_ty, loc, rewriter);
    }
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto maybeQuot =
        divideLeft(cvt, LinearLayout::identity1D(32 / bitwidth, kReg, kCol));
    assert(maybeQuot.has_value());
    auto quot = *maybeQuot;
    auto outValsOr = lowerTMemLdSt(loc, ctx, rewriter, quot, inVals, i32_ty,
                                   tmemBase, maxnreg, unpacked, pred);
    if (failed(outValsOr))
      return failure();
    auto outVals = std::move(*outValsOr);
    if (!isStore) {
      outVals = unpack(outVals, llvmElemTy, loc, rewriter);
    }
    return outVals;
  }
  assert(!isStore || cvt.getInDimSize(kReg) == vals.size());
  auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
  auto nRow = cvt.getOutDimSize(kRow);

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
    auto tile = getTileLayout(ctx, atom, bitwidth, unpacked, nRow);
    auto maybeReps = getVec(cvt, tile, maxnreg, bitwidth);
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
    auto tile =
        getTileLayout(ctx, TMemAccess16x32bx2, bitwidth, unpacked, nRow);
    auto maybeReps = getVec(cvt, tile, maxnreg, bitwidth);
    if (maybeReps) {
      auto [reps, perm, numRegsPerMessage] = std::move(*maybeReps);
      // Find the last kLane basis and use it as secondHalfOffset
      auto row = reps.getBasis(kLane, 4, kRow);
      auto col = reps.getBasis(kLane, 4, kCol);
      secondHalfOffset = (row << 16) | col;
      // We "quotient it out", meaning we remove the last basis from reps
      auto basis = reps.getBases();
      basis[kLane][4] = {0, 0};
      reps = LinearLayout(basis, reps.getOutDims(), /*isSurjective=*/false);
      msgInfo = {TMemAccess16x32bx2, reps, perm, numRegsPerMessage};
    }
  }

  if (!msgInfo) {
    emitError(loc, "Failed to lower TMEM load/store: unsupported dst layout");
    return failure();
  }
  auto [atom, reps, perm, numRegsPerMessage] = std::move(msgInfo.value());

  SmallVector<Value> inVals;
  if (isStore) {
    inVals = to_vector(vals);
    inVals = perm.apply(inVals);
  }
  auto outValsOr = lowerTMemLdSt(
      loc, ctx, rewriter, reps, inVals, atom, llvmElemTy, tmemBase, pred,
      numRegsPerMessage, unpacked && llvmElemTy.getIntOrFloatBitWidth() == 16,
      secondHalfOffset);
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
    Value pred, Type llvmElemTy, ArrayRef<Value> storeVals) {
  auto memLayout = toLinearLayout(memTy);
  auto regLayout = toLinearLayout(regTy);
  auto cvt = regLayout.invertAndCompose(memLayout);

  // tmemBase already encodes CTA/block offsets so we just remove them from the
  // cvt
  auto kBlock = str_attr("block");
  auto kCol = str_attr("col");
  auto nCTAs = cvt.getInDimSize(kBlock);
  auto maybeQuot =
      divideRight(cvt, LinearLayout::identity1D(nCTAs, kBlock, kCol));
  assert(maybeQuot.has_value());
  auto quot = maybeQuot->unsqueezeIn(kBlock);
  bool unpacked;
  if (auto enc = dyn_cast<TensorMemoryEncodingAttr>(memTy.getEncoding())) {
    unpacked = enc.getUnpacked();
  } else {
    assert(isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding()));
    unpacked = false;
  }

  // Handle K = 1 and K = 2 cases
  auto K = regTy.getDimSize(1);
  auto undefShmem =
      isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding()) && K < 4;
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  bool isStore = !storeVals.empty();
  auto inVals = to_vector(storeVals);
  auto packedLlvmElemTy = llvmElemTy;
  auto packedLayout = std::move(quot);
  if (undefShmem) {
    auto kReg = str_attr("register");
    auto tile = LinearLayout::identity1D(K, kReg, kCol) *
                LinearLayout::zeros1D(4 / K, kReg, kCol);
    auto maybePacked = divideLeft(packedLayout, tile);
    assert(maybePacked.has_value());
    packedLayout = std::move(*maybePacked);
    if (isStore) {
      inVals = pack(inVals, i32_ty, loc, rewriter);
    }
    packedLlvmElemTy = i32_ty;
  }

  auto resultValsOr =
      lowerTMemLdSt(loc, ctx, rewriter, packedLayout, inVals, packedLlvmElemTy,
                    tmemBase, maxnreg, unpacked, pred);
  if (failed(resultValsOr))
    return failure();
  auto resultVals = std::move(*resultValsOr);
  if (!isStore && undefShmem) {
    resultVals = unpack(resultVals, llvmElemTy, loc, rewriter);
  }
  return resultVals;
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
                            Value tmem_address, Value src_desc, Value pred,
                            bool scales) {
  PTXBuilder ptxBuilder;
  auto dst = ptxBuilder.newAddrOperand(tmem_address, "r");
  auto src = ptxBuilder.newOperand(src_desc, "l");
  std::string opcode = scales ? "tcgen05.cp.cta_group::1.warpx4.32x128b"
                              : "tcgen05.cp.cta_group::1.128x256b";
  auto &op = *ptxBuilder.create<PTXInstr>(opcode);
  op({dst, src}).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static void copyScales(ConversionPatternRewriter &rewriter, Location loc,
                       const TypeConverter *typeConverter,
                       triton::nvidia_gpu::TMEMCopyOp op, Value src, Value dst,
                       Value pred) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MemDescType srcTy = op.getSrc().getType();
  MemDescType dstTy = op.getDst().getType();
  Type elemTy = typeConverter->convertType(srcTy.getElementType());
  auto smemObj =
      LLVM::getSharedMemoryObjectFromStruct(loc, src, elemTy, rewriter);
  Value baseSrc = smemObj.getShmemAffineBase(loc, rewriter, srcTy);

  Value baseDst = dst;
  auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
  auto llvmElementTy = typeConverter->convertType(srcTy.getElementType());

  auto ll = toLinearLayout(srcTy);
  // flattenOuts flattens into fortran order, so need to transpose first to
  // get C-order
  auto ctx = op.getContext();
  auto outDimNames = standardOutDimNames(ctx, srcTy.getRank());
  std::reverse(outDimNames.begin(), outDimNames.end());
  ll = ll.transposeOuts(outDimNames).flattenOuts();
  auto invLayout = ll.flattenOuts().invert();
  auto kDim = *ll.getOutDimNames().begin();
  Value smemDesc = createBlockedScalesSMEMDescriptor(rewriter, loc, baseSrc);
  auto createCopy = [&](int repMorN, int repK) {
    for (int i = 0; i < repMorN; ++i) {
      for (int j = 0; j < repK; ++j) {
        // Multiple copies of 32x128b blocks are laid out along M/N first then
        // K
        auto colOffset = b.int_val(32, (j * repMorN + i) * 4);
        auto tmemAddr = b.add(b.ptrtoint(i32_ty, baseDst), colOffset);
        auto blockSize = (32 * 128) / llvmElementTy.getIntOrFloatBitWidth();
        auto linearIdx = (i * repK + j) * blockSize;
        auto smemOffset = applyLinearLayout(loc, rewriter, invLayout,
                                            {{kDim, b.i32_val(linearIdx)}})[0]
                              .second;
        auto smemAddr = b.gep(elemPtrTy, llvmElementTy, baseSrc, smemOffset);
        smemDesc = createBlockedScalesSMEMDescriptor(rewriter, loc, smemAddr);
        createTcgen05Cp(rewriter, loc, tmemAddr, smemDesc, pred,
                        /*scales=*/true);
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
  // * (1, rep_m, rep_k, 2, 256B), 5D scale load with TMA
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
        if (srcTy.getDimSize(0) == 1 &&
            srcTy.getDimSize(srcTy.getRank() - 1) == 256) {
          repMorN = srcTy.getDimSize(1);
          repK = srcTy.getDimSize(2);
        } else {
          repMorN = srcTy.getDimSize(0);
          repK = srcTy.getDimSize(1);
        }
      }
      break;
    }
  }
  createCopy(repMorN, repK);
}

static void copySharedToTmem(ConversionPatternRewriter &rewriter, Location loc,
                             const TypeConverter *typeConverter,
                             triton::nvidia_gpu::TMEMCopyOp op, Value src,
                             Value dst, Value pred) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MemDescType srcTy = op.getSrc().getType();
  MemDescType dstTy = op.getDst().getType();
  Type elemTy = typeConverter->convertType(srcTy.getElementType());
  auto smemObj =
      LLVM::getSharedMemoryObjectFromStruct(loc, src, elemTy, rewriter);
  Value baseSrc = smemObj.getShmemAffineBase(loc, rewriter, srcTy);

  Value baseDst = dst;
  assert(srcTy.getElementType().getIntOrFloatBitWidth() == 32);

  int blockN =
      cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(dstTy.getEncoding())
          .getBlockN();
  // Currently, hardcoded to 128x256b message.
  std::array<int, 2> instShape = {128, 8};
  int repNPerBlock = blockN / instShape[1];
  auto createCopy = [&](int repM, int repN) {
    Value zero = b.i32_val(0);
    SmallVector<int64_t> shape(op.getSrc().getType().getShape());
    DotOpMmaV5SmemLoader smemLoader = DotOpMmaV5SmemLoader(
        op.getSrc(), baseSrc, shape, op.getSrc().getType().getAllocShape(),
        zero, 1, /*trans=*/false, {128, 8},
        op.getSrc().getType().getElementType().getIntOrFloatBitWidth(),
        rewriter, loc);
    for (int m = 0; m < repM; m++) {
      for (int n = 0; n < repN; n++) {
        int colIndx =
            (n % repNPerBlock) * instShape[1] +
            m * repNPerBlock * instShape[1] +
            (n / repNPerBlock) * (srcTy.getDimSize(0) / instShape[0]) * blockN;
        auto colOffset = b.i32_val(colIndx);
        auto tmemAddr = b.add(b.ptrtoint(i32_ty, baseDst), colOffset);
        Value smemDesc = smemLoader.smemLoad(m, n, rewriter, loc);
        createTcgen05Cp(rewriter, loc, tmemAddr, smemDesc, pred,
                        /*scales=*/false);
      }
    }
  };

  int repM = srcTy.getDimSize(0) / instShape[0];
  int repN = srcTy.getDimSize(1) / instShape[1];
  createCopy(repM, repN);
}

struct TensorMemoryCopyOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMCopyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    Value pred = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);
    if (isa<TensorMemoryScalesEncodingAttr>(
            op.getDst().getType().getEncoding())) {
      // Special case for copy of scales as they behave differently from other
      // copies. This can be unified once we fix the smem layout representation
      // of the source.
      copyScales(rewriter, loc, typeConverter, op, adaptor.getSrc(),
                 adaptor.getDst(), pred);
    } else {
      copySharedToTmem(rewriter, loc, typeConverter, op, adaptor.getSrc(),
                       adaptor.getDst(), pred);
    }

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
    if (!isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
            op.getSrc().getType().getEncoding())) {
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

    if (!encoding.getUnpacked()) {
      // Adjust the column offset based on the element size.
      int numElementsPer32B = 32 / srcTy.getElementTypeBitWidth();
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
