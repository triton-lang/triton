#include "cpu/include/TritonCPUTransforms/OptCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "include/triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include <iostream>
#include <utility>

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTDOTTOAMX
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "triton-cpu-dot-to-amx"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// This struct describes buffers used to load/store AMX tiles.
struct AmxBuffer {
  Value memRef;
  SmallVector<Value, 2> indices;

  bool empty() const { return !memRef; }
};

// This structure is used to hold candidates for conversion to AMX
// Mul[F|I]Op operations.
struct AmxDotOpCandidate {
  // Operation to convert.
  cpu::DotOp op;
  // Available LHS, RHS, and accumulator types are limited in AMX and we might
  // require additional casts. Here we keep actual element types used by LHS,
  // RHS, and accumulator in AMX tiles.
  Type lhsTileElemTy;
  Type rhsTileElemTy;
  Type accTileElemTy;
  // AMX tile row size is limited by 64 bytes, so M and N dimensions are limited
  // by 16 because accumulator always has 4-byte elements. K dimension for tiles
  // is limited by 64 / <size of input element>. Here we keep actual tile sizes.
  int64_t tileM;
  int64_t tileN;
  int64_t tileK;
  // We have a limited number of available tiles, so if input/output is too
  // big to fit available tiles, we need to split them into blocks. Here we
  // keep a number of tiles in accumulator block. K dimension for input blocks
  // is always 1 tile now.
  int64_t tilesInBlockM;
  int64_t tilesInBlockN;
  // If accumulator is updated in a loop, then this flag indicates if we
  // should keep it in tiles the whole loop and move back to vectors only
  // after the loop.
  bool keepAccOnTiles = false;
  // If we want to keep accumulator in tiles but it's too big, then we might
  // keep it bufferized instead.
  bool keepAccInBuf = false;
  // If resulting tiles are not required to be trasfered to vectors and can be
  // directly stored to the output memory instead, then this field holds a
  // buffer to use.
  AmxBuffer outBuf;
  // If output buffer is used then keep the original vector store here.
  Operation *origStore = nullptr;
};

// Check if input and output types can be handled by AMX (possibly, using
// additional casts for input/output). Returns true if AMX usage is possible.
// In this case, tile element type fields of the candidate structure are
// filled with actual types to be used in lowering.
bool checkElemTypes(Type lhsElemTy, Type rhsElemTy, Type accElemTy,
                    Type resElemTy, bool supportInt8, bool supportFp16,
                    bool supportBf16, AmxDotOpCandidate &candidate) {
  MLIRContext *ctx = lhsElemTy.getContext();
  if (lhsElemTy.isInteger()) {
    if (!supportInt8) {
      LDBG("Drop candidate because AMX_INT8 is not available.");
      return false;
    }

    // For integer case only i8 is allowed for LHS and RHS.
    if (!lhsElemTy.isInteger(8) || !rhsElemTy.isInteger(8)) {
      LDBG("Drop candidate with unsupported input integer type.");
      return false;
    }

    // Accumulator should be i32. If it's smaller, we will use casts.
    if (!accElemTy.isInteger() || accElemTy.getIntOrFloatBitWidth() > 32 ||
        !resElemTy.isInteger() || resElemTy.getIntOrFloatBitWidth() > 32) {
      LDBG("Drop candidate with unsupported output integer type.");
      return false;
    }

    candidate.lhsTileElemTy = IntegerType::get(ctx, 8);
    candidate.rhsTileElemTy = IntegerType::get(ctx, 8);
    candidate.accTileElemTy = IntegerType::get(ctx, 32);

    return true;
  }

  // FP case. Expect no integer args or result.
  if (rhsElemTy.isInteger() || accElemTy.isInteger() || resElemTy.isInteger()) {
    LDBG("Drop candidate with mixed int/fp types.");
    return false;
  }

  // For fp case LHS and RHS types should match and can be either FP16 or
  // BF16.
  if (lhsElemTy.getIntOrFloatBitWidth() > 16 ||
      rhsElemTy.getIntOrFloatBitWidth() > 16) {
    LDBG("Drop candidate with unsupported input fp type.");
    return false;
  }

  // Try to find a common input type. There is currently no support
  // for FP8 types, so promote them to FP16/BF16.
  Type commonInputElemTy;
  if (lhsElemTy.getIntOrFloatBitWidth() == 16) {
    commonInputElemTy = lhsElemTy;
    if (rhsElemTy.getIntOrFloatBitWidth() == 16 &&
        rhsElemTy != commonInputElemTy) {
      LDBG("Drop candidate with mismatched input types.");
      return false;
    }
  } else if (rhsElemTy.getIntOrFloatBitWidth() == 16)
    commonInputElemTy = rhsElemTy;
  // Both inputs are FP8, choose 16-bit FP type to use.
  else if (supportBf16)
    commonInputElemTy = BFloat16Type::get(ctx);
  else
    commonInputElemTy = Float16Type::get(ctx);

  if (commonInputElemTy.isF16() && !supportFp16) {
    LDBG("Drop candidate because AMX_FP16 is not available.");
    return false;
  }

  if (commonInputElemTy.isBF16() && !supportBf16) {
    LDBG("Drop candidate because AMX_BF16 is not available.");
    return false;
  }

  // Accumulator type should be FP32, we can use casts if it is smaller.
  if (accElemTy.getIntOrFloatBitWidth() > 32) {
    LDBG("Drop candidate with unsupported accumulator type.");
    return false;
  }

  candidate.lhsTileElemTy = commonInputElemTy;
  candidate.rhsTileElemTy = commonInputElemTy;
  candidate.accTileElemTy = Float32Type::get(ctx);

  return true;
}

// Check input shapes. Currently, support only 2D cases and ignore small
// inputs.
bool checkInputShapes(VectorType lhsTy, VectorType resTy) {
  if (lhsTy.getRank() != 2)
    return false;

  if (lhsTy.getDimSize(0) < 8 || lhsTy.getDimSize(1) < 8 ||
      resTy.getDimSize(1) < 8)
    return false;

  return true;
}

// Check if accumulator value is updated in a loop and has no other
// usages than a dot op, that updates it. Tile loads/stores and casts
// for such accumulators can be done outside of the loop.
bool isLoopCarriedAcc(Value acc) {
  LDBG("Check if accumulator can be held in tiles: " << acc);
  if (!acc.hasOneUse()) {
    LDBG("  No. Has multiple uses.");
    for (auto op : acc.getUsers())
      LDBG("    " << *op);
    return false;
  }

  auto blockArg = dyn_cast<BlockArgument>(acc);
  if (!blockArg) {
    LDBG("  No. Not a block argument.");
    return false;
  }

  auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!forOp) {
    LDBG("  No. Not in a for-loop.");
    return false;
  }

  blockArg.getArgNumber();

  Value updAcc = acc.getUsers().begin()->getResult(0);
  if (!updAcc.hasOneUse()) {
    LDBG("  No. Has multiple uses.");
    return false;
  }

  auto &updAccUse = *updAcc.getUses().begin();
  if (!isa<scf::YieldOp>(updAccUse.getOwner()) ||
      updAccUse.getOperandNumber() !=
          (blockArg.getArgNumber() - forOp.getNumInductionVars())) {
    LDBG("  No. Loop carried dependency not detected.");
    return false;
  }

  LDBG("  Yes.");
  return true;
}

// Return a value that holds the resulting loop carried accumulator value.
// It's one of ForOp's results.
Value getResValueForLoopCarriedAcc(cpu::DotOp op) {
  Value updAcc = op.getResult();
  auto forOp = dyn_cast<scf::ForOp>(op->getParentOp());
  auto &use = *updAcc.getUses().begin();
  return forOp.getResult(use.getOperandNumber());
}

// Choose tile and block sizes for the candidate. Tile sizes are determined
// by input shapes and types. Block sizes are chosen to minimize number of
// tile loads/stores including tile register spills.
void setupBlockAndTileSizes(ArrayRef<int64_t> lhsShape,
                            ArrayRef<int64_t> rhsShape,
                            AmxDotOpCandidate &candidate) {
  int64_t m = lhsShape[0];
  int64_t n = rhsShape[1];
  int64_t k = rhsShape[0];
  int64_t tileM = std::min(m, (int64_t)16);
  int64_t tileN = std::min(n, (int64_t)16);
  int64_t tileK = std::min(
      k, (int64_t)512 / candidate.lhsTileElemTy.getIntOrFloatBitWidth());

  int64_t accBlocksM = m / tileM;
  int64_t accBlocksN = n / tileN;

  // All these sizes are power of 2. We have 8 tile registers and
  // cannot use them all for accumulator. So, we will use up to 4
  // tiles for accumulator in a single block.
  while (accBlocksM * accBlocksN > 4) {
    if (accBlocksM > accBlocksN)
      accBlocksM /= 2;
    else
      accBlocksN /= 2;
  }

  candidate.tileM = tileM;
  candidate.tileN = tileN;
  candidate.tileK = tileK;
  candidate.tilesInBlockM = accBlocksM;
  candidate.tilesInBlockN = accBlocksN;
}

// Check if vector transfer read/write operation uses a mask
// or involves a bounds check.
template <typename T> bool hasMaskOrBoundsCheck(T op) {
  auto inBounds = op.getInBounds();
  Value mask = op.getMask();
  bool hasBoundsCheck =
      std::any_of(inBounds.begin(), inBounds.end(), [](Attribute attr) {
        return !cast<mlir::BoolAttr>(attr).getValue();
      });
  return hasBoundsCheck || mask;
}

// Check if a value is used only for a store and that this store can be
// replaced with tile stores. In this case fill appropriate fields in the
// candidate structure.
void findOutputBuffer(Value val, AmxDotOpCandidate &candidate) {
  if (val.hasOneUse()) {
    auto store = dyn_cast<vector::TransferWriteOp>(*val.user_begin());
    if (store && !hasMaskOrBoundsCheck(store))
      candidate.outBuf = AmxBuffer{store.getSource(), store.getIndices()};
    candidate.origStore = store;
  }
}

// Check if specified ContractionOp can be lowered to AMX operations.
// If conversion is possible, then true is returned and candidate
// structure is filled with detailed transformation info.
bool isAmxCandidate(cpu::DotOp op, bool supportInt8, bool supportFp16,
                    bool supportBf16, AmxDotOpCandidate &candidate) {
  MLIRContext *ctx = op.getContext();
  VectorType lhsTy = cast<VectorType>(op.getA().getType());
  VectorType rhsTy = cast<VectorType>(op.getB().getType());
  VectorType accTy = cast<VectorType>(op.getC().getType());
  VectorType resTy = cast<VectorType>(op.getType());

  LDBG("Considering candidate op: " << op);

  // Check if input and output types match available hardware capabilities.
  // If check is successful then tile element types are filled with types
  // to use in AMX operations.
  if (!checkElemTypes(lhsTy.getElementType(), rhsTy.getElementType(),
                      accTy.getElementType(), resTy.getElementType(),
                      supportInt8, supportFp16, supportBf16, candidate))
    return false;

  // Check input shapes.
  if (!checkInputShapes(lhsTy, resTy))
    return false;

  candidate.op = op;
  setupBlockAndTileSizes(lhsTy.getShape(), rhsTy.getShape(), candidate);
  candidate.keepAccOnTiles = isLoopCarriedAcc(op.getC());

  // Can't keep acc in a tile the whole loop right now:
  // https://github.com/llvm/llvm-project/issues/109481
  if (candidate.keepAccOnTiles) {
    // We might not have enough tiles to hold accumulator. In this case
    // keep it in a bufffer.
    if (candidate.tilesInBlockM * candidate.tilesInBlockN > 1) {
      LDBG("Accumulator is too big to keep on tiles. Keep it bufferized "
           "insterad.");
      candidate.keepAccOnTiles = false;
      candidate.keepAccInBuf = true;
    } else {
      findOutputBuffer(getResValueForLoopCarriedAcc(op), candidate);
    }

    // TODO: fix LLVM bug and remove this code.
    LDBG("Avoid accumulator on tiles due to LLVM bug: "
         "https://github.com/llvm/llvm-project/issues/109481.");
    LDBG("Keep accumulator bufferized instead.");
    candidate.keepAccOnTiles = false;
    candidate.keepAccInBuf = true;
    candidate.outBuf = AmxBuffer{};
  } else {
    findOutputBuffer(op.getResult(), candidate);
  }

  return true;
}

// Cast vector to a specified element type using ext or trunc
// operations. Return the original value if it already matches
// the required element type.
Value maybeCast(Location loc, Value val, Type dstElemTy,
                PatternRewriter &rewriter) {
  VectorType srcTy = cast<VectorType>(val.getType());
  if (srcTy.getElementType() == dstElemTy)
    return val;

  VectorType dstTy = srcTy.cloneWith(std::nullopt, dstElemTy);
  if (srcTy.getElementType().isInteger()) {
    if (srcTy.getElementTypeBitWidth() < dstTy.getElementTypeBitWidth())
      return rewriter.create<arith::ExtSIOp>(loc, dstTy, val);
    return rewriter.create<arith::TruncIOp>(loc, dstTy, val);
  }

  if (srcTy.getElementTypeBitWidth() < dstTy.getElementTypeBitWidth())
    return rewriter.create<arith::ExtFOp>(loc, dstTy, val);
  return rewriter.create<arith::TruncFOp>(loc, dstTy, val);
}

// Get initial value for a loop-carried accumulator.
Value getInitAccValue(Value val) {
  auto blockArg = cast<BlockArgument>(val);
  auto forOp = cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
  int initValIdx = blockArg.getArgNumber() - forOp.getNumInductionVars();
  return forOp.getInitArgs()[initValIdx];
}

VectorType getSwizzledRhsTileType(VectorType origTileType) {
  int64_t rowsPerGroup = 32 / origTileType.getElementTypeBitWidth();
  SmallVector<int64_t> shape({origTileType.getDimSize(0) / rowsPerGroup,
                              origTileType.getDimSize(1) * rowsPerGroup});
  return origTileType.cloneWith(shape, origTileType.getElementType());
}

AmxBuffer allocateTmpBuffer(Location loc, VectorType vecTy,
                            Operation *allocaPoint, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(allocaPoint);
  auto memRefTy = MemRefType::get(vecTy.getShape(), vecTy.getElementType());
  Value memRef = rewriter.create<memref::AllocaOp>(
      loc, memRefTy, rewriter.getIntegerAttr(rewriter.getI64Type(), 64));
  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value, 2> indices(2, zeroIdx);
  return {memRef, indices};
}

// In AMX, element values shoud be packed to 32-bit groups that would be
// multiplied elementwise with following accumulation. It means that RHS
// needs to be pre-packed. E.g. for the following input
//   B(0,0) B(0,1) B(0,2) ... B(0,15)
//   B(1,0) B(1,1) B(1,2) ... B(1,15)
//   B(2,0) B(2,1) B(2,2) ... B(2,15)
//   B(3,0) B(3,1) B(3,2) ... B(3,15)
// and BF16/FP16 type we need to transform it to
//   B(0,0) B(1,0) B(0,1), B(1,1) ... B(0,15) B(1,15)
//   B(2,0) B(3,0) B(2,1), B(3,1) ... B(2,15) B(3,15)
// so that original columns are 32-bits now. In case of int8 type, the
// result would be:
//   B(0,0) B(1,0) B(2,0), B(3,0) ... B(0,15) B(1,15), B(2,15) B(3,15)
void interleaveAndStore(Location loc, Value val, Value buf,
                        PatternRewriter &rewriter) {
  LDBG("Repacking operand before storing to a buffer.");
  VectorType valTy = cast<VectorType>(val.getType());
  int64_t rowsPerGroup = 32 / valTy.getElementTypeBitWidth();
  assert(rowsPerGroup == 2 || rowsPerGroup == 4);
  assert(valTy.getDimSize(0) % rowsPerGroup == 0);
  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  for (int64_t i = 0; i < valTy.getDimSize(0); i += rowsPerGroup) {
    Value row1, row2;
    if (rowsPerGroup == 2) {
      row1 = rewriter.create<vector::ExtractOp>(loc, val, i);
      row2 = rewriter.create<vector::ExtractOp>(loc, val, i + 1);
    } else {
      row1 = rewriter.create<vector::InterleaveOp>(
          loc, rewriter.create<vector::ExtractOp>(loc, val, i),
          rewriter.create<vector::ExtractOp>(loc, val, i + 2));
      row2 = rewriter.create<vector::InterleaveOp>(
          loc, rewriter.create<vector::ExtractOp>(loc, val, i + 1),
          rewriter.create<vector::ExtractOp>(loc, val, i + 3));
    }
    Value shuffled = rewriter.create<vector::InterleaveOp>(loc, row1, row2);
    Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i / rowsPerGroup);
    rewriter.create<vector::StoreOp>(loc, shuffled, buf,
                                     SmallVector<Value>({idx, zeroIdx}));
  }
}

// Prepare temporary buffers to be used for tile loads. If the original
// value can be directly loaded to tiles from its original memory, then
// use it instead. Return empty buffer if source value is all zeros and
// skipForZeros is set.
//
// If interleave flag is set, then pre-pack RHS before store. See
// interleaveAndStore for more details.
AmxBuffer prepareTensorBuffer(Location loc, Value val, bool interleave,
                              bool skipForZeros, bool readOnly,
                              Operation *allocaPoint,
                              PatternRewriter &rewriter) {
  LDBG("Preparing buffer (interleave=" << interleave
                                       << ") for a vector: " << val);
  auto valLoad = val.getDefiningOp<vector::TransferReadOp>();
  if (valLoad && !interleave && readOnly && !hasMaskOrBoundsCheck(valLoad)) {
    Value memRef = valLoad.getSource();
    ValueRange indices = valLoad.getIndices();
    LDBG("  Reusing the original memref for a buffer: " << memRef);
    return {memRef, indices};
  }

  if (skipForZeros && isZeroConst(val)) {
    LDBG("Skip buffer for zero vector.");
    return {};
  }

  auto vecTy = cast<VectorType>(val.getType());
  if (interleave)
    vecTy = getSwizzledRhsTileType(vecTy);
  AmxBuffer buf = allocateTmpBuffer(loc, vecTy, allocaPoint, rewriter);

  if (interleave) {
    interleaveAndStore(loc, val, buf.memRef, rewriter);
  } else {
    rewriter.create<vector::TransferWriteOp>(loc, val, buf.memRef, buf.indices);
  }

  return buf;
}

// Return a buffer where the final result should be stored. If result can
// be directly stored to the output memory, then it is used as an output
// buffer. Otherwise, re-use accumulator buffer or create a new one.
AmxBuffer prepareResultBuffer(Location loc, Value val, const AmxBuffer &accBuf,
                              const AmxBuffer &outBuf, Operation *allocaPoint,
                              PatternRewriter &rewriter) {
  if (!outBuf.empty()) {
    LDBG("Output memory will be used for direct tile stores.");
    return outBuf;
  }

  if (!accBuf.empty()) {
    LDBG("Result will be stored to accumulator buffer.");
    return accBuf;
  }

  LDBG("Allocating buffer for the result.");
  return allocateTmpBuffer(loc, cast<VectorType>(val.getType()), allocaPoint,
                           rewriter);
}

Value shiftIndex(Location loc, Value index, int64_t offs,
                 PatternRewriter &rewriter) {
  if (!offs)
    return index;

  // Do constant folding right away here for better code readability
  // after the pass.
  auto cstOp = dyn_cast<arith::ConstantOp>(index.getDefiningOp());
  if (cstOp) {
    int64_t oldVal = cast<IntegerAttr>(cstOp.getValue()).getInt();
    return rewriter.create<arith::ConstantIndexOp>(loc, oldVal + offs);
  }

  Value offsVal = rewriter.create<arith::ConstantIndexOp>(loc, offs);
  return rewriter.create<arith::AddIOp>(loc, index.getType(), index, offsVal);
}

SmallVector<Value, 2> shiftIndices(Location loc, ArrayRef<Value> indices,
                                   VectorType tileTy, int64_t tilesInBlockM,
                                   int64_t tilesInBlockN, int64_t blockM,
                                   int64_t blockN, int64_t tileM, int64_t tileN,
                                   PatternRewriter &rewriter) {
  int64_t blockOffsM = blockM * tilesInBlockM * tileTy.getDimSize(0);
  int64_t blockOffsN = blockN * tilesInBlockN * tileTy.getDimSize(1);
  int64_t tileOffsM = blockOffsM + tileM * tileTy.getDimSize(0);
  int64_t tileOffsN = blockOffsN + tileN * tileTy.getDimSize(1);
  return {shiftIndex(loc, indices[0], tileOffsM, rewriter),
          shiftIndex(loc, indices[1], tileOffsN, rewriter)};
}

Value loadTile(Location loc, VectorType tileTy, const AmxBuffer &buf,
               int64_t tilesInBlockM, int64_t tilesInBlockN, int64_t blockM,
               int64_t blockN, int64_t tileM, int64_t tileN,
               PatternRewriter &rewriter) {
  auto indices =
      shiftIndices(loc, buf.indices, tileTy, tilesInBlockM, tilesInBlockN,
                   blockM, blockN, tileM, tileN, rewriter);
  return rewriter.create<amx::TileLoadOp>(loc, tileTy, buf.memRef, indices);
}

void storeTile(Location loc, VectorType tileTy, Value val, const AmxBuffer &buf,
               int64_t tilesInBlockM, int64_t tilesInBlockN, int64_t blockM,
               int64_t blockN, int64_t tileM, int64_t tileN,
               PatternRewriter &rewriter) {
  auto indices =
      shiftIndices(loc, buf.indices, tileTy, tilesInBlockM, tilesInBlockN,
                   blockM, blockN, tileM, tileN, rewriter);
  rewriter.create<amx::TileStoreOp>(loc, buf.memRef, indices, val);
}

SmallVector<SmallVector<Value>>
loadBlockTiles(Location loc, VectorType tileTy, const AmxBuffer &buf,
               int64_t tilesInBlockM, int64_t tilesInBlockN, int64_t blockM,
               int64_t blockN, PatternRewriter &rewriter) {
  SmallVector<SmallVector<Value>> res(tilesInBlockM);
  for (int64_t m = 0; m < tilesInBlockM; ++m) {
    for (int64_t n = 0; n < tilesInBlockN; ++n) {
      Value tile = buf.memRef
                       ? loadTile(loc, tileTy, buf, tilesInBlockM,
                                  tilesInBlockN, blockM, blockN, m, n, rewriter)
                       : rewriter.create<amx::TileZeroOp>(loc, tileTy);
      res[m].push_back(tile);
    }
  }
  return res;
}

// Move acc to a tile for the whole loop. It might be loads from memory or
// zero tiles.
SmallVector<SmallVector<Value>>
moveLoopAccToTiles(Location loc, VectorType tileTy, const AmxBuffer &buf,
                   int64_t tilesInBlockM, int64_t tilesInBlockN,
                   PatternRewriter &rewriter) {
  LDBG("Loading accumulator to tiles before the loop.");
  auto res = loadBlockTiles(loc, tileTy, buf, tilesInBlockM, tilesInBlockN, 0,
                            0, rewriter);

  // TODO: add new block args into ForOp and return them instead.
  // Yield directly uses them for now and will be patched after mul
  // ops generation.
  llvm_unreachable("Not yet supported.");

  return res;
}

// Multiply two blocks. LHS block is preloaded to tiles with the following
// iteration over RHS. Accumulator values are updated in accTiles.
// Optionally, results can also be stored to accBuf.
void multiplyBlocksPreloadLhs(Location loc, VectorType lhsTileTy,
                              VectorType rhsTileTy, VectorType accTileTy,
                              const AmxBuffer &lhsBuf, const AmxBuffer &rhsBuf,
                              const AmxBuffer &accBuf, int64_t blockM,
                              int64_t blockN, int64_t blockK,
                              int64_t tilesInBlockM, int64_t tilesInBlockN,
                              SmallVector<SmallVector<Value>> &accTiles,
                              bool storeResult, PatternRewriter &rewriter) {
  bool isInteger = accTileTy.getElementType().isInteger();
  SmallVector<SmallVector<Value>> lhsTiles = loadBlockTiles(
      loc, lhsTileTy, lhsBuf, tilesInBlockM, 1, blockM, blockK, rewriter);

  for (int64_t tileN = 0; tileN < tilesInBlockN; ++tileN) {
    Value rhsTile = loadTile(loc, rhsTileTy, rhsBuf, 1, tilesInBlockN, blockK,
                             blockN, 0, tileN, rewriter);

    for (int64_t tileM = 0; tileM < tilesInBlockM; ++tileM) {
      if (isInteger)
        accTiles[tileM][tileN] =
            rewriter.create<amx::TileMulIOp>(loc, accTileTy, lhsTiles[tileM][0],
                                             rhsTile, accTiles[tileM][tileN]);
      else
        accTiles[tileM][tileN] =
            rewriter.create<amx::TileMulFOp>(loc, accTileTy, lhsTiles[tileM][0],
                                             rhsTile, accTiles[tileM][tileN]);

      // Insert store here to better mix stores with multiplications.
      if (storeResult) {
        storeTile(loc, accTileTy, accTiles[tileM][tileN], accBuf, tilesInBlockM,
                  tilesInBlockN, blockM, blockN, tileM, tileN, rewriter);
      }
    }
  }
}

// Similar to multiplyBlocksPreloadLhs but here RHS is preloaded to tiles.
void multiplyBlocksPreloadRhs(Location loc, VectorType lhsTileTy,
                              VectorType rhsTileTy, VectorType accTileTy,
                              const AmxBuffer &lhsBuf, const AmxBuffer &rhsBuf,
                              const AmxBuffer &accBuf, int64_t blockM,
                              int64_t blockN, int64_t blockK,
                              int64_t tilesInBlockM, int64_t tilesInBlockN,
                              SmallVector<SmallVector<Value>> &accTiles,
                              bool storeResult, PatternRewriter &rewriter) {
  bool isInteger = accTileTy.getElementType().isInteger();
  SmallVector<SmallVector<Value>> rhsTiles = loadBlockTiles(
      loc, rhsTileTy, rhsBuf, 1, tilesInBlockN, blockK, blockN, rewriter);

  for (int64_t tileM = 0; tileM < tilesInBlockM; ++tileM) {
    Value lhsTile = loadTile(loc, lhsTileTy, lhsBuf, tilesInBlockM, 1, blockM,
                             blockK, tileM, 0, rewriter);

    for (int64_t tileN = 0; tileN < tilesInBlockN; ++tileN) {
      if (isInteger)
        accTiles[tileM][tileN] = rewriter.create<amx::TileMulIOp>(
            loc, accTileTy, lhsTile, rhsTiles[0][tileN],
            accTiles[tileM][tileN]);
      else
        accTiles[tileM][tileN] = rewriter.create<amx::TileMulFOp>(
            loc, accTileTy, lhsTile, rhsTiles[0][tileN],
            accTiles[tileM][tileN]);

      // Insert store here to better mix stores with multiplications.
      if (storeResult) {
        storeTile(loc, accTileTy, accTiles[tileM][tileN], accBuf, tilesInBlockM,
                  tilesInBlockN, blockM, blockN, tileM, tileN, rewriter);
      }
    }
  }
}

LogicalResult convertCandidate(AmxDotOpCandidate &candidate,
                               PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  VectorType lhsTy = cast<VectorType>(op.getA().getType());
  VectorType rhsTy = cast<VectorType>(op.getB().getType());
  VectorType accTy = cast<VectorType>(op.getC().getType());
  VectorType resTy = cast<VectorType>(op.getResult().getType());
  VectorType lhsTileTy =
      lhsTy.cloneWith(SmallVector<int64_t>({candidate.tileM, candidate.tileK}),
                      candidate.lhsTileElemTy);
  VectorType rhsTileTy = getSwizzledRhsTileType(
      rhsTy.cloneWith(SmallVector<int64_t>({candidate.tileK, candidate.tileN}),
                      candidate.rhsTileElemTy));
  VectorType accTileTy =
      accTy.cloneWith(SmallVector<int64_t>({candidate.tileM, candidate.tileN}),
                      candidate.accTileElemTy);

  // If we don't work with a loop and want to directly store tiles into output
  // memory, then use the original store as insertion point to have its buffer
  // values available for generated code.
  if (!candidate.keepAccInBuf && !candidate.keepAccOnTiles &&
      !candidate.outBuf.empty())
    rewriter.setInsertionPoint(candidate.origStore);

  Operation *allocaPoint = op;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();

  // Cast input data if required and prepare input buffer. It might be temporary
  // buffers with stored vectors or the original input memory.
  Value lhs = maybeCast(loc, op.getA(), candidate.lhsTileElemTy, rewriter);
  AmxBuffer lhsBuf =
      prepareTensorBuffer(loc, lhs, false, false, true, allocaPoint, rewriter);

  Value rhs = maybeCast(loc, op.getB(), candidate.rhsTileElemTy, rewriter);
  AmxBuffer rhsBuf =
      prepareTensorBuffer(loc, rhs, true, false, true, allocaPoint, rewriter);

  Value acc = maybeCast(loc, op.getC(), candidate.accTileElemTy, rewriter);
  Value accToStore = acc;
  scf::ForOp forOp;
  if (candidate.keepAccInBuf || candidate.keepAccOnTiles) {
    forOp = cast<scf::ForOp>(op->getParentOp());
    accToStore = getInitAccValue(acc);
  }
  AmxBuffer accBuf;
  {
    // If accumulator is bufferized then we should move initial values before
    // the loop.
    OpBuilder::InsertionGuard g(rewriter);
    if (candidate.keepAccInBuf)
      rewriter.setInsertionPoint(forOp);
    accBuf =
        prepareTensorBuffer(loc, accToStore, false, !candidate.keepAccInBuf,
                            false, allocaPoint, rewriter);
  }

  AmxBuffer resBuf = prepareResultBuffer(
      loc, op.getResult(), accBuf, candidate.outBuf, allocaPoint, rewriter);

  SmallVector<SmallVector<Value>> accTiles;
  if (candidate.keepAccOnTiles)
    accTiles =
        moveLoopAccToTiles(loc, accTileTy, accBuf, candidate.tilesInBlockM,
                           candidate.tilesInBlockN, rewriter);

  int64_t blocksInAccM =
      accTy.getDimSize(0) / candidate.tileM / candidate.tilesInBlockM;
  int64_t blocksInAccN =
      accTy.getDimSize(1) / candidate.tileN / candidate.tilesInBlockN;
  int64_t tilesInVectorK = lhsTy.getDimSize(1) / candidate.tileK;
  for (int64_t blockM = 0; blockM < blocksInAccM; ++blockM) {
    for (int64_t blockN = 0; blockN < blocksInAccN; ++blockN) {
      if (!candidate.keepAccOnTiles)
        accTiles =
            loadBlockTiles(loc, accTileTy, accBuf, candidate.tilesInBlockM,
                           candidate.tilesInBlockN, blockM, blockN, rewriter);

      for (int64_t blocK = 0; blocK < tilesInVectorK; ++blocK) {
        // We can store accumulator if it is the last block over K dimension.
        // TODO: enable forward store for acc kept in tiles.
        bool storeAcc =
            !candidate.keepAccOnTiles && (blocK == (tilesInVectorK - 1));
        // We need to choose which block (LHS or RHS) to keep on tiles.
        // E.g. for ACC block 4x1 tiles, LHS block is also 4 tiles, so
        // we would use all tile registers trying to keep both ACC and
        // LHS blocks on registers. To decrease register pressure, keep
        // the smallest block on tiles.
        if (candidate.tilesInBlockM <= candidate.tilesInBlockN)
          multiplyBlocksPreloadLhs(
              loc, lhsTileTy, rhsTileTy, accTileTy, lhsBuf, rhsBuf, resBuf,
              blockM, blockN, blocK, candidate.tilesInBlockM,
              candidate.tilesInBlockN, accTiles, storeAcc, rewriter);
        else
          multiplyBlocksPreloadRhs(
              loc, lhsTileTy, rhsTileTy, accTileTy, lhsBuf, rhsBuf, resBuf,
              blockM, blockN, blocK, candidate.tilesInBlockM,
              candidate.tilesInBlockN, accTiles, storeAcc, rewriter);
      }
    }
  }

  // TODO: For keepAccOnTiles fix YieldOp to use mul results.
  // TODO: For keepAccOnTiles move all new forOp results to vector through a
  // buffer.
  if (candidate.keepAccOnTiles)
    llvm_unreachable("Not yet supported.");

  if (candidate.keepAccInBuf) {
    int resIdx = op.getResult().getUses().begin()->getOperandNumber();
    Value loopRes = forOp.getResult(resIdx);
    LDBG(
        "Loading buffererized accumulator to a vector to replace loop result.");
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(forOp);
    Value newVal = rewriter.create<vector::TransferReadOp>(
        loc, cast<VectorType>(acc.getType()), resBuf.memRef, resBuf.indices);
    // We might need to cast back to the original type.
    newVal = maybeCast(loc, newVal, accTy.getElementType(), rewriter);
    rewriter.replaceAllUsesWith(loopRes, newVal);
    // For now, just use init value for unused ForOp result instead of
    // its removal.
    rewriter.replaceOp(op, op.getC());
  } else if (candidate.outBuf.empty()) {
    LDBG("Loading the result to a vector to replace orig op result.");
    Value newVal = rewriter.create<vector::TransferReadOp>(
        loc, cast<VectorType>(acc.getType()), resBuf.memRef, resBuf.indices);
    // We might need to cast back to the original type.
    newVal = maybeCast(loc, newVal, accTy.getElementType(), rewriter);
    rewriter.replaceOp(op, newVal);
  } else {
    LDBG("Removing original operation and its use.");
    rewriter.eraseOp(*op.getResult().user_begin());
    rewriter.eraseOp(op);
  }

  return success();
}

struct ConvertDotToAMX
    : public triton::cpu::impl::ConvertDotToAMXBase<ConvertDotToAMX> {
  ConvertDotToAMX() = default;
  ConvertDotToAMX(bool convertInt8, bool convertFp16, bool convertBf16) {
    this->convertInt8 = convertInt8;
    this->convertFp16 = convertFp16;
    this->convertBf16 = convertBf16;
  }

  void runOnOperation() override {
    if (!convertInt8 && !convertFp16 && !convertBf16)
      return;

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    SmallVector<AmxDotOpCandidate> candidates;
    mod->walk([this, &candidates](cpu::DotOp op) {
      AmxDotOpCandidate candidate;
      if (isAmxCandidate(op, convertInt8, convertFp16, convertBf16,
                         candidate)) {
        LLVM_DEBUG({
          LDBG("Found AMX candidate");
          LDBG("  Op: " << candidate.op);
          LDBG("  LhsTileElemTy: " << candidate.lhsTileElemTy);
          LDBG("  RhsTileElemTy: " << candidate.rhsTileElemTy);
          LDBG("  AccTileElemTy: " << candidate.accTileElemTy);
          LDBG("  TileM: " << candidate.tileM);
          LDBG("  TileN: " << candidate.tileN);
          LDBG("  TileK: " << candidate.tileK);
          LDBG("  TilesInBlockM: " << candidate.tilesInBlockM);
          LDBG("  TilesInBlockN: " << candidate.tilesInBlockN);
          LDBG("  KeepAccOnTiles: " << candidate.keepAccOnTiles);
          LDBG("  KeepAccInBuf: " << candidate.keepAccInBuf);
          LDBG("  Has output buffer: " << !candidate.outBuf.empty());
        });
        candidates.push_back(candidate);
      }
      return WalkResult::advance();
    });

    for (auto &candidate : candidates) {
      LDBG("Starting conversion of candidate: " << candidate.op);
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate.op);
      if (succeeded(convertCandidate(candidate, rewriter))) {
        LDBG("Conversion succeeded!");
      } else {
        LDBG("Conversion failed!");
      }
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotToAMX() {
  return std::make_unique<ConvertDotToAMX>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotToAMX(bool convertInt8, bool convertFp16, bool convertBf16) {
  return std::make_unique<ConvertDotToAMX>(convertInt8, convertFp16,
                                           convertBf16);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
