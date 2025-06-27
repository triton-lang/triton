#include "Utility.h"
#include "Dialect/NVGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir {
namespace LLVM {
namespace NVIDIA {
using namespace mlir::triton;

static Value shuffleCommonImpl(Location loc, RewriterBase &rewriter, Value val,
                               Value i, NVVM::ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32_ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32_ty, vec, b.i32_val(1));
    val0 = shuffleCommonImpl(loc, rewriter, val0, i, mode, clamp);
    val1 = shuffleCommonImpl(loc, rewriter, val1, i, mode, clamp);
    vec = b.undef(vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, val.getType());
  }
  Type type = val.getType();
  if (type != i32_ty) {
    val = b.bitcast(val, int_ty(bits));
    if (bits < 32)
      val = b.zext(i32_ty, val);
  }
  Value mask = b.i32_val(0xFFFFFFFF);
  Value result = rewriter.create<NVVM::ShflOp>(loc, i32_ty, mask, val, i, clamp,
                                               mode, UnitAttr());
  if (type != i32_ty) {
    if (bits < 32)
      result = b.trunc(int_ty(bits), result);
    result = b.bitcast(result, type);
  }
  return result;
}

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, NVVM::ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // To shuffle pointers, convert them to i64.
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result = shuffleCommonImpl(loc, rewriter, val, i, mode, clamp);
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), NVVM::ShflKind::bfly,
                       b.i32_val(0x1f));
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), NVVM::ShflKind::up,
                       b.i32_val(0x0));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleIdx(loc, rewriter, val, b.i32_val(i));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, i, NVVM::ShflKind::idx,
                       b.i32_val(0x1f));
}

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);

  // It is not easy to get the compute capability here, so we use numCTAs to
  // decide the semantic of GetProgramIdOp. If numCTAs = 1, then
  // GetProgramIdOp is converted to "%ctaid", otherwise it is converted to
  // "%clusterid".
  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

  std::string sreg = numCTAs == 1 ? "ctaid." : "clusterid.";
  sreg.append(1, 'x' + axis); // 0 -> 'x', 1 -> 'y', 2 -> 'z'
  return getSRegValue(rewriter, loc, sreg);
}

Value getSRegValue(OpBuilder &rewriter, Location loc, StringRef sRegStr) {
  ValueRange args;
  auto intrName = Twine("llvm.nvvm.read.ptx.sreg.") + sRegStr;
  auto callOp =
      createLLVMIntrinsicCallOp(rewriter, loc, intrName.str(), i32_ty, args);
  return callOp.getResult(0);
}

Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
              Value mask) {
  Value args[] = {a, b, mask};
  auto op =
      createLLVMIntrinsicCallOp(rewriter, loc, "llvm.nvvm.prmt", i32_ty, args);
  return op.getResult(0);
}

/// Create a predicate with just single active thread.
Value createElectPredicate(Location loc, RewriterBase &rewriter) {
  return rewriter.create<NVVM::ElectSyncOp>(loc, i1_ty);
}

void createSyncWarp(Location loc, OpBuilder &rewriter) {
  TritonLLVMOpBuilder b(loc, rewriter);
  Type resultTy = void_ty(rewriter.getContext());
  Value args[] = {b.i32_val(0xffffffff)};
  createLLVMIntrinsicCallOp(rewriter, loc, "llvm.nvvm.bar.warp.sync", resultTy,
                            args);
}

Value createElectPredicateWarp0(Location loc, RewriterBase &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value warpId = getLaneAndWarpId(rewriter, loc).second;
  Value warp0 = b.icmp_eq(warpId, b.i32_val(0));
  return b.and_(warp0, createElectPredicate(loc, rewriter));
}

LogicalResult lowerLdStMatrixImpl(
    Location loc, LinearLayout cvt, bool transpose,
    SmallVector<Value> &vals, // Input for stmatrix, output for ldmatrix
    Value smemBase, Value affineOffset, uint64_t maskSpanAffineOffset,
    Type llvmElemTy, ConversionPatternRewriter &rewriter,
    const ::triton::NVIDIA::TargetInfo &targetInfo) {
  // Lower load via ldmatrix, store via stmatrix

  bool isStore = !vals.empty();
  if (isStore && !targetInfo.supportStMatrix())
    return failure();
  if (!isStore && !targetInfo.supportLdMatrix())
    return failure();

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();

  auto S = [ctx](StringRef v) { return StringAttr::get(ctx, v); };
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto kBlock = S("block");
  auto kOffset = S("offset");
  auto smemPtrTy = ptr_ty(ctx, 3);
  auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
  // In the transpose case, consecutive elements are not stored contiguously
  // so we cannot split an fp32
  // We could support bitwidth == 8, but it'd be a rather weird layout
  // so we don't do that for now
  if ((!transpose && bitwidth > 32) || (transpose && bitwidth != 16))
    return failure();
  // Inter block stmatrix is not supported
  if (cvt.hasInDim(kBlock))
    return failure();

  // We must have at least 32-bits worth of registers to use these instructions
  if (transpose && cvt.getInDimSizeLog2(kReg) < llvm::Log2_32(32 / bitwidth)) {
    return failure();
  }

  std::optional<ColumnAction> maybePermutation;
  LinearLayout tile;
  if (!transpose) {
    tile = LinearLayout::identity1D(32 / bitwidth, kReg, kOffset) *
           LinearLayout::identity1D(4, kLane, kOffset);
    // Find if there is a register permutation that allows us to divideLeft
    // We need to pass the map from regs to offsets, as is cvt
    maybePermutation = regPermForDivide(cvt, tile, /*left=*/true);
    if (!maybePermutation.has_value()) {
      return failure();
    }
    auto permutation = maybePermutation.value();
    // Check if the action indeed allows us to divideLeft
    cvt = permutation.apply(cvt);
    if (isStore) {
      vals = permutation.apply(vals);
    }
  }

  LinearLayout reps;
  if (!transpose) {
    auto maybeQuot = divideLeft(cvt, tile);
    if (!maybeQuot.has_value()) {
      return failure();
    }
    reps = zerosLike(tile) * maybeQuot.value();
  } else {
    // Division does not quite work here. To define this properly, we would need
    // to define a different multiplication that does:
    // A *' B = [[0, A], [B, 0]] and define leftDivision for it
    // We do it ad-hoc for now, as I beleive there's not much demand for this op
    // outside of this lowering.

    // We implement leftDivision as above for B = identity1D(8, kLane, kOffset)
    // Divisibility in the sense above is the same as regular divisibility
    // You need to see that the tile A is a sublayout of the matrix, and that
    // it has zeros above it and to its right.

    // In particular, offsets lanes 4, 8, 16 map to offsets 1, 2, 4...
    const auto &laneBases = cvt.getBases().find(kLane)->second;
    for (int i = 0; i < 3; ++i) {
      if (laneBases[i + 2][0] != (1 << i))
        return failure();
    }
    // ... and no other basis should depend on 1, 2, 4
    // Note that this gives us the usual alignment condition, but we have
    // translated it to checking that the matrix to the left of A is all zeros
    for (auto dim : cvt.getInDimNames()) {
      const auto &bases = cvt.getBases().find(dim)->second;
      for (auto [i, basis] : llvm::enumerate(bases)) {
        if (dim == kLane && i >= 2)
          continue;
        if (basis[0] & 0b111)
          return failure();
      }
    }

    // Hack: We are not going to use in the rest of the function reps[kLane][2:]
    // so we don't need to zero them out
    reps = cvt;
  }

  // If we are lowering a subslice, the subslice offsets shall not touch the
  // contiguous part of the tile
  if (maskSpanAffineOffset & (llvm::Log2_32(128 / bitwidth) - 1)) {
    return failure();
  }

  // Choose up to 4 packs of 32-bit elements indexed by the next (at most) two
  // bases as the vectorisation factor. We don't consider the basis of the tile
  // for vectorisation so we substract them
  auto vec = std::min<int32_t>(2, reps.getInDimSizeLog2(kReg) -
                                      llvm::Log2_32(32 / bitwidth));

  // Map from kReg, kLane, kWarp to beginning of each tile
  assert(reps.getOutDimSize(kOffset) == cvt.getOutDimSize(kOffset));

  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  // Compute the addresses for the 0th tile
  // Here we implement the stmatrix.x4 addressing. As per the PTX docs, the
  // threads 0-7 hold the address of the first element of the 8 columns of the
  // first submatrix, threads 8-15 for the second submatrix, etc. In general we
  // map:
  // - The lowest 3 bits of the laneId to the columns of each submatrix, which
  // is
  //   given by the 3 kLane bases of quotient that are not part of the tile
  // - The top `vec` bits of the thread id to the submatrix number, which is
  // given
  //   by the first `vec` reg bases that are not part of the tile
  std::vector<std::vector<int32_t>> laneBases;
  if (!transpose) {
    auto tileDimSizeReg = llvm::Log2_32(32 / bitwidth);
    auto tileDimSizeLane = 2;
    for (int i = 0; i < 3; ++i) {
      laneBases.push_back(reps.getBasis(kLane, tileDimSizeLane + i));
    }
    for (int i = 0; i < vec; ++i) {
      laneBases.push_back(reps.getBasis(kReg, tileDimSizeReg + i));
    }
  } else {
    // We choose the first basis of the register. In the future we could choose
    // a basis that minimises the bank conflicts
    laneBases.push_back(reps.getBasis(kReg, 0));
    laneBases.push_back(reps.getBasis(kLane, 0));
    laneBases.push_back(reps.getBasis(kLane, 1));
    for (int i = 0; i < vec; ++i) {
      laneBases.push_back(reps.getBasis(kReg, i + 1));
    }
  }

  LinearLayout addrLayout =
      LinearLayout({{kLane, laneBases}, {kWarp, reps.getBases().lookup(kWarp)}},
                   {{kOffset, reps.getOutDimSize(kOffset)}}, false);
  auto regBase = applyLinearLayout(loc, rewriter, addrLayout,
                                   {{kLane, laneId}, {kWarp, warpId}})[0]
                     .second;
  regBase = b.xor_(regBase, affineOffset);

  if (isStore) {
    assert(vals.size() == cvt.getInDimSize(kReg));
  }

  // Elements per op
  auto nVecs = 1 << vec;
  auto elemsPerVec = 32 / bitwidth;
  auto step = nVecs * elemsPerVec;
  for (int i = 0; i < cvt.getInDimSize(kReg); i += step) {
    auto regIdx = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}})[0].second;
    Value offset = b.xor_(regBase, b.i32_val(regIdx));
    auto vecAddr = b.gep(smemPtrTy, llvmElemTy, smemBase, offset,
                         LLVM::GEPNoWrapFlags::inbounds);
    Type packedTy = vec_ty(llvmElemTy, 32 / bitwidth);
    auto layout = transpose ? NVVM::MMALayout::col : NVVM::MMALayout::row;
    if (isStore) {
      // Pack into vector of i32
      SmallVector<Value> inputs;
      for (int j = 0; j < nVecs; j++) {
        Value input = b.undef(packedTy);
        for (int k = 0; k < elemsPerVec; k++) {
          input = b.insert_element(packedTy, input,
                                   vals[i + j * elemsPerVec + k], b.i32_val(k));
        }
        inputs.push_back(b.bitcast(input, i32_ty));
      }
      rewriter.create<NVVM::StMatrixOp>(loc, vecAddr, inputs, layout);
    } else {
      Type matTy = nVecs == 1
                       ? i32_ty
                       : static_cast<Type>(LLVM::LLVMStructType::getLiteral(
                             ctx, SmallVector<Type>(nVecs, i32_ty)));
      auto res =
          rewriter
              .create<triton::nvgpu::LoadMatrixOp>(
                  loc, matTy, vecAddr, triton::nvgpu::LoadMatrixShape::m8n8,
                  /*bitWidth=*/16,
                  /*needTrans=*/transpose)
              .getResult();
      // Extract result into srcVals
      for (int j = 0; j < nVecs; j++) {
        Value output = nVecs == 1 ? res : b.extract_val(i32_ty, res, j);
        output = b.bitcast(output, vec_ty(llvmElemTy, elemsPerVec));
        for (int k = 0; k < elemsPerVec; k++) {
          vals.push_back(b.extract_element(llvmElemTy, output, b.i32_val(k)));
        }
      }
    }
  }

  if (!isStore) {
    assert(vals.size() == cvt.getInDimSize(kReg));
    // Undo the permutation and the removeBroadcast
    if (maybePermutation.has_value()) {
      auto invPerm = maybePermutation.value().inverse();
      vals = invPerm.apply(vals);
    }
  }
  return success();
}

} // namespace NVIDIA
} // namespace LLVM
} // namespace mlir
