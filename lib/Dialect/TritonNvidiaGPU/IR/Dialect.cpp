/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include <numeric>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

static constexpr int numTmemRows = 128;

TMemAllocation getTmemAllocSizes(MemDescType memDescType) {
  auto *ctx = memDescType.getContext();
  auto S = [&](StringRef str) { return StringAttr::get(ctx, str); };
  auto kRow = S("row");
  auto kCol = S("col");
  // Remove multibuffering if present
  auto shape = memDescType.getShape().take_back(2);
  auto ll = toLinearLayout(shape, memDescType.getEncoding());
  auto bitwidth = memDescType.getElementTypeBitWidth();
  int nRow = ll.getInDimSize(kRow);
  int nCol = ll.getInDimSize(kCol) / (32 / bitwidth);
  // If we have just one 16xcol block per warp, we don't allocate 128 rows
  // we use 64 rows instead.
  // We could generalise this to when we have more zeros in the layout, but
  // the allocator does not support this yet
  if (ll.getBasis(kRow, llvm::Log2_32(16)) == ArrayRef{0, 0}) {
    nRow /= 2;
  }

  // Hack: We should represent this in the LL. Remove the block dimension
  if (auto tmemEnc =
          dyn_cast<TensorMemoryEncodingAttr>(memDescType.getEncoding())) {
    nCol /= tmemEnc.getCTASplitM() * tmemEnc.getCTASplitN();
  } else if (auto tmemScaleEnc = dyn_cast<TensorMemoryScalesEncodingAttr>(
                 memDescType.getEncoding())) {
    nCol /= tmemScaleEnc.getCTASplitM() * tmemScaleEnc.getCTASplitN();
  }
  // If multibuffering is present, we need to allocate more cols
  if (memDescType.getRank() > 2) {
    assert(memDescType.getRank() == 3);
    nCol *= memDescType.getDimSize(0);
  }
  return {nRow, nCol};
}

LinearLayout getTileLayout(MLIRContext *ctx, TMemAccessAtom atom, bool unpacked,
                           bool withWarp) {
  auto str_attr = [&](StringRef str) { return StringAttr::get(ctx, str); };
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kRow = str_attr("row");
  auto kCol = str_attr("col");
  // Set the output order to be kRow, kCol and the input order to be kReg first
  LinearLayout tile = LinearLayout({{kReg, {}}, {kLane, {}}}, {kRow, kCol});
  // Each register moves 32/bitwidth (= 2) columns when unpacked
  if (unpacked) {
    tile *= LinearLayout::zeros1D(1, kReg, kCol, 2);
  }
  if (atom == TMemAccessAtom::I32x32b) {
    tile *= LinearLayout::identity1D(32, kLane, kRow);
  } else if (atom == TMemAccessAtom::I16x32bx2) {
    tile *= LinearLayout::identity1D(16, kLane, kRow);
  } else if (atom == TMemAccessAtom::I16x64b) {
    LinearLayout::BasesT bases;
    bases[kLane] = std::vector<std::vector<int32_t>>{
        {8, 0}, {0, 1}, {1, 0}, {2, 0}, {4, 0}};
    tile *= LinearLayout(std::move(bases), {kRow, kCol});
  } else if (atom == TMemAccessAtom::I16x128b) {
    tile *= LinearLayout::identity1D(4, kLane, kCol) *
            LinearLayout::identity1D(8, kLane, kRow) *
            LinearLayout::identity1D(2, kReg, kRow);
  } else if (atom == TMemAccessAtom::I16x256b) {
    tile *= LinearLayout::identity1D(2, kReg, kCol) *
            LinearLayout::identity1D(4, kLane, kCol) *
            LinearLayout::identity1D(8, kLane, kRow) *
            LinearLayout::identity1D(2, kReg, kRow);
  } else {
    llvm_unreachable("Unsupported TMEM access atom");
  }
  if (withWarp) {
    auto nCol = tile.getOutDimSize(kCol);
    auto bases = tile.getBases();
    bases[kWarp].push_back({32, 0});
    bases[kWarp].push_back({64, 0});
    tile = LinearLayout(std::move(bases), {{kRow, 128}, {kCol, nCol}}, false);
  }
  return tile;
}

static std::optional<LinearLayout> getDistributedLayoutForTmemLdSt(
    const LinearLayout &ll, TMemAccessAtom atom, unsigned numWarps,
    int bitwidth,
    std::optional<gpu::CGAEncodingAttr> cgaLayout = std::nullopt) {
  auto dims = to_vector(ll.getOutDimNames());
  assert(dims.size() == 2);
  auto rowColDims = to_vector(ll.getInDimNames());
  auto *ctx = dims[0].getContext();
  // Add block dimension
  if (cgaLayout) {
    // Get CGALayout without broadcasting to divide the ll
    // as the TMEM layout does not reflect CTA broadcasting
    auto cgaShape = to_vector(cgaLayout->getLinearLayout().getOutDimSizes());
    auto kBlock = StringAttr::get(ctx, "block");
    // The cta order in TMEM is always [0, 1]
    auto ctaCol =
        LinearLayout::identity1D(cgaShape[0], rowColDims[1], dims[0]) *
        LinearLayout::identity1D(cgaShape[1], rowColDims[1], dims[1]);
    auto quot = divideRight(ll, ctaCol);
    assert(quot.has_value());
    auto maybeRet =
        getDistributedLayoutForTmemLdSt(*quot, atom, numWarps, bitwidth);
    if (!maybeRet)
      return maybeRet;
    // Add the full block layout (with broadcasting)
    return *maybeRet * cgaLayout->getLinearLayout();
  }
  // This code is dual to the one in lowerTMemLdSt
  if (bitwidth != 32) {
    // TODO move this to a helper function
    auto kReg = StringAttr::get(ctx, "register");
    LinearLayout quot;
    int bestContig = 1;
    for (int contig = 1; bitwidth * contig <= 32; contig *= 2) {
      auto maybeQuot = divideLeft(
          ll, LinearLayout::identity1D(contig, rowColDims[1], dims[1]));
      if (!maybeQuot)
        break;
      quot = *maybeQuot;
      bestContig = contig;
    }

    // Pack contiguous elements
    // This works to pack b8 or b16 into b32 but also b8 into b16 and recurse
    if (bestContig > 1) {
      auto ret = getDistributedLayoutForTmemLdSt(quot, atom, numWarps,
                                                 bitwidth * bestContig);
      if (!ret)
        return ret;
      auto castbbitwidth = LinearLayout::identity1D(bestContig, kReg, dims[1]);
      return castbbitwidth * ret.value();
    }
    if (auto maybeQuot = divideLeft(
            ll, LinearLayout::zeros1D(32 / bitwidth, rowColDims[1], dims[1]) *
                    LinearLayout::identity1D(2, rowColDims[1], dims[1]));
        bitwidth == 16 && maybeQuot) {
      // Unpacked case
      auto ret =
          getDistributedLayoutForTmemLdSt(*maybeQuot, atom, numWarps, 32);
      if (!ret)
        return ret;
      auto castbbitwidth = LinearLayout::identity1D(2, kReg, dims[1]);
      return castbbitwidth * ret.value();
    } else if (auto maybeQuot =
                   divideLeft(ll, LinearLayout::zeros1D(
                                      32 / bitwidth, rowColDims[1], dims[1]))) {
      // Software padding
      assert(maybeQuot);
      return getDistributedLayoutForTmemLdSt(*maybeQuot, atom, numWarps, 32);
    } else if (ll.getInDimSize(rowColDims[1]) == 1) {
      // Software padding with just one column
      return getDistributedLayoutForTmemLdSt(ll, atom, numWarps, 32);
    } else {
      assert(false && "Should not happen");
    }
  }
  // getTileLayout returns the layout for a bitwidth of 32
  assert(bitwidth == 32);
  auto tile = getTileLayout(ctx, atom, false, /*withWarp=*/false);
  // Plan:
  // tile: register, lane -> row, cols
  // ll: row, cols -> dim0, dim1
  // We extend the tile to have the right vectorisation + warps and
  // the result is given by
  // ll o tile : register, lane, warp -> dim0, dim1

  auto nColsTile = tile.getOutDimSize(rowColDims[1]);
  auto nColsLL = ll.getInDimSize(rowColDims[1]);
  auto nColsMissing = nColsLL / nColsTile;
  if (nColsMissing == 0) {
    return std::nullopt;
  }
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  bool instr32Rows = atom == TMemAccessAtom::I32x32b;
  bool layout16Rows =
      ll.getBasis(rowColDims[0], llvm::Log2_32(16)) == ArrayRef{0, 0};

  // We are choosing the distributed layout (ll o tile). In the lowering
  // we will do ll^{-1} o (ll o tile) and we expect to get tile back.
  // For this to be possible, ll should accept a left-inverse, that is, it
  // should be injective
  // In less fancy words, we look for the `comp` layout not to have any zero
  // basis as that would disallow the resulting layout to be left-divisible by
  // the tile
  auto comp =
      tile.compose(ll).sublayout({kReg, kLane}, to_vector(ll.getOutDimNames()));
  if (instr32Rows) {
    // We will use 16x32bx2 instruction for lane=16 so we remove the last lane
    // basis
    comp = comp.resizeInDim(kLane, comp.getInDimSize(kLane) / 2);
  }
  if (!comp.isInjective())
    return std::nullopt;

  // Fit the warp bases either tiling on the RHS or in row=16
  StringAttr row16;
  // If we need to fit something (the instruction does not cover it
  // and the layout has 32 rows) we first try to fit a warp, and if we
  // can't we fit a register
  if (!instr32Rows && !layout16Rows) {
    if (numWarps > 4) {
      row16 = kWarp;
    } else {
      row16 = kReg;
    }
  }

  // We reserve enough columns to fit in the warps
  int warpsToTile = numWarps / ((row16 == kWarp) ? 8 : 4);
  // Cap warps to tile above by nColsMissing. The rest go to broadcasting
  int warpBroadcast = warpsToTile / std::min(nColsMissing, warpsToTile);
  warpsToTile /= warpBroadcast;
  nColsMissing /= warpsToTile;

  if (nColsMissing > 1) {
    if (instr32Rows && layout16Rows) {
      // If the lane 16 would load repeated data, instead we make it load half
      // of the data via the 16x32bx2 instruction
      tile = divideLeft(tile, LinearLayout::identity1D(2, kLane, rowColDims[0]))
                 .value();
      tile *= LinearLayout::identity1D(nColsMissing / 2, kReg, rowColDims[1]) *
              LinearLayout::identity1D(2, kLane, rowColDims[1]);

    } else {
      tile *= LinearLayout::identity1D(nColsMissing, kReg, rowColDims[1]);
    }
  }

  // add the warp bases. The M=64 + 2CTA case has already been handled
  auto bases = tile.getBases();
  auto &warpBases = bases[kWarp];
  warpBases.push_back({32, 0});
  warpBases.push_back({64, 0});

  if (row16) {
    bases[row16].push_back({16, 0});
  }
  tile = LinearLayout(std::move(bases),
                      {{rowColDims[0], 128},
                       {rowColDims[1], tile.getOutDimSize(rowColDims[1])}},
                      false);
  tile *= LinearLayout::identity1D(warpsToTile, kWarp, rowColDims[1]);
  tile *= LinearLayout::zeros1D(warpBroadcast, kWarp, rowColDims[1]);
  assert(tile.getOutDimSize(rowColDims[1]) == ll.getInDimSize(rowColDims[1]));

  auto ret = tile.compose(ll);
  return ret;
}

std::optional<LinearLayout>
getDistributedLayoutForTmemLdSt(gpu::MemDescType memType, TMemAccessAtom atom,
                                unsigned numWarps,
                                gpu::CGAEncodingAttr cgaLayout) {
  assert(memType.getMemorySpace() ==
         TensorMemorySpaceAttr::get(memType.getContext()));
  assert(numWarps >= 4 && llvm::isPowerOf2_32(numWarps) &&
         "numWarps must be a power of 2 and >= 4");
  assert(atom != TMemAccessAtom::I16x32bx2 &&
         "This layout is inferred sometimes for the 32x32b atom");
  auto ll = toLinearLayout(memType.getShape(), memType.getEncoding());
  auto bitwidth = memType.getElementTypeBitWidth();
  return getDistributedLayoutForTmemLdSt(ll, atom, numWarps, bitwidth,
                                         cgaLayout);
}

DistributedEncodingTrait
getDefaultLayoutForTmemLdSt(gpu::MemDescType memType, unsigned numWarps,
                            gpu::CGAEncodingAttr cgaLayout) {
  auto *ctx = memType.getContext();
  bool prefer16x256 =
      triton::tools::getBoolEnv("TRITON_PREFER_TMEM_16x256_LAYOUT");
  if (prefer16x256) {
    auto layout = getDistributedLayoutForTmemLdSt(
        memType, TMemAccessAtom::I16x256b, numWarps, cgaLayout);
    if (layout) {
      return LinearEncodingAttr::get(ctx, std::move(*layout));
    }
  }
  auto layout = getDistributedLayoutForTmemLdSt(
      memType, TMemAccessAtom::I32x32b, numWarps, cgaLayout);
  assert(layout);
  return LinearEncodingAttr::get(ctx, std::move(*layout));
}

std::optional<DistributedEncodingTrait>
getTmemLoadLayoutSplitLongM(RankedTensorType tensorType, MemDescType memType,
                            int numWarps) {
  if (numWarps != 8)
    return std::nullopt;

  auto cgaLayout = getCGALayout(tensorType.getEncoding());
  std::optional<LinearLayout> layout = getDistributedLayoutForTmemLdSt(
      memType, TMemAccessAtom::I32x32b, numWarps, cgaLayout);
  if (!layout)
    return std::nullopt;
  auto ret = std::move(*layout);

  // Optimisation for reductions:
  // We can map lane=16 to any dimension, and it will be lowered to 32x16bx2.
  // As such, if we have 8 warps and the basis warp=4 is mapped to a different
  // dimension than warp=1, warp=2, and lane=16 is mapped to the same dimension
  // as the first two warp bases, we can swap warp=4 and lane=16.
  // Generally, we don't want warp=4 to have data on a different dimension to
  // dim=1 and dim=2
  auto *ctx = tensorType.getContext();
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto dims = to_vector(ret.getOutDimNames());

  // In most cases this is going to be dim=0, but the optimization
  // also applies for scales where we may be able to have the layout
  // replicated across warps
  for (int dim : {0, 1}) {
    auto w1dim = ret.getBasis(kWarp, 0, dims[dim]) == 0;
    auto w2dim = ret.getBasis(kWarp, 1, dims[dim]) == 0;
    auto w4dim = ret.getBasis(kWarp, 2, dims[dim]) == 0;
    auto l16dim = ret.getBasis(kLane, 4, dims[dim]) == 0;
    if (l16dim != w4dim && w1dim == w2dim && w1dim == l16dim) {
      auto bases = ret.getBases();
      std::swap(bases[kWarp][2], bases[kLane][4]);
      return LinearEncodingAttr::get(
          tensorType.getContext(),
          LinearLayout(std::move(bases), ret.getOutDims(), ret.isSurjective()));
    }
  }
  return std::nullopt;
}

SmallVector<DistributedEncodingTrait>
getTmemCompatibleLayouts(Operation *op, RankedTensorType tensorType,
                         MemDescType memType) {
  int numWarps = lookupNumWarps(op);
  assert(numWarps % 4 == 0);
  auto cgaLayout = getCGALayout(tensorType.getEncoding());
  SmallVector<DistributedEncodingTrait> layouts;
  for (auto atom : {TMemAccessAtom::I32x32b, TMemAccessAtom::I16x256b,
                    TMemAccessAtom::I16x128b, TMemAccessAtom::I16x64b}) {
    auto ll =
        getDistributedLayoutForTmemLdSt(memType, atom, numWarps, cgaLayout);
    if (ll) {
      layouts.push_back(LinearEncodingAttr::get(tensorType.getContext(),
                                                std::move(ll.value())));
    }
  }
  // Small hack until we generalise isDistributedLayoutTMemCompatible
  auto ll = getTmemLoadLayoutSplitLongM(tensorType, memType, numWarps);
  if (ll) {
    layouts.push_back(ll.value());
  }
  return layouts;
}

// Verify if the distributed layout can be mapped onto tensor memory.
bool isDistributedLayoutTMemCompatible(Operation *op,
                                       RankedTensorType tensorType,
                                       gpu::MemDescType memType) {
  auto maxnreg = getContextualMaxNReg(op);
  return succeeded(computeTMemLdStEncodingInfo(tensorType, memType, maxnreg));
}

LogicalResult
TensorMemoryEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 unsigned blockM, unsigned blockN,
                                 unsigned colStride, unsigned CTASplitM,
                                 unsigned CTASplitN, bool) {
  if (!(CTASplitM >= 1 && CTASplitN >= 1 && llvm::isPowerOf2_32(CTASplitM) &&
        llvm::isPowerOf2_32(CTASplitN))) {
    return emitError()
           << "CTASplitM and CTASplitN must be greater than 0 and a power of 2";
  }
  if (blockM != 64 && blockM != 128) {
    return emitError() << "blockM must be 64 or 128 but got " << blockM;
  }
  if (!llvm::isPowerOf2_32(blockN)) {
    return emitError() << "blockN must be a power of 2 but got " << blockN;
  }
  if (blockN > 512) {
    return emitError() << "blockN must be less than or equal to 512 but got "
                       << blockN;
  }
  if (!(colStride == 1 || colStride == 2 || colStride == 4)) {
    return emitError() << "colStride must be 1, 2, or 4 but got "
                       << "but got " << colStride;
  }
  return success();
}

LogicalResult impl::verifyMMAv5Op(Operation *op) {
  auto isInterleaved = [](MemDescType memdesc) {
    auto enc = dyn_cast<TensorMemoryEncodingAttr>(memdesc.getEncoding());
    return enc && getTmemAllocSizes(memdesc).numRows != 64 &&
           enc.getBlockM() == 64;
  };

  auto itf = cast<MMAv5OpInterface>(op);
  if (isInterleaved(itf.getA().getType()) &&
      isInterleaved(itf.getAccumulator().getType())) {
    return op->emitOpError(
        "does not support blockM=64 with interleaved blocks in TMEM layout");
  }
  return success();
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// ASM Interface (i.e.: alias)
//===----------------------------------------------------------------------===//
namespace {
class TritonGPUOpAsmInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto sharedAttr = mlir::dyn_cast<TensorMemoryEncodingAttr>(attr)) {
      os << "tmem";
      return AliasResult::FinalAlias;
    }
    if (mlir::isa<TensorMemoryScalesEncodingAttr>(attr)) {
      os << "tmem_scales";
      return AliasResult::FinalAlias;
    }
    return OpAsmDialectInterface::getAlias(attr, os);
  }
};
} // namespace

//===----------------------------------------------------------------------===//

void TritonNvidiaGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonNvidiaGPU/IR/Ops.cpp.inc"
      >();
  addInterfaces<TritonGPUOpAsmInterface>();
  addInterfaces<TritonInlinerInterface>();
}

// verify TritonNvidiaGPU ops
LogicalResult
TritonNvidiaGPUDialect::verifyOperationAttribute(Operation *op,
                                                 NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
