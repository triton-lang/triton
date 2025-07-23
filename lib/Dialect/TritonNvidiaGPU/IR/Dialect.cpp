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
#include "triton/Tools/Sys/GetEnv.hpp"

#include <numeric>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
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
  const int rowSizeInBytes = 4;
  auto shapePerCTA = triton::gpu::getShapePerCTA(memDescType);
  if (isa<TensorMemoryScalesEncodingAttr>(memDescType.getEncoding())) {
    // For scales the data are packed and replicated 4 times.
    assert(memDescType.getElementType().getIntOrFloatBitWidth() == 8);
    assert(memDescType.getShape().size() == 2 &&
           "TODO handle multibuffering of scales.");
    int k = shapePerCTA[1];
    int m = shapePerCTA[0];
    int numColumn = ceil<int>(m, 32) * ceil<int>(k, 4);
    return TMemAllocation(numColumn, numTmemRows);
  }
  assert(isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
             memDescType.getEncoding()) &&
         "Expecting a tensor memory encoding attribute");
  triton::nvidia_gpu::TensorMemoryEncodingAttr attr =
      cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
          memDescType.getEncoding());
  bool isUnpacked = attr.getUnpacked();
  int64_t elementSizeInBytes =
      isUnpacked ? rowSizeInBytes
                 : memDescType.getElementType().getIntOrFloatBitWidth() / 8;
  int sizeInBytes = product(shapePerCTA) * elementSizeInBytes;
  int numRows = numTmemRows;
  // BlockM of 64 is and interleaved format, where for single message only the
  // first 16 rows are used. For multiple blocks, the rows are interleaved, i.e.
  //  0                   N/2                     N
  //  ---------------------------------------------
  // 0  0,0 0,1... 0,N/2-1   0,N/2 0,N/2+1 ... 0, N-1  \
  //...                                                  Block 0
  // 15 15,0 15,1..15,N/2-1  15,N/2 15,N/2+1...15, N-1 /
  // 16 0,0 0,1... 0,N/2-1   0,N/2 0,N/2+1 ... 0, N-1  \
  //...                                                  Block 1
  // 31 15,0 15,1..15,N/2-1  15,N/2 15,N/2+1...15, N-1 /
  // Note that allocations that consists of single block of 64 rows are
  // "sparse" and only half of the rows are used.
  // Note that even for 3D shapes for which 2D slices are big enough to fit
  // entire tensor block, we will use "sparse" allocation.
  int blockM = attr.getBlockM();
  int blockN = attr.getBlockN();
  int lastDim = shapePerCTA.size() - 1;
  int isSingleBlock =
      (shapePerCTA[lastDim - 1] <= blockM) && (shapePerCTA[lastDim] <= blockN);
  if (blockM == 64 && isSingleBlock)
    numRows = 64;
  int numColumn = ceil<int>(sizeInBytes, (numRows * rowSizeInBytes));
  return TMemAllocation(numColumn, numRows);
}

Attribute getTmemLoadStoreLayout32x32b(unsigned M, unsigned N,
                                       RankedTensorType oldType,
                                       unsigned numWarps) {
  assert(numWarps == 4 || numWarps == 8);
  auto shape = getShapePerCTA(oldType);
  assert(shape.size() == 2);
  SmallVector<unsigned> sizePerThread;
  SmallVector<unsigned> threadsPerWarp;
  SmallVector<unsigned> warpsPerCTA;
  SmallVector<unsigned> order;
  SmallVector<unsigned> blocksPerTile = {(unsigned)shape[0] / M,
                                         (unsigned)shape[1] / N};
  int numBlocks = blocksPerTile[0] * blocksPerTile[1];
  if (M == 64) {
    unsigned numWarpGroups = numWarps / 4;
    if (numBlocks == 1) {
      // Split along the N dimension
      sizePerThread = {1, ceil<unsigned>(N, numWarpGroups * 2)};
      threadsPerWarp = {16, 2};
      warpsPerCTA = {4, numWarpGroups};
    } else {
      sizePerThread = {1, ceil<unsigned>(N, 2)};
      threadsPerWarp = {16, 2};
      warpsPerCTA = {0, 0};
      // Distribute at most as many warp groups as there is blocks
      // along M dimension.
      warpsPerCTA[0] = 4 * std::min(blocksPerTile[0], numWarpGroups);
      // Distribute rest of the warp groups along N dimension.
      warpsPerCTA[1] = ceil<int>(numWarpGroups, warpsPerCTA[0] / 4);
    }
  } else {
    unsigned numWarpGroups = numWarps / 4;
    if (shape[0] > 128) {
      // Split along M dimension
      sizePerThread = {1, N};
      threadsPerWarp = {32, 1};
      warpsPerCTA = {4 * numWarpGroups, 1};
    } else {
      // Split along N dimension
      sizePerThread = {1, ceil<unsigned>(N, numWarpGroups)};
      threadsPerWarp = {32, 1};
      warpsPerCTA = {4, numWarpGroups};
    }
  }
  order = {0, 1};
  auto ctaLayout = getCTALayout(oldType.getEncoding());
  return triton::gpu::BlockedEncodingAttr::get(ctaLayout.getContext(),
                                               sizePerThread, threadsPerWarp,
                                               warpsPerCTA, order, ctaLayout);
}

Attribute getTmemCompatibleLayout(unsigned M, unsigned N,
                                  RankedTensorType oldType, unsigned numWarps) {
  bool prefer16x256 =
      triton::tools::getBoolEnv("TRITON_PREFER_TMEM_16x256_LAYOUT");
  if (prefer16x256) {
    std::optional<LinearLayout> ll =
        getTmemLoadStoreLayout16x256(M, N, oldType, numWarps);
    if (ll) {
      return LinearEncodingAttr::get(oldType.getContext(), *ll);
    }
  }
  return getTmemLoadStoreLayout32x32b(M, N, oldType, numWarps);
}

bool isDistributedLayoutSplitMTmemLoadStore(RankedTensorType tensorType,
                                            MemDescType memType, int numWarps) {
  auto tmemEnc = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
      memType.getEncoding());
  if (!tmemEnc || tmemEnc.getBlockM() != 128)
    return false;
  int M = tmemEnc.getBlockM();
  int N = tmemEnc.getBlockN();
  auto llEncoding = dyn_cast<LinearEncodingAttr>(tensorType.getEncoding());
  if (!llEncoding)
    return false;
  auto CTALayout = getCTALayout(tensorType.getEncoding());
  auto shapePerCTA = mlir::triton::gpu::getShapePerCTA(tensorType);
  if (numWarps != 8)
    return false;
  LinearLayout llLayout =
      getTmemLoadLayoutSplitLongM(M, N, tensorType, numWarps);
  return llEncoding.getLinearLayout() == llLayout;
}

// Verify if the distributed layout can be mapped onto tensor memory.
bool isDistributedLayoutTMemCompatible(Operation *op,
                                       RankedTensorType tensorType,
                                       MemDescType memType) {
  int numWarps = lookupNumWarps(op);
  assert(numWarps % 4 == 0);
  if (isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
          memType.getEncoding())) {
    return tensorType.getEncoding() ==
           triton::gpu::LinearEncodingAttr::get(
               tensorType.getContext(),
               getScaleTMEMStoreLinearLayout(tensorType, numWarps));
  }
  auto attr =
      cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(memType.getEncoding());
  int blockM = attr.getBlockM();
  int blockN = attr.getBlockN();
  if (isDistributedLayoutSplitMTmemLoadStore(tensorType, memType, numWarps))
    return true;

  auto ll16x256 =
      getTmemLoadStoreLayout16x256(blockM, blockN, tensorType, numWarps);
  auto enc =
      cast<triton::gpu::DistributedEncodingTrait>(tensorType.getEncoding());
  if (ll16x256.has_value() &&
      areLayoutsEquivalent(
          tensorType.getShape(),
          LinearEncodingAttr::get(tensorType.getContext(), ll16x256.value()),
          enc))
    return true;
  auto layout = cast<triton::gpu::DistributedEncodingTrait>(
      nvidia_gpu::getTmemLoadStoreLayout32x32b(blockM, blockN, tensorType,
                                               numWarps));
  // TODO: Add support for more layout compatible with tmem load/store. There
  // will only be a discret set of layout possible due to the limiations of
  // tmem_load/store.
  return areLayoutsEquivalent(tensorType.getShape(), layout, enc);
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
