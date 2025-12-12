#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::instrument;
using mlir::triton::BufferRegion;

namespace {

BlockedEncodingAttr getThreadLocalBlockedEncoding(MLIRContext *ctx,
                                                  unsigned int size,
                                                  unsigned int warps) {
  auto cgaLayout = CGAEncodingAttr::getDefault(ctx, /*rank=*/1);
  return BlockedEncodingAttr::get(ctx,
                                  /*sizePerThread=*/{size},
                                  /*threadsPerWarp=*/{32},
                                  /*warpsPerCTA=*/{warps},
                                  /*order=*/{0}, cgaLayout);
}

BlockedEncodingAttr getThreadLocalBlockedEncoding(MLIRContext *ctx,
                                                  unsigned int buffers,
                                                  unsigned int barriers,
                                                  unsigned int warps) {
  auto cgaLayout = CGAEncodingAttr::getDefault(ctx, /*rank=*/2);
  return BlockedEncodingAttr::get(ctx,
                                  /*sizePerThread=*/{buffers, barriers},
                                  /*threadsPerWarp=*/{1, 32},
                                  /*warpsPerCTA=*/{1, warps},
                                  /*order=*/{0, 1}, cgaLayout);
}

RankedTensorType getIntTensorType(Region *region, ArrayRef<int64_t> shape,
                                  unsigned bitWidth) {
  MLIRContext *ctx = region->getContext();
  unsigned int warps = lookupNumWarps(region);
  BlockedEncodingAttr encoding;
  if (shape.size() == 1) {
    encoding = getThreadLocalBlockedEncoding(
        ctx, static_cast<unsigned>(shape[0]), warps);
  } else {
    assert(shape.size() == 2 && "Only 1D and 2D shapes are supported");
    encoding =
        getThreadLocalBlockedEncoding(ctx, static_cast<unsigned>(shape[0]),
                                      static_cast<unsigned>(shape[1]), warps);
  }
  Type elType = IntegerType::get(ctx, bitWidth);
  return RankedTensorType::get(shape, elType, encoding);
}

std::pair<Value, RankedTensorType>
createBufferDescriptorsTensor(ImplicitLocOpBuilder &builder, MemType memType,
                              ArrayRef<BufferRegion> regions) {
  int64_t size = regions.size();
  assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
  auto tensorType =
      getIntTensorType(builder.getInsertionBlock()->getParent(), {size}, 64);
  SmallVector<int32_t> offsets;
  SmallVector<int32_t> lengths;
  offsets.reserve(size);
  lengths.reserve(size);
  for (const auto &region : regions) {
    offsets.push_back(static_cast<int32_t>(region.baseOffset));
    lengths.push_back(static_cast<int32_t>(region.length));
  }
  return {ExperimentalBufferDescriptorsOp::create(builder, tensorType, offsets,
                                                  lengths, memType),
          tensorType};
}

SmallVector<SmallVector<uint8_t>>
createAliasingMatrix(ArrayRef<BufferRegion> regions) {
  SmallVector<SmallVector<uint8_t>> matrix;
  size_t numRegions = regions.size();
  matrix.resize(numRegions);
  for (size_t i = 0; i < numRegions; ++i)
    matrix[i].assign(numRegions, /*Value=*/0);

  for (size_t i = 0; i < numRegions; ++i) {
    uint64_t startI = regions[i].baseOffset;
    uint64_t endI = startI + regions[i].length;
    if (regions[i].length == 0)
      continue;
    // Include self-aliasing
    for (size_t j = i; j < numRegions; ++j) {
      uint64_t startJ = regions[j].baseOffset;
      uint64_t endJ = startJ + regions[j].length;
      if (regions[j].length == 0)
        continue;
      bool alias = (startI < endJ) && (startJ < endI);
      if (alias) {
        matrix[i][j] = 1;
        matrix[j][i] = 1;
      }
    }
  }
  return matrix;
}

Value createInitializedScratchMemory(ImplicitLocOpBuilder &b,
                                     TypedValue<RankedTensorType> tensor) {
  Type elType = tensor.getType().getElementType();
  int elSize = elType.getIntOrFloatBitWidth() / 8;
  int numEls = product(tensor.getType().getShape());
  int64_t sizeInBytes = numEls * elSize;
  Type ptrType = triton::getPointerType(elType);
  auto alloc = GlobalScratchAllocOp::create(b, ptrType, sizeInBytes, elSize);
  createStoreScratchMemory(b, b.getLoc(), alloc, tensor, tensor.getType());
  return alloc;
}

Value createZeroInitStateTensor(ImplicitLocOpBuilder &b, int m, int n,
                                int bitWidth) {
  SmallVector<int64_t> shape = {m};
  if (n > 0) {
    shape.push_back(n);
  }
  auto type =
      getIntTensorType(b.getInsertionBlock()->getParent(), shape, bitWidth);
  TypedValue<RankedTensorType> tensor =
      createConstIntTensor(b, b.getLoc(), 0, type);
  return createInitializedScratchMemory(b, tensor);
}

TypedValue<RankedTensorType>
createAliasMatrixTensor(ImplicitLocOpBuilder &b,
                        ArrayRef<SmallVector<uint8_t>> matrix, Region *region) {
  size_t rows = matrix.size();
  if (rows == 0)
    return {};
  size_t cols = matrix.front().size();
  for (const auto &row : matrix)
    assert(row.size() == cols && "Expected square alias matrix");

  auto type = getIntTensorType(
      region, {static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
      /*bitWidth=*/1);
  SmallVector<APInt> values;
  values.reserve(rows * cols);
  for (const auto &row : matrix)
    for (uint8_t v : row)
      values.emplace_back(/*numBits=*/1, v);

  auto denseAttr = DenseElementsAttr::get(type, values);
  Value constValue = arith::ConstantOp::create(b, b.getLoc(), type, denseAttr);
  return cast<TypedValue<RankedTensorType>>(constValue);
}

bool hasCpAsync(ModuleOp module) {
  bool hasCpAsync = false;
  module.walk([&](Operation *op) {
    if (isa<AsyncCopyGlobalToLocalOp, AsyncCommitGroupOp, AsyncWaitOp>(op)) {
      hasCpAsync = true;
    }
  });
  return hasCpAsync;
}

bool hasWGMMA(ModuleOp module) {
  bool hasWGMMA = false;
  module.walk([&](Operation *op) {
    if (isa<WarpGroupDotOp, WarpGroupDotWaitOp>(op)) {
      hasWGMMA = true;
    }
  });
  return hasWGMMA;
}

bool hasTMAStore(ModuleOp module) {
  bool hasTMAStore = false;
  module.walk([&](Operation *op) {
    if (isa<AsyncTMACopyLocalToGlobalOp, TMAStoreWaitOp>(op)) {
      hasTMAStore = true;
    }
  });
  return hasTMAStore;
}

Value createLockVariable(ImplicitLocOpBuilder &b) {
  Type ptrType = triton::getPointerType(b.getI32Type());
  auto alloc = GlobalScratchAllocOp::create(b, ptrType, 4, 4);
  Value zero = arith::ConstantOp::create(b, b.getLoc(), b.getI32Type(),
                                         b.getI32IntegerAttr(0));
  triton::AtomicRMWOp::create(b, b.getI32Type(), RMWOp::XCHG, alloc, zero,
                              nullptr, MemSemantic::ACQUIRE_RELEASE,
                              MemSyncScope::GPU);
  return alloc;
}

} // namespace

namespace mlir::triton::instrument {

TypedValue<RankedTensorType> createConstIntTensor(OpBuilder &builder,
                                                  Location loc, int64_t val,
                                                  RankedTensorType tensorType,
                                                  bool isSigned /*= false*/) {
  int bitWidth = tensorType.getElementType().getIntOrFloatBitWidth();
  auto denseAttr =
      DenseElementsAttr::get(tensorType, APInt(bitWidth, val, isSigned));
  return cast<TypedValue<RankedTensorType>>(
      arith::ConstantOp::create(builder, loc, tensorType, denseAttr)
          .getResult());
}

DistributedEncodingTrait getSingleDimSliceEncoding(BlockedEncodingAttr encoding,
                                                   int dim) {
  int rank = encoding.getOrder().size();
  MLIRContext *ctx = encoding.getContext();
  assert(dim < rank && "Expected dim to be less than rank");
  DistributedEncodingTrait sliceEncoding = encoding;
  for (int i = 0; i < rank; ++i) {
    if (i != dim) {
      sliceEncoding = SliceEncodingAttr::get(ctx, i, sliceEncoding);
    }
  }
  return sliceEncoding;
}

Value expandOuterSlicedDim(OpBuilder &b, Location loc, Value tensor) {
  auto type = cast<RankedTensorType>(tensor.getType());
  auto sliceEncoding = dyn_cast<SliceEncodingAttr>(type.getEncoding());
  if (sliceEncoding) {
    int dim = sliceEncoding.getDim();
    auto shape = type.getShape();
    auto newShape = SmallVector<int64_t>(shape);
    newShape.insert(newShape.begin() + dim, 1);
    auto newType = RankedTensorType::get(newShape, type.getElementType(),
                                         sliceEncoding.getParent());
    tensor = ExpandDimsOp::create(b, loc, newType, tensor, dim);
  }
  return tensor;
}

static Value expandAllSlicedDims(OpBuilder &b, Location loc, Value tensor) {
  auto type = cast<RankedTensorType>(tensor.getType());
  auto sliceEncoding = dyn_cast<SliceEncodingAttr>(type.getEncoding());
  while (sliceEncoding) {
    tensor = expandOuterSlicedDim(b, loc, tensor);
    type = cast<RankedTensorType>(tensor.getType());
    sliceEncoding = dyn_cast<SliceEncodingAttr>(type.getEncoding());
  }
  return tensor;
}

static Value createPointerTensor(OpBuilder &b, Location loc, Value base,
                                 RankedTensorType tensorType) {
  auto encoding = cast<BlockedEncodingAttr>(tensorType.getEncoding());
  Value ptrTensor = SplatOp::create(
      b, loc,
      RankedTensorType::get(tensorType.getShape(), base.getType(), encoding),
      base);
  auto offsetsType =
      RankedTensorType::get(tensorType.getShape(), b.getI32Type(), encoding);
  SmallVector<int> strides(tensorType.getRank());
  strides[0] = 1;
  for (int i = 1; i < tensorType.getRank(); ++i) {
    strides[i] = strides[i - 1] * tensorType.getShape()[i - 1];
  }
  for (int i = 0; i < tensorType.getRank(); ++i) {
    auto partialEncoding = getSingleDimSliceEncoding(encoding, i);
    auto arangeType = RankedTensorType::get({tensorType.getShape()[i]},
                                            b.getI32Type(), partialEncoding);
    auto arange =
        MakeRangeOp::create(b, loc, arangeType, 0, arangeType.getShape()[0]);
    auto cstStride = createConstIntTensor(b, loc, strides[i], arangeType);
    auto arangeTimesStride =
        arith::MulIOp::create(b, loc, arangeType, arange, cstStride);
    auto expandDims = expandAllSlicedDims(b, loc, arangeTimesStride);
    if (cast<RankedTensorType>(expandDims.getType()).getShape() !=
        tensorType.getShape()) {
      expandDims = BroadcastOp::create(b, loc, offsetsType, expandDims);
    }
    ptrTensor =
        AddPtrOp::create(b, loc, ptrTensor.getType(), ptrTensor, expandDims);
  }
  return ptrTensor;
}

Operation *createStoreScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                    Value tensor, RankedTensorType tensorType) {
  auto ptrTensor = createPointerTensor(b, loc, alloc, tensorType);
  return StoreOp::create(b, loc, ptrTensor, tensor, CacheModifier::NONE,
                         EvictionPolicy::NORMAL);
}

Value createLoadScratchMemory(OpBuilder &b, Location loc, Value alloc,
                              RankedTensorType tensorType) {
  auto ptrTensor = createPointerTensor(b, loc, alloc, tensorType);
  return LoadOp::create(b, loc, ptrTensor, CacheModifier::NONE,
                        EvictionPolicy::NORMAL, false);
}

FuncOp getEntryPoint(ModuleOp module) {
  SmallVector<FuncOp> publicFuncs = llvm::to_vector(llvm::make_filter_range(
      module.getOps<FuncOp>(), [](FuncOp func) { return func.isPublic(); }));
  assert(publicFuncs.size() == 1 && "Expected exactly one public function");
  return publicFuncs.front();
}

Region *AuxDataMap::RegionToValueMap::getEnclosingParitionOrFunctionRegion(
    Operation *op) {
  Region *region = op->getParentRegion();
  while (region) {
    if (auto wsOp = dyn_cast<WarpSpecializeOp>(region->getParentOp())) {
      if (region == &wsOp.getDefaultRegion()) {
        return getEnclosingParitionOrFunctionRegion(wsOp);
      }
      return region;
    }
    if (auto wsOp =
            dyn_cast<WarpSpecializePartitionsOp>(region->getParentOp())) {
      return region;
    }
    if (isa<FuncOp>(region->getParentOp())) {
      ModuleOp module = op->getParentOfType<ModuleOp>();
      assert(getEntryPoint(module) == region->getParentOp() &&
             "Concurrency sanitizer supports only one instrumented "
             "function in the module");
      return region;
    }
    region = region->getParentRegion();
  }
  llvm_unreachable("Expected to find enclosing partition or function region");
  return nullptr;
}

void AuxDataMap::populateAndPassToWarpSpecialize(ModuleOp module) {
  SmallVector<SmallVector<BufferRegion>, numMemTypes> bufRegions(numMemTypes);
  SmallVector<BufferRegion> barrierRegions;
  getBuffersAndBarriers(module, bufRegions, barrierRegions);

  FuncOp entryPoint = getEntryPoint(module);
  assert(entryPoint);
  Region *entryRegion = &entryPoint.getBody();

  ImplicitLocOpBuilder b(entryPoint.getLoc(), entryPoint);
  b.setInsertionPointToStart(&entryPoint.getBody().front());

  for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
    int iMemType = (int)memType;
    if (bufRegions[iMemType].empty()) {
      continue;
    }

    buffers[iMemType].insert(
        entryRegion,
        {createBufferDescriptorsTensor(b, memType, bufRegions[iMemType])});
    // Buffer descriptors are rematerialized in the warp specialize region,
    // not passed as an argument.
    createInWarpSpecialize(
        entryPoint, buffers[iMemType], [&](ImplicitLocOpBuilder &b) {
          return ValueType{
              createBufferDescriptorsTensor(b, memType, bufRegions[iMemType])};
        });
    int numBufs = bufRegions[iMemType].size();

    auto aliasMatrixData = createAliasingMatrix(bufRegions[iMemType]);
    if (!aliasMatrixData.empty()) {
      auto aliasTensor =
          createAliasMatrixTensor(b, aliasMatrixData, entryRegion);
      aliasMatrices[iMemType].insert(entryRegion,
                                     {aliasTensor, aliasTensor.getType()});
      createInWarpSpecialize(
          entryPoint, aliasMatrices[iMemType],
          [aliasMatrixData](ImplicitLocOpBuilder &nestedBuilder) {
            Region *region = nestedBuilder.getInsertionBlock()->getParent();
            auto tensor =
                createAliasMatrixTensor(nestedBuilder, aliasMatrixData, region);
            return ValueType{tensor, tensor.getType()};
          });
    }

    writeVisibility[iMemType].insert(
        entryRegion, {createZeroInitStateTensor(b, numBufs, 0, 64),
                      getIntTensorType(entryRegion, {numBufs}, 64)});
    passToWarpSpecialize(entryPoint, writeVisibility[iMemType].at(entryRegion),
                         writeVisibility[iMemType]);
    readVisibility[iMemType].insert(
        entryRegion,
        {createZeroInitStateTensor(b, numBufs, THREADS_BITMASK_SIZE, 64),
         getIntTensorType(entryRegion, {numBufs, THREADS_BITMASK_SIZE}, 64)});
    passToWarpSpecialize(entryPoint, readVisibility[iMemType].at(entryRegion),
                         readVisibility[iMemType]);
  }

  if (!barrierRegions.empty()) {
    // Barriers allocations are in shared memory
    barriers.insert(entryRegion, {createBufferDescriptorsTensor(
                                     b, MemType::SHARED_MEM, barrierRegions)});
    // Barriers allocations are rematerialized in the warp specialize region,
    // not passed as an argument.
    createInWarpSpecialize(entryPoint, barriers, [&](ImplicitLocOpBuilder &b) {
      return ValueType{createBufferDescriptorsTensor(b, MemType::SHARED_MEM,
                                                     barrierRegions)};
    });

    int numBarriers = barrierRegions.size();
    barrierStates.insert(entryRegion,
                         {createZeroInitStateTensor(b, numBarriers, 0, 32),
                          getIntTensorType(entryRegion, {numBarriers}, 32)});
    passToWarpSpecialize(entryPoint, barrierStates.at(entryRegion),
                         barrierStates);

    // Deadlock detection aux data: waiting (i32[K]) storing waiting flag and
    // phase bits per thread (two bits per thread).
    waiting.insert(entryRegion,
                   {createZeroInitStateTensor(b, numBarriers, 0, 32),
                    getIntTensorType(entryRegion, {numBarriers}, 32)});
    passToWarpSpecialize(entryPoint, waiting.at(entryRegion), waiting);

    for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
      int iMemType = (int)memType;
      // Create state tensors:
      int numBufs = bufRegions[iMemType].size();
      int numBarriers = barrierRegions.size();
      if (numBufs > 0) {
        writeTracking[iMemType].insert(
            entryRegion,
            {createZeroInitStateTensor(b, numBufs, numBarriers, 8),
             getIntTensorType(entryRegion, {numBufs, numBarriers}, 8)});
        passToWarpSpecialize(entryPoint,
                             writeTracking[iMemType].at(entryRegion),
                             writeTracking[iMemType]);
        readTracking[iMemType].insert(
            entryRegion,
            {createZeroInitStateTensor(b, numBufs, numBarriers, 64),
             getIntTensorType(entryRegion, {numBufs, numBarriers}, 64)});
        passToWarpSpecialize(entryPoint, readTracking[iMemType].at(entryRegion),
                             readTracking[iMemType]);
      }
    }
  }

  // Create lock variable allocation
  Value lockVal = createLockVariable(b);
  lock.insert(entryRegion, {lockVal, lockVal.getType()});
  passToWarpSpecialize(entryPoint, lock.at(entryRegion), lock);

  auto createCommitTensor = [&](CommitKind::Kind commitKind) {
    int numBufs = bufRegions[(int)MemType::SHARED_MEM].size();
    if (numBufs == 0)
      return;
    // NUM_THREADS instead of THREADS_BITMASK_SIZE as commit-count tracking
    // operates on base threads.
    commits[commitKind].insert(
        entryRegion,
        {createZeroInitStateTensor(b, numBufs, NUM_THREADS, 8),
         getIntTensorType(entryRegion, {numBufs, NUM_THREADS}, 8)});
    passToWarpSpecialize(entryPoint, commits[commitKind].at(entryRegion),
                         commits[commitKind]);
  };

  // Create write commits tensor for cp-async
  if (hasCpAsync(module)) {
    createCommitTensor(CommitKind::AsyncCp);
  }

  // Create reads commits tensor for wgmma
  if (hasWGMMA(module)) {
    createCommitTensor(CommitKind::Wgmma);
  }

  if (hasTMAStore(module)) {
    createCommitTensor(CommitKind::TmaStore);
  }
}

void AuxDataMap::getBuffersAndBarriers(
    ModuleOp module, SmallVector<SmallVector<BufferRegion>, 2> &bufRegions,
    SmallVector<BufferRegion> &barrierRegions) {
  // Collect shared memory buffers allocated in the module
  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  triton::BufferRegionAnalysis *analysis =
      solver->load<triton::BufferRegionAnalysis>();
  if (failed(solver->initializeAndRun(module)))
    return;

  analysis->calculateUsedBufferRegions(module);
  bufRegions[(int)MemType::SHARED_MEM] = analysis->getAllUsedBufferRegions(
      BufferRegionAnalysis::RegionType::SHARED_MEMORY);
  bufRegions[(int)MemType::TENSOR_MEM] = analysis->getAllUsedBufferRegions(
      BufferRegionAnalysis::RegionType::TENSOR_MEMORY);
  barrierRegions = analysis->getAllUsedBufferRegions(
      BufferRegionAnalysis::RegionType::BARRIER);

  if (!barrierRegions.empty()) {
    barrierRegions.resize(llvm::NextPowerOf2(barrierRegions.size() - 1),
                          BufferRegion{0, 0});
  }

  for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
    int iMemType = (int)memType;
    if (bufRegions[iMemType].empty()) {
      continue;
    }
    bufRegions[iMemType].resize(
        llvm::NextPowerOf2(bufRegions[iMemType].size() - 1),
        BufferRegion{0, 0});
  }
}

void AuxDataMap::passToWarpSpecialize(FuncOp func, ValueType valueType,
                                      RegionToValueMap &map) {
  func.walk([&](WarpSpecializeOp op) {
    op->insertOperands(op.getNumOperands(), {valueType.value});
    for (Region *region : op.getPartitionRegions()) {
      // Pass the value as a pointer type (instead of the type of underlying
      // memory)
      region->addArgument(valueType.value.getType(), op.getLoc());
      Type newType = valueType.type;
      if (auto tensorType = dyn_cast<RankedTensorType>(newType)) {
        // If this is a tensor, make sure the layout matches the region's warp
        // count
        newType = getIntTensorType(
            region, tensorType.getShape(),
            tensorType.getElementType().getIntOrFloatBitWidth());
      }
      map.insert(region,
                 ValueType{region->getArgument(region->getNumArguments() - 1),
                           newType});
    }
  });
}

void AuxDataMap::createInWarpSpecialize(
    FuncOp func, RegionToValueMap &map,
    std::function<ValueType(ImplicitLocOpBuilder &)> createFn) {
  func.walk([&](WarpSpecializeOp op) {
    for (Region *region : op.getPartitionRegions()) {
      ImplicitLocOpBuilder b(region->getLoc(), region);
      b.setInsertionPointToStart(&region->getBlocks().front());
      map.insert(region, createFn(b));
    }
  });
}

} // namespace mlir::triton::instrument
