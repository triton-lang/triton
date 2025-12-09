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
createBufferPointersTensor(ImplicitLocOpBuilder &builder, MemType memType,
                           SmallVector<uint32_t> values) {
  int64_t size = values.size();
  assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
  auto tensorType =
      getIntTensorType(builder.getInsertionBlock()->getParent(), {size}, 64);
  auto valuesI32 = llvm::to_vector(llvm::map_range(
      values, [](uint32_t v) { return static_cast<int32_t>(v); }));
  return {ExperimentalBufferPointersOp::create(builder, tensorType, valuesI32,
                                               memType),
          tensorType};
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
             "For now we support"
             " only one function in the module");
      return region;
    }
    region = region->getParentRegion();
  }
  llvm_unreachable("Expected to find enclosing partition or function region");
  return nullptr;
}

void AuxDataMap::populateAndPassToWarpSpecialize(ModuleOp module) {
  SmallVector<SmallVector<uint32_t>, 2> bufValues(numMemTypes);
  SmallVector<uint32_t> barrierValues;
  getBuffersAndBarriers(module, bufValues, barrierValues);

  FuncOp entryPoint = getEntryPoint(module);
  assert(entryPoint);
  Region *entryRegion = &entryPoint.getBody();

  ImplicitLocOpBuilder b(entryPoint.getLoc(), entryPoint);
  b.setInsertionPointToStart(&entryPoint.getBody().front());

  for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
    int iMemType = (int)memType;
    if (bufValues[iMemType].empty()) {
      continue;
    }

    buffers[iMemType].insert(
        entryRegion,
        {createBufferPointersTensor(b, memType, bufValues[iMemType])});
    // Buffer pointers are rematerialized in the warp specialize region,
    // not passed as an argument.
    createInWarpSpecialize(
        entryPoint, buffers[iMemType], [&](ImplicitLocOpBuilder &b) {
          return ValueType{
              createBufferPointersTensor(b, memType, bufValues[iMemType])};
        });
    int numBufs = bufValues[iMemType].size();

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

  if (!barrierValues.empty()) {
    // Barriers allocations are in shared memory
    barriers.insert(entryRegion, {createBufferPointersTensor(
                                     b, MemType::SHARED_MEM, barrierValues)});
    // Barriers allocations are rematerialized in the warp specialize region,
    // not passed as an argument.
    createInWarpSpecialize(entryPoint, barriers, [&](ImplicitLocOpBuilder &b) {
      return ValueType{
          createBufferPointersTensor(b, MemType::SHARED_MEM, barrierValues)};
    });

    int numBarriers = barrierValues.size();
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
      int numBufs = bufValues[iMemType].size();
      int numBarriers = barrierValues.size();
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
    int numBufs = bufValues[(int)MemType::SHARED_MEM].size();
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
    ModuleOp module, SmallVector<SmallVector<uint32_t>, 2> &bufValues,
    SmallVector<uint32_t> &barrierValues) {
  // Collect shared memory buffers allocated in the module
  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  triton::BufferRegionAnalysis *analysis =
      solver->load<triton::BufferRegionAnalysis>();
  if (failed(solver->initializeAndRun(module)))
    return;

  analysis->calculateUsedBufferRegions(module);
  bufValues[(int)MemType::SHARED_MEM] = llvm::to_vector(llvm::map_range(
      analysis->getAllUsedBufferRegions(
          BufferRegionAnalysis::RegionType::SHARED_MEMORY),
      [](const BufferRegion &region) { return region.baseOffset; }));
  bufValues[(int)MemType::TENSOR_MEM] = llvm::to_vector(llvm::map_range(
      analysis->getAllUsedBufferRegions(
          BufferRegionAnalysis::RegionType::TENSOR_MEMORY),
      [](const BufferRegion &region) { return region.baseOffset; }));
  barrierValues = llvm::to_vector(llvm::map_range(
      analysis->getAllUsedBufferRegions(
          BufferRegionAnalysis::RegionType::BARRIER),
      [](const BufferRegion &region) { return region.baseOffset; }));

  if (!barrierValues.empty()) {
    barrierValues.resize(llvm::NextPowerOf2(barrierValues.size() - 1), 0);
  }

  for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
    int iMemType = (int)memType;
    if (bufValues[iMemType].empty()) {
      continue;
    }
    bufValues[iMemType].resize(
        llvm::NextPowerOf2(bufValues[iMemType].size() - 1), 0);
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
