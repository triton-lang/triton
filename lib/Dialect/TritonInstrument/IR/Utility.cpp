#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
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
  auto ctaLayout = CTALayoutAttr::getDefault(ctx, /*rank=*/1);
  return BlockedEncodingAttr::get(ctx,
                                  /*sizePerThread=*/{size},
                                  /*threadsPerWarp=*/{32},
                                  /*warpsPerCTA=*/{warps},
                                  /*order=*/{0}, ctaLayout);
}

BlockedEncodingAttr getThreadLocalBlockedEncoding(MLIRContext *ctx,
                                                  unsigned int buffers,
                                                  unsigned int barriers,
                                                  unsigned int warps) {
  auto ctaLayout = CTALayoutAttr::getDefault(ctx, /*rank=*/2);
  return BlockedEncodingAttr::get(ctx,
                                  /*sizePerThread=*/{buffers, barriers},
                                  /*threadsPerWarp=*/{1, 32},
                                  /*warpsPerCTA=*/{1, warps},
                                  /*order=*/{0, 1}, ctaLayout);
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

Value createBufferPointersTensor(ImplicitLocOpBuilder &builder, MemType memType,
                                 SmallVector<int32_t> values) {
  int64_t size = values.size();
  assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
  auto tensorType =
      getIntTensorType(builder.getInsertionBlock()->getParent(), {size}, 64);
  return builder.create<ExperimentalBufferPointersOp>(tensorType, values,
                                                      memType);
}

Value createInitializedScratchMemory(ImplicitLocOpBuilder &b,
                                     TypedValue<RankedTensorType> tensor) {
  Type elType = tensor.getType().getElementType();
  int elSize = elType.getIntOrFloatBitWidth() / 8;
  int numEls = product(tensor.getType().getShape());
  int64_t sizeInBytes = numEls * elSize;
  Type ptrType = triton::getPointerType(elType);
  auto alloc = b.create<GlobalScratchAllocOp>(ptrType, sizeInBytes, elSize);
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

bool canAllocBeInstrumented(Operation *op) {
  if (llvm::any_of(op->getUsers(),
                   [](Operation *user) { return isa<CallOp>(user); })) {
    op->emitWarning("Allocation is used in a function call, cannot instrument");
    return false;
  }
  if (llvm::all_of(op->getUsers(), [](Operation *user) {
        return !isa<MemDescIndexOp>(user);
      })) {
    return true;
  }
  if (llvm::all_of(op->getUsers(), [](Operation *user) {
        return isa<MemDescIndexOp>(user) || isa<LocalDeallocOp>(user) ||
               isa<WarpSpecializeOp>(user);
      })) {
    return true;
  }
  op->emitWarning(
      "Allocation is used in an inconsistent way, cannot instrument");
  return false;
}

// Interpret local_allocs that are used in ttg.memdesc_index as multibuffered
bool isMultiBuffered(Value v) {
  for (auto &use : v.getUses()) {
    if (isa<MemDescIndexOp>(use.getOwner())) {
      return true;
    }
    if (auto wsOp = dyn_cast<WarpSpecializeOp>(use.getOwner())) {
      int opNumber = use.getOperandNumber();
      for (Region *region : wsOp.getPartitionRegions()) {
        if (isMultiBuffered(region->getArguments()[opNumber])) {
          return true;
        }
      }
    }
  }
  return false;
}

uint64_t getAllocationOffset(LocalAllocOp op) {
  auto offsetAttr = op->getAttr("allocation.offset");
  if (!offsetAttr) {
    llvm::report_fatal_error(
        "ConcurrencySanitizer should run after AllocateSharedMemory pass.");
  }
  return cast<IntegerAttr>(offsetAttr).getInt();
}

uint64_t getAllocationOffset(TMEMAllocOp op) {
  auto colOffsetAttr = op->getAttr("tensor_memory_col_offset");
  auto rowOffsetAttr = op->getAttr("tensor_memory_row_offset");
  if (!colOffsetAttr || !rowOffsetAttr) {
    llvm::report_fatal_error(
        "ConcurrencySanitizer should run after AllocateSharedMemory and "
        "TensorMemoryAllocation pass.");
  }
  int colOffset = cast<IntegerAttr>(colOffsetAttr).getInt();
  int rowOffset = cast<IntegerAttr>(rowOffsetAttr).getInt();
  return colOffset | (rowOffset << 16);
}

bool isBarrier(triton::gpu::LocalAllocOp op) {
  // Is there InitBarrierOp in the forward slice of the op?
  bool foundInitBarrier = false;
  SetVector<Operation *> forwardSlice;
  ForwardSliceOptions options;
  options.filter = [&](Operation *op) {
    if (isa<InitBarrierOp>(op)) {
      foundInitBarrier = true;
      return false;
    }
    return true;
  };
  getForwardSlice(op.getOperation(), &forwardSlice, options);
  return foundInitBarrier;
}

unsigned getNumBuffers(Operation *op) {
  MemDescType ty = cast<MemDescType>(op->getResultTypes().front());
  return ty.getShape()[0];
}

unsigned getSubBufferSize(LocalAllocOp op) {
  MemDescType ty = op.getType();
  unsigned elSize = ty.getElementType().getIntOrFloatBitWidth() / 8;
  return product(ty.getShape().drop_front()) * elSize;
}

unsigned getSubBufferSize(TMEMAllocOp op) {
  int numCols = getTmemAllocSizes(op.getType()).numCols;
  int numSubBuffers = getNumBuffers(op);
  return numCols / numSubBuffers;
}

Value createLockVariable(ImplicitLocOpBuilder &b) {
  Type ptrType = triton::getPointerType(b.getI32Type());
  auto alloc = b.create<GlobalScratchAllocOp>(ptrType, 4, 4);
  Value zero = b.create<arith::ConstantOp>(b.getLoc(), b.getI32Type(),
                                           b.getI32IntegerAttr(0));
  b.create<triton::AtomicRMWOp>(b.getI32Type(), RMWOp::XCHG, alloc, zero,
                                nullptr, MemSemantic::ACQUIRE_RELEASE,
                                MemSyncScope::GPU);
  return alloc;
}

} // namespace

namespace mlir::triton::instrument {

TypedValue<RankedTensorType> createConstIntTensor(OpBuilder &builder,
                                                  Location loc, int64_t val,
                                                  RankedTensorType tensorType) {
  auto denseAttr = DenseElementsAttr::get(
      tensorType, APInt(tensorType.getElementType().getIntOrFloatBitWidth(),
                        val, /*isSigned=*/true));
  return cast<TypedValue<RankedTensorType>>(
      builder.create<arith::ConstantOp>(loc, tensorType, denseAttr)
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
    tensor = b.create<ExpandDimsOp>(loc, newType, tensor, dim);
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
  Value ptrTensor = b.create<SplatOp>(
      loc,
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
        b.create<MakeRangeOp>(loc, arangeType, 0, arangeType.getShape()[0]);
    auto cstStride = createConstIntTensor(b, loc, strides[i], arangeType);
    auto arangeTimesStride =
        b.create<arith::MulIOp>(loc, arangeType, arange, cstStride);
    auto expandDims = expandAllSlicedDims(b, loc, arangeTimesStride);
    if (cast<RankedTensorType>(expandDims.getType()).getShape() !=
        tensorType.getShape()) {
      expandDims = b.create<BroadcastOp>(loc, offsetsType, expandDims);
    }
    ptrTensor =
        b.create<AddPtrOp>(loc, ptrTensor.getType(), ptrTensor, expandDims);
  }
  return ptrTensor;
}

Operation *createStoreScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                    Value tensor, RankedTensorType tensorType) {
  auto ptrTensor = createPointerTensor(b, loc, alloc, tensorType);
  return b.create<StoreOp>(loc, ptrTensor, tensor, CacheModifier::NONE,
                           EvictionPolicy::NORMAL);
}

Operation *createLoadScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                   RankedTensorType tensorType) {
  auto ptrTensor = createPointerTensor(b, loc, alloc, tensorType);
  return b.create<LoadOp>(loc, ptrTensor, CacheModifier::NONE,
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
  SmallVector<SmallVector<int32_t>, 2> bufValues(numMemTypes);
  SmallVector<int32_t> barrierValues;
  getBuffersAndBarriers(module, bufValues, barrierValues);

  if (bufValues[(int)MemType::SHARED_MEM].empty() &&
      bufValues[(int)MemType::TENSOR_MEM].empty()) {
    return;
  }

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
    buffers[iMemType][entryRegion] = {
        createBufferPointersTensor(b, memType, bufValues[iMemType])};
    createInWarpSpecialize(
        entryPoint, buffers[iMemType], [&](ImplicitLocOpBuilder &b) {
          return AuxDataMap::RegionToValueMap::ValueType{
              createBufferPointersTensor(b, memType, bufValues[iMemType])};
        });
    int numBufs = bufValues[iMemType].size();

    writeVisibility[iMemType][entryRegion] = {
        createZeroInitStateTensor(b, numBufs, 0, 64),
        getIntTensorType(entryRegion, {numBufs}, 64)};
    passToWarpSpecialize(entryPoint, writeVisibility[iMemType][entryRegion],
                         writeVisibility[iMemType]);
    readVisibility[iMemType][entryRegion] = {
        createZeroInitStateTensor(b, numBufs, THREADS_BITMASK_SIZE, 64),
        getIntTensorType(entryRegion, {numBufs, THREADS_BITMASK_SIZE}, 64)};
    passToWarpSpecialize(entryPoint, readVisibility[iMemType][entryRegion],
                         readVisibility[iMemType]);
  }

  if (!barrierValues.empty()) {
    // Barriers allocations are in shared memory
    barriers[entryRegion] = {
        createBufferPointersTensor(b, MemType::SHARED_MEM, barrierValues)};
    createInWarpSpecialize(entryPoint, barriers, [&](ImplicitLocOpBuilder &b) {
      return AuxDataMap::RegionToValueMap::ValueType{
          createBufferPointersTensor(b, MemType::SHARED_MEM, barrierValues)};
    });

    for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
      int iMemType = (int)memType;
      // Create state tensors:
      int numBufs = bufValues[iMemType].size();
      int numBarriers = barrierValues.size();
      if (numBufs > 0) {
        writeTracking[iMemType][entryRegion] = {
            createZeroInitStateTensor(b, numBufs, numBarriers, 8),
            getIntTensorType(entryRegion, {numBufs, numBarriers}, 8)};
        passToWarpSpecialize(entryPoint, writeTracking[iMemType][entryRegion],
                             writeTracking[iMemType]);
        readTracking[iMemType][entryRegion] = {
            createZeroInitStateTensor(b, numBufs, numBarriers, 64),
            getIntTensorType(entryRegion, {numBufs, numBarriers}, 64)};
        passToWarpSpecialize(entryPoint, readTracking[iMemType][entryRegion],
                             readTracking[iMemType]);
      }
    }
  }

  // Create lock variable allocation
  Value lockVal = createLockVariable(b);
  lock[entryRegion] = {lockVal};
  passToWarpSpecialize(entryPoint, lock[entryRegion], lock);

  // Create write commits tensor for cp-async
  if (hasCpAsync(module)) {
    int iMemType = (int)MemType::SHARED_MEM;
    int numBufs = bufValues[iMemType].size();
    assert(numBufs > 0);
    // NUM_THREADS instead of THREADS_BITMASK_SIZE as cp_async can't work on the
    // helper threads of TMA and TC
    asyncCpCommits[entryRegion] = {
        createZeroInitStateTensor(b, numBufs, NUM_THREADS, 8),
        getIntTensorType(entryRegion, {numBufs, NUM_THREADS}, 8)};
    passToWarpSpecialize(entryPoint, asyncCpCommits[entryRegion],
                         asyncCpCommits);
  }

  // Create reads commits tensor for wgmma
  if (hasWGMMA(module)) {
    int iMemType = (int)MemType::SHARED_MEM;
    int numBufs = bufValues[iMemType].size();
    assert(numBufs > 0);
    // NUM_THREADS instead of THREADS_BITMASK_SIZE as wgmma can't work on the
    // helper threads of TMA and TC
    wgmmaCommits[entryRegion] = {
        createZeroInitStateTensor(b, numBufs, NUM_THREADS, 8),
        getIntTensorType(entryRegion, {numBufs, NUM_THREADS}, 8)};
    passToWarpSpecialize(entryPoint, wgmmaCommits[entryRegion], wgmmaCommits);
  }
}

void AuxDataMap::getBuffersAndBarriers(
    ModuleOp module, SmallVector<SmallVector<int32_t>, 2> &bufValues,
    SmallVector<int32_t> &barrierValues) {
  // Collect shared memory buffers allocated in the module
  llvm::SmallVector<llvm::SetVector<int32_t>> bufSets(numMemTypes);
  llvm::SetVector<int32_t> barrierSet;
  module.walk([&](LocalAllocOp op) {
    if (!canAllocBeInstrumented(op)) {
      return WalkResult::advance();
    }
    int32_t baseOffset = getAllocationOffset(op);
    auto &setToAdd =
        isBarrier(op) ? barrierSet : bufSets[(int)MemType::SHARED_MEM];
    setToAdd.insert(baseOffset);
    if (isMultiBuffered(op)) {
      unsigned numBuffers = getNumBuffers(op);
      assert(numBuffers > 0 && "Expected at least one buffer");
      unsigned subBufferSize = getSubBufferSize(op);
      for (unsigned i = 1; i < numBuffers; ++i) {
        setToAdd.insert(baseOffset + i * subBufferSize);
      }
    }
    return WalkResult::advance();
  });

  module.walk([&](TMEMAllocOp op) {
    if (!canAllocBeInstrumented(op)) {
      return WalkResult::advance();
    }
    int32_t baseOffset = getAllocationOffset(op);
    bufSets[(int)MemType::TENSOR_MEM].insert(baseOffset);
    if (isMultiBuffered(op)) {
      unsigned numBuffers = getNumBuffers(op);
      assert(numBuffers > 0 && "Expected at least one buffer");
      unsigned subBufferSize = getSubBufferSize(op);
      for (unsigned i = 1; i < numBuffers; ++i) {
        bufSets[(int)MemType::TENSOR_MEM].insert(baseOffset +
                                                 i * subBufferSize);
      }
    }
    return WalkResult::advance();
  });

  barrierValues = llvm::to_vector(barrierSet);
  if (!barrierValues.empty()) {
    barrierValues.resize(llvm::NextPowerOf2(barrierValues.size() - 1), 0);
  }

  for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
    int iMemType = (int)memType;
    bufValues[iMemType] = llvm::to_vector(bufSets[iMemType]);
    if (bufValues[iMemType].empty()) {
      continue;
    }
    bufValues[iMemType].resize(
        llvm::NextPowerOf2(bufValues[iMemType].size() - 1), 0);
  }
}

void AuxDataMap::passToWarpSpecialize(
    FuncOp func, AuxDataMap::RegionToValueMap::ValueType valueType,
    RegionToValueMap &map) {
  func.walk([&](WarpSpecializeOp op) {
    op->insertOperands(op.getNumOperands(), {valueType.value});
    for (Region *region : op.getPartitionRegions()) {
      // Pass the value as a pointer type (instead of the type of undelying
      // memory)
      region->addArgument(valueType.value.getType(), op.getLoc());
      Type newType = valueType.type;
      if (newType) {
        auto tensorType = cast<RankedTensorType>(newType);
        newType = getIntTensorType(
            region, tensorType.getShape(),
            tensorType.getElementType().getIntOrFloatBitWidth());
      }
      map[region] = AuxDataMap::RegionToValueMap::ValueType{
          region->getArgument(region->getNumArguments() - 1), newType};
    }
  });
}

void AuxDataMap::createInWarpSpecialize(
    FuncOp func, RegionToValueMap &map,
    std::function<RegionToValueMap::ValueType(ImplicitLocOpBuilder &)>
        createFn) {
  func.walk([&](WarpSpecializeOp op) {
    for (Region *region : op.getPartitionRegions()) {
      ImplicitLocOpBuilder b(region->getLoc(), region);
      b.setInsertionPointToStart(&region->getBlocks().front());
      map[region] = createFn(b);
    }
  });
}

} // namespace mlir::triton::instrument
