#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/FunctionBuilder.h"
#include "triton/Dialect/TritonInstrument/Transforms/ConSanTargetHooks.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::instrument;
using mlir::triton::BufferRegion;

namespace {

constexpr unsigned kMaxVectorLengthBits = 128;

DistributedEncodingTrait getWarpLocalEncoding(MLIRContext *ctx,
                                              ArrayRef<int64_t> shape,
                                              unsigned warps, unsigned numCTAs,
                                              unsigned bitwidth) {
  assert(!shape.empty() && "Expected non-empty shape");
  auto dims = standardOutDimNames(ctx, shape.size());
  auto kBlock = StringAttr::get(ctx, "block");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kRegister = StringAttr::get(ctx, "register");

  // A warp-local layout ensures each warp has a copy of the whole tensor, so
  // reductions, layout conversions, etc. don't require shared memory. Attempt
  // to pick a decent coalesced layout, assuming the inner dimension is
  // contiguous and the tensor is 16-byte aligned. However, pick the widest
  // vector length to reduce the number of instructions, speeding up
  // compilation.
  // unsigned vecLen = kMaxVectorLengthBits / bitwidth;
  unsigned vecLen = 1;

  // Broadcast along blocks and warps. Use the innermost dimension for the
  // lane/register mapping and keep the outer dimensions replicated.
  auto lastDim = dims.back();
  auto repOrder = llvm::to_vector(llvm::seq<unsigned>(0, shape.size()));
  auto trivialShape = SmallVector<unsigned>(shape.size(), 1);
  auto llReg = LinearLayout::identity1D(1, kRegister, lastDim);
  auto llLane = LinearLayout::identity1D(32, kLane, lastDim);
  auto llWarp = LinearLayout::zeros1D(warps, kWarp, lastDim);
  // ConSan's multi-CTA state is replicated in every CTA. The leading logical
  // CTA dimension is therefore broadcast across blocks instead of split.
  auto llBlock = LinearLayout::zeros1D(numCTAs, kBlock, dims.front());
  LinearLayout ll = identityStandardND(kRegister, trivialShape, repOrder) *
                    llReg * llLane * llWarp * llBlock;
  SmallVector<int64_t> layoutShape(shape.size(), 1);
  layoutShape.back() = ll.getTotalOutDimSize();
  ll = ll.reshapeOuts(standardOutDimPairs(ctx, layoutShape));

  llvm::SmallDenseMap<StringAttr, int64_t> bounds;
  for (auto [dim, size] : llvm::zip_equal(dims, shape))
    bounds.try_emplace(dim, size);
  ll = ensureLayoutNotLargerThan(ll, bounds);
  ll = ensureLayoutNotSmallerThan(ll, bounds);

  return LinearEncodingAttr::get(ctx, ll);
}

std::pair<Value, RankedTensorType>
createBufferDescriptorsTensor(ImplicitLocOpBuilder &builder, MemType memType,
                              ArrayRef<BufferRegion> regions) {
  Region *region = builder.getInsertionBlock()->getParent();
  int64_t size = regions.size();
  int64_t numCTAs = lookupNumCTAs(region->getParentOp());
  assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
  auto tensorType = getIntTensorType(region, {numCTAs, size}, 64);
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

bool hasCrossBufferAliasing(ArrayRef<BufferRegion> regions) {
  size_t numRegions = regions.size();
  for (size_t i = 0; i < numRegions; ++i) {
    if (regions[i].length == 0)
      continue;
    uint64_t startI = regions[i].baseOffset;
    uint64_t endI = startI + regions[i].length;
    for (size_t j = i + 1; j < numRegions; ++j) {
      if (regions[j].length == 0)
        continue;
      uint64_t startJ = regions[j].baseOffset;
      uint64_t endJ = startJ + regions[j].length;
      if ((startI < endJ) && (startJ < endI)) {
        return true;
      }
    }
  }
  return false;
}

Value createZeroInitStateTensor(ImplicitLocOpBuilder &b,
                                ArrayRef<int64_t> shape, int bitWidth,
                                FunctionBuilder &funcBuilder) {
  auto type =
      getIntTensorType(b.getInsertionBlock()->getParent(), shape, bitWidth);
  Type elType = type.getElementType();
  int elSize = elType.getIntOrFloatBitWidth() / 8;
  int numEls = product(type.getShape());
  int64_t sizeInBytes = numEls * elSize;
  Type ptrType = triton::getPointerType(elType);
  // Allocate scratch buffers with 16-byte alignment so global loads and stores
  // can be vectorized if possible.
  auto alloc = createThirdPartyScratchAlloc(b, b.getLoc(), ptrType, sizeInBytes,
                                            /*alignment=*/16,
                                            /*sharedClusterState=*/true);
  Value cstZero = arith::ConstantIntOp::create(b, 0, bitWidth);
  funcBuilder.createFillGlobalTensorCall(b, alloc, type, cstZero);
  return alloc;
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

  int64_t numCTAs = lookupNumCTAs(region->getParentOp());
  auto type = getIntTensorType(
      region, {numCTAs, static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
      /*bitWidth=*/1);
  auto sliceType = RankedTensorType::get(
      {static_cast<int64_t>(rows), static_cast<int64_t>(cols)}, b.getI1Type(),
      SliceEncodingAttr::get(
          b.getContext(), /*dim=*/0,
          cast<DistributedEncodingTrait>(type.getEncoding())));
  SmallVector<APInt> values;
  values.reserve(rows * cols);
  for (const auto &row : matrix)
    for (uint8_t v : row)
      values.emplace_back(/*numBits=*/1, v);

  auto denseAttr = DenseElementsAttr::get(sliceType, values);
  Value constValue =
      arith::ConstantOp::create(b, b.getLoc(), sliceType, denseAttr);
  constValue = expandOuterSlicedDim(b, b.getLoc(), constValue);
  constValue = BroadcastOp::create(b, b.getLoc(), type, constValue);
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

Value createLockVariable(ImplicitLocOpBuilder &b) {
  Type ptrType = triton::getPointerType(b.getI32Type());
  auto alloc = createThirdPartyScratchAlloc(b, b.getLoc(), ptrType, 4, 4,
                                            /*sharedClusterState=*/true);
  return alloc;
}

} // namespace

namespace mlir::triton::instrument {

static Value createCurrentCTAMask(OpBuilder &b, Location loc,
                                  RankedTensorType tensorType);

uint32_t getMemDescLength(Value buf) {
  auto memDescType = cast<MemDescType>(buf.getType());
  if (isa<SharedEncodingTrait>(memDescType.getEncoding())) {
    unsigned elSize = memDescType.getElementType().getIntOrFloatBitWidth() / 8;
    return static_cast<uint32_t>(product(getShapePerCTA(memDescType)) * elSize);
  }
  if (isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace())) {
    return getTmemAllocSizes(memDescType).numCols;
  }
  llvm_unreachable("Unsupported memory space for memdesc");
}

gpu::GlobalScratchAllocOp
createThirdPartyScratchAlloc(OpBuilder &b, Location loc, Type ptrType,
                             int64_t sizeInBytes, int64_t alignment,
                             bool sharedClusterState) {
  return gpu::GlobalScratchAllocOp::create(
      b, loc, ptrType, sizeInBytes, alignment, b.getUnitAttr(),
      sharedClusterState ? b.getUnitAttr() : UnitAttr());
}

void createAssertInThread(ImplicitLocOpBuilder &b, Value condition,
                          StringRef message) {
  if (isa<RankedTensorType>(condition.getType())) {
    auto conditionTy = cast<RankedTensorType>(condition.getType());
    if (conditionTy.getRank() > 0 && conditionTy.getShape()[0] > 1) {
      Value currentCTAMask = createCurrentCTAMask(b, b.getLoc(), conditionTy);
      Value trueTensor = createConstIntTensor(b, b.getLoc(), 1, conditionTy);
      condition = arith::SelectOp::create(b, b.getLoc(), currentCTAMask,
                                          condition, trueTensor);
    }
    triton::AssertOp::create(b, condition, message);
    return;
  }
  ExperimentalAssertUniformOp::create(b, condition, message);
}

RankedTensorType getIntTensorType(Region *region, ArrayRef<int64_t> shape,
                                  unsigned bitWidth) {
  MLIRContext *ctx = region->getContext();
  unsigned int warps = lookupNumWarps(region);
  unsigned int numCTAs = lookupNumCTAs(region->getParentOp());
  DistributedEncodingTrait encoding =
      getWarpLocalEncoding(ctx, shape, warps, numCTAs, bitWidth);
  Type elType = IntegerType::get(ctx, bitWidth);
  return RankedTensorType::get(shape, elType, encoding);
}

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

DistributedEncodingTrait
getSingleDimSliceEncoding(DistributedEncodingTrait encoding, int dim) {
  int rank = encoding.getRepOrder().size();
  MLIRContext *ctx = encoding.getContext();
  assert(dim < rank && "Expected dim to be less than rank");
  DistributedEncodingTrait sliceEncoding = encoding;
  for (int i = rank - 1; i >= 0; --i) {
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
  auto encoding = cast<DistributedEncodingTrait>(tensorType.getEncoding());
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

static Value createCurrentCTAMask(OpBuilder &b, Location loc,
                                  RankedTensorType tensorType) {
  assert(tensorType.getRank() > 0 && "expected ranked tensor");
  auto encoding = cast<DistributedEncodingTrait>(tensorType.getEncoding());
  auto sliceEncoding = getSingleDimSliceEncoding(encoding, /*dim=*/0);
  auto indexType = RankedTensorType::get({tensorType.getShape()[0]},
                                         b.getI32Type(), sliceEncoding);
  Value range = MakeRangeOp::create(b, loc, indexType, /*start=*/0,
                                    tensorType.getShape()[0]);
  Value ctaId = ExperimentalClusterCTAIdOp::create(b, loc);
  Value ctaIdTensor = SplatOp::create(b, loc, indexType, ctaId);
  Value mask1D = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, range,
                                       ctaIdTensor);
  auto maskType =
      RankedTensorType::get(tensorType.getShape(), b.getI1Type(), encoding);
  Value mask = expandAllSlicedDims(b, loc, mask1D);
  if (cast<RankedTensorType>(mask.getType()).getShape() !=
      tensorType.getShape())
    mask = BroadcastOp::create(b, loc, maskType, mask);
  return mask;
}

Operation *createStoreScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                    Value tensor, RankedTensorType tensorType,
                                    bool currentCTAOnly) {
  if (currentCTAOnly) {
    assert(tensorType.getRank() >= 2 &&
           "expected currentCTAOnly tensor to have a leading CTA dimension");
    int64_t numCTAs = lookupNumCTAs(b);
    assert(tensorType.getShape()[0] == numCTAs &&
           "expected leading dimension to match numCTAs");
    if (numCTAs > 1) {
      Value oldTensor = createLoadScratchMemory(b, loc, alloc, tensorType);
      Value currentCTAMask = createCurrentCTAMask(b, loc, tensorType);
      tensor =
          arith::SelectOp::create(b, loc, currentCTAMask, tensor, oldTensor);
    }
  }
  auto ptrTensor = createPointerTensor(b, loc, alloc, tensorType);
  return StoreOp::create(b, loc, ptrTensor, tensor, Value(),
                         CacheModifier::NONE, EvictionPolicy::NORMAL,
                         /*ignore_cta=*/true);
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

void AuxDataMap::populateAndPassToWarpSpecialize(
    ModuleOp module, FunctionBuilder &fb, const ConSanTargetHooks *hooks) {
  SmallVector<SmallVector<BufferRegion>, numMemTypes> bufRegions(numMemTypes);
  SmallVector<BufferRegion> barrierRegions;
  getBuffersAndBarriers(module, bufRegions, barrierRegions);
  int numCTAs = lookupNumCTAs(module);

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

    hasNonTrivialAliasing[iMemType] =
        hasCrossBufferAliasing(bufRegions[iMemType]);
    if (hasNonTrivialAliasing[iMemType]) {
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
              auto tensor = createAliasMatrixTensor(nestedBuilder,
                                                    aliasMatrixData, region);
              return ValueType{tensor, tensor.getType()};
            });
      }
    }

    writeVisibility[iMemType].insert(
        entryRegion, {createZeroInitStateTensor(b, {numCTAs, numBufs}, 64, fb),
                      getIntTensorType(entryRegion, {numCTAs, numBufs}, 64)});
    passToWarpSpecialize(entryPoint, writeVisibility[iMemType].at(entryRegion),
                         writeVisibility[iMemType]);
    readVisibility[iMemType].insert(
        entryRegion,
        {createZeroInitStateTensor(b, {numCTAs, numBufs, THREADS_BITMASK_SIZE},
                                   64, fb),
         getIntTensorType(entryRegion, {numCTAs, numBufs, THREADS_BITMASK_SIZE},
                          64)});
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
    barrierStates.insert(
        entryRegion,
        {createZeroInitStateTensor(b, {numCTAs, numBarriers}, 32, fb),
         getIntTensorType(entryRegion, {numCTAs, numBarriers}, 32)});
    passToWarpSpecialize(entryPoint, barrierStates.at(entryRegion),
                         barrierStates);

    // Deadlock detection aux data over [cta, barrier]: waiting
    // stores waiting flag and phase bits per thread (two bits per thread).
    waiting.insert(
        entryRegion,
        {createZeroInitStateTensor(b, {numCTAs, numBarriers}, 32, fb),
         getIntTensorType(entryRegion, {numCTAs, numBarriers}, 32)});
    passToWarpSpecialize(entryPoint, waiting.at(entryRegion), waiting);

    for (MemType memType : {MemType::SHARED_MEM, MemType::TENSOR_MEM}) {
      int iMemType = (int)memType;
      // Create state tensors:
      int numBufs = bufRegions[iMemType].size();
      if (numBufs > 0) {
        writeTracking[iMemType].insert(
            entryRegion,
            {createZeroInitStateTensor(b, {numCTAs, numBufs, numBarriers}, 8,
                                       fb),
             getIntTensorType(entryRegion, {numCTAs, numBufs, numBarriers},
                              8)});
        passToWarpSpecialize(entryPoint,
                             writeTracking[iMemType].at(entryRegion),
                             writeTracking[iMemType]);
        readTracking[iMemType].insert(
            entryRegion,
            {createZeroInitStateTensor(b, {numCTAs, numBufs, numBarriers}, 64,
                                       fb),
             getIntTensorType(entryRegion, {numCTAs, numBufs, numBarriers},
                              64)});
        passToWarpSpecialize(entryPoint, readTracking[iMemType].at(entryRegion),
                             readTracking[iMemType]);
      }
    }
  }

  // Create lock variable allocation
  Value lockVal = createLockVariable(b);
  // Initialize the shared-cluster lock once, then synchronize before any CTA
  // can enter the first instrumented critical section.
  Value ctaId = ExperimentalClusterCTAIdOp::create(b, b.getLoc());
  Value zero = arith::ConstantIntOp::create(b, 0, 32);
  Value isCTA0 =
      arith::CmpIOp::create(b, arith::CmpIPredicate::eq, ctaId, zero);
  ExperimentalLockReleaseOp::create(b, lockVal, isCTA0);
  if (numCTAs > 1) {
    ClusterBarrierOp::create(b, b.getLoc());
  } else {
    BarrierOp::create(b, b.getLoc(),
                      AddrSpace::GlobalRead | AddrSpace::GlobalWrite);
  }
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
        {createZeroInitStateTensor(b, {numCTAs, numBufs, NUM_THREADS}, 8, fb),
         getIntTensorType(entryRegion, {numCTAs, numBufs, NUM_THREADS}, 8)});
    passToWarpSpecialize(entryPoint, commits[commitKind].at(entryRegion),
                         commits[commitKind]);
  };

  // Create write commits tensor for cp-async
  if (hasCpAsync(module)) {
    createCommitTensor(CommitKind::AsyncCp);
  }

  if (hooks) {
    for (auto kind : hooks->getRequiredCommitKinds(module)) {
      if (commits[kind].empty())
        createCommitTensor(kind);
    }
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
  func.walk([&](WarpSpecializePartitionsOp op) {
    op->insertOperands(op.getNumOperands(), {valueType.value});
    for (Region &region : op.getPartitionRegions()) {
      // Pass the value as a pointer type (instead of the type of underlying
      // memory)
      region.addArgument(valueType.value.getType(), op.getLoc());
      Type newType = valueType.type;
      if (auto tensorType = dyn_cast<RankedTensorType>(newType)) {
        // If this is a tensor, make sure the layout matches the region's warp
        // count
        newType = getIntTensorType(
            &region, tensorType.getShape(),
            tensorType.getElementType().getIntOrFloatBitWidth());
      }
      map.insert(
          &region,
          ValueType{region.getArgument(region.getNumArguments() - 1), newType});
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
