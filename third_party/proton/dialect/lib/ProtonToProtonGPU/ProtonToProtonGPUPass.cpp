#include "Analysis/ScopeIdAllocation.h"
#include "Conversion/ProtonToProtonGPU/Passes.h"
#include "Dialect/Proton/IR/Dialect.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

namespace mlir {
namespace triton {
namespace proton {

#define GEN_PASS_DEF_CONVERTPROTONTOPROTONGPU
#include "Conversion/ProtonToProtonGPU/Passes.h.inc"

#define DEBUG_TYPE "proton-to-proton-gpu"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

constexpr float maxSharedMemRatio = 0.04; // 4 percent of max shared mem

namespace {

void parseSelectIds(llvm::StringRef selectIds,
                    llvm::SmallVectorImpl<int32_t> &selectIdVec) {
  auto rest = selectIds;
  while (!rest.empty()) {
    llvm::StringRef id;
    std::tie(id, rest) = rest.split(',');
    if (id.trim().size() > 0) {
      selectIdVec.push_back(std::stoi(id.str()));
    }
    if (rest.trim().size() == 0)
      break;
  }
  llvm::sort(selectIdVec);
  selectIdVec.erase(llvm::unique(selectIdVec), selectIdVec.end());
}

template <typename T, typename OP> bool hasOperator(T *o) {
  bool exist = false;
  o->walk([&](OP op) {
    exist = true;
    return WalkResult::interrupt();
  });
  return exist;
}

void instrumentWarpSpecializeOps(FuncOp func, Value buffer, Value profileMem) {
  for (auto wsOp : func.getOps<triton::gpu::WarpSpecializeOp>()) {
    auto loc = wsOp.getLoc();
    if (hasOperator<Operation, proton::RecordOp>(wsOp.getOperation())) {
      wsOp->insertOperands(wsOp->getNumOperands(), {buffer, profileMem});
      for (Region *region : wsOp.getPartitionRegions()) {
        region->addArgument(buffer.getType(), loc);
        region->addArgument(profileMem.getType(), loc);
      }
    }
  }
}

LogicalResult replaceProtonRecordOp(OpBuilder &builder, FuncOp func,
                                    Value segment, MetricType metricType,
                                    ModuleScopeIdAllocation &scopeInfo,
                                    bool clockExtension) {
  mlir::IntegerType clkType =
      clockExtension ? mlir::IntegerType::get(builder.getContext(), 64)
                     : mlir::IntegerType::get(builder.getContext(), 32);

  // Replace all proton::RecordOp in the worker warps.
  func->walk([&](triton::gpu::WarpSpecializePartitionsOp partitions) {
    auto loc = partitions.getLoc();
    for (auto &partition : partitions.getPartitionRegions()) {
      if (hasOperator<Region, proton::RecordOp>(&partition)) {
        Block &block = partition.front();
        builder.setInsertionPointToStart(&block);
        int argNum = block.getNumArguments();
        auto bufferArg = block.getArgument(argNum - 2);
        auto profileMemArg = block.getArgument(argNum - 1);

        // Create a new segment for the worker warp.
        Value newSegment = builder.create<gpu::SegmentAllocOp>(
            loc, segment.getType(), bufferArg);

        // Restore warp-level context before profiling.
        builder.create<gpu::RestoreCtxOp>(loc, newSegment, profileMemArg);

        // Replace all proton::RecordOp.
        partition.walk([&](proton::RecordOp record) {
          builder.setInsertionPoint(record);

          Value counter =
              builder.create<gpu::ReadCounterOp>(loc, clkType, metricType);
          int scopeId = scopeInfo.getOpScopeId(record);
          builder.create<gpu::CircularStoreOp>(loc, newSegment, counter,
                                               record.getIsStart(), scopeId);
          record.erase();
        });

        // Save warp-level context after profiling.
        partition.walk([&](triton::gpu::WarpReturnOp ret) {
          builder.setInsertionPoint(ret);
          builder.create<gpu::SaveCtxOp>(loc, newSegment, profileMemArg);
        });
      }
    }
  });

  // Replace all proton::RecordOp in the master warps. For the master warps, we
  // don't need to restore warp-level context and we save the context in the end
  // of kernel (right before FinalizeOp).
  auto loc = func.getLoc();
  func->walk([&](proton::RecordOp record) {
    builder.setInsertionPoint(record);
    Value counter =
        builder.create<gpu::ReadCounterOp>(loc, clkType, metricType);
    int scopeId = scopeInfo.getOpScopeId(record);
    builder.create<gpu::CircularStoreOp>(loc, segment, counter,
                                         record.getIsStart(), scopeId);
    record.erase();
  });

  return success();
}

int getAllocSharedMemSize(int maxSharedMemSize, int sharedMemUsed,
                          int segmentNum) {
  const int bytesPerEntry = gpu::getBytesPerClockEntry();
  const int wordsPerEntry = bytesPerEntry / 4; // 1 word = 4 bytes
  const int circularHeaderSize = gpu::getCircularHeaderSize(); // byte size
  sharedMemUsed = llvm::alignTo(sharedMemUsed, bytesPerEntry);
  if (sharedMemUsed >= maxSharedMemSize) {
    // We just assume there's enough shared memory and error out if not during
    // execution.
    maxSharedMemSize += sharedMemUsed;
  }

  int segmentByteSizeShared =
      llvm::NextPowerOf2((maxSharedMemSize - sharedMemUsed) / segmentNum) / 2;
  int numSharedEntries = segmentByteSizeShared * segmentNum / bytesPerEntry;
  int allocSharedMemSize = numSharedEntries * bytesPerEntry;

  int estimatedOccupany = maxSharedMemSize / std::max(1, sharedMemUsed);
  if (estimatedOccupany <= 1)
    return allocSharedMemSize;

  int maxAllocSharedMemSize = maxSharedMemSize * maxSharedMemRatio;
  while (allocSharedMemSize > maxAllocSharedMemSize)
    allocSharedMemSize /= 2;

  return allocSharedMemSize;
}
} // namespace

class ConvertProtonToProtonGPUPass
    : public impl::ConvertProtonToProtonGPUBase<ConvertProtonToProtonGPUPass> {
public:
  ConvertProtonToProtonGPUPass(
      MetricType metricType, SamplingStrategy samplingStrategy,
      llvm::StringRef samplingOptions, gpu::Granularity granularity,
      gpu::BufferStrategy bufferStrategy, gpu::BufferType bufferType,
      int32_t bufferSize, int32_t maxSharedMemSize, int64_t profileScratchSize,
      int32_t profileScratchAlignment, bool clockExtension)
      : ConvertProtonToProtonGPUBase<ConvertProtonToProtonGPUPass>() {
    this->metricType = metricType;
    this->samplingStrategy = samplingStrategy;
    this->granularity = granularity;
    this->samplingOptions = samplingOptions.str();
    this->bufferStrategy = bufferStrategy;
    this->bufferType = bufferType;
    this->bufferSize = bufferSize;
    this->maxSharedMemSize = maxSharedMemSize;
    this->profileScratchSize = profileScratchSize;
    this->profileScratchAlignment = profileScratchAlignment;
    this->clockExtension = clockExtension;
  }

  LogicalResult circularRecordStrategyLowering(FuncOp func) {
    MLIRContext *context = func.getContext();
    Location loc = func->getLoc();
    ModuleOp mod = llvm::cast<ModuleOp>(func->getParentOp());

    OpBuilder builder(context);
    builder.setInsertionPointToStart(&func.getBody().front());

    int numWarps = gpu::getTotalNumWarps(mod);

    llvm::SmallVector<int32_t, 8> selectIdVec;
    int segmentNum = numWarps;
    if (!samplingOptions.empty() &&
        samplingStrategy == SamplingStrategy::SELECTIVE) {
      parseSelectIds(samplingOptions, selectIdVec);
      segmentNum = selectIdVec.size();
      if (segmentNum && granularity != gpu::Granularity::WARP) {
        mlir::emitError(
            loc, "only warp granularity supports selective ids for now.");
        return failure();
      }
    }

    int sharedMemUsed = 0;
    if (mod->hasAttr("ttg.shared"))
      sharedMemUsed =
          mod->getAttrOfType<mlir::IntegerAttr>("ttg.shared").getInt();

    int allocSharedMemSize =
        getAllocSharedMemSize(maxSharedMemSize, sharedMemUsed, segmentNum);

    const int bytesPerEntry = gpu::getBytesPerClockEntry();

    if (bufferSize != 0)
      bufferSize = llvm::alignTo(bufferSize, bytesPerEntry);
    // Validate buffer size
    if (bufferSize != 0 && !llvm::isPowerOf2_32(bufferSize / segmentNum)) {
      mlir::emitError(loc, "buffer-size per segment(" +
                               llvm::Twine(segmentNum) +
                               ") must be power of 2");
      return failure();
    }

    int allocBufferSize;
    if (bufferType == gpu::BufferType::SHARED) {
      if (bufferSize > 0)
        allocBufferSize = std::min(allocSharedMemSize, bufferSize.getValue());
      else
        allocBufferSize = allocSharedMemSize;
    } else if (bufferType == gpu::BufferType::GLOBAL) {
      allocBufferSize = bufferSize;
    } else {
      mlir::emitError(loc, "buffer-type not supported");
      return failure();
    }

    if (allocBufferSize <= 0) {
      mlir::emitError(loc, "profiling buffer size should be greater than 0");
      return failure();
    }

    // Circular strategy memory layout (total: allocProfileScratchSize bytes)
    //  +-----------------------------------------------+
    //  | header (circularHeaderSize bytes)             |
    //  +-----------------------------------------------+
    //  | contexts for all warps (4 bytes x numWarps)   |
    //  +-----------------------------------------------+
    //  | profiled data (allocBufferSize bytes)         |
    //  +-----------------------------------------------+
    const int circularHeaderSize = gpu::getCircularHeaderSize(); // byte size

    int allocProfileScratchSize =
        llvm::alignTo(allocBufferSize + circularHeaderSize + numWarps * 4,
                      profileScratchAlignment);

    if (profileScratchSize < allocProfileScratchSize) {
      LDBG("Global scratch memory for proton profiling is not large "
           "enough, we allocate the scratch size as " +
           llvm::Twine(allocProfileScratchSize) + " bytes.");
    }

    Value buffer;
    auto ctaLayout =
        triton::gpu::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                                        /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto encoding = triton::gpu::SwizzledSharedEncodingAttr::get(
        context, 1, 1, 1, {0}, ctaLayout);

    if (bufferType == gpu::BufferType::SHARED) {
      Attribute sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(context);
      auto sharedBufferType = triton::gpu::MemDescType::get(
          {allocBufferSize / 4}, builder.getI32Type(), encoding,
          sharedMemorySpace, /*mutable_memory=*/true);
      buffer = builder.create<triton::gpu::LocalAllocOp>(loc, sharedBufferType);
    } else if (bufferType == gpu::BufferType::GLOBAL) {
      mlir::emitError(loc, "not implemented yet");
      return failure();
    } else {
      mlir::emitError(loc, "buffer-type not supported");
      return failure();
    }

    auto memorySpace =
        mlir::cast<triton::gpu::MemDescType>(buffer.getType()).getMemorySpace();
    auto segmentType = gpu::SegmentType::get(
        context, allocBufferSize, memorySpace, granularity, selectIdVec);
    Value segment =
        builder.create<gpu::SegmentAllocOp>(loc, segmentType, buffer);

    ModuleScopeIdAllocation &scopeInfo = getAnalysis<ModuleScopeIdAllocation>();

    // Set insertion point to the start of the function
    builder.setInsertionPointToStart(&func.getBody().front());

    Value profileMem = builder.create<gpu::GlobalScratchAllocOp>(
        loc, triton::getPointerType(builder.getI32Type()),
        allocProfileScratchSize, profileScratchAlignment);

    if (hasOperator<Operation, triton::gpu::WarpSpecializeOp>(
            func.getOperation()))
      builder.create<gpu::InitCtxOp>(loc, profileMem);

    instrumentWarpSpecializeOps(func, buffer, profileMem);

    if (failed(replaceProtonRecordOp(builder, func, segment, metricType,
                                     scopeInfo, clockExtension)))
      return failure();

    func.walk([&](triton::ReturnOp ret) {
      builder.setInsertionPoint(ret);
      builder.create<mlir::gpu::BarrierOp>(loc);
      builder.create<gpu::FinalizeOp>(loc, segment, profileMem);
    });

    return success();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    Location loc = m->getLoc();

    // Validate metric type at runtime instead of using assert
    if (metricType != MetricType::CYCLE) {
      mlir::emitError(loc, "only CYCLE metric type is supported currently");
      signalPassFailure();
      return;
    }

    // Check if there are any functions in the module
    int numFuncs = llvm::range_size(m.getOps<triton::FuncOp>());
    if (numFuncs == 0) {
      return; // No functions to process, silently return
    } else if (numFuncs > 1) {
      // We currently only support one function in the module
      mlir::emitError(loc, "only one function per module is supported");
      signalPassFailure();
      return;
    }

    FuncOp func = *m.getOps<triton::FuncOp>().begin();

    // Check if there are any proton records to process
    if (!hasOperator<Operation, proton::RecordOp>(func.getOperation())) {
      return; // No proton records to process, silently return
    }

    // Validate profile scratch alignment
    if (!llvm::isPowerOf2_32(profileScratchAlignment)) {
      mlir::emitError(loc, "profileScratchAlignment must be power of 2");
      signalPassFailure();
      return;
    }

    // Process based on buffer strategy
    if (bufferStrategy == gpu::BufferStrategy::CIRCULAR) {
      if (failed(circularRecordStrategyLowering(func))) {
        // No need to call signalPassFailure() here as it's already called in
        // circularRecordStrategyLowering
        signalPassFailure();
      }
    } else {
      mlir::emitError(
          loc, "buffer-strategy '" +
                   std::to_string(static_cast<int>(
                       static_cast<gpu::BufferStrategy>(bufferStrategy))) +
                   "' is not supported");
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createConvertProtonToProtonGPUPass(
    MetricType metricType, SamplingStrategy samplingStrategy,
    llvm::StringRef samplingOptions, gpu::Granularity granularity,
    gpu::BufferStrategy bufferStrategy, gpu::BufferType bufferType,
    int32_t bufferSize, int32_t maxSharedMemSize, int64_t profileScratchSize,
    int32_t profileScratchAlignment, bool clkExt) {
  return std::make_unique<ConvertProtonToProtonGPUPass>(
      metricType, samplingStrategy, samplingOptions, granularity,
      bufferStrategy, bufferType, bufferSize, maxSharedMemSize,
      profileScratchSize, profileScratchAlignment, clkExt);
}

} // namespace proton
} // namespace triton
} // namespace mlir
