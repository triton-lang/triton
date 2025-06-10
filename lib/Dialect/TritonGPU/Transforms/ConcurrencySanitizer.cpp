#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONGPUCONCURRENCYSANITIZER
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

// TODO: remove
Operation *createDebugPrintOp(ImplicitLocOpBuilder &builder, StringRef str,
                              Value value) {
  auto strAttr = StringAttr::get(builder.getLoc().getContext(), str);
  return builder.create<tt::PrintOp>(strAttr, false, std::vector<Value>{value},
                                     std::vector<int32_t>{0});
}

void createAssertExpected(ImplicitLocOpBuilder &builder, Value value,
                          Value expected, StringRef msg) {
  auto cmp =
      builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, value, expected);
  builder.create<tt::AssertOp>(cmp, msg);
}

BlockedEncodingAttr getBlockedEncoding(ModuleOp module, unsigned int size) {
  MLIRContext *ctx = module.getContext();
  unsigned int warps =
      mlir::cast<mlir::IntegerAttr>(module->getAttr("ttg.num-warps")).getInt();
  auto ctaLayout = CTALayoutAttr::getDefault(ctx, /*rank=*/1);
  return BlockedEncodingAttr::get(ctx,
                                  /*sizePerThread=*/{size},
                                  /*threadsPerWarp=*/{32},
                                  /*warpsPerCTA=*/{warps},
                                  /*order=*/{0}, ctaLayout);
}

// Interpret local_allocs that are used in ttg.memdesc_subview as multibuffered
bool isMultiBuffered(triton::gpu::LocalAllocOp op) {
  return llvm::any_of(op->getUsers(), [](Operation *user) {
    return isa<ttg::MemDescSubviewOp>(user);
  });
}

uint64_t getAllocationOffset(triton::gpu::LocalAllocOp op) {
  auto offsetAttr = op->getAttr("allocation.offset");
  if (!offsetAttr) {
    return 0;
  }
  return cast<IntegerAttr>(offsetAttr).getInt();
}

unsigned getNumBuffers(triton::gpu::LocalAllocOp op) {
  MemDescType ty = op.getType();
  return ty.getShape().size();
}

unsigned getSubBufferSize(triton::gpu::LocalAllocOp op) {
  MemDescType ty = op.getType();
  unsigned elSize = ty.getElementType().getIntOrFloatBitWidth();
  return product(ty.getShape().drop_front()) * elSize;
}

tt::FuncOp getEntryPoint(ModuleOp module) {
  SmallVector<tt::FuncOp> publicFuncs = llvm::to_vector(
      llvm::make_filter_range(module.getOps<tt::FuncOp>(),
                              [](tt::FuncOp func) { return func.isPublic(); }));
  assert(publicFuncs.size() == 1 && "Expected exactly one public function");
  return publicFuncs.front();
}

Value createCmpIntTensorScalar(ImplicitLocOpBuilder &builder, Value tensor,
                               Value scalar) {
  auto tensorTy = cast<RankedTensorType>(tensor.getType());
  auto scalarTy = scalar.getType();
  auto elemTy = tensorTy.getElementType();
  assert(scalarTy == elemTy &&
         "Expected scalar to be of the same type as the tensor elements");
  auto splat = builder.create<triton::SplatOp>(tensorTy, scalar);
  auto cmp =
      builder.create<arith::CmpIOp>(arith::CmpIPredicate::eq, tensor, splat);
  return cmp;
}

Operation *createSumReduction(ImplicitLocOpBuilder &builder, Value tensor) {
  OpBuilder::InsertionGuard guard(builder);

  auto tensorTy = cast<RankedTensorType>(tensor.getType());
  auto elemTy = tensorTy.getElementType();
  auto reduce = builder.create<tt::ReduceOp>(std::vector<Value>{tensor}, 0);
  auto block = builder.createBlock(&reduce->getRegion(0));
  builder.setInsertionPointToStart(block);
  block->addArguments({elemTy, elemTy}, {builder.getLoc(), builder.getLoc()});
  Value sum = builder.create<arith::AddIOp>(block->getArgument(0),
                                            block->getArgument(1));
  builder.create<tt::ReduceReturnOp>(std::vector<Value>{sum});
  return reduce;
}

} // namespace

class ConcurrencySanitizerPass
    : public impl::TritonGPUConcurrencySanitizerBase<ConcurrencySanitizerPass> {
public:
  constexpr static int8_t WRITE_BIT = 1 << 0;
  constexpr static int8_t READ_BIT = 1 << 1;

  void runOnOperation() override {
    return;
    module = getOperation();
    // Collect shared memory buffers allocated in the module
    // TODO: We should actually map the region in IR + the offset in the buffer
    // to the local_alloc to give user a better error message
    llvm::SetVector<int64_t> shMemBufsSet;
    module.walk([&](triton::gpu::LocalAllocOp op) {
      int64_t baseOffset = getAllocationOffset(op);
      shMemBufsSet.insert(baseOffset);
      if (isMultiBuffered(op)) {
        unsigned numBuffers = getNumBuffers(op);
        assert(numBuffers > 0 && "Expected at least one buffer");
        unsigned subBufferSize = getSubBufferSize(op);
        for (unsigned i = 1; i < numBuffers; ++i) {
          shMemBufsSet.insert(baseOffset + i * subBufferSize);
        }
      }
    });

    tt::FuncOp entryPoint = getEntryPoint(module);
    assert(entryPoint);

    SmallVector<int64_t> shMemBufsValues = llvm::to_vector(shMemBufsSet);
    // Pad to the next power of 2 with zeros
    uint64_t nextPowerOf2 = llvm::NextPowerOf2(shMemBufsValues.size());
    shMemBufsValues.resize(nextPowerOf2, 0);

    ImplicitLocOpBuilder b(entryPoint.getLoc(), entryPoint);
    b.setInsertionPointToStart(&entryPoint.getBody().front());
    Value shMemBufs = createInitializedIntTensor(b, shMemBufsValues);
    Value shMemBase = b.create<ttg::SharedMemoryBaseOp>();
    shMemBufs = b.create<arith::AddIOp>(
        shMemBufs, b.create<triton::SplatOp>(shMemBufs.getType(), shMemBase));

    // Create state tensors:
    // 1. Barrier, tracking which barriers are tracking the buffer
    // 2. State, a bitfield tracking if the buffer is written (0x1) or read
    // (0x2)
    Value barriers = createConstIntTensor(b, 0, b.getIntegerType(64),
                                          shMemBufsValues.size());
    Value state =
        createConstIntTensor(b, 0, b.getIntegerType(8), shMemBufsValues.size());

    // debug
    // auto strAttr = StringAttr::get(loc.getContext(), "shMemBufs: ");
    // b.create<tt::PrintOp>(loc, strAttr, false, std::vector<Value>{shMemBufs},
    // std::vector<int32_t>{0});
    instrumentMemoryOperations(b, shMemBufs, barriers, state);
  }

private:
  Value createInitializedIntTensor(ImplicitLocOpBuilder &builder,
                                   SmallVector<int64_t> values) {
    int64_t size = values.size();
    assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
    auto tensorType = RankedTensorType::get({size}, builder.getIntegerType(64),
                                            getBlockedEncoding(module, size));
    SmallVector<APInt> apInts = llvm::to_vector(
        llvm::map_range(values, [](int64_t v) { return APInt(64, v); }));
    auto denseAttr = DenseElementsAttr::get(tensorType, apInts);
    return builder.create<arith::ConstantOp>(tensorType, denseAttr);
  }

  Value createConstIntTensor(ImplicitLocOpBuilder &builder, int val,
                             Type elType, int64_t size) {
    assert(llvm::isPowerOf2_64(size) && "Expected power of 2");
    auto tensorType =
        RankedTensorType::get({size}, elType, getBlockedEncoding(module, size));
    auto denseAttr = DenseElementsAttr::get(
        tensorType, APInt(elType.getIntOrFloatBitWidth(), val));
    return builder.create<arith::ConstantOp>(tensorType, denseAttr);
  }

  Value createConstIntTensor(ImplicitLocOpBuilder &builder, int val,
                             RankedTensorType tensorType) {
    auto denseAttr = DenseElementsAttr::get(
        tensorType,
        APInt(tensorType.getElementType().getIntOrFloatBitWidth(), val));
    return builder.create<arith::ConstantOp>(tensorType, denseAttr);
  }

  void instrumentMemoryOperations(ImplicitLocOpBuilder &b, Value buffers,
                                  Value barriers, Value state) {
    SmallVector<ttng::AsyncTMACopyGlobalToLocalOp> copyOps;
    module.walk(
        [&](ttng::AsyncTMACopyGlobalToLocalOp op) { copyOps.push_back(op); });
    int i = 0;
    for (auto op : copyOps) {
      b.setLoc(op.getLoc());
      b.setInsertionPoint(op);
      RankedTensorType barriersTy = cast<RankedTensorType>(barriers.getType());
      RankedTensorType stateTy = cast<RankedTensorType>(state.getType());
      Value zero_64b = createConstIntTensor(b, 0, barriersTy);
      Value zero_8b = createConstIntTensor(b, 0, stateTy);
      Value buffer = b.create<ttg::MemDescToI64Op>(op.getResult());
      Value bar = b.create<ttg::MemDescToI64Op>(op.getBarrier());
      Value barSplat = b.create<triton::SplatOp>(barriersTy, bar);
      Value mask = createCmpIntTensorScalar(b, buffers, buffer);

      // 1. Check if the buffer has outstanding accesses
      Value rwSplat = createConstIntTensor(b, WRITE_BIT | READ_BIT, stateTy);
      Value writeSplat = createConstIntTensor(b, WRITE_BIT, stateTy);
      Value isRW = b.create<arith::AndIOp>(state, rwSplat);
      Value isRWMask = b.create<arith::SelectOp>(mask, isRW, zero_8b);
      // Value isCurrentRW = createSumReduction(b, isRWMask)->getResult(0);

      std::stringstream ss;
      ss << "isRWMask: " << i;
      createDebugPrintOp(b, ss.str(), isRWMask);

      // Assert that the buffer is not being read or written
      createAssertExpected(b, isRWMask, zero_8b,
                           "TMA copy to buffer being read or written");

      // 2. Update the access state
      // TODO: Implicit assumption we are instrumenting buffers in the program
      // order, so the next instruction uses the updated state
      state = b.create<arith::SelectOp>(
          mask, b.create<arith::OrIOp>(state, writeSplat), state);

      ss.str("");
      ss << "updated state: " << i;
      createDebugPrintOp(b, ss.str(), state);

      barriers = b.create<arith::SelectOp>(mask, barSplat, barriers);

      createDebugPrintOp(b, "updated barriers: ", barriers);
      // createDebugPrintOp(b, "buffer: ", loc, buffer);
      // createDebugPrintOp(b, "all buffers: ", loc, buffers);
      // createDebugPrintOp(b, "mask: ", loc, mask);
      i++;
    }
  }

  ModuleOp module;
};

} // namespace gpu
} // namespace triton
} // namespace mlir
