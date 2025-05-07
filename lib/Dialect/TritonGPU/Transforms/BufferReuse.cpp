#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/AddressRanges.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

//===----------------------------------------------------------------------===//
// BufferLiveRangeAnalysis
//===----------------------------------------------------------------------===//

namespace {
struct BufferState {
  llvm::BitVector liveness;
};

struct BufferRange {
  Value alloc;
  llvm::AddressRange range;
};

struct LiveRangeState {
  llvm::MapVector<Value, BufferState> states;
};

struct BufferLiveRangeAnalysis {
  llvm::AddressRange getAllocRange(Value value);

  uint64_t curAllocOffset = 0;
  DenseMap<Value, llvm::AddressRange> allocRanges;
};
} // namespace

llvm::AddressRange BufferLiveRangeAnalysis::getAllocRange(Value value) {
  auto it = allocRanges.find(value);
  if (it != allocRanges.end())
    return it->second;

  auto memdesc = cast<MemDescType>(value.getType());
  assert(memdesc.getAllocShape() == memdesc.getShape());
  // We are not tracking real in-memory representations, just logical values
  // that happen to live in memory. That is, we don't need to use
  // `getAllocationShapePerCTA` or respect alignment. We just need to know the
  // subrange that is live or dead.
  uint64_t allocNumElts = product(memdesc.getAllocShape());
  llvm::AddressRange range(curAllocOffset, curAllocOffset + allocNumElts);
  curAllocOffset += allocNumElts;

  allocRanges.insert({value, range});
  return range;
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUBUFFERREUSE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct BufferReuse
    : public triton::gpu::impl::TritonGPUBufferReuseBase<BufferReuse> {
  using TritonGPUBufferReuseBase::TritonGPUBufferReuseBase;

  void runOnOperation() override;
};
} // namespace

void BufferReuse::runOnOperation() {}
