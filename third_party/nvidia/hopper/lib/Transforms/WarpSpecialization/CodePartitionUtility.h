#ifndef NV_DIALECT_HOPPER_TRANSFORMS_CODEPARTITIONUTILITY_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_CODEPARTITIONUTILITY_H_

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <algorithm>
#include <numeric>

namespace mlir {
// class DominanceInfo;
// class PostDominanceInfo;

namespace tt = mlir::triton;

/*
namespace triton {
class ModuleAxisInfoAnalysis;
class LoadOp;
class StoreOp;
class FuncOp;
namespace gpu {
class SwizzledSharedEncodingAttr;
}
} // namespace triton
*/

enum class DataChannelKind { SMEM, TMEM };

struct Channel {
public:
  using Relation = std::pair<int, SmallVector<int>>;

  Channel(int producer, SmallVector<int> &consumers, Operation *op,
          unsigned operandIdx, unsigned numBuffers)
      : relation(producer, consumers), op(op), operandIdx(operandIdx),
        numBuffers(numBuffers) {}

  bool operator==(const Channel &c) {
    return relation == c.relation && operandIdx == c.operandIdx && op == c.op;
  }

  Operation *getDstOp() { return op; }
  unsigned getDstOperandIdx() { return operandIdx; }
  virtual Value getSrcOperand() { return op->getOperand(operandIdx); }
  virtual Operation *getSrcOp() { return getSrcOperand().getDefiningOp(); }

  Relation relation; // producer task Id, a list of consumer task Ids
  Operation *op;
  unsigned operandIdx;
  unsigned numBuffers;
  DataChannelKind channelKind = DataChannelKind::SMEM;
};

struct CommChannel {
  DenseMap<int, Value> tokens;
  // Producer barrier is only needed when the producer op itself can update the
  // barrier inline, such as the TMA load.
  std::optional<Value> producerBarrier;
  // Consumer barrier is only needed when the consumer op itself can update the
  // barrier inline, such as the TCGen5MMAOp.
  DenseMap<int, Value> consumerBarriers;
};

namespace ttng = ::mlir::triton::nvidia_gpu;
struct TmemDataChannel : Channel {
  ttng::TMEMAllocOp tmemAllocOp;
  ttng::TCGen5MMAOp tmemMmaOp;
  Operation *tmemProducerOp;

  TmemDataChannel(int producer, SmallVector<int> &consumers,
                  ttng::TMEMAllocOp tmemAllocOp, ttng::TCGen5MMAOp tmemMmaOp,
                  Operation *tmemLoadOp, unsigned operandIdx,
                  unsigned numBuffers)
      : Channel(producer, consumers, tmemLoadOp, operandIdx, numBuffers),
        tmemAllocOp(tmemAllocOp), tmemProducerOp(tmemAllocOp),
        tmemMmaOp(tmemMmaOp) {
    assert(consumers.size() == 1 &&
           "TmemDataChannel must have a single consumer");
    channelKind = DataChannelKind::TMEM;
  }

  ttng::TMEMAllocOp getAllocOp() { return tmemAllocOp; }
  ttng::TCGen5MMAOp getMmaOp() { return tmemMmaOp; }
  virtual Operation *getSrcOp() { return tmemProducerOp; }
};

bool enclosing(scf::IfOp ifOp, Operation *op);
bool enclosing(scf::ForOp forOp, Operation *op);

bool channelWithReuse(Operation *dstOp,
                      SmallVector<Operation *> &opsWithBufferReuse);
void excludeChannelsWithReuse(const DenseSet<Operation *> &opsWithChannels,
                              SmallVector<Operation *> &opsWithBufferReuse,
                              DenseSet<Operation *> &excludeReuse);

bool needAccumulatedLoopCntForReuse(
    scf::IfOp ifOp, SmallVector<Operation *> &opsWithBufferReuse);

// Return number of AccumCnts for the given ctrlOp. Add a single
// AccumCnt for all channels under opsWithBufferReuse and it will be the
// last AccumCnt.
unsigned getAccumCnts(Operation *ctrlOp,
                      const DenseSet<Operation *> &opsWithChannels,
                      SmallVector<Operation *> &opsWithBufferReuse);

unsigned getAccumArgIdx(scf::ForOp parentForOp, Operation *ctrlOp,
                        const DenseSet<Operation *> &opsWithChannels,
                        SmallVector<Operation *> &opsWithBufferReuse);

Value appendBufferIdxArgs(
    SmallVector<Operation *> &taskTopOps, unsigned numBuffers,
    const SmallVector<Channel *> &channels,
    const DenseMap<Channel *, Channel *> &mapToRepresenting,
    SmallVector<Operation *> &opsWithBufferReuse,
    DenseSet<Operation *> &opsWithChannels);

void reuseBuffers(SmallVector<Operation *> &taskTopOps,
                  const SmallVector<Channel *> &channels,
                  DenseMap<Channel *, Channel *> &mapToRepresenting,
                  SmallVector<Operation *> &opsWithBufferReuse);
void updateAccumRegions(SmallVector<Operation *> &opList,
                        const SmallVector<Channel *> &channels,
                        DenseSet<Operation *> &opsWithChannels);
void insertAsyncCopy(
    triton::FuncOp funcOp,
    const DenseMap<Channel *, SmallVector<Channel *>>
        &channelsGroupedByProducers,
    const DenseMap<Channel *, Value> &bufferMap,
    DenseMap<Channel *, std::pair<Operation *, Operation *>> &copyOpMap,
    DenseSet<Operation *> &opsWithChannels,
    SmallVector<Operation *> &opsWithBufferReuse);

Value getAccumCount(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                    const DenseSet<Operation *> &opsWithChannels,
                    SmallVector<Operation *> &opsWithBufferReuse);
std::pair<Value, Value> getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder,
                                             Location loc, Value accumCnt,
                                             unsigned numBuffers);
void getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                          unsigned numBuffers,
                          const DenseSet<Operation *> &opsWithChannels,
                          Value &bufferIdx, Value &phase,
                          SmallVector<Operation *> &opsWithBufferReuse);

Value getBarrierForPipelineStage(OpBuilderWithAsyncTaskIds &builder,
                                 Value barrierAlloc, Value bufferIdx);

Operation *optimizeTMALoads(OpBuilderWithAsyncTaskIds &builder,
                            SmallVector<tt::DescriptorLoadOp> &tmaLoads,
                            SmallVector<Value> &buffers, Value barrierAlloc,
                            Value bufferIdx, Value bufferIdxExtract,
                            Value phase, Operation *headProducer,
                            Operation *headConsumer);
void SpecializeRegion(triton::FuncOp funcOp, int regDecProducer,
                      int regIncConsumer);

} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_CODEPARTITIONUTILITY_H_
