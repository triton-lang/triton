#ifndef NV_DIALECT_HOPPER_TRANSFORMS_CODEPARTITIONUTILITY_H_
#define NV_DIALECT_HOPPER_TRANSFORMS_CODEPARTITIONUTILITY_H_

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "Utility.h"
#include <algorithm>
#include <numeric>

namespace mlir {

namespace tt = mlir::triton;

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
  virtual ~Channel() = default;

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
namespace triton {
namespace nvidia_gpu {
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
} // namespace nvidia_gpu
} // namespace triton

bool enclosing(scf::IfOp ifOp, Operation *op);
bool enclosing(scf::ForOp forOp, Operation *op);

// Return number of AccumCnts for the given ctrlOp. Add a single
// AccumCnt for all channels under opsWithBufferReuse and it will be the
// last AccumCnt.
unsigned getAccumCnts(Operation *ctrlOp,
                      const DenseSet<Operation *> &regionsWithChannels);

unsigned getAccumArgIdx(scf::ForOp parentForOp, Operation *ctrlOp,
                        const DenseSet<Operation *> &regionsWithChannels);

SmallVector<Operation *>
getTaskTopRegion(triton::FuncOp funcOp, const SmallVector<Channel *> &channels);

void appendAccumCntsForOps(SmallVector<Operation *> &taskTopOps,
                           const SmallVector<Channel *> &channels,
                           DenseSet<Operation *> &regionsWithChannels);

void collectRegionsWithChannels(const SmallVector<Channel *> &channels,
                                DenseSet<Operation *> &regionsWithChannels);
void insertAsyncCopy(
    triton::FuncOp funcOp,
    const DenseMap<Channel *, SmallVector<Channel *>>
        &channelsGroupedByProducers,
    const DenseMap<Channel *, Value> &bufferMap,
    DenseMap<Channel *, std::pair<Operation *, Operation *>> &copyOpMap,
    DenseSet<Operation *> &regionsWithChannels);

Value getAccumCount(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                    const DenseSet<Operation *> &regionsWithChannels);
std::pair<Value, Value> getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder,
                                             Location loc, Value accumCnt,
                                             unsigned numBuffers);
void getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                          unsigned numBuffers,
                          const DenseSet<Operation *> &regionsWithChannels,
                          Value &bufferIdx, Value &phase);

Value getBarrierForPipelineStage(OpBuilderWithAsyncTaskIds &builder,
                                 Value barrierAlloc, Value bufferIdx);

Operation *optimizeTMALoads(OpBuilderWithAsyncTaskIds &builder,
                            SmallVector<tt::DescriptorLoadOp> &tmaLoads,
                            SmallVector<Value> &buffers, Value barrierAlloc,
                            Value bufferIdx, Value bufferIdxExtract,
                            Value phase, Operation *headProducer,
                            Operation *headConsumer);
void specializeRegion(triton::FuncOp funcOp, unsigned requestedRegisters);

} // namespace mlir

#endif // NV_DIALECT_HOPPER_TRANSFORMS_CODEPARTITIONUTILITY_H_
