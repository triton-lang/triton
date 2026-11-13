#ifndef TRITON_ANALYSIS_MEMBAR_TMEMMODEL_H
#define TRITON_ANALYSIS_MEMBAR_TMEMMODEL_H

#include "triton/Analysis/Membar/SmemModel.h"
#include <memory>
#include <optional>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace mlir::triton::membar {

struct TMemAccess {
  const BufferRegionFootprint *footprint;
  bool isWrite;
};

enum class Need {
  WaitLoad,
  WaitStore,
  RegionOrder,
  WarpOrder,
  WarpConvergence
};

constexpr unsigned bit(Need need) { return 1u << unsigned(need); }
constexpr unsigned warpBarrier =
    bit(Need::WarpOrder) | bit(Need::WarpConvergence);
constexpr unsigned fullBarrier = bit(Need::RegionOrder) | warpBarrier;

enum class Boundary {
  None,
  WaitAll,     // Complete issued TMEM accesses.
  Publication, // Order peer-visible TMEM effects.
};

// Observe the input before the operation and start new producers at its output.
// A scratch rendezvous lies between those two sets of effects.
struct OpModel {
  unsigned fixedBefore = 0, fixedInside = 0, fixedAfter = 0;
  BlockInfo sharedIn, sharedOut;
  SmallVector<TMemAccess, 1> tmemAccesses;
  // Native transfers need a completion wait; descriptor asm needs region order.
  std::optional<Need> tmemProducerNeed;
  Boundary boundary = Boundary::None;
  bool needsWarp = false;
};

bool usesTensorMemory(ModuleOp module);

// These target rules describe TMEM instruction effects and word ownership.
class TMemModel {
public:
  explicit TMemModel(BufferRegionAnalysis &regions);
  ~TMemModel();
  void describe(Operation *op, OpModel &model, ArrayRef<MemoryAccess> accesses);
  bool isSameWarp(Operation *lhs, Operation *rhs);

private:
  struct Info;
  const Info *getInfo(Operation *op);

  BufferRegionAnalysis &regions;
  DenseMap<Operation *, std::unique_ptr<Info>> info;
};

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_TMEMMODEL_H
