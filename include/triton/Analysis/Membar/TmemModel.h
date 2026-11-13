#ifndef TRITON_ANALYSIS_MEMBAR_TMEMMODEL_H
#define TRITON_ANALYSIS_MEMBAR_TMEMMODEL_H

#include "triton/Analysis/BufferRegion.h"
#include <memory>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace mlir::triton::membar {

struct OpModel;

struct TMemAccess {
  const BufferRegionFootprint *footprint;
  bool isWrite;

  bool operator==(const TMemAccess &) const = default;
};

bool usesTensorMemory(ModuleOp module);

// These target rules describe TMEM instruction effects and word ownership.
class TMemModel {
public:
  TMemModel(ModuleOp module, BufferRegionAnalysis &regions);
  ~TMemModel();
  void describe(Operation *op, OpModel &model, ArrayRef<MemoryAccess> accesses);
  // Transfers in one placement graph execute in the same warp role.
  bool isSameWarp(Operation *lhs, Operation *rhs);

private:
  struct Info;
  const Info *getInfo(Operation *op);
  bool isPrivate(const BufferRegionFootprint *footprint, Region *issuer) const;

  BufferRegionAnalysis &regions;
  SmallVector<std::pair<const BufferRegionFootprint *, Region *>> allocations;
  DenseMap<Operation *, std::unique_ptr<Info>> info;
};

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_TMEMMODEL_H
