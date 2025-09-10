#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

namespace mlir::triton::instrument {

constexpr int numMemTypes = getMaxEnumValForMemType() + 1;

constexpr int NUM_THREADS = 16;
constexpr int TMA_THREAD_OFFSET = NUM_THREADS;
constexpr int TC_THREAD_OFFSET = TMA_THREAD_OFFSET + NUM_THREADS;
constexpr int TOTAL_NUM_THREADS = TC_THREAD_OFFSET + NUM_THREADS;
constexpr int THREADS_BITMASK_SIZE = llvm::NextPowerOf2(TOTAL_NUM_THREADS);

Operation *createStoreScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                    Value tensor, RankedTensorType tensorType);
Operation *createLoadScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                   RankedTensorType tensorType);
Value expandOuterSlicedDim(OpBuilder &b, Location loc, Value tensor);
TypedValue<RankedTensorType> createConstIntTensor(OpBuilder &builder,
                                                  Location loc, int64_t val,
                                                  RankedTensorType tensorType);
FuncOp getEntryPoint(ModuleOp module);

// Map from region to auxiliary data
struct AuxDataMap {
  struct RegionToValueMap {
    DenseMap<Region *, Value> values;
    void
    createInRegions(ImplicitLocOpBuilder &b, SmallVector<Region *> regions,
                    std::function<Value(ImplicitLocOpBuilder &)> createFn) {
      for (Region *region : regions) {
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(&region->getBlocks().front());
        values[region] = createFn(b);
      }
    }
    Value &operator[](Region *region) { return values[region]; }
    Value &operator[](Operation *op) {
      return values[getEnclosingParitionOrFunctionRegion(op)];
    }
    bool empty() const { return values.empty(); }

  private:
    Region *getEnclosingParitionOrFunctionRegion(Operation *op);
  };

  RegionToValueMap buffers[numMemTypes];
  RegionToValueMap barriers;

  RankedTensorType writeVisibilityType[numMemTypes];
  RegionToValueMap writeVisibility[numMemTypes];

  RankedTensorType writeTrackingType[numMemTypes];
  RegionToValueMap writeTracking[numMemTypes];

  RankedTensorType readVisibilityType[numMemTypes];
  RegionToValueMap readVisibility[numMemTypes];

  RankedTensorType readTrackingType[numMemTypes];
  RegionToValueMap readTracking[numMemTypes];

  RankedTensorType asyncCpCommitsType;
  RegionToValueMap asyncCpCommits;

  RankedTensorType wgmmaCommitsType;
  RegionToValueMap wgmmaCommits;

  RegionToValueMap lock;

  void populateAndPassToWarpSpecialize(ModuleOp module);

private:
  void getBuffersAndBarriers(ModuleOp module,
                             SmallVector<SmallVector<int32_t>, 2> &bufValues,
                             SmallVector<int32_t> &barrierValues);
  void passToWarpSpecialize(triton::FuncOp func, Value value,
                            RegionToValueMap &map);
};

} // namespace mlir::triton::instrument
