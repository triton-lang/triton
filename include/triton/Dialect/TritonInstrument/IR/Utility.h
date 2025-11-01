#ifndef TRITONINSTRUMENT_UTILITY_H
#define TRITONINSTRUMENT_UTILITY_H

#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
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
Value createLoadScratchMemory(OpBuilder &b, Location loc, Value alloc,
                              RankedTensorType tensorType);
Value expandOuterSlicedDim(OpBuilder &b, Location loc, Value tensor);
TypedValue<RankedTensorType> createConstIntTensor(OpBuilder &builder,
                                                  Location loc, int64_t val,
                                                  RankedTensorType tensorType,
                                                  bool isSigned = false);
FuncOp getEntryPoint(ModuleOp module);
gpu::DistributedEncodingTrait
getSingleDimSliceEncoding(gpu::BlockedEncodingAttr encoding, int dim);

struct ValueType {
  Value value;
  Type type;

  ValueType() = default;
  ValueType(Value value, Type type) : value(value), type(type) {}
  ValueType(std::pair<Value, Type> value)
      : value(value.first), type(value.second) {}
};

// Map from IR region to ConSan auxiliary data. Auxiliary data is a value
// and an optional type, for values that are stored in the scratch memory.
struct AuxDataMap {
  struct RegionToValueMap {
    DenseMap<Region *, ValueType> values;
    ValueType &operator[](Region *region) { return values[region]; }
    ValueType &operator[](Operation *op) {
      return values[getEnclosingParitionOrFunctionRegion(op)];
    }
    bool empty() const { return values.empty(); }

  private:
    Region *getEnclosingParitionOrFunctionRegion(Operation *op);
  };

  // Please see TritonInstrumentOps.td for more information on the auxiliary
  // data structures.
  RegionToValueMap buffers[numMemTypes];
  RegionToValueMap barriers;
  RegionToValueMap barrierStates;

  RegionToValueMap writeVisibility[numMemTypes];
  RegionToValueMap writeTracking[numMemTypes];
  RegionToValueMap readVisibility[numMemTypes];
  RegionToValueMap readTracking[numMemTypes];
  RegionToValueMap asyncCpCommits;
  RegionToValueMap wgmmaCommits;
  RegionToValueMap lock;
  RegionToValueMap waiting;

  void populateAndPassToWarpSpecialize(ModuleOp module);

private:
  void getBuffersAndBarriers(ModuleOp module,
                             SmallVector<SmallVector<int32_t>, 2> &bufValues,
                             SmallVector<int32_t> &barrierValues);
  void passToWarpSpecialize(triton::FuncOp func, ValueType value,
                            RegionToValueMap &map);
  void createInWarpSpecialize(
      triton::FuncOp func, RegionToValueMap &map,
      std::function<ValueType(ImplicitLocOpBuilder &)> createFn);
};

} // namespace mlir::triton::instrument

#endif // TRITONINSTRUMENT_UTILITY_H
