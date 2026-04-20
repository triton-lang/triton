#ifndef TRITONINSTRUMENT_UTILITY_H
#define TRITONINSTRUMENT_UTILITY_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

#include <array>

namespace mlir::triton::gpu {
class GlobalScratchAllocOp;
}

namespace mlir::triton::instrument {
class ConSanTargetHooks;
class FunctionBuilder;

constexpr int numMemTypes = getMaxEnumValForMemType() + 1;

constexpr int NUM_THREADS = 16;
constexpr int TMA_THREAD_OFFSET = NUM_THREADS;
constexpr int TC_THREAD_OFFSET = TMA_THREAD_OFFSET + NUM_THREADS;
constexpr int CLC_THREAD_OFFSET = TC_THREAD_OFFSET + NUM_THREADS;
constexpr int TOTAL_NUM_THREADS = CLC_THREAD_OFFSET + NUM_THREADS;
static_assert(TOTAL_NUM_THREADS <= 64,
              "ConSan thread bitsets are stored in i64 masks");
const int THREADS_BITMASK_SIZE = llvm::PowerOf2Ceil(TOTAL_NUM_THREADS);

namespace CommitKind {
enum Kind { None = -1, AsyncCp = 0, Wgmma, TmaStore, NumCommitKinds };
}

// -- ConSan capture-count constants -----------------------------------------
// Each constant corresponds to specific passToWarpSpecialize() calls in
// populateAndPassToWarpSpecialize().  Keep in sync with that function and
// with estimateConSanCaptureCount() below.

// writeVisibility + readVisibility per active memory type.
constexpr int kCapturesPerMemType = 2;

// barrierStates + waiting + barrierWriteRecipients (only when barriers exist).
constexpr int kBarrierBaseCaptures = 3;

// writeTracking + readTracking per active memory type (only when barriers
// exist and the memory type has buffers).
constexpr int kBarrierTrackingCapturesPerMemType = 2;

// The lock variable (always present).
constexpr int kFixedCaptures = 1;

// Size in bytes of each capture (a global-scratch pointer).
constexpr int kCaptureSizeBytes = 8;

/// Estimate the number of WarpSpecialize captures that the
/// ConcurrencySanitizer pass will add via passToWarpSpecialize().
/// \p numActiveMemTypes  Number of memory types with buffers.
/// \p hasBarriers        Whether barriers exist in the module.
/// \p numCommitKinds     Number of distinct commit kinds required.
inline int estimateConSanCaptureCount(int numActiveMemTypes, bool hasBarriers,
                                      int numCommitKinds) {
  int perMemType = kCapturesPerMemType * numActiveMemTypes;
  int barrierCaptures =
      hasBarriers ? kBarrierBaseCaptures +
                        kBarrierTrackingCapturesPerMemType * numActiveMemTypes
                  : 0;
  return perMemType + barrierCaptures + kFixedCaptures + numCommitKinds;
}

void createAssertInThread(ImplicitLocOpBuilder &b, Value condition,
                          StringRef message);
Operation *createStoreScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                    Value tensor, RankedTensorType tensorType,
                                    bool currentCTAOnly = false);
Value createLoadScratchMemory(OpBuilder &b, Location loc, Value alloc,
                              RankedTensorType tensorType);
gpu::GlobalScratchAllocOp
createThirdPartyScratchAlloc(OpBuilder &b, Location loc, Type ptrType,
                             int64_t sizeInBytes, int64_t alignment,
                             bool sharedClusterState = false);
Value expandOuterSlicedDim(OpBuilder &b, Location loc, Value tensor);
RankedTensorType getIntTensorType(Region *region, ArrayRef<int64_t> shape,
                                  unsigned bitWidth);
TypedValue<RankedTensorType> createConstIntTensor(OpBuilder &builder,
                                                  Location loc, int64_t val,
                                                  RankedTensorType tensorType,
                                                  bool isSigned = false);
uint32_t getMemDescLength(Value buf);
FuncOp getEntryPoint(ModuleOp module);
gpu::DistributedEncodingTrait
getSingleDimSliceEncoding(gpu::DistributedEncodingTrait encoding, int dim);

inline Value maybeAnd(ImplicitLocOpBuilder &b, Value lhs, Value rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return arith::AndIOp::create(b, lhs, rhs);
}

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
    ValueType at(Region *region) {
      if (values.find(region) == values.end()) {
        assert(false && "Region not found in AuxDataMap");
      }
      return values[region];
    }
    ValueType at(Operation *op) {
      return at(getEnclosingParitionOrFunctionRegion(op));
    }
    void insert(Region *region, ValueType value) { values[region] = value; }
    bool empty() const { return values.empty(); }

  private:
    Region *getEnclosingParitionOrFunctionRegion(Operation *op);
  };

  // Please see TritonInstrumentOps.td for more information on the auxiliary
  // data structures.
  RegionToValueMap buffers[numMemTypes];
  RegionToValueMap barriers;
  RegionToValueMap barrierStates;
  RegionToValueMap barrierWriteRecipients;

  RegionToValueMap writeVisibility[numMemTypes];
  RegionToValueMap writeTracking[numMemTypes];
  RegionToValueMap readVisibility[numMemTypes];
  RegionToValueMap readTracking[numMemTypes];
  RegionToValueMap commits[CommitKind::NumCommitKinds];
  RegionToValueMap aliasMatrices[numMemTypes];
  RegionToValueMap lock;
  RegionToValueMap waiting;
  std::array<bool, numMemTypes> hasNonTrivialAliasing{};

  void populateAndPassToWarpSpecialize(ModuleOp module,
                                       FunctionBuilder &funcBuilder,
                                       const ConSanTargetHooks *hooks);

private:
  void getBuffersAndBarriers(
      ModuleOp module,
      SmallVector<SmallVector<triton::BufferRegion>, 2> &bufRegions,
      SmallVector<triton::BufferRegion> &barrierRegions);
  void passToWarpSpecialize(triton::FuncOp func, ValueType value,
                            RegionToValueMap &map, int &captureCounter);
  void createInWarpSpecialize(
      triton::FuncOp func, RegionToValueMap &map,
      std::function<ValueType(ImplicitLocOpBuilder &)> createFn);
};

} // namespace mlir::triton::instrument

#endif // TRITONINSTRUMENT_UTILITY_H
