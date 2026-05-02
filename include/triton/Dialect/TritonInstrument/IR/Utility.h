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

// Map from IR region to ConSan auxiliary data.
//
// Aux data is created in the entry function and then either rematerialized or
// captured into warp-specialize partition regions. Each map member below is
// keyed by the IR region that owns the value visible at an instrumentation
// insertion point. For scratch-backed state, ValueType::value is the scratch
// pointer and ValueType::type is the logical tensor type loaded from/stored to
// that pointer. For tensor descriptors and constants, ValueType::value is the
// tensor itself and ValueType::type is its type.
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

  // Shape notation:
  //   C = CTAs in the cluster.
  //   B = tracked buffers for one memory type, power-of-two padded.
  //   K = tracked mbarriers, power-of-two padded.
  //   T = logical ConSan thread bit slots, padded to 64.
  //   P = base-thread commit columns, currently 16.
  //
  // Storage notation:
  //   tensor  = distributed tensor value.
  //   scratch = pointer to shared-cluster global scratch memory.

  // tensor, <C x B x i64>
  // Per-memory-type packed buffer descriptors. Each i64 stores the 32-bit base
  // offset and 32-bit length of one shared-memory or tensor-memory region.
  RegionToValueMap buffers[numMemTypes];

  // tensor, <C x K x i64>
  // Packed descriptors for tracked mbarrier allocations. Barriers are shared
  // memory descriptors.
  RegionToValueMap barriers;

  // scratch, <C x K x i64>
  // Packed barrier lifecycle state. Zero means invalid/uninitialized. Bit 0 is
  // phase, bits [1..20] are the initial arrival count, bits [21..40] are the
  // current arrival count, and bits [41..61] hold a signed tx-count.
  RegionToValueMap barrierStates;

  // scratch, <C x K x i32>
  // Per-barrier CTA bitsets of write-recipient rows reached by outstanding
  // EffectWrites operations such as TMA and CLC. Used when a later wait
  // transfers tracked writes.
  RegionToValueMap barrierWriteRecipients;

  // scratch, <C x B x i64>
  // Per-memory-type write frontier. Bit i means logical ConSan thread i can see
  // the latest write to the buffer row.
  RegionToValueMap writeVisibility[numMemTypes];

  // scratch, <C x B x K x i8>
  // Per-memory-type buffer/barrier map for writes that a barrier tracks.
  RegionToValueMap writeTracking[numMemTypes];

  // scratch, <C x B x T x i64>
  // Per-memory-type read frontier. For each buffer and logical thread lane, the
  // i64 value is a bitmask of reads visible to that lane's thread.
  RegionToValueMap readVisibility[numMemTypes];

  // scratch, <C x B x K x i64>
  // Per-memory-type buffer/barrier map for read visibility masks that a barrier
  // tracks.
  RegionToValueMap readTracking[numMemTypes];

  // scratch, <C x B x P x i8>
  // Per-commit-kind outstanding commit counters for shared-memory buffers.
  // Entries are 0 for none, -1 for staged but uncommitted, and positive for a
  // committed access with an outstanding-group distance.
  RegionToValueMap commits[CommitKind::NumCommitKinds];

  // tensor, <C x B x B x i1>
  // Optional per-memory-type alias matrix. Created only when BufferRegion
  // analysis finds cross-buffer aliasing; checks expand selected buffer rows
  // through this matrix.
  RegionToValueMap aliasMatrices[numMemTypes];

  // scratch pointer, i32
  // Shared-cluster lock used to serialize ConSan instrumentation updates.
  RegionToValueMap lock;

  // scratch, <C x K x i32>
  // Deadlock-detection bitfield. Each base thread uses two bits: waiting flag
  // and stored phase.
  RegionToValueMap waiting;

  // True when a memory type has cross-buffer aliasing and therefore requires
  // aliasMatrices to make visibility and commit checks conservative.
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
