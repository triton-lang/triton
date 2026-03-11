#include "GSan.h"
#include "Hash.cuh"

#include <cstddef>
#include <cstdint>
#include <device_launch_parameters.h>
#include <limits>

namespace gsan {

struct Location {
  const char *file;
  unsigned line;
};

__device__ const char *getSourceFile(Location loc) {
  return loc.file == nullptr ? "<unknown>" : loc.file;
}

} // namespace gsan

#define assert_msg(loc, cond, msg)                                             \
  do {                                                                         \
    if (!(cond)) {                                                             \
      __assert_fail((msg), gsan::getSourceFile(loc), (loc).line, "");          \
    }                                                                          \
  } while (false)

namespace gsan {
namespace {
static constexpr uint32_t writerFlag = 1u << 31;

enum class AtomicSem : uint8_t {
  Relaxed = 1,
  Acquire = 2,
  Release = 3,
  AcquireRelease = 4,
};

__device__ void rwLockAcquireRead(uint32_t &lock) {
  uint32_t old = __scoped_atomic_fetch_add(&lock, 1, __ATOMIC_ACQUIRE,
                                           __MEMORY_SCOPE_WRKGRP);
  if ((old & writerFlag) == 0)
    return;

  do {
    old =
        __scoped_atomic_load_n(&lock, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_WRKGRP);
  } while ((old & writerFlag) != 0);
}

__device__ void rwLockAcquireWrite(uint32_t &lock) {
  uint32_t actual = 0;
  while (!__scoped_atomic_compare_exchange_n(&lock, &actual, writerFlag, true,
                                             __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                                             __MEMORY_SCOPE_WRKGRP)) {
    actual = 0;
  }
}

__device__ void rwLockReleaseRead(uint32_t &lock) {
  __scoped_atomic_fetch_sub(&lock, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
}

__device__ void rwLockReleaseWrite(uint32_t &lock) {
  // Note we don't set 0 as there may be readers who've already
  // incremented optimistically
  __scoped_atomic_fetch_and(&lock, ~writerFlag, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_WRKGRP);
}

__device__ inline uintptr_t roundUp(uintptr_t ptr, uintptr_t align) {
  return ptr % align == 0 ? ptr : ptr + align - (ptr % align);
}

__device__ uint32_t getSmId() { return __nvvm_read_ptx_sreg_smid(); }

__device__ uintptr_t getThreadStateStrideBytes(GlobalState *globals) {
  auto clocksPerThread = 1u + globals->clockBufferSize;
  return sizeof(ThreadState) +
         sizeof(epoch_t) * globals->numThreads * clocksPerThread;
}

__device__ thread_id_t getDeviceThreadId(GlobalState *globals, uint32_t smid) {
  auto globalsBase = static_cast<uintptr_t>(globals->globalsBase);
  auto deviceBase = reinterpret_cast<uintptr_t>(globals);
  auto deviceIdx = (deviceBase - globalsBase) / kPerDeviceStateStride;
  return static_cast<thread_id_t>(deviceIdx * globals->numSms + smid);
}

__device__ uintptr_t getThreadStateBaseAddress(uintptr_t globalsAddr) {
  uintptr_t stateBase = globalsAddr;
  stateBase = roundUp(stateBase + sizeof(GlobalState), alignof(ThreadState));
  return stateBase;
}

__device__ ThreadState *getThreadStateById(GlobalState *globals,
                                           thread_id_t tid) {
  uint32_t deviceIdx = tid / globals->numSms;
  uint32_t smid = tid % globals->numSms;
  uintptr_t stateBase = static_cast<uintptr_t>(globals->globalsBase) +
                        deviceIdx * kPerDeviceStateStride;
  stateBase = getThreadStateBaseAddress(stateBase);
  auto stateStride = getThreadStateStrideBytes(globals);
  return reinterpret_cast<ThreadState *>(stateBase + stateStride * smid);
}

__device__ ThreadState *getThreadState(GlobalState *globals) {
  uint32_t smid = getSmId();
  uintptr_t stateBase =
      getThreadStateBaseAddress(reinterpret_cast<uintptr_t>(globals));
  auto stateStride = getThreadStateStrideBytes(globals);
  auto *state = reinterpret_cast<ThreadState *>(stateBase + stateStride * smid);

  if (state->globals == nullptr) {
    // Per-SM thread state is persistent across launches; initialize lazily.
    // Multiple threads may race here but they write identical values.
    state->reserveBase = globals->reserveBase;
    state->numReads = 0;
    state->clockBufferDirty = 0;
    state->clockBufferHead = 0;
    state->threadId = getDeviceThreadId(globals, smid);
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    state->globals = globals;
  }

  return state;
}

__device__ epoch_t *getClockBufferBase(ThreadState *state) {
  auto *globals = getGlobalState(state);
  return state->vectorClock + globals->numThreads;
}

__device__ epoch_t *getClockBufferSlot(ThreadState *state, epoch_t token,
                                       Location loc) {
  assert_msg(loc, token != 0, "Invalid GSan clock token");
  assert_msg(loc, token <= state->clockBufferHead, "Future GSan clock token");
  auto *globals = getGlobalState(state);
  assert_msg(loc, state->clockBufferHead - token < globals->clockBufferSize,
             "GSan clock buffer token overwritten");
  uint32_t slot = (token - 1) % globals->clockBufferSize;
  return getClockBufferBase(state) + slot * globals->numThreads;
}

__device__ epoch_t publishClockBuffer(ThreadState *state, Location loc) {
  auto *globals = getGlobalState(state);
  uint32_t nextHead = state->clockBufferHead + 1;
  assert_msg(loc, nextHead <= std::numeric_limits<epoch_t>::max(),
             "GSan clock buffer token overflowed");
  epoch_t *slot =
      getClockBufferBase(state) +
      ((nextHead - 1) % globals->clockBufferSize) * globals->numThreads;
  for (int i = 0; i < globals->numThreads; ++i)
    slot[i] = state->vectorClock[i];
  state->clockBufferHead = nextHead;
  state->clockBufferDirty = 0;
  return static_cast<epoch_t>(nextHead);
}

__device__ AtomicSem decodeAtomicSem(uint32_t sem) {
  switch (sem) {
  case 1:
    return AtomicSem::Relaxed;
  case 2:
    return AtomicSem::Acquire;
  case 3:
    return AtomicSem::Release;
  case 4:
    return AtomicSem::AcquireRelease;
  default:
    return AtomicSem::Relaxed;
  }
}

__device__ AtomicScope decodeAtomicScope(uint32_t scope) {
  switch (scope) {
  case 1:
    return AtomicScope::GPU;
  case 2:
    return AtomicScope::CTA;
  case 3:
    return AtomicScope::System;
  default:
    return AtomicScope::NonAtomic;
  }
}

__device__ bool hasAcquire(AtomicSem sem) {
  return sem == AtomicSem::Acquire || sem == AtomicSem::AcquireRelease;
}

__device__ bool hasRelease(AtomicSem sem) {
  return sem == AtomicSem::Release || sem == AtomicSem::AcquireRelease;
}

__device__ bool scopeCoversPair(AtomicScope scope, thread_id_t lhs,
                                thread_id_t rhs, GlobalState *globals) {
  switch (getBaseAtomicScope(scope)) {
  case AtomicScope::CTA:
    return lhs == rhs;
  case AtomicScope::GPU:
    return lhs / globals->numSms == rhs / globals->numSms;
  case AtomicScope::System:
    return true;
  case AtomicScope::NonAtomic:
  case AtomicScope::CTAToken:
  case AtomicScope::GPUToken:
  case AtomicScope::SystemToken:
    return false;
  }
  return false;
}

__device__ bool areAtomicScopesCompatible(AtomicScope lhs, thread_id_t lhsTid,
                                          AtomicScope rhs, thread_id_t rhsTid,
                                          GlobalState *globals) {
  if (!isAtomicScope(lhs) || !isAtomicScope(rhs))
    return false;
  return scopeCoversPair(lhs, lhsTid, rhsTid, globals) &&
         scopeCoversPair(rhs, lhsTid, rhsTid, globals);
}

__device__ void initThread(GlobalState *globals, Location loc) {
  auto *state = getThreadState(globals);

  if (threadIdx.x == 0) {
    auto smid = getSmId();
    auto tid = getDeviceThreadId(globals, smid);

    // Preserve the synchronized vector clock from prior launches on this
    // stream and advance the local epoch for the new kernel entry.
    auto *clock = state->vectorClock;
    assert_msg(loc, clock[tid] != std::numeric_limits<epoch_t>::max(),
               "Vector clock overflowed");
    clock[tid] += 1;
    state->clockBufferDirty = 1;
  }
}

struct Range {
  uintptr_t start;
  uintptr_t end;
};

__device__ Range roundRange(Range x) {
  // Round start down to shadow granularity
  x.start = x.start - (x.start % kShadowMemGranularityBytes);
  // Round end up to shadow granularity
  auto mod = x.end % kShadowMemGranularityBytes;
  x.end = x.end + (mod == 0 ? 0 : kShadowMemGranularityBytes - mod);
  return x;
}

__device__ ShadowCell *acquireShadow(uintptr_t shadowAddr) {
  auto cell = reinterpret_cast<ShadowCell *>(shadowAddr);
  uint16_t actual = 0;

  while (!__scoped_atomic_compare_exchange_n(&cell->lock, &actual, 1, true,
                                             __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                                             __MEMORY_SCOPE_SYSTEM)) {
    actual = 0;
  }
  return cell;
}

__device__ void releaseShadow(ShadowCell *cell) {
  __scoped_atomic_store_n(&cell->lock, 0, __ATOMIC_RELEASE,
                          __MEMORY_SCOPE_SYSTEM);
}

__device__ epoch_t appendClockBufferSnapshot(ThreadState *state,
                                             const epoch_t *snapshot,
                                             Location loc) {
  auto *globals = getGlobalState(state);
  assert_msg(loc, globals->clockBufferSize != 0,
             "GSan clock buffer size must be non-zero");
  uint32_t nextHead = state->clockBufferHead + 1;
  assert_msg(loc, nextHead <= std::numeric_limits<epoch_t>::max(),
             "GSan clock buffer token overflowed");
  epoch_t *slot =
      getClockBufferBase(state) +
      ((nextHead - 1) % globals->clockBufferSize) * globals->numThreads;
  for (int i = 0; i < globals->numThreads; ++i)
    slot[i] = snapshot[i];
  state->clockBufferHead = nextHead;
  return static_cast<epoch_t>(nextHead);
}

__device__ epoch_t publishCurrentVectorClock(ThreadState *state, Location loc) {
  auto token = appendClockBufferSnapshot(state, state->vectorClock, loc);
  state->clockBufferDirty = 0;
  return token;
}

__device__ const epoch_t *getSnapshotForWrite(ThreadState *state,
                                              const ScalarClock &write,
                                              Location loc) {
  if (!isTokenScope(write.scope))
    return nullptr;
  auto *writerState = getThreadStateById(getGlobalState(state), write.threadId);
  return getClockBufferSlot(writerState, write.epoch, loc);
}

__device__ epoch_t propagateClockBufferSnapshot(ThreadState *state,
                                                const ScalarClock &write,
                                                Location loc) {
  auto *snapshot = getSnapshotForWrite(state, write, loc);
  assert_msg(loc, snapshot != nullptr, "Invalid GSan propagated clock token");
  return appendClockBufferSnapshot(state, snapshot, loc);
}

__device__ void incrementThreadEpoch(ThreadState *state, Location loc) {
  auto tid = state->threadId;
  auto *clock = state->vectorClock;
  assert_msg(loc, clock[tid] != std::numeric_limits<epoch_t>::max(),
             "Vector clock overflowed");
  clock[tid] += 1;
  state->clockBufferDirty = 1;
}

__device__ bool dominatesSnapshot(ThreadState *state, const epoch_t *snapshot) {
  auto *globals = getGlobalState(state);
  for (int i = 0; i < globals->numThreads; ++i) {
    if (state->vectorClock[i] < snapshot[i])
      return false;
  }
  return true;
}

__device__ bool clockHappensBefore(ThreadState *state, const ScalarClock &clock,
                                   Location loc) {
  if (clock.epoch == 0)
    return true;
  if (const epoch_t *snapshot = getSnapshotForWrite(state, clock, loc))
    return dominatesSnapshot(state, snapshot);
  return state->vectorClock[clock.threadId] >= clock.epoch;
}

__device__ void assertOrderedOrCompatible(ThreadState *state,
                                          AtomicScope currentScope,
                                          const ScalarClock &prior,
                                          Location loc, const char *message) {
  if (prior.epoch == 0)
    return;
  if (isAtomicScope(currentScope) &&
      areAtomicScopesCompatible(currentScope, state->threadId, prior.scope,
                                prior.threadId, getGlobalState(state))) {
    return;
  }
  assert_msg(loc, clockHappensBefore(state, prior, loc), message);
}

__device__ void maybeMergeAcquire(ThreadState *state, AtomicScope currentScope,
                                  const ScalarClock &prior, Location loc) {
  if (!isTokenScope(prior.scope))
    return;
  if (!areAtomicScopesCompatible(currentScope, state->threadId, prior.scope,
                                 prior.threadId, getGlobalState(state))) {
    return;
  }
  auto *snapshot = getSnapshotForWrite(state, prior, loc);
  bool changed = false;
  auto *globals = getGlobalState(state);
  for (int i = 0; i < globals->numThreads; ++i) {
    if (state->vectorClock[i] < snapshot[i]) {
      state->vectorClock[i] = snapshot[i];
      changed = true;
    }
  }
  if (changed)
    state->clockBufferDirty = 1;
}

__device__ ScalarClock makeScalarClock(ThreadState *state, AtomicScope scope) {
  auto tid = state->threadId;
  return ScalarClock{state->vectorClock[tid], tid, scope};
}

__device__ ScalarClock makeTokenClock(ThreadState *state, AtomicScope scope,
                                      epoch_t token) {
  return ScalarClock{token, state->threadId, makeTokenScope(scope)};
}

__device__ void recordRead(ThreadState *state, ShadowCell *cell,
                           AtomicScope scope) {
  auto numReads = cell->numReads;
  if (numReads < std::numeric_limits<decltype(cell->numReads)>::max())
    ++cell->numReads;

  auto scalarClock = makeScalarClock(state, scope);
  for (int iRead = 0; iRead < ShadowCell::kReadClockSize; ++iRead) {
    auto readClock = cell->readClocks[iRead];
    if (readClock.threadId == state->threadId || readClock.epoch == 0) {
      cell->readClocks[iRead] = scalarClock;
      return;
    }
  }

  auto threadNumReads = __scoped_fetch_add_n(
      &state->numReads, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  auto seed = getGlobalState(state)->rngSeed;
  uint32_t rand = hash2x32(threadNumReads, state->threadId, seed);
  if ((rand >> 8) % numReads != 0)
    return;
  auto clockIdx = rand % ShadowCell::kReadClockSize;
  cell->readClocks[clockIdx] = scalarClock;
}

__device__ void doWrite(ThreadState *state, ShadowCell *cell, Location loc) {
  // Check WAR
  for (int iRead = 0; iRead < ShadowCell::kReadClockSize; ++iRead) {
    assertOrderedOrCompatible(state, AtomicScope::NonAtomic,
                              cell->readClocks[iRead], loc,
                              "Write after read race detected");
  }
  // Check WAW
  assertOrderedOrCompatible(state, AtomicScope::NonAtomic, cell->writeClock,
                            loc, "Write after write race detected");
  // Update write
  cell->writeClock = makeScalarClock(state, AtomicScope::NonAtomic);
}

__device__ void writeRange(ThreadState *state, uintptr_t write_addr, int nBytes,
                           Location loc) {
  auto range = roundRange(Range{write_addr, write_addr + nBytes});

  auto reserveBase = state->reserveBase;
  rwLockAcquireRead(state->lock);

  for (uintptr_t addr = range.start; addr < range.end;
       addr += kShadowMemGranularityBytes) {
    if (!isGsanManaged(addr, reserveBase))
      continue;
    auto shadowAddr = getShadowAddress(addr);
    auto cell = acquireShadow(shadowAddr);
    doWrite(state, cell, loc);
    releaseShadow(cell);
  }

  rwLockReleaseRead(state->lock);
}

// Handles tl.store(ptrs, values, mask)
__device__ void tensorStore(ThreadState *state, const char *stackPtr,
                            int nElems, int bytesPerElem, Location loc) {
  const uintptr_t *ptrsPtr = reinterpret_cast<const uintptr_t *>(stackPtr);
  const char *maskPtr = stackPtr + nElems * sizeof(uintptr_t);
  for (int i = 0; i < nElems; ++i) {
    auto ptr = ptrsPtr[i];
    auto mask = maskPtr[i];
    if (mask)
      writeRange(state, ptr, bytesPerElem, loc);
  }
}

__device__ void doRead(ThreadState *state, ShadowCell *cell, Location loc) {
  assertOrderedOrCompatible(state, AtomicScope::NonAtomic, cell->writeClock,
                            loc, "Read after write race detected");
  recordRead(state, cell, AtomicScope::NonAtomic);
}

__device__ void readRange(ThreadState *state, uintptr_t read_addr, int nBytes,
                          Location loc) {
  auto range = roundRange(Range{read_addr, read_addr + nBytes});

  auto reserveBase = state->reserveBase;
  rwLockAcquireRead(state->lock);

  for (uintptr_t addr = range.start; addr < range.end;
       addr += kShadowMemGranularityBytes) {
    if (!isGsanManaged(addr, reserveBase))
      continue;
    auto shadowAddr = getShadowAddress(addr);
    auto cell = acquireShadow(shadowAddr);
    doRead(state, cell, loc);
    releaseShadow(cell);
  }

  rwLockReleaseRead(state->lock);
}

// Handles tl.load(ptrs, mask)
__device__ void tensorLoad(ThreadState *state, const char *stackPtr, int nElems,
                           int bytesPerElem, Location loc) {
  const uintptr_t *ptrsPtr = reinterpret_cast<const uintptr_t *>(stackPtr);
  const char *maskPtr = stackPtr + nElems * sizeof(uintptr_t);
  for (int i = 0; i < nElems; ++i) {
    auto ptr = ptrsPtr[i];
    auto mask = maskPtr[i];
    if (mask)
      readRange(state, ptr, bytesPerElem, loc);
  }
}

__device__ void initAtomicEventState(AtomicEventState *event) {
  event->threadState = nullptr;
  event->numCells = 0;
  for (auto &cell : event->cells)
    cell = nullptr;
}

__device__ void acquireAtomicShadowRange(ThreadState *state,
                                         AtomicEventState *event,
                                         uintptr_t address, int nBytes,
                                         Location loc) {
  auto range = roundRange(Range{address, address + nBytes});
  auto reserveBase = state->reserveBase;
  uint8_t numCells = 0;
  for (uintptr_t addr = range.start; addr < range.end;
       addr += kShadowMemGranularityBytes) {
    if (isGsanManaged(addr, reserveBase))
      ++numCells;
  }
  assert_msg(loc, numCells <= kMaxAtomicShadowCells,
             "Atomic access spans too many GSan shadow cells");
  if (numCells == 0)
    return;

  // FIXME: Deadlock risk, if two concurrent accesses have different types, they
  // may partially acquire the shadow cells and block other threads from making
  // progress.
  rwLockAcquireWrite(state->lock);
  event->threadState = state;
  event->numCells = 0;
  for (uintptr_t addr = range.start; addr < range.end;
       addr += kShadowMemGranularityBytes) {
    if (!isGsanManaged(addr, reserveBase))
      continue;
    event->cells[event->numCells++] = acquireShadow(getShadowAddress(addr));
  }
}

__device__ void releaseAtomicShadowRange(AtomicEventState *event) {
  if (event->threadState == nullptr)
    return;
  for (uint8_t i = 0; i < event->numCells; ++i)
    releaseShadow(event->cells[i]);
  rwLockReleaseWrite(event->threadState->lock);
  initAtomicEventState(event);
}

__device__ void beginAtomicAccess(GlobalState *globals, AtomicEventState *event,
                                  bool pred, uintptr_t address, int nBytes,
                                  uint32_t semRaw, uint32_t scopeRaw,
                                  Location loc) {
  initAtomicEventState(event);
  if (!pred)
    return;

  auto *state = getThreadState(globals);
  acquireAtomicShadowRange(state, event, address, nBytes, loc);
  if (event->threadState == nullptr)
    return;

  auto sem = decodeAtomicSem(semRaw);
  auto scope = decodeAtomicScope(scopeRaw);
  for (uint8_t i = 0; i < event->numCells; ++i) {
    auto *cell = event->cells[i];
    auto write = cell->writeClock;
    assertOrderedOrCompatible(state, scope, write, loc,
                              "Read after write race detected");
    if (hasAcquire(sem))
      maybeMergeAcquire(state, scope, write, loc);
    recordRead(state, cell, scope);
  }
}

__device__ void endAtomicAccess(AtomicEventState *event, bool pred,
                                bool didWrite, uint32_t semRaw,
                                uint32_t scopeRaw, Location loc) {
  if (!pred || event->threadState == nullptr)
    return;

  auto *state = event->threadState;
  auto sem = decodeAtomicSem(semRaw);
  auto scope = decodeAtomicScope(scopeRaw);

  if (didWrite) {
    for (uint8_t i = 0; i < event->numCells; ++i) {
      auto *cell = event->cells[i];
      for (int iRead = 0; iRead < ShadowCell::kReadClockSize; ++iRead) {
        assertOrderedOrCompatible(state, scope, cell->readClocks[iRead], loc,
                                  "Write after read race detected");
      }
      assertOrderedOrCompatible(state, scope, cell->writeClock, loc,
                                "Write after write race detected");
    }

    ScalarClock newWriteClock;
    if (hasRelease(sem)) {
      auto token = publishCurrentVectorClock(state, loc);
      newWriteClock = makeTokenClock(state, scope, token);
    } else {
      auto previousWrite = event->cells[0]->writeClock;
      if (isTokenScope(previousWrite.scope)) {
        auto token = propagateClockBufferSnapshot(state, previousWrite, loc);
        newWriteClock = makeTokenClock(state, scope, token);
      } else {
        newWriteClock = makeScalarClock(state, scope);
      }
    }

    for (uint8_t i = 0; i < event->numCells; ++i)
      event->cells[i]->writeClock = newWriteClock;

    if (hasRelease(sem))
      incrementThreadEpoch(state, loc);
  }

  releaseAtomicShadowRange(event);
}

} // namespace
} // namespace gsan

extern "C" __device__ void
__triton_gsan_load_tensor(void *globalState, const char *stackPtr, int numElems,
                          int bytesPerElem, const char *file, unsigned line) {
  auto loc = gsan::Location{file, line};
  auto *threadState =
      gsan::getThreadState(reinterpret_cast<gsan::GlobalState *>(globalState));
  gsan::tensorLoad(threadState, stackPtr, numElems, bytesPerElem, loc);
}

extern "C" __device__ void __triton_gsan_init(void *globalState,
                                              const char *file, unsigned line) {
  auto loc = gsan::Location{file, line};
  gsan::initThread(reinterpret_cast<gsan::GlobalState *>(globalState), loc);
}

extern "C" __device__ void
__triton_gsan_store_tensor(void *globalState, const char *stackPtr,
                           int numElems, int bytesPerElem, const char *file,
                           unsigned line) {
  auto loc = gsan::Location{file, line};
  auto *threadState =
      gsan::getThreadState(reinterpret_cast<gsan::GlobalState *>(globalState));
  gsan::tensorStore(threadState, stackPtr, numElems, bytesPerElem, loc);
}

extern "C" __device__ void
__triton_gsan_atomic_begin_scalar(void *globalState, void *eventState, int pred,
                                  uintptr_t address, int bytesPerElem, int sem,
                                  int scope, const char *file, int line) {
  auto loc = gsan::Location{file, static_cast<unsigned>(line)};
  gsan::beginAtomicAccess(
      reinterpret_cast<gsan::GlobalState *>(globalState),
      reinterpret_cast<gsan::AtomicEventState *>(eventState), pred != 0,
      address, bytesPerElem, sem, scope, loc);
}

extern "C" __device__ void
__triton_gsan_atomic_end_scalar(void *eventState, int pred, int didWrite,
                                int sem, int scope, const char *file,
                                int line) {
  auto loc = gsan::Location{file, static_cast<unsigned>(line)};
  gsan::endAtomicAccess(reinterpret_cast<gsan::AtomicEventState *>(eventState),
                        pred != 0, didWrite != 0, sem, scope, loc);
}
