#include "GSan.h"
#include "Hash.cuh"

extern "C" GSAN_DEVICE void __assertfail(const char *assertion,
                                         const char *file, unsigned line,
                                         const char *function,
                                         __SIZE_TYPE__ charSize);

static GSAN_DEVICE inline void __assert_fail(const char *assertion,
                                             const char *file, unsigned line,
                                             const char *function) {
  __assertfail(assertion, file, line, function, sizeof(char));
}

namespace gsan {

struct Location {
  const char *file;
  unsigned line;
};

GSAN_DEVICE const char *getSourceFile(Location loc) {
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
static constexpr uint32_t kWriterFlag = 1u << 31;
static constexpr epoch_t kMaxEpoch = static_cast<epoch_t>(~0u);
static constexpr uint16_t kMaxUint16 = static_cast<uint16_t>(~0u);

enum class AtomicSem : uint8_t {
  Relaxed = 1,
  Acquire = 2,
  Release = 3,
  AcquireRelease = 4,
};

GSAN_DEVICE void rwLockAcquireRead(uint32_t &lock) {
  uint32_t old = __scoped_atomic_fetch_add(&lock, 1, __ATOMIC_ACQUIRE,
                                           __MEMORY_SCOPE_WRKGRP);
  if ((old & kWriterFlag) == 0)
    return;

  do {
    old =
        __scoped_atomic_load_n(&lock, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_WRKGRP);
  } while ((old & kWriterFlag) != 0);
}

GSAN_DEVICE void rwLockAcquireWrite(uint32_t &lock) {
  uint32_t actual = 0;
  while (!__scoped_atomic_compare_exchange_n(&lock, &actual, kWriterFlag, true,
                                             __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                                             __MEMORY_SCOPE_WRKGRP)) {
    actual = 0;
  }
}

GSAN_DEVICE void rwLockReleaseRead(uint32_t &lock) {
  __scoped_atomic_fetch_sub(&lock, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
}

GSAN_DEVICE void rwLockReleaseWrite(uint32_t &lock) {
  // Note we don't set 0 as there may be readers who've already
  // incremented optimistically
  __scoped_atomic_fetch_and(&lock, ~kWriterFlag, __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_WRKGRP);
}

GSAN_DEVICE inline uintptr_t roundUp(uintptr_t ptr, uintptr_t align) {
  return ptr % align == 0 ? ptr : ptr + align - (ptr % align);
}

GSAN_DEVICE uint32_t getSmId() { return __nvvm_read_ptx_sreg_smid(); }

GSAN_DEVICE void syncThreads(uint32_t barrierId, uint32_t numThreads) {
  asm volatile("bar.sync %0, %1;"
               :
               : "r"(barrierId), "r"(numThreads)
               : "memory");
}

GSAN_DEVICE uintptr_t getThreadStateStrideBytes(GlobalState *globals) {
  auto clocksPerThread = 1u + globals->clockBufferSize;
  return sizeof(ThreadState) +
         sizeof(epoch_t) * globals->numThreads * clocksPerThread;
}

GSAN_DEVICE thread_id_t getDeviceThreadId(GlobalState *globals, uint32_t smid) {
  auto globalsBase = static_cast<uintptr_t>(globals->globalsBase);
  auto deviceBase = reinterpret_cast<uintptr_t>(globals);
  auto deviceIdx = (deviceBase - globalsBase) / kPerDeviceStateStride;
  return static_cast<thread_id_t>(deviceIdx * globals->numSms + smid);
}

GSAN_DEVICE uintptr_t getThreadStateBaseAddress(uintptr_t globalsAddr) {
  uintptr_t stateBase = globalsAddr;
  stateBase = roundUp(stateBase + sizeof(GlobalState), alignof(ThreadState));
  return stateBase;
}

GSAN_DEVICE ThreadState *getThreadStateById(GlobalState *globals,
                                            thread_id_t tid) {
  uint32_t deviceIdx = tid / globals->numSms;
  uint32_t smid = tid % globals->numSms;
  uintptr_t stateBase = static_cast<uintptr_t>(globals->globalsBase) +
                        deviceIdx * kPerDeviceStateStride;
  stateBase = getThreadStateBaseAddress(stateBase);
  auto stateStride = getThreadStateStrideBytes(globals);
  return reinterpret_cast<ThreadState *>(stateBase + stateStride * smid);
}

GSAN_DEVICE ThreadState *getThreadState(GlobalState *globals) {
  uint32_t smid = getSmId();
  uintptr_t stateBase =
      getThreadStateBaseAddress(reinterpret_cast<uintptr_t>(globals));
  auto stateStride = getThreadStateStrideBytes(globals);
  return reinterpret_cast<ThreadState *>(stateBase + stateStride * smid);
}

GSAN_DEVICE epoch_t *getClockBufferBase(ThreadState *state) {
  auto *globals = getGlobalState(state);
  return state->vectorClock + globals->numThreads;
}

GSAN_DEVICE epoch_t *getClockBufferSlot(ThreadState *state, epoch_t token,
                                        Location loc) {
  assert_msg(loc, token != 0, "Invalid GSan clock token");
  assert_msg(loc, token <= state->clockBufferHead, "Future GSan clock token");
  auto *globals = getGlobalState(state);
  assert_msg(loc, state->clockBufferHead - token < globals->clockBufferSize,
             "GSan clock buffer token overwritten");
  uint32_t slot = token % globals->clockBufferSize;
  return getClockBufferBase(state) + slot * globals->numThreads;
}

GSAN_DEVICE epoch_t publishClockBuffer(ThreadState *state, Location loc) {
  auto *globals = getGlobalState(state);
  uint32_t nextHead = state->clockBufferHead + 1;
  assert_msg(loc, nextHead <= kMaxEpoch, "GSan clock buffer token overflowed");
  epoch_t *slot =
      getClockBufferBase(state) +
      ((nextHead - 1) % globals->clockBufferSize) * globals->numThreads;
  for (int i = 0; i < globals->numThreads; ++i)
    slot[i] = state->vectorClock[i];
  state->clockBufferHead = nextHead;
  state->clockBufferDirty = 0;
  return static_cast<epoch_t>(nextHead);
}

GSAN_DEVICE AtomicSem decodeAtomicSem(uint32_t sem) {
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
    __builtin_trap();
    return AtomicSem::Relaxed;
  }
}

GSAN_DEVICE AtomicScope decodeAtomicScope(uint32_t scope) {
  switch (scope) {
  case 1:
    return AtomicScope::GPU;
  case 2:
    return AtomicScope::CTA;
  case 3:
    return AtomicScope::System;
  default:
    __builtin_trap();
    return AtomicScope::NonAtomic;
  }
}

GSAN_DEVICE bool hasAcquire(AtomicSem sem) {
  return sem == AtomicSem::Acquire || sem == AtomicSem::AcquireRelease;
}

GSAN_DEVICE bool hasRelease(AtomicSem sem) {
  return sem == AtomicSem::Release || sem == AtomicSem::AcquireRelease;
}

GSAN_DEVICE bool scopeCoversPair(AtomicScope scope, thread_id_t lhs,
                                 thread_id_t rhs, GlobalState *globals) {
  switch (scope) {
  case AtomicScope::CTA:
    return lhs == rhs;
  case AtomicScope::GPU:
    return lhs / globals->numSms == rhs / globals->numSms;
  case AtomicScope::System:
    return true;
  case AtomicScope::NonAtomic:
    return false;
  }
  return false;
}

GSAN_DEVICE bool areAtomicScopesCompatible(AtomicScope lhs, thread_id_t lhsTid,
                                           AtomicScope rhs, thread_id_t rhsTid,
                                           GlobalState *globals) {
  if (!isAtomicScope(lhs) || !isAtomicScope(rhs))
    return false;
  return scopeCoversPair(lhs, lhsTid, rhsTid, globals) &&
         scopeCoversPair(rhs, lhsTid, rhsTid, globals);
}

GSAN_DEVICE bool canAccumulateReleaseRmw(ThreadState *state, AtomicScope scope,
                                         const ScalarClock &previousWrite) {
  // A propagated snapshot currently carries only one scope. Only combine
  // concurrent releases when that scope remains unchanged.
  if (!previousWrite.isRelease || scope != previousWrite.scope)
    return false;
  return areAtomicScopesCompatible(scope, state->threadId, previousWrite.scope,
                                   previousWrite.threadId,
                                   getGlobalState(state));
}

GSAN_DEVICE void acquireStreamClock(ThreadState *state,
                                    const uint32_t *streamClock,
                                    uint32_t threadIdx, uint32_t numThreads,
                                    uint32_t barrierId) {
  syncThreads(barrierId, numThreads);
  if (threadIdx == 0)
    rwLockAcquireWrite(state->lock);
  syncThreads(barrierId, numThreads);

  auto *globals = getGlobalState(state);
  for (int i = threadIdx; i < globals->numThreads; i += numThreads) {
    auto epoch = streamClock[i];
    if (state->vectorClock[i] < epoch)
      state->vectorClock[i] = static_cast<epoch_t>(epoch);
  }
  if (threadIdx == 0) {
    state->clockBufferDirty = 1;
    state->gdcWaitCalled = 1;
  }

  syncThreads(barrierId, numThreads);
  if (threadIdx == 0)
    rwLockReleaseWrite(state->lock);
}

GSAN_DEVICE void publishStreamClock(ThreadState *state, uint32_t *streamClock,
                                    uint32_t threadIdx, uint32_t numThreads,
                                    uint32_t barrierId, Location loc) {
  syncThreads(barrierId, numThreads);
  if (threadIdx == 0) {
    rwLockAcquireRead(state->lock);
    assert_msg(loc, state->gdcWaitCalled,
               "kernel launched with programmatic dependent launch did not "
               "call gdc_wait");
  }
  syncThreads(barrierId, numThreads);

  auto *globals = getGlobalState(state);
  for (int i = threadIdx; i < globals->numThreads; i += numThreads) {
    uint32_t current = state->vectorClock[i];
    asm volatile("red.relaxed.gpu.max.u32 [%0], %1;"
                 :
                 : "l"(streamClock + i), "r"(current)
                 : "memory");
  }

  syncThreads(barrierId, numThreads);
  if (threadIdx == 0)
    rwLockReleaseRead(state->lock);
}

GSAN_DEVICE uint32_t *getStreamClock(uint32_t *streamClocks,
                                     __UINT64_TYPE__ kernelId, unsigned offset,
                                     GlobalState *globals) {
  return streamClocks + ((kernelId + offset) % 3) * globals->numThreads;
}

GSAN_DEVICE void initThread(GlobalState *globals, uint32_t *streamClocks,
                            __UINT64_TYPE__ kernelId, bool acquirePrevious,
                            uint32_t threadIdx, uint32_t numThreads,
                            uint32_t barrierId, Location loc) {
  auto *state = getThreadState(globals);
  if (threadIdx == 0) {
    rwLockAcquireWrite(state->lock);
    auto st_globals = __scoped_atomic_load_n(&state->globals, __ATOMIC_ACQUIRE,
                                             __MEMORY_SCOPE_DEVICE);
    if (st_globals == nullptr) {
      // Lazily initialize per-SM thread state
      state->reserveBase = globals->reserveBase;
      state->numReads = 0;
      state->clockBufferDirty = 0;
      state->gdcWaitCalled = 0;
      state->clockBufferHead = 0;
      state->threadId = getDeviceThreadId(globals, getSmId());
      __scoped_atomic_store_n(&state->globals, globals, __ATOMIC_RELEASE,
                              __MEMORY_SCOPE_DEVICE);
    }
    state->gdcWaitCalled = acquirePrevious;
  }
  syncThreads(barrierId, numThreads);

  // A dependent grid can begin once its predecessor has signaled, but the
  // grid two launches back has completed by then. Replace all non-local clock
  // entries so persistent SM state cannot acquire the predecessor implicitly.
  auto *streamClock =
      getStreamClock(streamClocks, kernelId, acquirePrevious ? 2 : 1, globals);
  auto tid = state->threadId;
  for (int i = threadIdx; i < globals->numThreads; i += numThreads) {
    if (i == tid)
      continue;
    auto epoch = streamClock[i];
    state->vectorClock[i] = static_cast<epoch_t>(epoch);
  }

  if (threadIdx == 0) {
    // Preserve the synchronized vector clock from prior launches on this
    // stream and advance the local epoch for the new kernel entry.
    auto *clock = state->vectorClock;
    assert_msg(loc, clock[tid] != kMaxEpoch, "Vector clock overflowed");
    clock[tid] += 1;
    state->clockBufferDirty = 1;
  }

  syncThreads(barrierId, numThreads);
  if (threadIdx == 0)
    rwLockReleaseWrite(state->lock);
}

struct Range {
  uintptr_t start;
  uintptr_t end;
};

GSAN_DEVICE Range roundRange(Range x) {
  // Round start down to shadow granularity
  x.start = x.start - (x.start % kShadowMemGranularityBytes);
  // Round end up to shadow granularity
  auto mod = x.end % kShadowMemGranularityBytes;
  x.end = x.end + (mod == 0 ? 0 : kShadowMemGranularityBytes - mod);
  return x;
}

GSAN_DEVICE uint8_t getAccessMask(Range access, uintptr_t cellAddress) {
  uintptr_t cellEnd = cellAddress + kShadowMemGranularityBytes;
  uint32_t firstByte =
      access.start > cellAddress ? access.start - cellAddress : 0;
  uint32_t lastByte = access.end < cellEnd ? access.end - cellAddress
                                           : kShadowMemGranularityBytes;
  return static_cast<uint8_t>((1u << lastByte) - (1u << firstByte));
}

GSAN_DEVICE bool maskIncludesByte(uint8_t mask, int byte) {
  return (mask & (1u << byte)) != 0;
}

GSAN_DEVICE ShadowCell *acquireShadow(uintptr_t shadowAddr) {
  auto cell = reinterpret_cast<ShadowCell *>(shadowAddr);
  uint16_t actual = 0;

  while (!__scoped_atomic_compare_exchange_n(&cell->lock, &actual, 1, true,
                                             __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
                                             __MEMORY_SCOPE_SYSTEM)) {
    actual = 0;
  }
  return cell;
}

GSAN_DEVICE void releaseShadow(ShadowCell *cell) {
  __scoped_atomic_store_n(&cell->lock, 0, __ATOMIC_RELEASE,
                          __MEMORY_SCOPE_SYSTEM);
}

GSAN_DEVICE epoch_t appendClockBufferSnapshot(ThreadState *state,
                                              const epoch_t *snapshot,
                                              Location loc) {
  auto *globals = getGlobalState(state);
  assert_msg(loc, globals->clockBufferSize != 0,
             "GSan clock buffer size must be non-zero");
  uint32_t curHead = state->clockBufferHead;
  uint32_t nextHead = curHead + 1;
  assert_msg(loc, nextHead <= kMaxEpoch, "GSan clock buffer token overflowed");
  epoch_t *slot = getClockBufferBase(state) +
                  (nextHead % globals->clockBufferSize) * globals->numThreads;
  for (int i = 0; i < globals->numThreads; ++i)
    slot[i] = snapshot[i];
  state->clockBufferHead = nextHead;
  return static_cast<epoch_t>(nextHead);
}

GSAN_DEVICE epoch_t publishCurrentVectorClock(ThreadState *state,
                                              Location loc) {
  if (state->clockBufferDirty) {
    auto token = appendClockBufferSnapshot(state, state->vectorClock, loc);
    state->clockBufferDirty = 0;
    return token;
  }
  return state->clockBufferHead;
}

GSAN_DEVICE const epoch_t *getSnapshotForWrite(ThreadState *state,
                                               const ScalarClock &write,
                                               Location loc) {
  if (!write.isRelease)
    return nullptr;
  auto *writerState = getThreadStateById(getGlobalState(state), write.threadId);
  return getClockBufferSlot(writerState, write.epoch, loc);
}

GSAN_DEVICE epoch_t publishCurrentVectorClockWithPriorRelease(
    ThreadState *state, const ScalarClock &previousWrite, Location loc) {
  const auto *previousSnapshot = getSnapshotForWrite(state, previousWrite, loc);
  auto *globals = getGlobalState(state);
  assert_msg(loc, globals->clockBufferSize != 0,
             "GSan clock buffer size must be non-zero");
  uint32_t nextHead = state->clockBufferHead + 1;
  assert_msg(loc, nextHead <= kMaxEpoch, "GSan clock buffer token overflowed");
  auto *slot = getClockBufferBase(state) +
               (nextHead % globals->clockBufferSize) * globals->numThreads;
  for (int i = 0; i < globals->numThreads; ++i) {
    auto current = state->vectorClock[i];
    auto previous = previousSnapshot[i];
    slot[i] = current > previous ? current : previous;
  }
  state->clockBufferHead = nextHead;
  // The joined snapshot extends the release sequence without acquiring the
  // prior writer into this thread's vector clock.
  state->clockBufferDirty = 1;
  return static_cast<epoch_t>(nextHead);
}

GSAN_DEVICE epoch_t propagateClockBufferSnapshot(ThreadState *state,
                                                 const ScalarClock &write,
                                                 Location loc) {
  auto *snapshot = getSnapshotForWrite(state, write, loc);
  assert_msg(loc, snapshot != nullptr, "Invalid GSan propagated clock token");
  auto token = appendClockBufferSnapshot(state, snapshot, loc);
  state->clockBufferDirty = 1;
  return token;
}

GSAN_DEVICE void incrementThreadEpoch(ThreadState *state, Location loc) {
  auto tid = state->threadId;
  auto *clock = state->vectorClock;
  assert_msg(loc, clock[tid] != kMaxEpoch, "Vector clock overflowed");
  clock[tid] += 1;
  state->clockBufferDirty = 1;
}

GSAN_DEVICE bool dominatesSnapshot(ThreadState *state,
                                   const epoch_t *snapshot) {
  auto *globals = getGlobalState(state);
  for (int i = 0; i < globals->numThreads; ++i) {
    if (state->vectorClock[i] < snapshot[i])
      return false;
  }
  return true;
}

GSAN_DEVICE bool clockHappensBefore(ThreadState *state,
                                    const ScalarClock &clock, Location loc) {
  if (clock.epoch == 0)
    return true;
  if (const epoch_t *snapshot = getSnapshotForWrite(state, clock, loc))
    return dominatesSnapshot(state, snapshot);
  return state->vectorClock[clock.threadId] >= clock.epoch;
}

GSAN_DEVICE void assertOrderedOrCompatible(ThreadState *state,
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

GSAN_DEVICE void maybeMergeAcquire(ThreadState *state, AtomicScope currentScope,
                                   const ScalarClock &prior, Location loc) {
  if (!prior.isRelease)
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

GSAN_DEVICE ScalarClock makeScalarClock(ThreadState *state, AtomicScope scope) {
  auto tid = state->threadId;
  return ScalarClock{state->vectorClock[tid], tid, scope, false};
}

GSAN_DEVICE ScalarClock makePublishedClock(ThreadState *state,
                                           AtomicScope scope, epoch_t token) {
  return ScalarClock{token, state->threadId, scope, true};
}

GSAN_DEVICE bool clocksEqual(const ScalarClock &lhs, const ScalarClock &rhs) {
  return lhs.epoch == rhs.epoch && lhs.threadId == rhs.threadId &&
         lhs.scope == rhs.scope && lhs.isRelease == rhs.isRelease;
}

GSAN_DEVICE bool isFirstWriteClock(const ShadowCell *cell, uint8_t mask,
                                   int byte) {
  for (int previousByte = 0; previousByte < byte; ++previousByte) {
    if (maskIncludesByte(mask, previousByte) &&
        clocksEqual(cell->writeClocks[byte], cell->writeClocks[previousByte]))
      return false;
  }
  return true;
}

GSAN_DEVICE void replaceReadClock(ThreadState *state, ShadowCell *cell,
                                  int index, ScalarClock clock, uint8_t mask) {
  for (int i = 0; i < ShadowCell::kReadClockSize; ++i) {
    if (i == index)
      continue;
    auto &prior = cell->readClocks[i];
    if (prior.threadId == state->threadId && prior.scope == clock.scope) {
      cell->readMasks[i] &= static_cast<uint8_t>(~mask);
      if (cell->readMasks[i] == 0)
        prior = ScalarClock{};
    }
  }
  cell->readClocks[index] = clock;
  cell->readMasks[index] = mask;
}

GSAN_DEVICE void recordRead(ThreadState *state, ShadowCell *cell,
                            AtomicScope scope, uint8_t mask) {
  auto numReads = cell->numReads;
  if (numReads < kMaxUint16)
    ++cell->numReads;

  auto scalarClock = makeScalarClock(state, scope);
  int reusableIndex = -1;
  int emptyIndex = -1;
  for (int iRead = 0; iRead < ShadowCell::kReadClockSize; ++iRead) {
    auto readClock = cell->readClocks[iRead];
    auto readMask = cell->readMasks[iRead];
    if (readClock.epoch == 0 || readMask == 0) {
      if (emptyIndex == -1)
        emptyIndex = iRead;
      continue;
    }
    if (clocksEqual(readClock, scalarClock)) {
      replaceReadClock(state, cell, iRead, scalarClock, readMask | mask);
      return;
    }
    if (readClock.threadId == state->threadId && readClock.scope == scope &&
        (readMask & static_cast<uint8_t>(~mask)) == 0)
      reusableIndex = iRead;
  }

  if (reusableIndex != -1 || emptyIndex != -1) {
    auto index = reusableIndex != -1 ? reusableIndex : emptyIndex;
    replaceReadClock(state, cell, index, scalarClock, mask);
    return;
  }

  auto threadNumReads = __scoped_atomic_fetch_add(
      &state->numReads, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  auto seed = getGlobalState(state)->rngSeed;
  uint32_t rand = hash2x32(threadNumReads, state->threadId, seed);
  rand = rand % numReads;
  if (rand < ShadowCell::kReadClockSize)
    replaceReadClock(state, cell, rand, scalarClock, mask);
}

GSAN_DEVICE void doWrite(ThreadState *state, ShadowCell *cell, uint8_t mask,
                         Location loc) {
  // Check WAR
  for (int iRead = 0; iRead < ShadowCell::kReadClockSize; ++iRead) {
    if ((cell->readMasks[iRead] & mask) == 0)
      continue;
    assertOrderedOrCompatible(state, AtomicScope::NonAtomic,
                              cell->readClocks[iRead], loc,
                              "Write after read race detected");
  }
  // Check WAW
  for (int byte = 0; byte < kShadowMemGranularityBytes; ++byte) {
    if (maskIncludesByte(mask, byte) && isFirstWriteClock(cell, mask, byte)) {
      assertOrderedOrCompatible(state, AtomicScope::NonAtomic,
                                cell->writeClocks[byte], loc,
                                "Write after write race detected");
    }
  }
  // Update write
  auto clock = makeScalarClock(state, AtomicScope::NonAtomic);
  for (int byte = 0; byte < kShadowMemGranularityBytes; ++byte) {
    if (maskIncludesByte(mask, byte))
      cell->writeClocks[byte] = clock;
  }
}

GSAN_DEVICE void writeRange(ThreadState *state, uintptr_t write_addr,
                            int nBytes, Location loc) {
  Range access{write_addr, write_addr + nBytes};
  auto range = roundRange(access);

  auto reserveBase = state->reserveBase;

  for (uintptr_t addr = range.start; addr < range.end;
       addr += kShadowMemGranularityBytes) {
    if (!isGsanManaged(addr, reserveBase))
      continue;
    auto shadowAddr = getShadowAddress(addr);
    auto cell = acquireShadow(shadowAddr);
    doWrite(state, cell, getAccessMask(access, addr), loc);
    releaseShadow(cell);
  }
}

// Handles tl.store(ptrs, values, mask)
GSAN_DEVICE void tensorStore(ThreadState *state, const char *stackPtr,
                             int nElems, int bytesPerElem, Location loc) {
  const uintptr_t *ptrsPtr = reinterpret_cast<const uintptr_t *>(stackPtr);
  const char *maskPtr = stackPtr + nElems * sizeof(uintptr_t);
  bool acquired = false;
  for (int i = 0; i < nElems; ++i) {
    auto ptr = ptrsPtr[i];
    auto mask = maskPtr[i];
    if (mask) {
      if (!acquired) {
        rwLockAcquireRead(state->lock);
        acquired = true;
      }
      writeRange(state, ptr, bytesPerElem, loc);
    }
  }
  if (acquired)
    rwLockReleaseRead(state->lock);
}

GSAN_DEVICE void doRead(ThreadState *state, ShadowCell *cell, uint8_t mask,
                        Location loc) {
  for (int byte = 0; byte < kShadowMemGranularityBytes; ++byte) {
    if (maskIncludesByte(mask, byte) && isFirstWriteClock(cell, mask, byte)) {
      assertOrderedOrCompatible(state, AtomicScope::NonAtomic,
                                cell->writeClocks[byte], loc,
                                "Read after write race detected");
    }
  }
  recordRead(state, cell, AtomicScope::NonAtomic, mask);
}

GSAN_DEVICE void readRange(ThreadState *state, uintptr_t read_addr, int nBytes,
                           Location loc) {
  Range access{read_addr, read_addr + nBytes};
  auto range = roundRange(access);

  auto reserveBase = state->reserveBase;
  if (range.start >= reserveBase + kReserveSize || reserveBase >= range.end)
    return;

  for (uintptr_t addr = range.start; addr < range.end;
       addr += kShadowMemGranularityBytes) {
    if (!isGsanManaged(addr, reserveBase))
      continue;
    auto shadowAddr = getShadowAddress(addr);
    auto cell = acquireShadow(shadowAddr);
    doRead(state, cell, getAccessMask(access, addr), loc);
    releaseShadow(cell);
  }
}

// Handles tl.load(ptrs, mask)
GSAN_DEVICE void tensorLoad(ThreadState *state, const char *stackPtr,
                            int nElems, int bytesPerElem, Location loc) {
  const uintptr_t *ptrsPtr = reinterpret_cast<const uintptr_t *>(stackPtr);
  const char *maskPtr = stackPtr + nElems * sizeof(uintptr_t);
  bool acquired = false;
  for (int i = 0; i < nElems; ++i) {
    auto ptr = ptrsPtr[i];
    auto mask = maskPtr[i];
    if (mask) {
      if (!acquired) {
        rwLockAcquireRead(state->lock);
        acquired = true;
      }
      readRange(state, ptr, bytesPerElem, loc);
    }
  }
  if (acquired)
    rwLockReleaseRead(state->lock);
}

GSAN_DEVICE void initAtomicEventState(AtomicEventState *event) {
  event->threadState = nullptr;
  event->numCells = 0;
  for (int i = 0; i < kMaxAtomicShadowCells; ++i) {
    event->cells[i] = nullptr;
    event->masks[i] = 0;
  }
}

GSAN_DEVICE void acquireAtomicShadowRange(ThreadState *state,
                                          AtomicEventState *event,
                                          uintptr_t address, int nBytes,
                                          Location loc) {
  Range access{address, address + nBytes};
  auto range = roundRange(access);
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

  // FIXME: Deadlock risk. If two concurrent accesses have different types, they
  // may partially acquire the shadow cells and block other threads from making
  // progress.
  rwLockAcquireWrite(state->lock);
  event->threadState = state;
  event->numCells = 0;
  for (uintptr_t addr = range.start; addr < range.end;
       addr += kShadowMemGranularityBytes) {
    if (!isGsanManaged(addr, reserveBase))
      continue;
    auto index = event->numCells++;
    event->cells[index] = acquireShadow(getShadowAddress(addr));
    event->masks[index] = getAccessMask(access, addr);
  }
}

GSAN_DEVICE void releaseAtomicShadowRange(AtomicEventState *event) {
  if (event->threadState == nullptr)
    return;
  for (uint8_t i = 0; i < event->numCells; ++i)
    releaseShadow(event->cells[i]);
  rwLockReleaseWrite(event->threadState->lock);
  initAtomicEventState(event);
}

GSAN_DEVICE bool isFirstAtomicWriteClock(const AtomicEventState *event,
                                         uint8_t cellIndex, int byteIndex) {
  auto clock = event->cells[cellIndex]->writeClocks[byteIndex];
  for (uint8_t previousCell = 0; previousCell <= cellIndex; ++previousCell) {
    int byteLimit =
        previousCell == cellIndex ? byteIndex : kShadowMemGranularityBytes;
    for (int previousByte = 0; previousByte < byteLimit; ++previousByte) {
      if (maskIncludesByte(event->masks[previousCell], previousByte) &&
          clocksEqual(clock,
                      event->cells[previousCell]->writeClocks[previousByte]))
        return false;
    }
  }
  return true;
}

GSAN_DEVICE epoch_t publishMixedAtomicReleaseSnapshots(
    ThreadState *state, const AtomicEventState *event, bool includeCurrent,
    Location loc) {
  auto *globals = getGlobalState(state);
  assert_msg(loc, globals->clockBufferSize != 0,
             "GSan clock buffer size must be non-zero");
  uint32_t nextHead = state->clockBufferHead + 1;
  assert_msg(loc, nextHead <= kMaxEpoch, "GSan clock buffer token overflowed");
  auto *slot = getClockBufferBase(state) +
               (nextHead % globals->clockBufferSize) * globals->numThreads;
  for (int i = 0; i < globals->numThreads; ++i) {
    epoch_t epoch = includeCurrent ? state->vectorClock[i] : 0;
    for (uint8_t cellIndex = 0; cellIndex < event->numCells; ++cellIndex) {
      auto *cell = event->cells[cellIndex];
      for (int byte = 0; byte < kShadowMemGranularityBytes; ++byte) {
        if (!maskIncludesByte(event->masks[cellIndex], byte) ||
            !isFirstAtomicWriteClock(event, cellIndex, byte))
          continue;
        auto write = cell->writeClocks[byte];
        if (!write.isRelease)
          continue;
        auto *snapshot = getSnapshotForWrite(state, write, loc);
        if (epoch < snapshot[i])
          epoch = snapshot[i];
      }
    }
    slot[i] = epoch;
  }
  state->clockBufferHead = nextHead;
  // The propagated snapshot must not become this thread's acquired clock.
  state->clockBufferDirty = 1;
  return static_cast<epoch_t>(nextHead);
}

GSAN_DEVICE ScalarClock makeAtomicWriteClock(ThreadState *state,
                                             const AtomicEventState *event,
                                             AtomicSem sem, AtomicScope scope,
                                             Location loc) {
  bool isRelease = hasRelease(sem);
  bool canAccumulate = false;
  int numReleases = 0;
  ScalarClock previousRelease{};
  for (uint8_t cellIndex = 0; cellIndex < event->numCells; ++cellIndex) {
    auto *cell = event->cells[cellIndex];
    for (int byte = 0; byte < kShadowMemGranularityBytes; ++byte) {
      if (!maskIncludesByte(event->masks[cellIndex], byte) ||
          !isFirstAtomicWriteClock(event, cellIndex, byte))
        continue;
      auto write = cell->writeClocks[byte];
      if (!write.isRelease)
        continue;
      if (numReleases++ == 0)
        previousRelease = write;
      if (isRelease && canAccumulateReleaseRmw(state, scope, write)) {
        canAccumulate = true;
      } else if (isRelease) {
        auto *snapshot = getSnapshotForWrite(state, write, loc);
        assert_msg(loc, dominatesSnapshot(state, snapshot),
                   "GSan detected atomic release accumulation with mixed "
                   "scopes, which is not supported.");
      }
    }
  }

  if (!isRelease && numReleases == 0)
    return makeScalarClock(state, scope);

  epoch_t token;
  if (isRelease && !canAccumulate) {
    token = publishCurrentVectorClock(state, loc);
  } else if (numReleases == 1) {
    token = isRelease
                ? publishCurrentVectorClockWithPriorRelease(
                      state, previousRelease, loc)
                : propagateClockBufferSnapshot(state, previousRelease, loc);
  } else {
    token = publishMixedAtomicReleaseSnapshots(state, event, isRelease, loc);
  }
  return makePublishedClock(state, scope, token);
}

GSAN_DEVICE void beginAtomicAccess(GlobalState *globals,
                                   AtomicEventState *event, bool pred,
                                   uintptr_t address, int nBytes,
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
    for (int byte = 0; byte < kShadowMemGranularityBytes; ++byte) {
      if (maskIncludesByte(event->masks[i], byte) &&
          isFirstAtomicWriteClock(event, i, byte)) {
        assertOrderedOrCompatible(state, scope, cell->writeClocks[byte], loc,
                                  "Read after write race detected");
      }
    }
    recordRead(state, cell, scope, event->masks[i]);
  }
  if (hasAcquire(sem)) {
    for (uint8_t i = 0; i < event->numCells; ++i) {
      auto *cell = event->cells[i];
      for (int byte = 0; byte < kShadowMemGranularityBytes; ++byte) {
        if (maskIncludesByte(event->masks[i], byte) &&
            isFirstAtomicWriteClock(event, i, byte)) {
          maybeMergeAcquire(state, scope, cell->writeClocks[byte], loc);
        }
      }
    }
  }
}

GSAN_DEVICE void endAtomicAccess(AtomicEventState *event, bool pred,
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
        if ((cell->readMasks[iRead] & event->masks[i]) != 0) {
          assertOrderedOrCompatible(state, scope, cell->readClocks[iRead], loc,
                                    "Write after read race detected");
        }
      }
      for (int byte = 0; byte < kShadowMemGranularityBytes; ++byte) {
        if (maskIncludesByte(event->masks[i], byte) &&
            isFirstAtomicWriteClock(event, i, byte)) {
          assertOrderedOrCompatible(state, scope, cell->writeClocks[byte], loc,
                                    "Write after write race detected");
        }
      }
    }

    auto newWriteClock = makeAtomicWriteClock(state, event, sem, scope, loc);
    for (uint8_t i = 0; i < event->numCells; ++i) {
      auto *cell = event->cells[i];
      for (int byte = 0; byte < kShadowMemGranularityBytes; ++byte) {
        if (maskIncludesByte(event->masks[i], byte))
          cell->writeClocks[byte] = newWriteClock;
      }
    }

    if (hasRelease(sem))
      incrementThreadEpoch(state, loc);
  }

  releaseAtomicShadowRange(event);
}

} // namespace
} // namespace gsan

extern "C" GSAN_DEVICE void
__triton_gsan_load_tensor(void *globalState, const char *stackPtr, int numElems,
                          int bytesPerElem, const char *file, unsigned line) {
  auto loc = gsan::Location{file, line};
  auto *threadState =
      gsan::getThreadState(reinterpret_cast<gsan::GlobalState *>(globalState));
  gsan::tensorLoad(threadState, stackPtr, numElems, bytesPerElem, loc);
}

extern "C" GSAN_DEVICE void
__triton_gsan_init(void *globalState, gsan::uint32_t *streamClocks,
                   __UINT64_TYPE__ kernelId, int acquirePrevious,
                   unsigned threadIdx, unsigned numThreads, unsigned barrierId,
                   const char *file, unsigned line) {
  auto loc = gsan::Location{file, line};
  gsan::initThread(reinterpret_cast<gsan::GlobalState *>(globalState),
                   streamClocks, kernelId, acquirePrevious != 0, threadIdx,
                   numThreads, barrierId, loc);
}

extern "C" GSAN_DEVICE void
__triton_gsan_kernel_exit(void *globalState, gsan::uint32_t *streamClocks,
                          __UINT64_TYPE__ kernelId, unsigned threadIdx,
                          unsigned numThreads, unsigned barrierId,
                          const char *file, unsigned line) {
  auto loc = gsan::Location{file, line};
  auto *globals = reinterpret_cast<gsan::GlobalState *>(globalState);
  auto *state = gsan::getThreadState(globals);
  gsan::publishStreamClock(
      state, gsan::getStreamClock(streamClocks, kernelId, 0, globals),
      threadIdx, numThreads, barrierId, loc);
}

extern "C" GSAN_DEVICE void __triton_gsan_grid_dependency_wait(
    void *globalState, gsan::uint32_t *streamClocks, __UINT64_TYPE__ kernelId,
    unsigned threadIdx, unsigned numThreads, unsigned barrierId) {
  auto *globals = reinterpret_cast<gsan::GlobalState *>(globalState);
  auto *state = gsan::getThreadState(globals);
  gsan::acquireStreamClock(
      state, gsan::getStreamClock(streamClocks, kernelId, 2, globals),
      threadIdx, numThreads, barrierId);
}

extern "C" GSAN_DEVICE void
__triton_gsan_store_tensor(void *globalState, const char *stackPtr,
                           int numElems, int bytesPerElem, const char *file,
                           unsigned line) {
  auto loc = gsan::Location{file, line};
  auto *threadState =
      gsan::getThreadState(reinterpret_cast<gsan::GlobalState *>(globalState));
  gsan::tensorStore(threadState, stackPtr, numElems, bytesPerElem, loc);
}

extern "C" GSAN_DEVICE void
__triton_gsan_atomic_tensor(void *globalState, const char *stackPtr,
                            int numElems, int bytesPerElem, int sem, int scope,
                            const char *file, unsigned line) {
  auto loc = gsan::Location{file, line};
  auto *globals = reinterpret_cast<gsan::GlobalState *>(globalState);
  const auto *ptrsPtr = reinterpret_cast<const gsan::uintptr_t *>(stackPtr);
  const auto *maskPtr = stackPtr + numElems * sizeof(gsan::uintptr_t);

  for (int i = 0; i < numElems; ++i) {
    if (!maskPtr[i])
      continue;
    gsan::AtomicEventState event;
    gsan::beginAtomicAccess(globals, &event, /*pred=*/true, ptrsPtr[i],
                            bytesPerElem, sem, scope, loc);
    gsan::endAtomicAccess(&event, /*pred=*/true, /*didWrite=*/true, sem, scope,
                          loc);
  }
}

extern "C" GSAN_DEVICE void __triton_gsan_atomic_begin_scalar(
    void *globalState, void *eventState, int pred, gsan::uintptr_t address,
    int bytesPerElem, int sem, int scope, const char *file, unsigned line) {
  auto loc = gsan::Location{file, line};
  gsan::beginAtomicAccess(
      reinterpret_cast<gsan::GlobalState *>(globalState),
      reinterpret_cast<gsan::AtomicEventState *>(eventState), pred != 0,
      address, bytesPerElem, sem, scope, loc);
}

extern "C" GSAN_DEVICE void
__triton_gsan_atomic_end_scalar(void *eventState, int pred, int didWrite,
                                int sem, int scope, const char *file,
                                unsigned line) {
  auto loc = gsan::Location{file, line};
  gsan::endAtomicAccess(reinterpret_cast<gsan::AtomicEventState *>(eventState),
                        pred != 0, didWrite != 0, sem, scope, loc);
}
