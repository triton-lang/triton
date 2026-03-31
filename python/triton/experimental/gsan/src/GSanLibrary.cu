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

__device__ ThreadState *getThreadState(GlobalState *globals) {
  uint32_t smid = getSmId();
  uintptr_t stateBase = reinterpret_cast<uintptr_t>(globals);
  stateBase = roundUp(stateBase + sizeof(GlobalState), alignof(ThreadState));
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

__device__ void doWrite(ThreadState *state, ShadowCell *cell, Location loc) {
  epoch_t *clock = state->vectorClock;
  // Check WAR
  for (int iRead = 0; iRead < ShadowCell::kReadClockSize; ++iRead) {
    auto read = cell->readClocks[iRead];
    assert_msg(loc, clock[read.threadId] >= read.epoch,
               "Write after read race detected");
  }
  // Check WAW
  auto write = cell->writeClock;
  assert_msg(loc, clock[write.threadId] >= write.epoch,
             "Write after write race detected");
  // Update write
  auto tid = state->threadId;
  cell->writeClock = ScalarClock{clock[tid], tid, AtomicScope::NonAtomic};
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
  // Update read count
  auto numReads = cell->numReads;
  if (numReads < std::numeric_limits<decltype(cell->numReads)>::max())
    ++cell->numReads;

  epoch_t *clock = state->vectorClock;
  // Check RAW
  auto write = cell->writeClock;
  assert_msg(loc, clock[write.threadId] >= write.epoch,
             "Read after write race detected");

  auto tid = state->threadId;
  auto scalarClock = ScalarClock{clock[tid], tid, AtomicScope::NonAtomic};
  // First, try to update in-place
  for (int iRead = 0; iRead < ShadowCell::kReadClockSize; ++iRead) {
    auto readClock = cell->readClocks[iRead];
    if (readClock.threadId == tid || readClock.epoch == 0) {
      cell->readClocks[iRead] = scalarClock;
      return;
    }
  }

  // Otherwise, do stochastic replacement
  auto threadNumReads = __scoped_atomic_fetch_add(
      &state->numReads, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  auto seed = getGlobalState(state)->rngSeed;
  uint32_t rand = hash2x32(threadNumReads, state->threadId, seed);
  if ((rand >> 8) % numReads != 0)
    return;
  auto clockIdx = rand % ShadowCell::kReadClockSize;
  cell->readClocks[clockIdx] = scalarClock;
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
