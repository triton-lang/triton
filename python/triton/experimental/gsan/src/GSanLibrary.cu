#include "GSan.h"
#include "Hash.cuh"

#include <cstdint>
#include <cuda/atomic>
#include <device_launch_parameters.h>
#include <limits>

// HACK: Because we aren't compiling as cuda, clang doesn't provide these
// automatically.
extern "C" void __assertfail(const char *__message, const char *__file,
                             unsigned __line, const char *__function);
void __assert_fail(const char *__message, const char *__file, unsigned __line,
                   const char *__function) {
  __assertfail(__message, __file, __line, __function);
}

namespace gsan {
namespace {
static constexpr uint32_t writerFlag = 1u << 31;

__device__ void rwLockAcquireRead(uint32_t &lock) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_block> atom{lock};
  auto old = atom.fetch_add(1, cuda::memory_order_acquire);
  if ((old & writerFlag) == 0)
    return;

  do {
    old = atom.load(cuda::memory_order_acquire);
  } while ((old & writerFlag) != 0);
}

__device__ void rwLockAcquireWrite(uint32_t &lock) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_block> atom{lock};

  uint32_t actual = 0;
  while (!atom.compare_exchange_weak(actual, writerFlag,
                                     cuda::memory_order_acquire,
                                     cuda::memory_order_relaxed)) {
    actual = 0;
  }
}

__device__ void rwLockReleaseRead(uint32_t &lock) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_block> atom{lock};
  atom.fetch_sub(1, cuda::memory_order_relaxed);
}

__device__ void rwLockReleaseWrite(uint32_t &lock) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_block> atom{lock};
  // Note we don't set 0 as there may be readers who've already
  // incremented optimistically
  atom.fetch_and(~writerFlag, cuda::memory_order_release);
}

inline uintptr_t roundUp(uintptr_t ptr, uintptr_t align) {
  return ptr % align == 0 ? ptr : ptr + align - (ptr % align);
}

__device__ uint32_t getSmId() {
  uint32_t smid;
  asm("mov.b32 %0, %%smid;" : "=r"(smid));
  return smid;
}

__device__ void ctaBarrier() { asm volatile("bar.sync 0;" : : : "memory"); }

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
    cuda::atomic_thread_fence(cuda::memory_order_release,
                              cuda::thread_scope_device);
    state->globals = globals;
  }

  return state;
}

__device__ void initThread(GlobalState *globals) {
  auto *state = getThreadState(globals);

  if (threadIdx.x == 0) {
    auto smid = getSmId();
    auto tid = getDeviceThreadId(globals, smid);

    // Model each kernel entry as an independent logical thread with a fresh
    // vector clock. This allows sequential unsynchronized kernels to produce
    // RAW/WAR/WAW violations deterministically.
    auto *clock = state->vectorClock;
    for (int i = 0; i < globals->numThreads; ++i)
      if (i != tid)
        clock[i] = 0;

    clock[tid] += 1;
  }

  ctaBarrier();
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

  cuda::atomic_ref<uint16_t, cuda::thread_scope_system> lock(cell->lock);

  while (!lock.compare_exchange_weak(actual, 1, cuda::memory_order_acquire,
                                     cuda::memory_order_relaxed)) {
    actual = 0;
  }
  return cell;
}

__device__ void releaseShadow(ShadowCell *cell) {
  cuda::atomic_ref<uint16_t, cuda::thread_scope_system> lock(cell->lock);
  lock.store(0, cuda::memory_order_release);
}

__device__ void doWrite(ThreadState *state, ShadowCell *cell) {
  epoch_t *clock = state->vectorClock;
  // Check WAR
  for (int iRead = 0; iRead < ShadowCell::kReadClockSize; ++iRead) {
    auto read = cell->readClocks[iRead];
    assert(clock[read.threadId] >= read.epoch);
  }
  // Check WAW
  auto write = cell->writeClock;
  assert(clock[write.threadId] >= write.epoch);
  // Update write
  auto tid = state->threadId;
  cell->writeClock = ScalarClock{clock[tid], tid, AtomicScope::NonAtomic};
}

__device__ void writeRange(ThreadState *state, uintptr_t write_addr,
                           int nBytes) {
  auto range = roundRange(Range{write_addr, write_addr + nBytes});

  auto reserveBase = state->reserveBase;
  rwLockAcquireRead(state->lock);

  for (uintptr_t addr = range.start; addr < range.end;
       addr += kShadowMemGranularityBytes) {
    if (!isGsanManaged(addr, reserveBase))
      continue;
    auto shadowAddr = getShadowAddress(addr);
    auto cell = acquireShadow(shadowAddr);
    doWrite(state, cell);
    releaseShadow(cell);
  }

  rwLockReleaseRead(state->lock);
}

// Handles tl.store(ptrs, values, mask)
__device__ void tensorStore(ThreadState *state, const char *stackPtr,
                            int nElems, int bytesPerElem) {
  const uintptr_t *ptrsPtr = reinterpret_cast<const uintptr_t *>(stackPtr);
  const char *maskPtr = stackPtr + nElems * sizeof(uintptr_t);
  for (int i = 0; i < nElems; ++i) {
    auto ptr = ptrsPtr[i];
    auto mask = maskPtr[i];
    if (mask)
      writeRange(state, ptr, bytesPerElem);
  }
}

__device__ void doRead(ThreadState *state, ShadowCell *cell) {
  // Update read count
  auto numReads = cell->numReads;
  if (numReads < std::numeric_limits<decltype(cell->numReads)>::max())
    ++cell->numReads;

  epoch_t *clock = state->vectorClock;
  // Check RAW
  auto write = cell->writeClock;
  assert(clock[write.threadId] >= write.epoch);

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
  cuda::atomic_ref<uint32_t, cuda::thread_scope_block> threadNumReadsAtomic{
      state->numReads};
  auto threadNumReads =
      threadNumReadsAtomic.fetch_add(1, cuda::memory_order_relaxed);
  auto seed = getGlobalState(state)->rngSeed;
  uint32_t rand = hash2x32(threadNumReads, state->threadId, seed);
  if ((rand >> 8) % numReads != 0)
    return;
  auto clockIdx = rand % ShadowCell::kReadClockSize;
  cell->readClocks[clockIdx] = scalarClock;
}

__device__ void readRange(ThreadState *state, uintptr_t read_addr, int nBytes) {
  auto range = roundRange(Range{read_addr, read_addr + nBytes});

  auto reserveBase = state->reserveBase;
  rwLockAcquireRead(state->lock);

  for (uintptr_t addr = range.start; addr < range.end;
       addr += kShadowMemGranularityBytes) {
    if (!isGsanManaged(addr, reserveBase))
      continue;
    auto shadowAddr = getShadowAddress(addr);
    auto cell = acquireShadow(shadowAddr);
    doRead(state, cell);
    releaseShadow(cell);
  }

  rwLockReleaseRead(state->lock);
}

// Handles tl.load(ptrs, mask)
__device__ void tensorLoad(ThreadState *state, const char *stackPtr, int nElems,
                           int bytesPerElem) {
  const uintptr_t *ptrsPtr = reinterpret_cast<const uintptr_t *>(stackPtr);
  const char *maskPtr = stackPtr + nElems * sizeof(uintptr_t);
  for (int i = 0; i < nElems; ++i) {
    auto ptr = ptrsPtr[i];
    auto mask = maskPtr[i];
    if (mask)
      readRange(state, ptr, bytesPerElem);
  }
}

} // namespace
} // namespace gsan

extern "C" void __triton_gsan_load_tensor(void *globalState,
                                          const char *stackPtr, int numElems,
                                          int bytesPerElem) {
  auto *threadState =
      gsan::getThreadState(reinterpret_cast<gsan::GlobalState *>(globalState));
  gsan::tensorLoad(threadState, stackPtr, numElems, bytesPerElem);
}

extern "C" void __triton_gsan_init(void *globalState) {
  gsan::initThread(reinterpret_cast<gsan::GlobalState *>(globalState));
}

extern "C" void __triton_gsan_store_tensor(void *globalState,
                                           const char *stackPtr, int numElems,
                                           int bytesPerElem) {
  auto *threadState =
      gsan::getThreadState(reinterpret_cast<gsan::GlobalState *>(globalState));
  gsan::tensorStore(threadState, stackPtr, numElems, bytesPerElem);
}
