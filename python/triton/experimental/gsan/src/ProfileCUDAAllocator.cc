#include "AllocationProfile.h"
#include "AllocatorCommon.h"
#include "ProfileAllocatorRuntime.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
namespace ca = c10::cuda::CUDACachingAllocator;
namespace pool = triton::cuda_allocator;

namespace {
struct Context {
  CUdevice device;
  CUcontext context = nullptr;
  bool pushed = false;
  CUresult error;
  explicit Context(int ordinal) {
    error = cuDeviceGet(&device, ordinal);
    if (error == CUDA_SUCCESS)
      error = cuDevicePrimaryCtxRetain(&context, device);
    if (error == CUDA_SUCCESS) {
      error = cuCtxPushCurrent(context);
      pushed = error == CUDA_SUCCESS;
    }
  }
  ~Context() {
    if (pushed) {
      CUcontext previous;
      cuCtxPopCurrent(&previous);
    }
    if (context)
      cuDevicePrimaryCtxRelease(device);
  }
};

void check(CUresult error) {
  if (error == CUDA_SUCCESS)
    return;
  const char *message = "unknown CUDA error";
  cuGetErrorString(error, &message);
  TORCH_CHECK(false, "profiling allocator: ", message);
}

struct Stream {
  int device;
  CUstream stream;
  bool operator==(const Stream &other) const {
    return device == other.device && stream == other.stream;
  }
};

struct Record {
  CUdeviceptr shadow, output;
  CUfunction function;
  int device, grid, block, shared;
  CUevent event = nullptr;
  bool released = false, enabled = false, frozen = false;
  bool pending = false, launched = false;
  CUstream stream = nullptr;
  CUresult error = CUDA_SUCCESS;
};

void rememberError(Record &record, CUresult error) {
  if (record.error == CUDA_SUCCESS)
    record.error = error;
}

void summarize(Record &record, CUstream stream) {
  // The compiled reduction has two pointer arguments and two unused scratch
  // arguments. Its sizes are compile-time constants.
  CUdeviceptr scratch = 0;
  void *args[] = {&record.shadow, &record.output, &scratch, &scratch};
  auto result = cuLaunchKernel(record.function, record.grid, 1, 1, record.block,
                               1, 1, record.shared, stream, args, nullptr);
  if (result == CUDA_SUCCESS) {
    record.launched = true;
    record.stream = stream;
    result = cuEventRecord(record.event, stream);
    record.pending = result == CUDA_SUCCESS;
  }
  rememberError(record, result);
  check(result);
}

class ProfileCUDAAllocator;
struct Segment {
  pool::AllocNode root;
  Stream allocationStream;
};
struct Block {
  Segment *segment;
  pool::AllocNode *node;
  size_t requested;
  std::vector<Stream> users;
  std::vector<std::weak_ptr<Record>> records;
  CUevent ready = nullptr;
  // A release during capture cannot reset memory referenced by a replay.
  std::vector<c10::MempoolId_t> captures;
  bool quarantined = false;
};
struct Storage {
  std::shared_ptr<ProfileCUDAAllocator> allocator;
  std::unique_ptr<Block> block;
};
struct Holder {
  std::shared_ptr<ProfileCUDAAllocator> allocator;
  std::shared_ptr<Record> record;
  // Python owns the buffers/kernel; the data deleter never touches Python.
  PyObject *resources;
};
constexpr const char *capsuleName = "gsan_profile_lifetime";

class ProfileCUDAAllocator final
    : public ca::CUDAAllocator,
      public std::enable_shared_from_this<ProfileCUDAAllocator> {
  ca::CUDAAllocator *fallback_;
  // Retain an existing custom allocator if one was installed before ours.
  std::shared_ptr<ca::CUDAAllocator> fallbackOwner_;
  mutable std::mutex mutex_;
  bool enabled_ = false;
  std::thread::id thread_;
  std::vector<int> devices_;
  std::vector<std::unique_ptr<Segment>> segments_;
  std::unordered_map<uintptr_t, Block *> live_;
  std::vector<std::unique_ptr<Block>> pending_;
  std::map<int, CUstream> retirementStreams_;
  struct ActivePool {
    int device;
    c10::MempoolId_t id;
    std::function<bool(cudaStream_t)> filter;
  };
  std::vector<ActivePool> activePools_;
  std::vector<c10::MempoolId_t> explicitPools_;
  std::vector<c10::MempoolId_t> liveGraphs_;
  std::string error_;

  bool route(int device, CUstream stream) {
    if (!enabled_ || thread_ != std::this_thread::get_id() ||
        std::find(devices_.begin(), devices_.end(), device) == devices_.end())
      return false;
    // Explicit third-party MemPools continue to use the original allocator.
    // Graph capture is handled separately below.
    for (auto &entry : activePools_)
      if (entry.device == device &&
          std::find(explicitPools_.begin(), explicitPools_.end(), entry.id) !=
              explicitPools_.end() &&
          entry.filter(stream))
        return false;
    return true;
  }

  void collect() {
    for (auto it = pending_.begin(); it != pending_.end();) {
      auto &block = **it;
      if (block.quarantined) {
        ++it;
        continue;
      }
      Context context(block.segment->allocationStream.device);
      check(context.error);
      auto result = cuEventQuery(block.ready);
      if (result == CUDA_ERROR_NOT_READY) {
        ++it;
        continue;
      }
      check(result);
      check(cuEventDestroy(block.ready));
      pool::freeNode(block.node);
      it = pending_.erase(it);
    }
  }

  c10::DataPtr allocateOnStream(size_t bytes, int device, CUstream stream) {
    std::unique_lock lock(mutex_);
    if (!bytes || !route(device, stream)) {
      lock.unlock();
      return fallback_->allocate(bytes);
    }
    TORCH_CHECK(error_.empty(), error_);
    CUstreamCaptureStatus capture;
    check(cuStreamIsCapturing(stream, &capture));
    // Captured allocation would also capture the shadow initialization, which
    // would erase observations on every replay. Require preallocated storage.
    TORCH_CHECK(capture == CU_STREAM_CAPTURE_STATUS_NONE,
                "allocate profiling storage before CUDA graph capture");
    collect();
    TORCH_CHECK(bytes <= gsan_profile::kDataSize,
                "profiling allocation exceeds the address reservation");
    const size_t size = pool::roundUp(bytes, 512);
    Segment *segment = nullptr;
    pool::AllocNode *node = nullptr;
    for (auto &candidate : segments_) {
      if (candidate->allocationStream == Stream{device, stream} &&
          candidate->root.maxFreeBlockSize >= size) {
        segment = candidate.get();
        node = pool::allocateNode(&segment->root, size);
        break;
      }
    }
    if (!node) {
      // Share VMM pages between small storages; the second tree suballocates
      // data and metadata together, without involving PyTorch's MemPool.
      size_t capacity = std::max<size_t>(2 << 20, size);
      if (pool::roundDownToPowerOfTwo(capacity) != capacity)
        capacity = pool::roundDownToPowerOfTwo(capacity) * 2;
      auto base = reinterpret_cast<uintptr_t>(
          gsanProfileMalloc(capacity, device, stream));
      TORCH_CHECK_WITH(OutOfMemoryError, base,
                       "profiling allocator out of memory");
      size_t mapped;
      uintptr_t mappedBase;
      bool found = profileGetLayout(base, mappedBase, mapped);
      TORCH_INTERNAL_ASSERT(found && mappedBase == base && mapped >= capacity);
      auto created = std::make_unique<Segment>();
      created->root.virtualAddress = base;
      created->root.size = pool::roundDownToPowerOfTwo(mapped);
      created->root.maxFreeBlockSize = created->root.size;
      created->allocationStream = {device, stream};
      segment = created.get();
      segments_.push_back(std::move(created));
      node = pool::allocateNode(&segment->root, size);
    }
    auto storage = std::make_unique<Storage>();
    storage->allocator = shared_from_this();
    storage->block = std::make_unique<Block>();
    storage->block->segment = segment;
    storage->block->node = node;
    storage->block->requested = bytes;
    auto address = node->virtualAddress;
    live_.emplace(address, storage->block.get());
    return {reinterpret_cast<void *>(address), storage.release(),
            [](void *pointer) {
              std::unique_ptr<Storage> storage(static_cast<Storage *>(pointer));
              storage->allocator->retire(std::move(storage->block));
            },
            c10::Device(c10::DeviceType::CUDA, device)};
  }

public:
  explicit ProfileCUDAAllocator(
      std::shared_ptr<ca::CUDAAllocator> fallbackOwner)
      : fallback_(ca::get()), fallbackOwner_(std::move(fallbackOwner)) {}

  bool setEnabled(bool enabled, const std::vector<int> &devices) {
    std::lock_guard lock(mutex_);
    bool previous = enabled_;
    TORCH_CHECK(!enabled_ || thread_ == std::this_thread::get_id(),
                "AllocationProfile is active on another host thread");
    enabled_ = enabled;
    thread_ = std::this_thread::get_id();
    if (enabled && !devices.empty())
      devices_ = devices;
    return previous;
  }
  bool owns(uintptr_t address) {
    std::lock_guard lock(mutex_);
    return live_.count(address);
  }
  bool hasLiveStorage() {
    std::lock_guard lock(mutex_);
    return !live_.empty();
  }
  void resetMetadata(const std::vector<int> &devices) {
    std::lock_guard lock(mutex_);
    for (auto &segment : segments_) {
      auto stream = segment->allocationStream;
      if (std::find(devices.begin(), devices.end(), stream.device) ==
          devices.end())
        continue;
      Context context(stream.device);
      check(context.error);
      check(cuMemsetD32Async(
          gsan_profile::metadataAddress(segment->root.virtualAddress), 0,
          segment->root.size / gsan_profile::kGranularity, stream.stream));
    }
  }
  c10::DataPtr allocate(size_t bytes) override {
    auto device = c10::cuda::current_device();
    return allocateOnStream(bytes, device,
                            c10::cuda::getCurrentCUDAStream(device).stream());
  }
  // Raw-pointer clients keep the original allocator's ownership and stream
  // semantics. Profiling is confined to the DataPtr allocation path above.
  void *raw_alloc(size_t bytes) override { return fallback_->raw_alloc(bytes); }
  void *raw_alloc_with_stream(size_t bytes, cudaStream_t stream) override {
    return fallback_->raw_alloc_with_stream(bytes, stream);
  }
  void raw_delete(void *pointer) override { fallback_->raw_delete(pointer); }

  void recordStream(const c10::DataPtr &pointer,
                    c10::cuda::CUDAStream stream) override {
    std::unique_lock lock(mutex_);
    auto it = live_.find(reinterpret_cast<uintptr_t>(pointer.get()));
    if (it == live_.end()) {
      lock.unlock();
      fallback_->recordStream(pointer, stream);
      return;
    }
    Stream user{stream.device_index(), stream.stream()};
    auto &block = *it->second;
    if (!(user == block.segment->allocationStream) &&
        std::find(block.users.begin(), block.users.end(), user) ==
            block.users.end())
      block.users.push_back(user);
  }

  void retire(std::unique_ptr<Block> block) noexcept {
    std::lock_guard lock(mutex_);
    live_.erase(block->node->virtualAddress);
    std::vector<std::shared_ptr<Record>> records;
    for (auto &weak : block->records)
      if (auto record = weak.lock()) {
        if (!record->frozen)
          record->released = true;
        records.push_back(std::move(record));
      }
    try {
      auto allocationStream = block->segment->allocationStream;
      Context context(allocationStream.device);
      check(context.error);
      for (auto &capture : activePools_)
        if (std::find(explicitPools_.begin(), explicitPools_.end(),
                      capture.id) == explicitPools_.end())
          block->captures.push_back(capture.id);
      // Keep storage referenced by a graph alive even after its host owner
      // dies. The profile is invalidated; reset can reclaim it once the graph
      // dies.
      if (!block->captures.empty()) {
        for (auto &record : records)
          rememberError(*record, CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
        block->quarantined = true;
        pending_.push_back(std::move(block));
        return;
      }
      CUstream stream = allocationStream.stream;
      if (!block->users.empty()) {
        auto &retirement = retirementStreams_[allocationStream.device];
        if (!retirement)
          check(cuStreamCreate(&retirement, CU_STREAM_NON_BLOCKING));
        stream = retirement;
        block->users.push_back(allocationStream);
        for (auto user : block->users) {
          CUevent event;
          {
            Context userContext(user.device);
            check(userContext.error);
            check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
            auto result = cuEventRecord(event, user.stream);
            if (result != CUDA_SUCCESS) {
              cuEventDestroy(event);
              check(result);
            }
          }
          auto result = cuStreamWaitEvent(stream, event, 0);
          cuEventDestroy(event);
          check(result);
        }
      }
      for (auto &record : records) {
        if (record->frozen) {
          // Finalization can race storage destruction on another host thread.
          // Finish its one summary before clearing the shadow, without a CPU
          // wait in the deleter. A failed event publication forbids reuse.
          if (record->pending)
            check(cuStreamWaitEvent(stream, record->event, 0));
          else if (record->launched)
            check(record->error);
          continue;
        }
        if (record->enabled && record->error == CUDA_SUCCESS)
          summarize(*record, stream);
      }
      check(cuMemsetD32Async(
          gsan_profile::metadataAddress(block->node->virtualAddress), 0,
          pool::cdiv(block->requested, gsan_profile::kGranularity), stream));
      if (block->users.empty()) {
        // The whole segment is confined to this allocation stream, so reuse
        // naturally follows the summary and reset, without a CPU wait.
        pool::freeNode(block->node);
      } else {
        check(cuEventCreate(&block->ready, CU_EVENT_DISABLE_TIMING));
        check(cuEventRecord(block->ready, stream));
        pending_.push_back(std::move(block));
      }
    } catch (const std::exception &error) {
      error_ = error.what();
      for (auto &record : records)
        rememberError(*record, CUDA_ERROR_UNKNOWN);
      // Never recycle a block after a failed reset or dependency publication.
      block->quarantined = true;
      pending_.push_back(std::move(block));
    }
  }

  py::capsule watch(uintptr_t address, uintptr_t shadow, int device,
                    uintptr_t function, uintptr_t output, int grid, int block,
                    int shared, py::object resources) {
    Context context(device);
    check(context.error);
    auto record = std::make_shared<Record>();
    record->shadow = shadow;
    record->device = device;
    record->function = reinterpret_cast<CUfunction>(function);
    record->output = output;
    record->grid = grid;
    record->block = block;
    record->shared = shared;
    check(cuEventCreate(&record->event, CU_EVENT_DISABLE_TIMING));
    auto holder = std::make_unique<Holder>();
    holder->allocator = shared_from_this();
    holder->record = record;
    holder->resources = resources.release().ptr();
    py::capsule capsule(holder.get(), capsuleName, [](PyObject *capsule) {
      std::unique_ptr<Holder> holder(
          static_cast<Holder *>(PyCapsule_GetPointer(capsule, capsuleName)));
      if (!holder)
        return;
      holder->allocator->destroyRecord(*holder->record);
      Py_DECREF(holder->resources);
    });
    holder.release();
    std::lock_guard lock(mutex_);
    auto it = live_.find(address);
    TORCH_CHECK(it != live_.end(), "profiled storage is no longer live");
    it->second->records.push_back(record);
    return capsule;
  }
  static Record &getRecord(py::capsule capsule) {
    return *static_cast<Holder *>(capsule.get_pointer())->record;
  }
  bool enable(py::capsule capsule) {
    auto &record = getRecord(capsule);
    std::lock_guard lock(mutex_);
    if (!record.released)
      record.enabled = true;
    return record.enabled;
  }
  bool released(py::capsule capsule) {
    std::lock_guard lock(mutex_);
    return getRecord(capsule).released;
  }
  void finalize(py::capsule capsule, uintptr_t streamAddress) {
    auto &record = getRecord(capsule);
    Context context(record.device);
    check(context.error);
    CUevent event = nullptr;
    {
      std::lock_guard lock(mutex_);
      TORCH_CHECK(
          record.error != CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
          "storage retirement during CUDA graph capture is unsupported");
      check(record.error);
      if (!record.frozen) {
        record.frozen = true;
        // Retirement and finalization share one output. Whichever happens
        // first queues the summary; the other must not overwrite it.
        if (record.enabled && !record.released)
          summarize(record, reinterpret_cast<CUstream>(streamAddress));
      }
      event = record.pending ? record.event : nullptr;
    }
    if (event) {
      py::gil_scoped_release unlock;
      check(cuEventSynchronize(event));
    }
  }
  void destroyRecord(Record &record) {
    {
      std::lock_guard lock(mutex_);
      record.frozen = true;
    }
    Context context(record.device);
    if (context.error != CUDA_SUCCESS)
      return;
    {
      py::gil_scoped_release unlock;
      if (record.pending)
        cuEventSynchronize(record.event);
      else if (record.launched)
        cuStreamSynchronize(record.stream);
    }
    cuEventDestroy(record.event);
  }

  void emptyCache(c10::MempoolId_t id = {0, 0}) override {
    fallback_->emptyCache(id);
    if (id != c10::MempoolId_t{0, 0})
      return;
    std::lock_guard lock(mutex_);
    collect();
    for (auto it = pending_.begin(); it != pending_.end();) {
      auto &block = **it;
      if (!block.quarantined || block.captures.empty() ||
          std::any_of(block.captures.begin(), block.captures.end(),
                      [&](auto id) {
                        return std::find(liveGraphs_.begin(), liveGraphs_.end(),
                                         id) != liveGraphs_.end();
                      })) {
        ++it;
        continue;
      }
      Context context(block.segment->allocationStream.device);
      check(context.error);
      check(cuCtxSynchronize());
      check(cuMemsetD32(
          gsan_profile::metadataAddress(block.node->virtualAddress), 0,
          pool::cdiv(block.requested, gsan_profile::kGranularity)));
      pool::freeNode(block.node);
      it = pending_.erase(it);
    }
    for (auto it = segments_.begin(); it != segments_.end();) {
      auto &segment = **it;
      if (segment.root.maxFreeBlockSize != segment.root.size) {
        ++it;
        continue;
      }
      Context context(segment.allocationStream.device);
      check(context.error);
      gsanProfileFree(reinterpret_cast<void *>(segment.root.virtualAddress), 0,
                      segment.allocationStream.device,
                      segment.allocationStream.stream);
      it = segments_.erase(it);
    }
  }

  // Preserve the original allocator for ordinary tensors, third-party pools,
  // graph bookkeeping, and its introspection/observer APIs.
  void init(int count) override { fallback_->init(count); }
  bool initialized() override { return fallback_->initialized(); }
  std::string name() override { return "gsan-profile"; }
  void enable(bool value) override { fallback_->enable(value); }
  bool isEnabled() const override { return fallback_->isEnabled(); }
  void copy_data(void *dst, const void *src, size_t count) const override {
    fallback_->copy_data(dst, src, count);
  }
  double getMemoryFraction(c10::DeviceIndex d) override {
    return fallback_->getMemoryFraction(d);
  }
  void setMemoryFraction(double f, c10::DeviceIndex d) override {
    fallback_->setMemoryFraction(f, d);
  }
  std::vector<ca::StreamSegmentSize>
  getExpandableSegmentSizes(c10::DeviceIndex d) override {
    return fallback_->getExpandableSegmentSizes(d);
  }
  void cacheInfo(c10::DeviceIndex d, size_t *s) override {
    fallback_->cacheInfo(d, s);
  }
  void *getBaseAllocation(void *ptr, size_t *size) override {
    uintptr_t base;
    size_t mapped;
    if (profileGetLayout(reinterpret_cast<uintptr_t>(ptr), base, mapped)) {
      if (size)
        *size = mapped;
      return reinterpret_cast<void *>(base);
    }
    return fallback_->getBaseAllocation(ptr, size);
  }
  ca::DeviceStats getDeviceStats(c10::DeviceIndex d) override {
    return fallback_->getDeviceStats(d);
  }
  void resetAccumulatedStats(c10::DeviceIndex d) override {
    fallback_->resetAccumulatedStats(d);
  }
  void resetPeakStats(c10::DeviceIndex d) override {
    fallback_->resetPeakStats(d);
  }
#ifdef TRITON_TORCH_EXTENDED_ALLOCATOR
  bool getFreeDisabled(int lifetime) override {
    return fallback_->getFreeDisabled(lifetime);
  }
  void setFreeDisabled(int lifetime, bool value) override {
    fallback_->setFreeDisabled(lifetime, value);
  }
  ca::SnapshotInfo snapshot(c10::MempoolId_t id, bool traces,
                            bool extra) override {
    return fallback_->snapshot(id, traces, extra);
  }
  void recordHistory(bool enabled, ca::CreateContextFn fn, size_t count,
                     ca::RecordContext when, bool clear,
                     const std::vector<std::string> &skip, int64_t a, int64_t b,
                     bool c) override {
    fallback_->recordHistory(enabled, fn, count, when, clear, skip, a, b, c);
  }
  void attachCudaAllocObserver(ca::CudaAllocObserver observer) override {
    fallback_->attachCudaAllocObserver(std::move(observer));
  }
  std::string setAllocatorDebugContext(const char *context) override {
    return fallback_->setAllocatorDebugContext(context);
  }
#else
  ca::SnapshotInfo snapshot(c10::MempoolId_t id, bool traces) override {
    return fallback_->snapshot(id, traces);
  }
  void recordHistory(bool enabled, ca::CreateContextFn fn, size_t count,
                     ca::RecordContext when, bool clear,
                     const std::vector<std::string> &skip) override {
    fallback_->recordHistory(enabled, fn, count, when, clear, skip);
  }
#endif
  void beginAllocateToPool(c10::DeviceIndex d, c10::MempoolId_t id,
                           std::function<bool(cudaStream_t)> filter) override {
    fallback_->beginAllocateToPool(d, id, filter);
    std::lock_guard lock(mutex_);
    activePools_.push_back({d, id, std::move(filter)});
    if (std::find(explicitPools_.begin(), explicitPools_.end(), id) ==
        explicitPools_.end())
      liveGraphs_.push_back(id);
  }
  void endAllocateToPool(c10::DeviceIndex d, c10::MempoolId_t id) override {
    fallback_->endAllocateToPool(d, id);
    std::lock_guard lock(mutex_);
    activePools_.erase(std::remove_if(activePools_.begin(), activePools_.end(),
                                      [&](const auto &pool) {
                                        return pool.device == d &&
                                               pool.id == id;
                                      }),
                       activePools_.end());
  }
  void releasePool(c10::DeviceIndex d, c10::MempoolId_t id) override {
    fallback_->releasePool(d, id);
    std::lock_guard lock(mutex_);
    auto it = std::find(liveGraphs_.begin(), liveGraphs_.end(), id);
    if (it != liveGraphs_.end())
      liveGraphs_.erase(it);
  }
  void createOrIncrefPool(
      c10::DeviceIndex d, c10::MempoolId_t id,
      std::shared_ptr<ca::CUDAAllocator> allocator = nullptr) override {
    TORCH_CHECK(allocator.get() != this, "the profiling allocator is installed "
                                         "directly and cannot back a MemPool");
    fallback_->createOrIncrefPool(d, id, std::move(allocator));
    std::lock_guard lock(mutex_);
    explicitPools_.push_back(id);
  }
  int getPoolUseCount(c10::DeviceIndex d, c10::MempoolId_t id) override {
    return fallback_->getPoolUseCount(d, id);
  }
  void setUseOnOOM(c10::DeviceIndex d, c10::MempoolId_t id, bool use) override {
    fallback_->setUseOnOOM(d, id, use);
  }
  void setNoSplit(c10::DeviceIndex d, c10::MempoolId_t id) override {
    fallback_->setNoSplit(d, id);
  }
  bool checkPoolLiveAllocations(
      c10::DeviceIndex d, c10::MempoolId_t id,
      const std::unordered_set<void *> &expected) override {
    return fallback_->checkPoolLiveAllocations(d, id, expected);
  }
  ca::ShareableHandle shareIpcHandle(void *pointer) override {
    uintptr_t base;
    size_t size;
    TORCH_CHECK(
        !profileGetLayout(reinterpret_cast<uintptr_t>(pointer), base, size),
        "share profiling storage with profile.export_allocation_handles");
    return fallback_->shareIpcHandle(pointer);
  }
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    return fallback_->getIpcDevPtr(std::move(handle));
  }
  bool isHistoryEnabled() override { return fallback_->isHistoryEnabled(); }
  void attachOutOfMemoryObserver(ca::OutOfMemoryObserver observer) override {
    fallback_->attachOutOfMemoryObserver(std::move(observer));
  }
  void attachAllocatorTraceTracker(ca::AllocatorTraceTracker tracker) override {
    fallback_->attachAllocatorTraceTracker(std::move(tracker));
  }
  void enablePeerAccess(c10::DeviceIndex d, c10::DeviceIndex peer) override {
    fallback_->enablePeerAccess(d, peer);
  }
  cudaError_t memcpyAsync(void *dst, int dstDevice, const void *src,
                          int srcDevice, size_t size, cudaStream_t stream,
                          bool p2p) override {
    uintptr_t base;
    size_t mapped;
    if (dstDevice != srcDevice &&
        (profileGetLayout(reinterpret_cast<uintptr_t>(dst), base, mapped) ||
         profileGetLayout(reinterpret_cast<uintptr_t>(src), base, mapped)))
      return cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, size, stream);
    return fallback_->memcpyAsync(dst, dstDevice, src, srcDevice, size, stream,
                                  p2p);
  }
  std::shared_ptr<ca::AllocatorState>
  getCheckpointState(c10::DeviceIndex d, c10::MempoolId_t id) override {
    return fallback_->getCheckpointState(d, id);
  }
  ca::CheckpointDelta
  setCheckpointPoolState(c10::DeviceIndex d,
                         std::shared_ptr<ca::AllocatorState> state) override {
    return fallback_->setCheckpointPoolState(d, std::move(state));
  }
};
} // namespace

bool registerProfileCUDAAllocatorBindings(PyObject *module) {
  try {
    auto m = py::reinterpret_borrow<py::module_>(module);
    if (py::hasattr(m, "CUDAAllocator"))
      return true;
    py::class_<ProfileCUDAAllocator, ca::CUDAAllocator,
               std::shared_ptr<ProfileCUDAAllocator>>(m, "CUDAAllocator")
        .def(py::init<std::shared_ptr<ca::CUDAAllocator>>())
        .def("set_enabled", &ProfileCUDAAllocator::setEnabled)
        .def("owns", &ProfileCUDAAllocator::owns)
        .def("reset_metadata", &ProfileCUDAAllocator::resetMetadata)
        .def("has_live_storage", &ProfileCUDAAllocator::hasLiveStorage)
        .def(
            "empty_cache",
            [](ProfileCUDAAllocator &self) { self.emptyCache(); },
            py::call_guard<py::gil_scoped_release>())
        .def("watch", &ProfileCUDAAllocator::watch)
        .def("enable",
             py::overload_cast<py::capsule>(&ProfileCUDAAllocator::enable))
        .def("released", &ProfileCUDAAllocator::released)
        .def("finalize", &ProfileCUDAAllocator::finalize);
    return true;
  } catch (py::error_already_set &error) {
    error.restore();
  } catch (const std::exception &error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
  }
  return false;
}
