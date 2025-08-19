#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <string>
#include <map>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdexcept>
#include <chrono>
#include <cstdlib>      // getenv, atoi
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <unistd.h>     // getpid()

// Reuse style from RuntimeTest: a small FakeMetalLibrary, but make it thread-safe
// and observable for the purposes of this concurrency unit test.
class FakeMetalLibrary {
public:
  FakeMetalLibrary(bool stub_mode, const std::vector<std::string> &kernels = {})
      : stub_(stub_mode), available_kernels_(kernels), next_id_(1), creation_count_(0) {}

  bool is_stub() const { return stub_; }

  // Thread-safe create or return cached pipeline id for kernel name.
  int get_or_create_pipeline(const std::string &kernel_name) {
    // Quick check under cache mutex to avoid taking creation lock when already cached.
    {
      std::lock_guard<std::mutex> lk(cache_mutex_);
      auto it = pipeline_cache_.find(kernel_name);
      if (it != pipeline_cache_.end()) return it->second;
    }

    // Create under exclusive lock to ensure only one creation happens.
    std::lock_guard<std::mutex> lk(create_mutex_);
    // Re-check while holding creation lock.
    auto it = pipeline_cache_.find(kernel_name);
    if (it != pipeline_cache_.end()) return it->second;

    int id = next_id_++;
    pipeline_cache_[kernel_name] = id;
    creation_count_.fetch_add(1, std::memory_order_relaxed);
    return id;
  }

  // Kernel lookup helper (keeps parity with RuntimeTest style helpers).
  void ensure_kernel_exists(const std::string &kernel_name) const {
    for (const auto &k : available_kernels_) {
      if (k == kernel_name) return;
    }
    throw std::runtime_error("kernel not found: " + kernel_name);
  }

  size_t pipeline_cache_size() const {
    std::lock_guard<std::mutex> lk(cache_mutex_);
    return pipeline_cache_.size();
  }

  int creation_count() const { return creation_count_.load(std::memory_order_relaxed); }

private:
  bool stub_;
  std::vector<std::string> available_kernels_;
  mutable std::mutex cache_mutex_;
  std::map<std::string, int> pipeline_cache_;
  std::mutex create_mutex_;
  int next_id_;
  std::atomic<int> creation_count_;
};

 // DIAGNOSTIC: serialize startup via TRITON_TEST_PIPELINE_SLEEP_MS (temporary).
 // TODO: remove this instrumentation once flakiness investigation is complete.
 namespace {
 static std::mutex g_pipeline_repro_log_mutex;
 
 static std::string iso_timestamp_with_ms() {
   using namespace std::chrono;
   auto now = system_clock::now();
   auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
   std::time_t t = system_clock::to_time_t(now);
   std::tm tm;
 #if defined(__APPLE__) || defined(_POSIX_THREAD_SAFE_FUNCTIONS)
   localtime_r(&t, &tm);
 #else
   tm = *std::localtime(&t);
 #endif
   std::ostringstream oss;
   oss << std::put_time(&tm, "%FT%T") << '.' << std::setfill('0') << std::setw(3) << ms.count();
   return oss.str();
 }
 
 // Helper used only for diagnostic serialization of test startup.
 // Reads TRITON_TEST_PIPELINE_SLEEP_MS and if >0 logs a before/after pair
 // (timestamp, pid, thread-hash, test name, message) to stderr and to a
 // per-run file in build_unittest/test-logs/pipeline_repro for reliable capture.
 // Uses a mutex to ensure thread-safe logging.
 // Keep this helper minimal and easily revertible.
 static void maybe_sleep_and_log(const char *test_name) {
   const char *env = std::getenv("TRITON_TEST_PIPELINE_SLEEP_MS");
   if (!env) return;
   int ms = std::atoi(env);
   if (ms <= 0) return;
 
   const std::string log_dir = "build_unittest/test-logs/pipeline_repro";
   // Best-effort directory creation; ignore failures (diagnostic only).
   try {
     std::filesystem::create_directories(log_dir);
   } catch (...) {
     // ignore
   }
 
   pid_t pid = getpid();
   auto tid_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
   using namespace std::chrono;
   auto now_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
   // Per-run file to avoid races on a single shared file and ensure persistence.
   const std::string per_run_path = log_dir + "/targeted_log." + std::to_string(pid) + "." +
                                    std::to_string(tid_hash) + "." + std::to_string(now_ms) + ".txt";
 
   std::string ts_before = iso_timestamp_with_ms();
   {
     std::lock_guard<std::mutex> lk(g_pipeline_repro_log_mutex);
     // Write to stderr for quick visibility.
     std::ostringstream serr;
     serr << ts_before << " pid=" << pid << " tid=" << tid_hash << " test=" << test_name
          << " before-sleep ms=" << ms << std::endl;
     std::cerr << serr.str();
 
     // Append to per-run file and flush immediately.
     std::ofstream ofs(per_run_path, std::ios::app);
     if (ofs) {
       ofs << serr.str();
       ofs.flush();
     }
   }
 
   // Sleep outside file I/O hold to avoid blocking other threads doing logging.
   std::this_thread::sleep_for(std::chrono::milliseconds(ms));
 
   std::string ts_after = iso_timestamp_with_ms();
   {
     std::lock_guard<std::mutex> lk(g_pipeline_repro_log_mutex);
     std::ostringstream serr;
     serr << ts_after << " pid=" << pid << " tid=" << tid_hash << " test=" << test_name
          << " after-sleep ms=" << ms << std::endl;
     std::cerr << serr.str();
     std::ofstream ofs(per_run_path, std::ios::app);
     if (ofs) {
       ofs << serr.str();
       ofs.flush();
     }
   }
 }
 } // namespace
 
 // Simple reusable barrier for C++17 (std::barrier is C++20).
 class SimpleBarrier {
 public:
   explicit SimpleBarrier(unsigned count) : threshold_(count), count_(count), generation_(0) {}
 
   void arrive_and_wait() {
     std::unique_lock<std::mutex> lk(mutex_);
     auto gen = generation_;
     if (--count_ == 0) {
       // last arriving thread advances generation and wakes everyone
       generation_++;
       count_ = threshold_;
       cv_.notify_all();
       return;
     }
     cv_.wait(lk, [this, gen] { return gen != generation_; });
   }
 
 private:
   std::mutex mutex_;
   std::condition_variable cv_;
   const unsigned threshold_;
   unsigned count_;
   unsigned generation_;
 }
 
 ;

TEST(MetalPipeline, ConcurrentCreationProducesSinglePipeline) {
  FakeMetalLibrary lib(false, {"kernelA"});
  // DIAGNOSTIC: optionally serialize test startup and log timing.
  // Controlled by TRITON_TEST_PIPELINE_SLEEP_MS (ms). See maybe_sleep_and_log().
  maybe_sleep_and_log("ConcurrentCreationProducesSinglePipeline");
  const unsigned kThreads = 16;
  SimpleBarrier start_barrier(kThreads);
  std::vector<int> results(kThreads, -1);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (unsigned i = 0; i < kThreads; ++i) {
    threads.emplace_back([&lib, &start_barrier, &results, i]() {
      // Synchronize start so threads attempt creation concurrently.
      start_barrier.arrive_and_wait();
      try {
        results[i] = lib.get_or_create_pipeline("kernelA");
      } catch (...) {
        results[i] = -1;
      }
    });
  }

  for (auto &t : threads) t.join();

  // All threads should receive a valid id and the same id.
  EXPECT_GE(results[0], 1);
  for (unsigned i = 1; i < kThreads; ++i) {
    EXPECT_EQ(results[i], results[0]);
  }

  // Only one pipeline should have been created.
  EXPECT_EQ(lib.pipeline_cache_size(), 1u);
  EXPECT_EQ(lib.creation_count(), 1);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}