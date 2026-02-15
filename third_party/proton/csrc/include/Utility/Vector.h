#ifndef PROTON_UTILITY_VECTOR_H_
#define PROTON_UTILITY_VECTOR_H_

#include <algorithm>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace proton {

/// A simple thread safe vector with read/write lock.
template <typename Value, typename Container = std::vector<Value>>
class ThreadSafeVector {
public:
  ThreadSafeVector() = default;

  void push_back(const Value &value) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    vector.push_back(value);
  }

  void push_back(Value &&value) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    vector.push_back(std::move(value));
  }

  template <typename... Args> void emplace_back(Args &&...args) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    vector.emplace_back(std::forward<Args>(args)...);
  }

  bool contain(const Value &value) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return std::find(vector.begin(), vector.end(), value) != vector.end();
  }

  bool erase(const Value &value) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    auto it = std::find(vector.begin(), vector.end(), value);
    if (it == vector.end())
      return false;
    vector.erase(it);
    return true;
  }

  bool pop_back(Value &value) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    if (vector.empty())
      return false;
    value = vector.back();
    vector.pop_back();
    return true;
  }

  void clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    vector.clear();
  }

  size_t size() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return vector.size();
  }

  bool empty() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return vector.empty();
  }

  Container snapshot() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return vector;
  }

private:
  Container vector;
  std::shared_mutex mutex;
};

} // namespace proton

#endif // PROTON_UTILITY_VECTOR_H_
