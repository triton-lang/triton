#ifndef PROTON_UTILITY_MAP_H_
#define PROTON_UTILITY_MAP_H_

#include <map>
#include <mutex>
#include <shared_mutex>
#include <utility>

namespace proton {

/// A simple thread safe map with read/write lock.
template <typename Key, typename Value,
          typename Container = std::map<Key, Value>>
class ThreadSafeMap {
public:
  ThreadSafeMap() = default;

  template <typename FnT> void upsert(const Key &key, FnT &&fn) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    fn(map[key]);
  }

  template <typename FnT> bool withRead(const Key &key, FnT &&fn) const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto it = map.find(key);
    if (it == map.end()) {
      return false;
    }
    fn(it->second);
    return true;
  }

  template <typename FnT> bool withWrite(const Key &key, FnT &&fn) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    auto it = map.find(key);
    if (it == map.end()) {
      return false;
    }
    fn(it->second);
    return true;
  }

  Value &operator[](const Key &key) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    return map[key];
  }

  Value &operator[](Key &&key) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    return map[std::move(key)];
  }

  Value &at(const Key &key) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return map.at(key);
  }

  Value &at(const Key &key) const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return map.at(key);
  }

  void insert(const Key &key, const Value &value) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    map[key] = value;
  }

  bool contain(const Key &key) const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto it = map.find(key);
    if (it == map.end())
      return false;
    return true;
  }

  bool erase(const Key &key) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    return map.erase(key) > 0;
  }

  void clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    map.clear();
  }

  size_t size() const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return map.size();
  }

  std::optional<std::reference_wrapper<Value>> find(const Key &key) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    auto it = map.find(key);
    if (it == map.end()) {
      return std::nullopt;
    }
    return std::ref(it->second);
  }

private:
  Container map;
  mutable std::shared_mutex mutex;
};

} // namespace proton

#endif // PROTON_UTILITY_MAP_H_
