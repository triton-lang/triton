#ifndef PROTON_UTILITY_SMALL_MAP_H_
#define PROTON_UTILITY_SMALL_MAP_H_

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace proton {

/// A simple small map optimized for tiny cardinalities with linear lookup.
template <typename Key, typename Value, size_t InlineReserve = 4>
class SmallMap {
public:
  SmallMap() { data.reserve(InlineReserve); }

  bool insert(const Key &key, const Value &value) {
    auto it = findIt(key);
    if (it != data.end())
      return false;
    data.push_back({key, value});
    return true;
  }

  void insertOrAssign(const Key &key, const Value &value) {
    auto it = findIt(key);
    if (it != data.end()) {
      it->second = value;
      return;
    }
    data.push_back({key, value});
  }

  Value &operator[](const Key &key) {
    auto it = findIt(key);
    if (it != data.end())
      return it->second;
    data.push_back({key, Value()});
    return data.back().second;
  }

  bool erase(const Key &key) {
    auto it = findIt(key);
    if (it == data.end())
      return false;
    *it = data.back();
    data.pop_back();
    return true;
  }

  bool contains(const Key &key) const { return findIt(key) != data.end(); }

  Value *find(const Key &key) {
    auto it = findIt(key);
    if (it == data.end())
      return nullptr;
    return &it->second;
  }

  const Value *find(const Key &key) const {
    auto it = findIt(key);
    if (it == data.end())
      return nullptr;
    return &it->second;
  }

  void clear() { data.clear(); }

  size_t size() const { return data.size(); }

  bool empty() const { return data.empty(); }

  auto begin() { return data.begin(); }
  auto end() { return data.end(); }
  auto begin() const { return data.begin(); }
  auto end() const { return data.end(); }

private:
  using StorageT = std::vector<std::pair<Key, Value>>;

  typename StorageT::iterator findIt(const Key &key) {
    return std::find_if(data.begin(), data.end(),
                        [&](const auto &item) { return item.first == key; });
  }

  typename StorageT::const_iterator findIt(const Key &key) const {
    return std::find_if(data.begin(), data.end(),
                        [&](const auto &item) { return item.first == key; });
  }

  StorageT data;
};

} // namespace proton

#endif // PROTON_UTILITY_SMALL_MAP_H_
