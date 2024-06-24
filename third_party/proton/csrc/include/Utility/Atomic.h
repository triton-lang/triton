#ifndef PROTON_UTILITY_ATOMIC_H_
#define PROTON_UTILITY_ATOMIC_H_

#include <atomic>

namespace proton {

template <typename T> T atomicMax(std::atomic<T> &target, T value) {
  T current = target.load();
  while (current < value && !target.compare_exchange_weak(current, value))
    ;
  return current;
}

template <typename T> T atomicMin(std::atomic<T> &target, T value) {
  T current = target.load();
  while (current > value && !target.compare_exchange_weak(current, value))
    ;
  return current;
}

} // namespace proton

#endif // PROTON_UTILITY_ATOMIC_H_
