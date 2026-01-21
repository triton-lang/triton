#ifndef PROTON_DATA_PHASE_STORE_H_
#define PROTON_DATA_PHASE_STORE_H_

#include <cstddef>
#include <map>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace proton {

class PhaseStoreBase {
public:
  virtual ~PhaseStoreBase() = default;

  virtual void *getPtr(size_t phase) = 0;
  virtual void *createPtr(size_t phase) = 0;
  virtual void clearUpToInclusive(size_t phase) = 0;
  virtual void clearPhase(size_t phase) = 0;
};

template <typename T> class PhaseStore final : public PhaseStoreBase {
public:
  PhaseStore() = default;
  ~PhaseStore() override = default;

  struct Slot {
    mutable std::shared_mutex mutex;
    std::unique_ptr<T> value;
  };

  void *createPtr(size_t phase) override {
    std::shared_ptr<Slot> slot;
    {
      std::unique_lock<std::shared_mutex> lock(phasesMutex);
      auto &entry = phases[phase];
      if (!entry)
        entry = std::make_shared<Slot>();
      slot = entry;
    }
    {
      std::unique_lock<std::shared_mutex> slotLock(slot->mutex);
      if (!slot->value) // slot value might not exist yet or been cleared
        slot->value = std::make_unique<T>();
      return slot->value.get();
    }
  }

  void *getPtr(size_t phase) override { return getSlot(phase)->value.get(); }

  void clearUpToInclusive(size_t phase) override {
    clearRangeInclusive(0, phase);
  }

  void clearPhase(size_t phase) override { clearRangeInclusive(phase, phase); }

  template <typename FnT> decltype(auto) withPtr(size_t phase, FnT &&fn) const {
    auto slot = getSlot(phase);
    std::shared_lock<std::shared_mutex> slotLock(slot->mutex);
    return std::forward<FnT>(fn)(slot->value.get());
  }

private:
  void clearRangeInclusive(size_t beginPhase, size_t endPhase) {
    std::vector<std::shared_ptr<Slot>> slotsToClear;
    {
      std::shared_lock<std::shared_mutex> lock(phasesMutex);
      auto it = phases.lower_bound(beginPhase);
      auto endIt = phases.upper_bound(endPhase);
      for (; it != endIt; ++it) {
        if (it->second) {
          slotsToClear.push_back(it->second);
        }
      }
    }

    // Free the heavy per-phase payloads under per-phase locks, without blocking
    // unrelated phases from being accessed via the store map.
    for (auto &slot : slotsToClear) {
      std::unique_lock<std::shared_mutex> slotLock(slot->mutex);
      slot->value.reset();
    }

    // Finally, prune the cleared phases from the map.
    {
      std::unique_lock<std::shared_mutex> lock(phasesMutex);
      phases.erase(phases.lower_bound(beginPhase),
                   phases.upper_bound(endPhase));
    }
  }

  std::shared_ptr<Slot> getSlot(size_t phase) const {
    std::shared_lock<std::shared_mutex> lock(phasesMutex);
    auto it = phases.find(phase);
    if (it == phases.end() || !it->second) {
      throw std::runtime_error("[PROTON] Phase " + std::to_string(phase) +
                               " has no data.");
    }
    return it->second;
  }

  mutable std::shared_mutex phasesMutex;
  std::map<size_t, std::shared_ptr<Slot>> phases;
};

} // namespace proton

#endif // PROTON_DATA_PHASE_STORE_H_
