#ifndef PROTON_DATA_PHASE_STORE_H_
#define PROTON_DATA_PHASE_STORE_H_

#include <cstddef>
#include <map>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <string>

namespace proton {

class PhaseStoreBase {
public:
  virtual ~PhaseStoreBase() = default;

  virtual void ensure(size_t phase) = 0;
  virtual void *getOrCreatePtr(size_t phase) = 0;
  virtual void *getPtr(size_t phase) = 0;
  virtual const void *getPtr(size_t phase) const = 0;
  virtual void clearUpToInclusive(size_t phase) = 0;
};

template <typename T> class PhaseStore final : public PhaseStoreBase {
public:
  PhaseStore() = default;
  ~PhaseStore() override = default;

  struct Slot {
    mutable std::shared_mutex mutex;
    std::unique_ptr<T> value;
  };

  void ensure(size_t phase) override {
    auto slot = getOrCreateSlot(phase);
    std::unique_lock<std::shared_mutex> slotLock(slot->mutex);
    if (!slot->value) {
      slot->value = std::make_unique<T>();
    }
  }

  void *getOrCreatePtr(size_t phase) override {
    ensure(phase);
    return getPtr(phase);
  }

  void *getPtr(size_t phase) override {
    auto slot = getSlot(phase);
    std::shared_lock<std::shared_mutex> slotLock(slot->mutex);
    if (!slot->value) {
      throw std::runtime_error("[PROTON] Phase " + std::to_string(phase) +
                               " has no data.");
    }
    return slot->value.get();
  }

  const void *getPtr(size_t phase) const override {
    auto slot = getSlot(phase);
    std::shared_lock<std::shared_mutex> slotLock(slot->mutex);
    if (!slot->value) {
      throw std::runtime_error("[PROTON] Phase " + std::to_string(phase) +
                               " has no data.");
    }
    return slot->value.get();
  }

  void clearUpToInclusive(size_t phase) override {
    std::vector<std::shared_ptr<Slot>> slotsToClear;
    {
      std::shared_lock<std::shared_mutex> lock(phasesMutex);
      for (auto it = phases.begin(); it != phases.end() && it->first <= phase;
           ++it) {
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
      phases.erase(phases.begin(), phases.upper_bound(phase));
    }
  }

  std::shared_ptr<Slot> getOrCreateSlot(size_t phase) {
    {
      std::shared_lock<std::shared_mutex> lock(phasesMutex);
      auto it = phases.find(phase);
      if (it != phases.end() && it->second) {
        return it->second;
      }
    }
    std::unique_lock<std::shared_mutex> lock(phasesMutex);
    auto &slot = phases[phase];
    if (!slot) {
      slot = std::make_shared<Slot>();
    }
    return slot;
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

private:
  mutable std::shared_mutex phasesMutex;
  std::map<size_t, std::shared_ptr<Slot>> phases;
};

} // namespace proton

#endif // PROTON_DATA_PHASE_STORE_H_
