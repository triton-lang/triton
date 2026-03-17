#include "Context/Shadow.h"

#include <atomic>
#include <stdexcept>
#include <thread>

namespace proton {

namespace {

struct ShadowThreadDebugCache {
  size_t initializedKeys{0};
  size_t initializedTrue{0};
  size_t stackKeys{0};
  size_t totalContexts{0};
  bool registered{false};
};

std::atomic<size_t> shadowDebugThreads{0};
std::atomic<size_t> shadowDebugInitializedKeys{0};
std::atomic<size_t> shadowDebugInitializedTrue{0};
std::atomic<size_t> shadowDebugStackKeys{0};
std::atomic<size_t> shadowDebugTotalContexts{0};

thread_local ShadowThreadDebugCache shadowThreadDebugCache{};

void applyShadowDebugDelta(std::atomic<size_t> &target, size_t oldValue,
                           size_t newValue) {
  if (newValue >= oldValue) {
    target.fetch_add(newValue - oldValue, std::memory_order_relaxed);
  } else {
    target.fetch_sub(oldValue - newValue, std::memory_order_relaxed);
  }
}

void refreshShadowDebugState(
    const std::map<ShadowContextSource *, bool> &threadContextInitialized,
    const std::map<ShadowContextSource *, std::vector<Context>>
        &threadContextStack) {
  if (!shadowThreadDebugCache.registered) {
    shadowThreadDebugCache.registered = true;
    shadowDebugThreads.fetch_add(1, std::memory_order_relaxed);
  }

  const auto initializedKeys = threadContextInitialized.size();
  size_t initializedTrue = 0;
  for (const auto &[_, value] : threadContextInitialized) {
    initializedTrue += value ? 1 : 0;
  }
  const auto stackKeys = threadContextStack.size();
  size_t totalContexts = 0;
  for (const auto &[_, stack] : threadContextStack) {
    totalContexts += stack.size();
  }

  applyShadowDebugDelta(shadowDebugInitializedKeys,
                        shadowThreadDebugCache.initializedKeys,
                        initializedKeys);
  applyShadowDebugDelta(shadowDebugInitializedTrue,
                        shadowThreadDebugCache.initializedTrue,
                        initializedTrue);
  applyShadowDebugDelta(shadowDebugStackKeys, shadowThreadDebugCache.stackKeys,
                        stackKeys);
  applyShadowDebugDelta(shadowDebugTotalContexts,
                        shadowThreadDebugCache.totalContexts, totalContexts);

  shadowThreadDebugCache.initializedKeys = initializedKeys;
  shadowThreadDebugCache.initializedTrue = initializedTrue;
  shadowThreadDebugCache.stackKeys = stackKeys;
  shadowThreadDebugCache.totalContexts = totalContexts;
}

struct ShadowThreadDebugCleanup {
  ~ShadowThreadDebugCleanup() {
    if (!shadowThreadDebugCache.registered) {
      return;
    }
    shadowDebugThreads.fetch_sub(1, std::memory_order_relaxed);
    shadowDebugInitializedKeys.fetch_sub(shadowThreadDebugCache.initializedKeys,
                                         std::memory_order_relaxed);
    shadowDebugInitializedTrue.fetch_sub(shadowThreadDebugCache.initializedTrue,
                                         std::memory_order_relaxed);
    shadowDebugStackKeys.fetch_sub(shadowThreadDebugCache.stackKeys,
                                   std::memory_order_relaxed);
    shadowDebugTotalContexts.fetch_sub(shadowThreadDebugCache.totalContexts,
                                       std::memory_order_relaxed);
  }
};

thread_local ShadowThreadDebugCleanup shadowThreadDebugCleanup{};

} // namespace

void ShadowContextSource::initializeThreadContext() {
  if (!threadContextInitialized[this]) {
    threadContextStack[this] = *mainContextStack;
    threadContextInitialized[this] = true;
  }
  refreshShadowDebugState(threadContextInitialized, threadContextStack);
}

void ShadowContextSource::enterScope(const Scope &scope) {
  initializeThreadContext();
  threadContextStack[this].push_back(scope);
  refreshShadowDebugState(threadContextInitialized, threadContextStack);
}

std::vector<Context> ShadowContextSource::getContextsImpl() {
  initializeThreadContext();
  return threadContextStack[this];
}

size_t ShadowContextSource::getDepth() {
  initializeThreadContext();
  return threadContextStack[this].size();
}

void ShadowContextSource::exitScope(const Scope &scope) {
  if (threadContextStack[this].empty()) {
    throw std::runtime_error("Context stack is empty");
  }
  if (threadContextStack[this].back() != scope) {
    throw std::runtime_error("Context stack is not balanced");
  }
  threadContextStack[this].pop_back();
  refreshShadowDebugState(threadContextInitialized, threadContextStack);
}

void ShadowContextSource::clear() {
  ContextSource::clear();
  threadContextStack[this].clear();
  threadContextInitialized[this] = false;
  refreshShadowDebugState(threadContextInitialized, threadContextStack);
}

ShadowContextSource::DebugStats ShadowContextSource::debugStats() {
  return DebugStats{
      shadowDebugThreads.load(std::memory_order_relaxed),
      shadowDebugInitializedKeys.load(std::memory_order_relaxed),
      shadowDebugInitializedTrue.load(std::memory_order_relaxed),
      shadowDebugStackKeys.load(std::memory_order_relaxed),
      shadowDebugTotalContexts.load(std::memory_order_relaxed)};
}

/*static*/ thread_local std::map<ShadowContextSource *, bool>
    ShadowContextSource::threadContextInitialized;

/*static*/ thread_local std::map<ShadowContextSource *, std::vector<Context>>
    ShadowContextSource::threadContextStack;

} // namespace proton
