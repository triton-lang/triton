#ifndef PROTON_CONTEXT_CONTEXT_H_
#define PROTON_CONTEXT_CONTEXT_H_

#include <atomic>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace proton {

inline constexpr std::string_view kMetadataScopeName =
    "__proton_launch_metadata";
inline constexpr std::string_view kMetadataScopePrefix =
    "__proton_launch_metadata:";

/// A context is a named object.
struct Context {
private:
  static bool isMetadataStateName(std::string_view name) {
    return name == kMetadataScopeName ||
           (name.size() > kMetadataScopePrefix.size() &&
            name.substr(0, kMetadataScopePrefix.size()) ==
                kMetadataScopePrefix);
  }

  const bool state{};
  const bool metadataState{};

public:
  const std::string name{};

  Context() = default;
  Context(std::string name, bool isState = false)
      : state(isState), metadataState(isState && isMetadataStateName(name)),
        name(std::move(name)) {}
  virtual ~Context() = default;

  bool operator==(const Context &other) const { return name == other.name; }
  bool operator!=(const Context &other) const { return !(*this == other); }
  bool operator<(const Context &other) const { return name < other.name; }
  bool operator>(const Context &other) const { return name > other.name; }
  bool operator<=(const Context &other) const { return !(*this > other); }
  bool operator>=(const Context &other) const { return !(*this < other); }

  bool isState() const { return state; }

  bool isMetadataState() const { return metadataState; }
};

/// A context source is an object that can provide a list of contexts.
class ContextSource {
public:
  ContextSource() = default;
  virtual ~ContextSource() = default;

  std::vector<Context> getContexts(bool withState = true) {
    auto contexts = getContextsImpl();
    if (withState && state.has_value()) {
      auto splitMetadataStateName = [](std::string_view name,
                                       std::string_view &rawName) {
        if (name.size() <= kMetadataScopePrefix.size() ||
            name.substr(0, kMetadataScopePrefix.size()) !=
                kMetadataScopePrefix) {
          return false;
        }
        rawName = name.substr(kMetadataScopePrefix.size());
        return true;
      };
      std::string_view rawName;
      if (splitMetadataStateName(state->name, rawName)) {
        contexts.reserve(contexts.size() + 2);
        contexts.emplace_back(std::string(kMetadataScopeName),
                              /*isState=*/true);
        contexts.emplace_back(std::string(rawName));
      } else {
        contexts.reserve(contexts.size() + 1);
        contexts.push_back(state.value());
      }
    }
    return contexts;
  }

  void setState(std::optional<Context> state) {
    if (state.has_value()) {
      ContextSource::state.emplace(state->name, /*isState=*/true);
    } else {
      ContextSource::state.reset();
    }
  }

  virtual void clear() { ContextSource::state = std::nullopt; }

  virtual size_t getDepth() = 0;

protected:
  virtual std::vector<Context> getContextsImpl() = 0;
  static thread_local std::optional<Context> state;
};

/// A scope is a context with a unique identifier.
struct Scope : public Context {
  const static size_t DummyScopeId = std::numeric_limits<size_t>::max();
  static std::atomic<size_t> scopeIdCounter;

  static size_t getNewScopeId() { return scopeIdCounter++; }

  size_t scopeId{};

  explicit Scope(size_t scopeId) : Context(), scopeId(scopeId) {}

  explicit Scope(const std::string &name) : Context(name) {
    scopeId = getNewScopeId();
  }

  Scope(size_t scopeId, const std::string &name)
      : scopeId(scopeId), Context(name) {}

  Scope() : Scope(DummyScopeId, "") {}

  bool operator==(const Scope &other) const {
    return scopeId == other.scopeId && name == other.name;
  }
  bool operator!=(const Scope &other) const { return !(*this == other); }
  bool operator<(const Scope &other) const {
    return scopeId < other.scopeId || name < other.name;
  }
  bool operator>(const Scope &other) const {
    return scopeId > other.scopeId || name > other.name;
  }
  bool operator<=(const Scope &other) const { return !(*this > other); }
  bool operator>=(const Scope &other) const { return !(*this < other); }
};

/// A scope interface allows to instrument handles before and after a scope.
/// Scopes can be nested.
class ScopeInterface {
public:
  ScopeInterface() = default;
  virtual ~ScopeInterface() = default;
  virtual void enterScope(const Scope &scope) = 0;
  virtual void exitScope(const Scope &scope) = 0;
};

/// An op interface allows to instrument handles before and after an operation,
/// which cannot be nested.
class OpInterface {
public:
  OpInterface() = default;
  virtual ~OpInterface() = default;

  void enterOp(const Scope &scope) {
    if (isOpInProgress()) {
      return;
    }
    startOp(scope);
    setOpInProgress(true);
  }

  void exitOp(const Scope &scope) {
    if (!isOpInProgress()) {
      return;
    }
    stopOp(scope);
    setOpInProgress(false);
  }

protected:
  bool isOpInProgress() { return opInProgress[this]; }
  void setOpInProgress(bool value) {
    opInProgress[this] = value;
    if (opInProgress.size() > MAX_CACHE_OBJECTS && !value)
      opInProgress.erase(this);
  }
  virtual void startOp(const Scope &scope) = 0;
  virtual void stopOp(const Scope &scope) = 0;

private:
  inline static const int MAX_CACHE_OBJECTS = 10;
  static thread_local std::map<OpInterface *, bool> opInProgress;
};

class InstrumentationInterface {
public:
  InstrumentationInterface() = default;
  virtual ~InstrumentationInterface() = default;

  virtual void initFunctionMetadata(
      uint64_t functionId, const std::string &functionName,
      const std::vector<std::pair<size_t, std::string>> &scopeIdNames,
      const std::vector<std::pair<size_t, size_t>> &scopeIdParents,
      const std::string &metadataPath) = 0;
  virtual void destroyFunctionMetadata(uint64_t functionId) = 0;
  virtual void enterInstrumentedOp(uint64_t streamId, uint64_t functionId,
                                   uint8_t *buffer, size_t size) = 0;
  virtual void exitInstrumentedOp(uint64_t streamId, uint64_t functionId,
                                  uint8_t *buffer, size_t size) = 0;
};

} // namespace proton

#endif // PROTON_CONTEXT_CONTEXT_H_
