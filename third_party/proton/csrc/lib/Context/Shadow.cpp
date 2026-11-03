#include "Context/Shadow.h"
#include "Utility/Errors.h"

#include <stdexcept>
#include <thread>

namespace proton {

void ShadowContextSource::initializeThreadContext() {
  if (!threadContextInitialized[this]) {
    threadContextStack.erase(this);
    threadContextStack.emplace(this, *mainContextStack);
    threadContextInitialized[this] = true;
  }
}

void ShadowContextSource::enterScope(const Scope &scope) {
  initializeThreadContext();
  threadContextStack[this].push_back(scope);
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
    throw makeLogicError("Context stack is empty");
  }
  if (threadContextStack[this].back() != scope) {
    throw makeLogicError("Context stack is not balanced");
  }
  threadContextStack[this].pop_back();
}

void ShadowContextSource::clear() {
  ContextSource::clear();
  threadContextStack[this].clear();
  threadContextInitialized[this] = false;
}

/*static*/ thread_local std::map<ShadowContextSource *, bool>
    ShadowContextSource::threadContextInitialized;

/*static*/ thread_local std::map<ShadowContextSource *, std::vector<Context>>
    ShadowContextSource::threadContextStack;

} // namespace proton
