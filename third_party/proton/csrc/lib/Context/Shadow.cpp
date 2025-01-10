#include "Context/Shadow.h"

#include <stdexcept>
#include <thread>

namespace proton {

void ShadowContextSource::initializeThreadContext() {
  if (!mainContextStack) {
    mainContextStack = &threadContextStack;
    contextInitialized = true;
  }
  if (!contextInitialized) {
    threadContextStack = *mainContextStack;
    contextInitialized = true;
  }
}

void ShadowContextSource::enterScope(const Scope &scope) {
  initializeThreadContext();
  threadContextStack.push_back(scope);
}

std::vector<Context> ShadowContextSource::getContextsImpl() {
  initializeThreadContext();
  return threadContextStack;
}

void ShadowContextSource::exitScope(const Scope &scope) {
  if (threadContextStack.empty()) {
    throw std::runtime_error("Context stack is empty");
  }
  if (threadContextStack.back() != scope) {
    throw std::runtime_error("Context stack is not balanced");
  }
  threadContextStack.pop_back();
}

/*static*/ thread_local std::vector<Context>
    ShadowContextSource::threadContextStack;

/*static*/ thread_local bool ShadowContextSource::contextInitialized = false;

} // namespace proton
