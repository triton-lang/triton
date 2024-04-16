#include "Context/Shadow.h"

#include <stdexcept>

namespace proton {

void ShadowContextSource::enterScope(const Scope &scope) {
  contextStack.push_back(scope);
}

void ShadowContextSource::exitScope(const Scope &scope) {
  if (contextStack.empty()) {
    throw std::runtime_error("Context stack is empty");
  }
  if (contextStack.back() != scope) {
    throw std::runtime_error("Context stack is not balanced");
  }
  contextStack.pop_back();
}

} // namespace proton
