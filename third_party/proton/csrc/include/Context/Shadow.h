#ifndef PROTON_CONTEXT_SHADOW_H_
#define PROTON_CONTEXT_SHADOW_H_

#include "Context.h"
#include <vector>

namespace proton {

/// Incrementally build a list of contexts by shadowing the stack with
/// user-defined scopes.
class ShadowContextSource : public ContextSource, public ScopeInterface {
public:
  ShadowContextSource() = default;

  std::vector<Context> getContexts() override { return contextStack; }

  void enterScope(const Scope &scope) override;

  void exitScope(const Scope &scope) override;

private:
  std::vector<Context> contextStack;
};

} // namespace proton

#endif // PROTON_CONTEXT_CONTEXT_H_
