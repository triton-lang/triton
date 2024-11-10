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
  ~ShadowContextSource() = default;

  void enterScope(const Scope &scope) override final;

  void exitScope(const Scope &scope) override final;

protected:
  std::vector<Context> getContextsImpl() override final { return contextStack; }

private:
  std::vector<Context> contextStack;
};

} // namespace proton

#endif // PROTON_CONTEXT_CONTEXT_H_
