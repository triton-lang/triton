#ifndef PROTON_CONTEXT_PYTHON_H_
#define PROTON_CONTEXT_PYTHON_H_

#include "Context.h"
#include <optional>

namespace proton {

/// Unwind the Python stack and early return a list of contexts.
class PythonContextSource : public ContextSource {
public:
  PythonContextSource() = default;
  ~PythonContextSource() = default;

private:
  std::vector<Context> getContextsImpl() override final;
};

} // namespace proton

#endif // PROTON_CONTEXT_PYTHON_H_
