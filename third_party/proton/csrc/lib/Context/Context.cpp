#include "Context/Context.h"

namespace proton {

std::atomic<size_t> Scope::scopeIdCounter{1};

/*static*/ thread_local std::map<ThreadLocalOpInterface *, bool>
    ThreadLocalOpInterface::opInProgress;

} // namespace proton
