#ifndef TDL_INCLUDE_CODEGEN_OPTIMIZE_CSE_H
#define TDL_INCLUDE_CODEGEN_OPTIMIZE_CSE_H

#include <tuple>
#include <vector>
#include <set>

namespace triton {

namespace ir {
  class module;
}

namespace codegen{
class tune;

class optimize_cse {
public:
  optimize_cse() {}
  void run(ir::module &mod);
};


}
}

#endif
