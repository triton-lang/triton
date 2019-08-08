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
namespace transform{

class optimize_dce {
public:
  optimize_dce() {}
  void run(ir::module &mod);
};

}
}
}

#endif
