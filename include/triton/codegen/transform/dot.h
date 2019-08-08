#ifndef TDL_INCLUDE_CODEGEN_OPTIMIZE_DOT_H
#define TDL_INCLUDE_CODEGEN_OPTIMIZE_DOT_H

#include <tuple>
#include <vector>
#include <set>

namespace triton {

namespace ir {
  class module;
}

namespace codegen{

namespace analysis{
class tune;
}

namespace transform{

class optimize_dot {
public:
  optimize_dot(analysis::tune* params): params_(params) {}
  void run(ir::module &mod);

private:
  analysis::tune* params_;
};

}
}
}

#endif
