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

class tune;

class optimize_dot {
public:
  optimize_dot(tune* params): params_(params) {}
  void run(ir::module &mod);

private:
  tune* params_;
};


}
}

#endif
