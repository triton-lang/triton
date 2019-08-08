#ifndef TDL_INCLUDE_CODEGEN_OPTIMIZE_TRANS_H
#define TDL_INCLUDE_CODEGEN_OPTIMIZE_TRANS_H

#include <tuple>
#include <vector>
#include <set>

namespace triton {

namespace ir {
  class module;
  class value;
  class instruction;
  class trans_inst;
  class builder;
  class constant_int;
}

namespace codegen{
namespace transform{

class optimize_trans {
private:
  ir::value *replace_phi(ir::value* value, ir::builder &builder, const std::vector<ir::constant_int *> &perm);

public:
  optimize_trans() {}
  void run(ir::module &mod);
};


}
}
}

#endif
