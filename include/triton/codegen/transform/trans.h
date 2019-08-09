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

class peephole {
private:
  bool rewrite_trans_phi(ir::instruction* value, ir::builder &builder);
  bool rewrite_dot(ir::instruction *value, ir::builder& builder);
  bool rewrite_unit_red(ir::instruction *value, ir::builder& builder);
  bool rewrite_gep_ptr_min_off_plus_off(ir::instruction *value, ir::builder& builder);

private:

public:
  peephole() {}
  void run(ir::module &mod);
};


}
}
}

#endif
