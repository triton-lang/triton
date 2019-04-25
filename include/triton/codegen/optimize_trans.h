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
}

namespace codegen{

class optimize_trans {
private:
  ir::value *replace_phi(ir::value* value, std::vector<ir::instruction*>& to_delete, ir::builder &builder);

public:
  optimize_trans() {}
  void run(ir::module &mod);
};


}
}

#endif
