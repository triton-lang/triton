#ifndef TDL_INCLUDE_IR_CODEGEN_REASSOCIATE_H
#define TDL_INCLUDE_IR_CODEGEN_REASSOCIATE_H

#include <map>
#include <set>
#include <vector>
#include <iostream>

namespace triton {

// forward declaration
namespace ir {
class module;
class value;
class builder;
class instruction;
class getelementptr_inst;
}

namespace codegen{

class tune;

class reassociate {
  struct cst_info {
    ir::value* sta;
    ir::value* dyn;
  };

private:
  ir::instruction* is_bin_add(ir::value *x);
  ir::value *reassociate_idx(ir::value *value, ir::builder &builder, std::vector<ir::instruction*>& to_delete, ir::value *&noncst, ir::value *&cst);
  ir::value *reassociate_ptr(ir::getelementptr_inst* pz, ir::builder &builder, std::map<ir::value*, cst_info> &offsets);

public:
  reassociate(tune *params);
  void run(ir::module& module);

private:
  tune* params_;
};

}

}

#endif
