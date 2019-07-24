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
}

namespace codegen{

class tune;

class reassociate {
private:
  ir::instruction* is_bin_add(ir::value *x);
  ir::value *reorder_op(ir::value *value, ir::builder &builder, std::vector<ir::instruction*>& to_delete, ir::value *&noncst, ir::value *&cst);

public:
  reassociate(tune *params);
  void run(ir::module& module);

private:
  tune* params_;
};

}

}

#endif
