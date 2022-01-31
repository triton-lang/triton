#ifndef TDL_INCLUDE_CODEGEN_BUFFER_INFO_PASS_H
#define TDL_INCLUDE_CODEGEN_BUFFER_INFO_PASS_H

#include <set>
#include <map>

namespace triton {

namespace ir {
  class module;
  class value;
  class phi_node;
  class instruction;
  class builder;
}

namespace codegen{
namespace transform{

class cts {
private:
  void add_copy(ir::instruction *parent, ir::value *x, ir::builder &builder, bool to_shared);

public:
  cts(bool use_async = false): use_async_(use_async) {}
  void run(ir::module &mod);

private:
  bool use_async_;
};

}
}
}

#endif