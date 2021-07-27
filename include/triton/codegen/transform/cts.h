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
}

namespace codegen{
namespace transform{

class cts {
public:
  void run(ir::module &mod);
};

}
}
}

#endif
