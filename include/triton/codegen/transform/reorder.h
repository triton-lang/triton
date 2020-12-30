#ifndef TRITON_INCLUDE_IR_CODEGEN_REORDER_H
#define TRITON_INCLUDE_IR_CODEGEN_REORDER_H

namespace triton {

// forward declaration
namespace ir {
class module;
}

namespace codegen{

namespace transform{

class reorder {
public:
  void run(ir::module& module);
};

}

}

}

#endif
