#ifndef TRITON_INCLUDE_IR_CODEGEN_PIPELINE_H
#define TRITON_INCLUDE_IR_CODEGEN_PIPELINE_H

// forward declaration
namespace triton {
namespace ir {
class module;
}
} // namespace triton

namespace triton {
namespace codegen {
namespace transform {

class pipeline {
public:
  void run(ir::module &module);
};

} // namespace transform
} // namespace codegen
} // namespace triton

#endif
