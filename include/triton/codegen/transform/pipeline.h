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
  pipeline(bool has_copy_async): has_copy_async_(has_copy_async) {}
  void run(ir::module &module);

private:
  bool has_copy_async_;
};

} // namespace transform
} // namespace codegen
} // namespace triton

#endif
