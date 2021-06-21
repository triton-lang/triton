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
  pipeline(bool has_copy_async, int num_stages)
      : has_copy_async_(has_copy_async), num_stages_(num_stages) {}
  void run(ir::module &module);

private:
  bool has_copy_async_;
  int num_stages_;
};

} // namespace transform
} // namespace codegen
} // namespace triton

#endif
