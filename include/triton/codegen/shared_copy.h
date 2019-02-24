#ifndef TDL_INCLUDE_CODEGEN_SHARED_COPY_H
#define TDL_INCLUDE_CODEGEN_SHARED_COPY_H

#include <tuple>
#include <vector>

namespace triton {

namespace ir {
  class module;
  class value;
  class builder;
  class basic_block;
}

namespace codegen{

class buffer_info_pass;

class place_shared_copy {
private:
  typedef std::pair<unsigned, unsigned> interval_t;
  typedef std::vector<interval_t> interval_vec_t;

private:
  bool intersect(const interval_vec_t &I, interval_t i);
  void add_copy(ir::value *x, ir::builder &builder);

public:
  place_shared_copy(buffer_info_pass *info): info_(info) { }
  void run(ir::module &mod);

private:
  buffer_info_pass *info_;
};


}
}

#endif
