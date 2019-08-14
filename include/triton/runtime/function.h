#ifndef TDL_INCLUDE_FUNCTION_H
#define TDL_INCLUDE_FUNCTION_H

#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "arg.h"
// codegen
#include "triton/codegen/selection/selection.h"
#include "triton/codegen/selection/target.h"
#include "triton/codegen/analysis/tune.h"
#include "triton/codegen/analysis/shmem/allocation.h"
#include "triton/codegen/analysis/shmem/liveness.h"
#include "triton/codegen/analysis/shmem/info.h"
#include "triton/codegen/analysis/alignment.h"
#include "triton/codegen/transform/dce.h"
#include "triton/codegen/transform/peephole.h"
#include "triton/codegen/transform/shmem/barriers.h"
#include "triton/codegen/transform/reassociate.h"
#include "triton/codegen/transform/vectorize.h"

namespace llvm {
  class Module;
  class LLVMContext;
}

namespace triton {

namespace driver{
  class module;
  class stream;
  class kernel;
  class context;
  class device;
}

namespace lang{
class translation_unit;
}

namespace codegen{
namespace analysis{
class tune;
}
}

namespace ir {
class module;
class function;
class context;
class metaparameter;
}

namespace runtime{


typedef std::array<size_t, 3> grid_t;
typedef std::map<std::string, size_t> params_t;

struct options {
  size_t num_warps;
  params_t params;
};


class function {
public:
  typedef std::function<grid_t(const params_t&)> grid_fn_ty;

private:
  class caller {
  public:
    caller(ir::function *ir, std::shared_ptr<driver::module> program, size_t n_threads);
    void operator()(driver::stream *stream, const std::array<size_t, 3>& grid, const std::vector<arg>& args) const;

  private:
    std::shared_ptr<driver::module> parent_;
    std::shared_ptr<driver::kernel> bin_;
    std::vector<arg_type> param_tys_;
    size_t n_threads_;
  };

private:
  typedef std::pair<driver::device*, std::vector<int64_t>> cache_key_t;
  typedef std::pair<options, caller> cache_val_t;

private:
  triton::lang::translation_unit *make_ast(const char *src);
  std::unique_ptr<ir::module> make_ir(triton::lang::translation_unit *program);
  options autotune(lang::translation_unit *ast, driver::stream *stream, const grid_fn_ty& grid, const std::vector<arg> &args);
  std::unique_ptr<driver::module> make_bin(ir::module &function, driver::context *context, const options &opt);


public:
  function(const std::string& src);
  void operator()(const std::vector<arg>& args, const std::array<size_t, 3>& grid, driver::stream* stream);
  void operator()(const std::vector<arg>& args, const grid_fn_ty& grid, driver::stream *stream);

private:
  // execution context
  ir::context ctx_;
  // program representations
  std::string src_;
  lang::translation_unit *ast_;
  std::map<cache_key_t, cache_val_t> cache_;
};

}
}

#endif
