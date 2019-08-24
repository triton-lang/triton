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
#include "triton/lang/wgtcc/parser.h"

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
class grids;
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

template<typename T> T convert(const std::string& name);
template<> long convert<long>(const std::string& name) { return std::stol(name); }
template<> int convert<int>(const std::string& name) { return std::stoi(name); }

class function {
public:
  struct options_space_t {
    typedef std::pair<std::string, std::vector<std::string>> define_t;
    std::vector<define_t> defines;
    std::vector<size_t> num_warps;
  };

  struct options_t {
    template<class T>
    T D(const std::string& name) const {
      return convert<T>(defines.at(name));
    }

    std::map<std::string, std::string> defines;
    size_t num_warps;
  };

  typedef std::function<grid_t(const options_t&)> grid_fn_ty;


private:
  class caller {
  public:
    caller(ir::function *ir, std::shared_ptr<driver::module> program, const options_t& opt_);
    void operator()(driver::stream *stream, const std::array<size_t, 3>& grid, const std::vector<arg>& args) const;
    const options_t opt() const { return opt_; }

  private:
    std::shared_ptr<driver::kernel> bin_;
    std::shared_ptr<driver::module> parent_;
    std::vector<arg_type> param_tys_;
    options_t opt_;
  };

private:
  typedef std::pair<driver::device*, std::vector<int64_t>> cache_key_t;

private:
  triton::lang::translation_unit *make_ast(const std::string &src);
  std::unique_ptr<ir::module> make_ir(Parser &parser);
  std::unique_ptr<driver::module> make_bin(ir::module &function, driver::context *context, const options_t &opt);
  caller autotune(driver::stream *stream, const grid_fn_ty& grid, const std::vector<arg> &args);


public:
  function(const std::string& src, const options_space_t& opt = options_space_t());
  void operator()(const std::vector<arg>& args, const std::array<size_t, 3>& grid, driver::stream* stream);
  void operator()(const std::vector<arg>& args, const grid_fn_ty& grid, driver::stream *stream);
  std::string make_tensorflow_src(const std::vector<size_t> &outputs, const std::string &macro);

private:
  // execution context
  ir::context ctx_;
  // program representations
  std::string src_;
  std::map<cache_key_t, caller> cache_;
  // options
  options_space_t opt_space_;
};

}
}

#endif
