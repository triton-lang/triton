#pragma once

#ifndef _TRITON_RUNTIME_FUNCTION_H_
#define _TRITON_RUNTIME_FUNCTION_H_


#include <vector>
#include <string>
#include <memory>
#include <functional>
// codegen
#include "triton/ir/context.h"
#include "triton/codegen/target.h"
#include "triton/runtime/arg.h"

namespace llvm {
  class Module;
  class LLVMContext;
}

class Parser;

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
class tiles;
}
}

namespace ir {
class module;
class function;
class context;
}

namespace runtime{


typedef std::vector<size_t> grid_t;
typedef std::map<std::string, size_t> params_t;
template<typename T> inline T convert(const std::string& name);
template<> inline long convert<long>(const std::string& name) { return std::stol(name); }
template<> inline int convert<int>(const std::string& name) { return std::stoi(name); }

class function {
public:
  struct options_space_t {
    typedef std::pair<std::string, std::vector<std::string>> define_t;
    std::vector<define_t> defines;
    std::vector<int> num_warps;
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
    void operator()(driver::stream *stream, const grid_t& grid, const std::vector<arg>& args) const;
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
  static std::string preheader();

public:
  function(const std::string& src, const options_space_t& opt = options_space_t());
  void operator()(const std::vector<arg>& args, const grid_t& grid, driver::stream* stream);
  void operator()(const std::vector<arg>& args, const grid_fn_ty& grid, driver::stream *stream);
  void set_cst(const std::string& name, void* data, size_t n_bytes);

private:
  ir::context ctx_;
  std::string src_;
  options_space_t opt_space_;
  std::map<cache_key_t, caller> cache_;
  std::map<std::string, std::vector<char>> cst_;
};

}
}

#endif
