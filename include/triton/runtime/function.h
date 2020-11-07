#pragma once

#ifndef _TRITON_RUNTIME_FUNCTION_H_
#define _TRITON_RUNTIME_FUNCTION_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <set>
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
    std::vector<int> recompile_key;
  };

  struct options_t {
    template<class T>
    T D(const std::string& name) const {
      return convert<T>(defines.at(name));
    }
    bool operator<(const options_t& other) const {
      return std::make_pair(defines, num_warps) <
             std::make_pair(other.defines, other.num_warps);
    }
    std::string to_str() const;

    std::map<std::string, std::string> defines;
    size_t num_warps;
  };

  typedef std::function<grid_t(const options_t&)> grid_fn_ty;


private:
  class caller {
  public:
    // constructors
    caller(driver::context* ctx, std::ifstream& ifs, const options_t& opt);
    caller(ir::function *ir, std::shared_ptr<driver::module> program, const options_t& opt);
    // serialization
    void write(std::ofstream& ofs);
    void read(driver::context* ctx, std::ifstream& ifs);
    // accessors
    const options_t opt() const { return opt_; }
    const driver::module* parent() const { return &*parent_; }
    const driver::kernel* bin() const { return &*bin_; }
    arg_type param_ty(size_t i) const { return param_tys_.at(i);}
    const std::vector<arg_type>& param_tys() const { return param_tys_; }

    std::vector<int> retune() const { return retune_; }
    // entry points
    void operator()(driver::stream *stream, const grid_t& grid, void **args, size_t args_size) const;

  private:
    std::shared_ptr<driver::kernel> bin_;
    std::shared_ptr<driver::module> parent_;
    std::vector<arg_type> param_tys_;
    std::vector<int> retune_;
    options_t opt_;
    std::string name_;
  };

private:
  typedef std::pair<driver::device*, std::vector<int32_t>> cache_key_t;

private:
  // cache
  static std::string get_cache_prefix();
  // make
  triton::lang::translation_unit *make_ast(const std::string &src);
  std::unique_ptr<ir::module> make_ir(Parser &parser);
  std::unique_ptr<driver::module> make_bin(ir::module &function, driver::context *context, const options_t &opt);
  void make(driver::stream *stream, options_t opt);
  void precompile(driver::stream *stream, const options_space_t& tuning_space);
  // autotune
  caller* autotune(driver::stream *stream, const grid_fn_ty& grid, void **args, size_t args_size);

public:
  static std::string preheader();

public:
  function(const std::string& src, const options_space_t& opt, const std::string &cache_ref = "");
  void operator()(void** args, size_t args_size, const grid_t& grid, driver::stream* stream);
  void operator()(void** args, size_t args_size, const grid_fn_ty& grid, driver::stream *stream);
  void set_cst(const std::string& name, void* data, size_t n_bytes);
  std::string ptx(driver::stream *stream, const options_t& opt);

private:
  std::map<std::string, std::vector<char>> cst_;
  // pre-compilation
  ir::context ctx_;
  std::string src_;
  options_space_t opt_;
  std::set<options_t> compiled_;
  std::map<options_t, std::unique_ptr<caller>> callers_;
  std::vector<int> args_off_;
  size_t args_size_;
  // caching
  std::string cache_ref_;
  std::string cache_path_;
  std::map<cache_key_t, caller*> cache_;
};

}
}

#endif
