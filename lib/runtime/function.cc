#include <unistd.h>
#include <fstream>
#include <string>
#include <mutex>
#include <regex>
#include <functional>
#include <algorithm>
#include <memory>
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/analysis/liveness.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/transform/coalesce.h"
#include "triton/codegen/transform/dce.h"
#include "triton/codegen/transform/peephole.h"
#include "triton/codegen/transform/membar.h"
#include "triton/codegen/transform/reassociate.h"
#include "triton/codegen/transform/cts.h"
#include "triton/codegen/transform/disassociate.h"
#include "triton/codegen/selection/generator.h"
#include "triton/runtime/function.h"
#include "triton/lang/cpp.h"
#include "triton/lang/parser.h"
#include "triton/lang/code_gen.h"
#include "triton/driver/device.h"
#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include "triton/driver/module.h"
#include "triton/driver/error.h"
#include "triton/ir/module.h"
#include "triton/ir/function.h"
#include "triton/ir/print.h"
#include "triton/tools/bench.hpp"
#include "triton/tools/sha1.hpp"
#include "triton/tools/sys/getenv.hpp"
#include "triton/tools/sys/mkdir.hpp"
#include "llvm/IR/Module.h"

std::mutex mut;

namespace triton{
namespace runtime {

/* --------------------- */
/*    HELPERS            */
/* --------------------- */

void _loop_nest(std::vector<size_t> const & ranges,
                       std::function<void(std::vector<size_t> const &)> const & f){
  size_t D = ranges.size();
  std::vector<size_t> values(D, 0);
  size_t i = D - 1;
  while(true){
    f(values);
    while(values[i]++ == ranges[i] - 1){
      if(i == 0)
        return;
      values[i--] = 0;
    }
    i = D - 1;
  }
}


/* --------------------- */
/*    OPTIONS            */
/* --------------------- */

std::string function::options_t::to_str() const{
  std::string ret = "nw-" + std::to_string(num_warps);
  for(const auto& x : defines){
    ret += '-';
    ret += x.first;
    ret += '-';
    ret += x.second;
  }
  // legalize
  for(char& x: ret){
    if(x == ' ' || x == '^' || x == ',' || x == ':')
      x = '_';
  }
  return ret;
}


/* --------------------- */
/*     CALLER OBJECT     */
/* --------------------- */

arg_type convert(ir::type *ty) {
  if(ty->is_integer_ty(1))
    return INT1_T;
  if(ty->is_integer_ty(8))
    return INT8_T;
  if(ty->is_integer_ty(16))
    return INT16_T;
  if(ty->is_integer_ty(32))
    return INT32_T;
  if(ty->is_integer_ty(64))
    return INT64_T;
  if(ty->is_half_ty())
    return HALF_T;
  if(ty->is_float_ty())
    return FLOAT_T;
  if(ty->is_double_ty())
    return DOUBLE_T;
  if(ty->is_pointer_ty())
    return BUFFER_T;
  throw std::runtime_error("unknown type");
}

void function::caller::write(std::ofstream &ofs) {
  // write name
  ofs << name_ << std::endl;
  // write signature
  for(size_t i = 0; i < param_tys_.size(); i++)
    ofs << param_tys_[i] << " ";
  ofs << std::endl;
  // write module
  std::string source = ((driver::cu_module*)(&*parent_))->source();
  ofs << source;
}

void function::caller::read(driver::context* ctx, std::ifstream &ifs) {
  // read name
  std::getline(ifs, name_);
  // read signature
  std::string line;
  std::getline(ifs, line);
  std::istringstream current(line);
  int param;
  param_tys_.clear();
  while(current >> param)
    param_tys_.push_back((arg_type)param);
  // read module
  std::string src((std::istreambuf_iterator<char>(ifs)),
                   std::istreambuf_iterator<char>());
  parent_.reset(new driver::cu_module(ctx, src));
  bin_.reset(driver::kernel::create(&*parent_, name_.c_str()));

}

function::caller::caller(driver::context* ctx, std::ifstream &ifs, const options_t& opt)
  : opt_(opt) {
  read(ctx, ifs);
}

function::caller::caller(ir::function *ir,
                         std::shared_ptr<driver::module> parent, const options_t& opt)
  : parent_(parent), opt_(opt), name_(ir->get_name()) {
  bin_.reset(driver::kernel::create(&*parent, name_.c_str()));
  // extract signature
  ir::function_type* ty = ir->get_fn_type();
  for(size_t i = 0; i < ty->get_num_params(); i++)
    param_tys_.push_back(convert(ty->get_param_ty(i)));
}


void function::caller::operator ()(driver::stream *stream, const grid_t& _grid, const std::vector<arg>& args) const {
  if(args.size() != param_tys_.size())
    throw std::runtime_error("invalid number of arguments");
  // set arguments
  for(size_t i = 0; i < args.size(); i++){
    arg arg_i = args.at(i);
    arg_type ty = arg_i.type();
    if(ty != param_tys_.at(i))
      throw std::runtime_error("invalid type for argument " + std::to_string(i));
    if(ty == BUFFER_T){
      driver::buffer* buf = *((driver::buffer**)arg_i.data());
      bin_->setArg(i, buf->size() == 0 ? nullptr : buf);
    }
    else
      bin_->setArg(i, size_of(ty), arg_i.data());
  }
  // set grid
  if(_grid.size() > 3)
    throw std::runtime_error("grid size must be no greater than 3");
  std::array<size_t, 3> grid;
  for(size_t i = 0; i < 3; i++)
    grid[i] = (i < _grid.size()) ? _grid[i] : 1;
  // enqueue
  stream->enqueue(&*bin_, grid, {opt_.num_warps * 32, 1, 1});
}


/* --------------------- */
/*    FUNCTION           */
/* --------------------- */

// create Triton-IR from AST
std::unique_ptr<ir::module> function::make_ir(Parser& parser) {
  ir::module* module = new ir::module("", ctx_);
  Generator gen(&parser);
  gen.Gen(module);
  return std::unique_ptr<ir::module>(module);
}

// create Binary from Triton-IR
std::unique_ptr<driver::module> function::make_bin(ir::module &module,
                                                   driver::context *context,
                                                   const options_t& opt) {
  std::unique_ptr<codegen::target> target = context->device()->make_target();
  // generate llvm code
  llvm::LLVMContext ctx;
  std::unique_ptr<llvm::Module> llvm(new llvm::Module(module.get_name(), ctx));
  // create passes
  codegen::analysis::align align;
  codegen::analysis::axes axes;
  codegen::transform::disassociate disassociate;
  codegen::analysis::layouts layouts(&axes, &align, opt.num_warps);
  codegen::analysis::liveness liveness(&layouts);
  codegen::analysis::allocation allocation(&liveness);
  codegen::transform::membar barriers(&liveness, &layouts, &allocation);
  codegen::transform::dce dce;
  codegen::transform::peephole peephole;
  codegen::transform::reassociate reassociate;
  codegen::transform::coalesce coalesce(&align, &layouts);
  codegen::transform::cts cts;
  codegen::generator isel(&axes, &layouts, &align, &allocation, target.get(), opt.num_warps);
  // run passes
  dce.run(module);
  disassociate.run(module);
  dce.run(module);
  peephole.run(module);
  dce.run(module);
  align.run(module);
  cts.run(module);
  axes.run(module);
  layouts.run(module);
  coalesce.run(module);
  dce.run(module);
  align.run(module);
  dce.run(module);
  reassociate.run(module);
  cts.run(module);
  peephole.run(module);
  dce.run(module);
  align.run(module);
  axes.run(module);
  layouts.run(module);
  liveness.run(module);
  allocation.run(module);
  if(allocation.allocated_size() > context->device()->max_shared_memory())
    throw std::runtime_error("using too much shared memory");
  barriers.run(module);
  isel.visit(module, *llvm);
  std::unique_ptr<driver::module> res(driver::module::create(context, std::move(llvm)));
  return res;
}


// create Binary from options
function::caller* function::make(driver::stream *stream, options_t opt) {
  // cache path
  std::string cache_path = cache_path_ + opt.to_str() + ".ptx";
  int ref_mtime = tools::mtime(cache_ref_);
  int ptx_mtime = tools::mtime(cache_path);
  // if cached ptx is newer than reference library
  if(!ref_mtime || !ptx_mtime || ref_mtime < ptx_mtime){
    std::ifstream ifs(cache_path);
    // file is empty -- invalid
    if(ifs && ifs.peek() == std::ifstream::traits_type::eof())
      return nullptr;
    // load cached caller
    if(ifs)
      return new caller(stream->context(), ifs, opt);
  }
  // pre-process
  TokenSequence tokens;
  Preprocessor cpp(&src_, true);
  for(auto it: opt.defines)
    cpp.AddMacro(it.first, &it.second);
  cpp.Process(tokens);
  // src -> ast
  Parser parser(tokens);
  parser.Parse();
  // ast -> triton-ir
  auto ir = make_ir(parser);
  // triton-ir -> binary
  std::unique_ptr<driver::module> bin;
  try{
    bin = make_bin(*ir, stream->context(), opt);
  }catch(const std::runtime_error&){
    if(!cache_path_.empty())
      std::ofstream ofs(cache_path);
    return nullptr;
  }
  // create callable
  ir::function *tmp = ir->get_function_list()[0];
  caller* ret = new caller(tmp, std::move(bin), opt);
  // serialize callable
  if(!cache_path_.empty()){
    int pid = getpid();
    char pid_str[100];
    sprintf(pid_str, "%d", pid);
    std::string tmp_cache_path = cache_path + "." + pid_str;
    std::ofstream ofs(tmp_cache_path);
    ret->write(ofs);
    std::string command = "rm -f " + tmp_cache_path;
    std::ifstream ifs(cache_path);
    // file is empty -- invalid
    if(ifs && ifs.peek() == std::ifstream::traits_type::eof()) {
      system(const_cast<char*>(command.c_str()));
      return nullptr;
    }
    // load cached caller
    if(ifs) {
      system(const_cast<char*>(command.c_str()));
      return new caller(stream->context(), ifs, opt);
    }
    command = "mv " + tmp_cache_path + " " + cache_path;
    system(const_cast<char*>(command.c_str()));
  }
  return ret;
}

// precompile all kernels spanned by given options space
void function::precompile(driver::stream* stream,
                          const options_space_t& space) {
  // all ranges
  std::vector<size_t> ranges;
  ranges.push_back(space.num_warps.size());
  for(const auto& x: space.defines)
    ranges.push_back(x.second.size());
  // functor for source with given option
  auto do_make = [&](std::vector<size_t> params) {
    // compilation options
    unsigned i = 0;
    options_t opt;
    opt.num_warps = space.num_warps[params[i++]];
    for(auto D: space.defines)
      opt.defines[D.first] = D.second[params[i++]];
    // compile
    caller* call = make(stream, opt);
    if(!call)
      return;
    // copy constants
    std::unique_ptr<driver::buffer> buffer;
    for(const auto& cst: cst_){
      buffer = call->parent()->symbol(cst.first.c_str());
      stream->write(&*buffer, true, 0, cst.second);
    }
    callers_[opt].reset(call);
  };
  // multi-threaded compilation
  _loop_nest(ranges, do_make);
  if(callers_.empty())
    throw std::runtime_error("could not compile kernel");
}

// return auto-tuning key for given function arguments
function::cache_key_t function::get_key(driver::stream *stream, const std::vector<arg>& args) {
  cache_key_t ret;
  ret.first = stream->context()->device();
  for(size_t i = 0; i < args.size(); i++){
    arg_type ty = args.at(i).type();
    if(!is_int_type(ty))
      continue;
    long val = 0;
    std::memcpy((void*)&val, args.at(i).data(), size_of(ty));
    ret.second.push_back(val);
  }
  return ret;
}
// returns program with best compilation options for given parameter
function::caller* function::autotune(driver::stream* stream, const grid_fn_ty& grid_fn,
                                       const std::vector<arg>& args) {
  // fast path -- no autotuning necessary
  if(callers_.size() == 1)
    return &*callers_.begin()->second;
  // slow path -- autotuning necessary
  // copy buffer argument so that auto-tuning doesn't corrupt data
  std::list<std::shared_ptr<driver::cu_buffer>> copies;
  std::vector<arg> _args = args;
  for(size_t i = 0; i < args.size(); i++)
    if(_args[i].type() == BUFFER_T){
      driver::buffer* old = _args[i].buffer();
      size_t size = old->size();
      // only copy scalars
      // TODO: change that
      if(size != 4 && size != 2)
        continue;
      copies.push_back(std::make_shared<driver::cu_buffer>(old->context(), size));
      _args[i] = arg(copies.back().get());
    }
  double best_ts = INFINITY;
  caller* ret = nullptr;
  for(auto &x : callers_){
    if(x.second == nullptr)
      throw std::runtime_error("configuration not compiled");
    caller* current = &*x.second;
    double ts = tools::bench([&]() { (*current)(stream, grid_fn(x.first), args); },
                                     stream, true);
    ret = (ts < best_ts) ? current : ret;
    best_ts = std::min(ts, best_ts);
  }
  stream->synchronize();
  return ret;
}

// set copy host buffer "data" into constant memory buffer "name"
void function::set_cst(const std::string& name, void* data, size_t n_bytes) {
   cst_[name] = std::vector<char>((char*)data, (char*)data + n_bytes);
}


std::string function::preheader() {
  return  R"(
#define bool _Bool
#define true 1
#define false 0

#define __readonly      __attribute__((readonly))
#define __writeonly     __attribute__((writeonly))
#define __noalias       __attribute__((noalias))
#define __aligned(A)    __attribute__((aligned(A)))
#define __multipleof(A) __attribute__((multipleof(A)))

#define F32_INFINITY bitcast<float>(0x7F800000)
#define F16_INFINITY bitcast<half>((int16)0x7C00)

extern int atomic_cas(int*, int, int);
extern int atomic_xchg(int*, int);
extern float f32_atomic_add(float*, float);
extern int get_program_id(int);
extern int get_num_programs(int);
extern float sqrtf(float);
extern int select(bool, int, int);
extern char __constant__ * calloc(int);

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long uint64;
typedef char int8;
typedef short int16;
typedef int int32;
typedef long int64;
)";
}

std::string function::get_cache_prefix() {
  //user-specified cache path
  std::string result = tools::getenv("TRITON_CACHE_PATH");
  if(!result.empty()){
    if(tools::mkpath(result)==0)
      return result;
  }
  //create in home
  result = tools::getenv("HOME");
  if(!result.empty())
  {
    result = result + "/.triton/cache/";
    if(tools::mkpath(result)==0)
      return result;
  }
  return "";
}

function::function(const std::string &src,
                   const options_space_t& opt,
                   const std::string &cache_ref):
    src_(src), opt_(opt), cache_ref_(cache_ref) {
  // hash source code
  unsigned char hash[20];
  sha1::calc((void*)src_.data(), src_.size(), hash);
  // create cache path
  char _hex[40];
  sha1::toHexString(hash, _hex);
  std::string hex(_hex, _hex + 40);
  cache_path_ = get_cache_prefix() + hex + "/";
  tools::mkpath(cache_path_);
  // append pre-header to source
  src_ = preheader() + src_;
}

void function::operator()(const std::vector<arg>& args,
                          const grid_fn_ty& grid_fn,
                          driver::stream *stream) {
  // pre-compile kernels
  if(callers_.empty())
    precompile(stream, opt_);
  // auto-tune if necessary
  auto key = get_key(stream, args);
  auto it = cache_.find(key);
  if(it == cache_.end()){
    auto best = autotune(stream, grid_fn, args);
    it = cache_.insert({key, best}).first;
  }
  // run
  (*it->second)(stream, grid_fn(it->second->opt()), args);
}

void function::operator()(const std::vector<arg>& args,
                          const grid_t& grid,
                          driver::stream *stream) {
  return this->operator()(args, [&grid](const options_t&){ return grid; }, stream);
}



}
}
