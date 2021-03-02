#include <string>
#include <mutex>
#include <regex>
#include <functional>
#include <algorithm>
#include <sstream>
#include <memory>
#include "triton/codegen/analysis/axes.h"
#include "triton/codegen/analysis/allocation.h"
#include "triton/codegen/analysis/liveness.h"
#include "triton/codegen/analysis/align.h"
#include "triton/codegen/analysis/swizzle.h"
#include "triton/codegen/transform/coalesce.h"
#include "triton/codegen/transform/dce.h"
#include "triton/codegen/transform/peephole.h"
#include "triton/codegen/transform/membar.h"
#include "triton/codegen/transform/reassociate.h"
#include "triton/codegen/transform/cts.h"
#include "triton/codegen/transform/disassociate.h"
#include "triton/codegen/selection/generator.h"
#include "triton/codegen/transform/pipeline.h"
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
#include "triton/runtime/error.h"
#include "triton/tools/bench.hpp"
#include "triton/tools/sha1.hpp"
#include "triton/tools/sys/getenv.hpp"
#include "triton/tools/sys/mkdir.hpp"
#include "llvm/IR/Module.h"
#include <mutex>
#include <fstream>


namespace triton{
namespace runtime {

/* --------------------------------- */
/* --------------------------------- */
/* --------------------------------- */

std::shared_ptr<ir::module> kernel::src_to_ir(const std::string& _src, const options_t& opt) {
  std::string src =
R"(
#define bool _Bool
#define true 1
#define false 0

#define __readonly      __attribute__((readonly))
#define __writeonly     __attribute__((writeonly))
#define __noalias       __attribute__((noalias))
#define __aligned(A)    __attribute__((aligned(A)))
#define __multipleof(A) __attribute__((multipleof(A)))
#define __retune        __attribute__((retune))

#define F32_INFINITY bitcast<float>(0x7F800000)
#define F16_INFINITY bitcast<half>((int16)0x7C00)

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

#define PASTER(a, b, _) a ## _ ## b
#define EVALUATOR(a, b, _)  PASTER(a, b, _)
#define atomic_add(TYPE, TM, TN) EVALUATOR(atomic_add, EVALUATOR(TYPE, EVALUATOR(TM, TN, x), _), _)
#define DECLARATION(TYPE, TM, TN) extern void atomic_add(TYPE, TM, TN)(TYPE*[TM, TN], TYPE[TM, TN], bool[TM, TN])

DECLARATION(float, 64, 64);
DECLARATION(float, 64, 128);
DECLARATION(float, 128, 64);
DECLARATION(float, 128, 128);
extern void atomic_add_half_1x1(half*, half, bool);

DECLARATION(half , 64, 64);
DECLARATION(half , 64, 128);
DECLARATION(half , 128, 64);
DECLARATION(half , 128, 128);
extern void atomic_add_float_1x1(float*, float, bool);

extern int atomic_cas(int*, int, int);
extern int atomic_xchg(int*, int);
extern int get_program_id(int);
extern void __debug_barrier();
extern int get_num_programs(int);
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
  src += _src;
  // pre-process
  TokenSequence tokens;
  Preprocessor cpp(&src, true);
  for(auto it: opt.defines)
    cpp.AddMacro(it.first, &it.second);
  cpp.Process(tokens);
  // src -> ast
  Parser parser(tokens);
  parser.Parse();
  // ast -> triton-ir
  auto ret = std::make_shared<ir::module>("");
  Generator gen(&parser);
  gen.Gen(&*ret);
  return ret;
}

std::tuple<std::shared_ptr<driver::module>,
           std::shared_ptr<driver::kernel>,
           size_t> kernel::ir_to_bin(ir::module &ir, driver::device* dev, const options_t& opt) {
  // generate llvm code
  llvm::LLVMContext ctx;
  std::string name = ir.get_function_list()[0]->get_name();
  std::unique_ptr<llvm::Module> llvm(new llvm::Module(name, ctx));
  // optimizations
  std::unique_ptr<codegen::target> target = dev->make_target();
  bool cts_use_async = target->as_nvidia()->sm() >= 80;
  // create passes
  codegen::analysis::align align;
  codegen::analysis::axes axes;
  codegen::transform::cts cts(cts_use_async);
  codegen::transform::pipeline pipeline(cts_use_async);
  codegen::transform::disassociate disassociate;
  codegen::analysis::layouts layouts(&axes, &align, opt.num_warps, target.get());
  codegen::analysis::liveness liveness(&layouts);
  codegen::analysis::swizzle swizzle(&layouts, target.get());
  codegen::analysis::allocation allocation(&liveness);
  codegen::transform::membar barriers(&liveness, &layouts, &allocation);
  codegen::transform::dce dce;
  codegen::transform::peephole peephole(target.get(), &layouts);
  codegen::transform::reassociate reassociate;
  codegen::transform::coalesce coalesce(&align, &layouts);
  codegen::generator isel(&axes, &layouts, &align, &allocation, &swizzle, target.get(), opt.num_warps);
  // run passes
  dce.run(ir);
  pipeline.run(ir);
  dce.run(ir);
  disassociate.run(ir);
  dce.run(ir);
  align.run(ir);
  axes.run(ir);
  layouts.run(ir);
  peephole.run(ir);
  dce.run(ir);
  if(target->is_gpu())
    cts.run(ir);
  align.run(ir);
  axes.run(ir);
  layouts.run(ir);
  coalesce.run(ir);
  dce.run(ir);
  align.run(ir);
  dce.run(ir);
  if(target->is_gpu()){
    reassociate.run(ir);
    cts.run(ir);
  }
  dce.run(ir);
  align.run(ir);
  axes.run(ir);
  layouts.run(ir);
  peephole.run(ir);
  dce.run(ir);
  align.run(ir);
  axes.run(ir);
  layouts.run(ir);
  swizzle.run(ir);
  liveness.run(ir);
  allocation.run(ir);
  barriers.run(ir);
  isel.visit(ir, *llvm);
  std::shared_ptr<driver::module> mod(driver::module::create(dev, std::move(llvm)));
  std::shared_ptr<driver::kernel> ker(driver::kernel::create(&*mod, name.c_str()));
  size_t shared_mem = allocation.allocated_size();
  return std::make_tuple(mod, ker, shared_mem);
}

kernel::kernel(const std::string& src, const options_t& opt, driver::device *dev):
  opt(opt), dev_(dev) {
  ir_ = src_to_ir(src, opt);
  std::tie(mod_, ker_, shared_mem_) = ir_to_bin(*ir_, dev, opt);
}

void kernel::operator()(const std::string& args, driver::stream *stream, const std::vector<size_t>& _grid) const{
  // set grid
  if(_grid.size() > 3)
    throw std::runtime_error("grid size must be no greater than 3");
  std::array<size_t, 3> grid;
  for(size_t i = 0; i < 3; i++)
    grid[i] = (i < _grid.size()) ? _grid[i] : 1;
  // enqueue
  stream->enqueue(&*ker_, grid, {(size_t)opt.num_warps * 32, 1, 1}, (void*)args.data(), args.size(), shared_mem_);
}

std::string kernel::get_asm(asm_mode_t mode) {
  switch(mode){
      case ASM_LLIR:{
        return  ((driver::cu_module*)mod_.get())->llir();
      }
      case ASM_NV_PTX:
      case ASM_NV_SASS:{
        std::string ptx = ((driver::cu_module*)mod_.get())->ptx();
        // SASS
        std::string input = std::tmpnam(nullptr);
        std::string output = std::tmpnam(nullptr);
        std::ofstream ofs(input);
        ofs << ptx;
        ofs.close();
        if(mode == ASM_NV_PTX)
          return ptx;
        std::string cmd;
        int err;
        // compile ptx
        driver::cu_device* cu_device = (driver::cu_device*)dev_;
        cmd = "ptxas --gpu-name=sm_" + std::to_string(cu_device->compute_capability()) + " " + input + " -o " + input + ".o";
        err = system(cmd.c_str());
        // disassemble
        cmd = "cuobjdump --dump-sass " + input + ".o >> " + output;
        err = system(cmd.c_str());
        std::regex comment(" *\\/\\* 0x[0-9a-f]+ \\*\\/");
        std::string to_delete = "                                                                                           /*";
        std::ifstream ifs(output);
        std::string line;
        std::string sass;
        while(std::getline(ifs, line))
          if(!std::regex_match(line, comment))
            sass += line + "\n";
        return sass;
      }
      default:
        return "";
    }
}
/* --------------------------------- */
/* --------------------------------- */
/* --------------------------------- */


void function::init_kernels(const std::string& src, const options_t& opt,
                            const autotune_vals_t& confs, driver::device *device) {
  // list of all possible configs
  // just augment `opt` with each define of `confs`
  // and override warp count
  size_t num_opts = std::max(confs.size(), (size_t)1);
  std::vector<options_t> opts(num_opts, opt);
  for(size_t i = 0; i < confs.size(); i++){
    opts[i].defines.insert(confs[i].first.begin(), confs[i].first.end());
    opts[i].num_warps = confs[i].second;
  }
  // compile all possible configs
  // compilation errors (e.g., too much shared mem)
  // will populate `err`
  std::vector<std::pair<options_t, std::string>> err;
  for(const options_t& opt: opts) {
    try{
      kernels_.push_back({opt, std::make_shared<kernel>(src, opt, device)});
    }catch(const exception::base& e){
      err.push_back({opt, e.what()});
    }
  }
  // throw an exception if `err` is not empty
  if(kernels_.empty()){
    std::ostringstream dbg;
    dbg << "Auto-Tuner could not find any valid configuration:" << std::endl;
    for(auto x: err){
      dbg << "[ ";
      dbg << x.first.num_warps << ", ";
      dbg << "{ ";
      for(const auto& y: x.first.defines)
        dbg << '"' << y.first << "\"= \"" << y.second << "\", ";
      dbg << " } ] -> " << x.second << std::endl;
    }
    throw exception::no_valid_configuration(dbg.str());
  }
}



function::function(const std::string& src, const options_t &opt, driver::device *device,
                   const autotune_vals_t& autotune_vals, const std::vector<std::string>& autotune_key)  {  

  std::shared_ptr<ir::module> ir = kernel::src_to_ir(src, opt);
  std::vector<ir::argument*> args = ir->get_function_list()[0]->args();
  // find indices of autotune keys
  for(const std::string& name: autotune_key){
    auto pred = [&](ir::argument* arg) { return arg->get_name() == name; };
    auto it = std::find_if(args.begin(), args.end(), pred);
    if(it == args.end())
      throw std::runtime_error(name + " is not a valid argument name");
    key_idxs_.push_back(std::distance(args.begin(), it));
  }
  // signature
  auto convert = [](ir::type *ty) {
    if(ty->is_integer_ty(1))  return INT1_T;
    if(ty->is_integer_ty(8))  return INT8_T;
    if(ty->is_integer_ty(16)) return INT16_T;
    if(ty->is_integer_ty(32)) return INT32_T;
    if(ty->is_integer_ty(64)) return INT64_T;
    if(ty->is_half_ty())      return HALF_T;
    if(ty->is_float_ty())     return FLOAT_T;
    if(ty->is_double_ty())    return DOUBLE_T;
    if(ty->is_pointer_ty())   return BUFFER_T;
    throw std::runtime_error("unknown type");
  };
  for(ir::argument* arg: args)
    sig_.push_back(convert(arg->get_type()));
  // argument size and offset
  size_t curr = 0;
  for(arg_type ty: sig_){
    arg_size_.push_back(size_of(ty));
    arg_off_.push_back(curr);
    curr += arg_size_.back();
  }
  // pre-compile all kernels
  init_kernels(src, opt, autotune_vals, device);
}

kernel* function::autotune(const std::string &args, const grid_fn_ty& grid_fn, driver::stream* stream) {
  // fast path -- no autotuning necessary
  if(kernels_.size() == 1)
    return &*kernels_.begin()->second;
  // auto-tuning key
  std::vector<uint64_t> key(key_idxs_.size());
  for(size_t i = 0; i < key.size(); i++){
    int idx = key_idxs_[i];
    std::memcpy((void*)&key[i], (void*)((char*)args.data() + arg_off_[idx]), arg_size_[idx]);
  }
  auto it = cache_.find(key);
  if(it != cache_.end())
    return it->second;
  // run auto-tuner
  double best_ts = INFINITY;
  kernel* ret = nullptr;
  for(auto &x : kernels_){
    kernel* current = &*x.second;
    auto grid = grid_fn(x.first);
    while(grid.size() < 3)
      grid.push_back(1);
    double ts = tools::bench([&]() { (*current)(args, stream, grid); },
                                     stream, 5, 20);
    ret = (ts < best_ts) ? current : ret;
    best_ts = std::min(ts, best_ts);
  }
  stream->synchronize();
  it = cache_.insert({key, ret}).first;
  return it->second;
}

void function::operator()(const std::string& args, const grid_fn_ty& grid_fn, driver::stream *stream) {
  runtime::kernel* fn = autotune(args, grid_fn, stream);
  (*fn)(args, stream, grid_fn(fn->opt));
}


}
}
