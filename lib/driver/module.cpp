/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <iostream>
#include <fstream>

#include "isaac/driver/module.h"
#include "isaac/driver/context.h"
#include "isaac/driver/error.h"

#include "isaac/tools/sys/getenv.hpp"

namespace isaac
{
namespace driver
{

CUjit_target_enum cutarget(Device::Architecture arch){
  switch(arch){
    case Device::Architecture::SM_2_0: return CU_TARGET_COMPUTE_20;
    case Device::Architecture::SM_2_1: return CU_TARGET_COMPUTE_21;
    case Device::Architecture::SM_3_0: return CU_TARGET_COMPUTE_30;
    case Device::Architecture::SM_3_5: return CU_TARGET_COMPUTE_35;
    case Device::Architecture::SM_3_7: return CU_TARGET_COMPUTE_37;
    case Device::Architecture::SM_5_0: return CU_TARGET_COMPUTE_50;
    case Device::Architecture::SM_5_2: return CU_TARGET_COMPUTE_52;
    case Device::Architecture::SM_6_0: return CU_TARGET_COMPUTE_60;
    case Device::Architecture::SM_6_1: return CU_TARGET_COMPUTE_61;
    default: throw;
  }
}

inline std::pair<int, int> ptx(std::pair<int, int> sm){
  if(sm.first == 7) return {6, 0};
  if(sm.first == 6) return {5, 0};
  if(sm.first == 5) return {4, 3};
  throw;
}

std::string Module::header(Device const & device){
  auto cc = device.compute_capability();
  auto vptx = ptx(cc);
  std::string header;
  header += ".version " + std::to_string(vptx.first) + "." + std::to_string(vptx.second) + "\n";
  header += ".target sm_" + std::to_string(cc.first) + std::to_string(cc.second) + "\n";
  header += ".address_size 64\n";
  return header;
}

Module::Module(Context const & context, std::string const & source) : context_(context), source_(header(context.device()) + source){
  ContextSwitcher ctx_switch(context_);

  //Path to custom PTX compiler
  std::string compiler = tools::getenv("ISAAC_PTXAS");
  if(compiler.size()){
    auto cc = context.device().compute_capability();
    std::string out = context.cache_path() + "tmp.o";
    std::string opt = " --gpu-name sm_" + std::to_string(cc.first) + std::to_string(cc.second)
                    + "  -o " + out
                    + "  -ias \"" + source_ + "\"";
    std::string cmd = compiler + opt;
    if(std::system(cmd.c_str()) != 0)
      throw;
    dispatch::cuModuleLoad(&*cu_, out.c_str());
  }
  //JIT Compilation
  else{
    CUjit_option opt[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER};
    unsigned int errbufsize = 8096;
    std::string errbuf(errbufsize, 0);
    //CUjit_target_enum target = cutarget(context.device().architecture());
    void* optval[] = {(void*)(uintptr_t)errbufsize, (void*)errbuf.data()};
    try{
      dispatch::cuModuleLoadDataEx(&*cu_, source_.data(), 2, opt, optval);
    }catch(exception::cuda::base const &){
      std::cerr << "Compilation Failed! Log: " << std::endl;
      std::cerr << errbuf << std::endl;
      throw;
    }
  }
}

Context const & Module::context() const
{ return context_; }

Handle<CUmodule> const & Module::cu() const
{ return cu_; }

}

}

