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

Module::Module(Context const & context, std::string const & source, bool is_ir) : context_(context), source_(source){
   ContextSwitcher ctx_switch(context_);

  //PTX passed directly
  if(is_ir){
    CUjit_option opt[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER};
    unsigned int errbufsize = 8096;
    std::string errbuf(errbufsize, 0);
    //CUjit_target_enum target = cutarget(context.device().architecture());
    void* optval[] = {(void*)(uintptr_t)errbufsize, (void*)errbuf.data()};
    try{
      dispatch::cuModuleLoadDataEx(&*cu_, source.data(), 2, opt, optval);
    }catch(exception::cuda::base const &){
      std::cerr << "Compilation Failed! Log: " << std::endl;
      std::cerr << errbuf << std::endl;
      throw;
    }
    return;
  }
  //Creates program
  nvrtcProgram prog;
  const char ** includes = NULL;
  const char ** src = NULL;
  dispatch::nvrtcCreateProgram(&prog, source.c_str(), NULL, 0, src, includes);
  try{
    std::pair<size_t, size_t> capability = context_.device().compute_capability();
    std::string capability_opt = "--gpu-architecture=compute_";
    capability_opt += std::to_string(capability.first) + std::to_string(capability.second);
    const char * options[] = {capability_opt.c_str(), "--restrict"};
    dispatch::nvrtcCompileProgram(prog, 2, options);
  }catch(exception::nvrtc::compilation const &){
    size_t logsize;
    dispatch::nvrtcGetProgramLogSize(prog, &logsize);
    std::string log(logsize, 0);
    dispatch::nvrtcGetProgramLog(prog, (char*)log.data());
    std::cerr << "Compilation failed:" << std::endl;
    std::cerr << log << std::endl;
  }
  size_t ptx_size;
  dispatch::nvrtcGetPTXSize(prog, &ptx_size);
  std::vector<char> ptx(ptx_size);
  dispatch::nvrtcGetPTX(prog, ptx.data());
  //Create binary
  dispatch::cuModuleLoadDataEx(&*cu_, ptx.data(), 0, NULL, NULL);
}

Context const & Module::context() const
{ return context_; }

Handle<CUmodule> const & Module::cu() const
{ return cu_; }

}

}

