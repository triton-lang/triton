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

#include "triton/driver/module.h"
#include "triton/driver/context.h"
#include "triton/driver/error.h"

#include "triton/tools/sys/getenv.hpp"

namespace triton
{
namespace driver
{

Module::Module(Context const & context, std::string const & source) : context_(context), source_(source){
  ContextSwitcher ctx_switch(context_);
  // JIT compile source-code
  CUjit_option opt[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER};
  unsigned int errbufsize = 8096;
  std::string errbuf(errbufsize, 0);
  void* optval[] = {(void*)(uintptr_t)errbufsize, (void*)errbuf.data()};
  try{
    dispatch::cuModuleLoadDataEx(&*cu_, source_.data(), 2, opt, optval);
  }catch(exception::cuda::base const &){
    std::cerr << "Compilation Failed! Log: " << std::endl;
    std::cerr << errbuf << std::endl;
    throw;
  }
}

Context const & Module::context() const
{ return context_; }

Handle<CUmodule> const & Module::cu() const
{ return cu_; }

Buffer Module::symbol(const char *name) const{
  CUdeviceptr handle;
  size_t size;
  dispatch::cuModuleGetGlobal_v2(&handle, &size, *cu_, name);
  return Buffer(context_, handle, false);
}


}
}

