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

#ifndef TDL_INCLUDE_DRIVER_MODULE_H
#define TDL_INCLUDE_DRIVER_MODULE_H

#include <map>
#include "triton/driver/handle.h"
#include "triton/driver/context.h"
#include "triton/driver/buffer.h"

namespace llvm
{
  class Module;
  template<class T>
  class SmallVectorImpl;
}

namespace triton
{

namespace driver
{

class cu_context;
class cu_device;

// Base
class module: public polymorphic_resource<CUmodule, cl_program, host_module_t> {
protected:
  void init_llvm();

  enum file_type_t{
    Object,
    Assembly
  };

public:
  module(driver::context* ctx, CUmodule mod, bool has_ownership);
  module(driver::context* ctx, cl_program mod, bool has_ownership);
  module(driver::context* ctx, host_module_t mod, bool has_ownership);
  static module* create(driver::context* ctx, llvm::Module *src);
  driver::context* context() const;
  void compile_llvm_module(llvm::Module* module, const std::string& triple,
                                  const std::string &proc, std::string layout,
                           llvm::SmallVectorImpl<char> &buffer,
                           const std::string &features,
                           file_type_t file_type);

protected:
  driver::context* ctx_;
};

// CPU
class host_module: public module{
public:
  host_module(driver::context* context, llvm::Module *module);
};

// OpenCL
class ocl_module: public module{

public:
  ocl_module(driver::context* context, llvm::Module *module);
};

// CUDA
class cu_module: public module {
  std::string compile_llvm_module(llvm::Module* module);

public:
  cu_module(driver::context* context, llvm::Module *module);
  cu_module(driver::context* context, const std::string& source);
  cu_buffer* symbol(const char * name) const;

private:
  std::string source_;
};


}

}

#endif
