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
}

namespace triton
{

namespace driver
{

class cu_context;
class cu_device;

class module: public polymorphic_resource<CUmodule, cl_program> {
public:
  module(driver::context* ctx, CUmodule mod, bool has_ownership);
  module(driver::context* ctx, cl_program mod, bool has_ownership);
  driver::context* context() const;

protected:
  driver::context* ctx_;
};

class cu_module: public module {
  static std::string header(driver::cu_device const & device);
  std::string compile_llvm_module(llvm::Module* module);
  void init_llvm();

public:
  cu_module(driver::context* context, llvm::Module *module);
  cu_module(driver::context* context, const std::string& source);
  cu_buffer symbol(const char * name) const;

private:
  std::string source_;
};


}

}

#endif
