#pragma once

#ifndef _TRITON_DRIVER_MODULE_H_
#define _TRITON_DRIVER_MODULE_H_

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
