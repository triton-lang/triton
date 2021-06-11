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
class module: public polymorphic_resource<CUmodule, host_module_t> {
protected:
  void init_llvm();

  enum file_type_t{
    Object,
    Assembly
  };

public:
  module(CUmodule mod, bool has_ownership);
  module(host_module_t mod, bool has_ownership);
  static module* create(driver::device* device, std::unique_ptr<llvm::Module> src);
  void compile_llvm_module(std::unique_ptr<llvm::Module> module, const std::string& triple,
                           const std::string &proc, std::string layout,
                           llvm::SmallVectorImpl<char> &buffer,
                           const std::string &features,
                           file_type_t file_type);
  virtual std::unique_ptr<buffer> symbol(const char * name) const = 0;
  int spilled() const { return spilled_; }

protected:
  int spilled_;
};

// CPU
class host_module: public module{
public:
  host_module(std::unique_ptr<llvm::Module> module);
  std::unique_ptr<buffer> symbol(const char * name) const;
};

// CUDA
class cu_module: public module {
  std::string compile_llvm_module(llvm::Module* module, driver::device* device);
  void init_from_ptx(const std::string& ptx, cu_device *device);

public:
  cu_module(driver::device* device, std::unique_ptr<llvm::Module> module);
  cu_module(driver::device* device, const std::string& source);
  std::unique_ptr<buffer> symbol(const char * name) const;
  std::string llir() const { return llir_; }
  const std::string& ptx() const { return ptx_; }
  const std::string& cubin() const { return cubin_; }

private:
  std::string ptx_;
  std::string cubin_;
  std::string llir_;
};


}

}

#endif
