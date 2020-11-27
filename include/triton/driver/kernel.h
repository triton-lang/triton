#pragma once

#ifndef _TRITON_DRIVER_KERNEL_H_
#define _TRITON_DRIVER_KERNEL_H_

#include "triton/driver/module.h"
#include "triton/driver/handle.h"
#include <memory>

namespace llvm
{
class GenericValue;
}

namespace triton
{

namespace driver
{

class cu_buffer;

// Base
class kernel: public polymorphic_resource<CUfunction, host_function_t> {
public:
  kernel(driver::module* program, CUfunction fn, bool has_ownership);
  kernel(driver::module* program, host_function_t fn, bool has_ownership);
  driver::module* module();
  static kernel* create(driver::module* program, const char* name);
private:
  driver::module* program_;
};

// Host
class host_kernel: public kernel {
public:
  //Constructors
  host_kernel(driver::module* program, const char* name);
};

// CUDA
class cu_kernel: public kernel {
public:
  //Constructors
  cu_kernel(driver::module* program, const char * name);
};

}

}

#endif

