#pragma once

#ifndef _TRITON_DRIVER_BUFFER_H_
#define _TRITON_DRIVER_BUFFER_H_

#include "triton/driver/handle.h"
#include "triton/driver/context.h"

namespace triton
{
namespace driver
{

class stream;

// Base
class buffer : public polymorphic_resource<CUdeviceptr, hipDeviceptr_t, host_buffer_t> {
public:
  buffer(size_t size, hipDeviceptr_t hip, bool take_ownership);
  buffer(size_t size, CUdeviceptr cu, bool take_ownership);
  buffer(size_t size, host_buffer_t hst, bool take_ownership);
  uintptr_t addr_as_uintptr_t();
  static buffer* create(driver::context* ctx, size_t size);
  size_t size();

protected:
  size_t size_;
};

// CPU
class host_buffer: public buffer
{
public:
  host_buffer(size_t size);
};

// CUDA
class cu_buffer: public buffer
{
public:
  cu_buffer(size_t size);
  cu_buffer(size_t size, CUdeviceptr cu, bool take_ownership);
  void set_zero(triton::driver::stream *queue, size_t size);
};

// HIP
class hip_buffer: public buffer
{
public:
  hip_buffer(size_t size);
  hip_buffer(size_t size, hipDeviceptr_t hip, bool take_ownership);
  void set_zero(triton::driver::stream *queue, size_t size);
};

}
}

#endif


