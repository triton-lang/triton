#pragma once

#ifndef _TRITON_DRIVER_CONTEXT_H_
#define _TRITON_DRIVER_CONTEXT_H_

#include "triton/driver/device.h"
#include "triton/driver/handle.h"

namespace triton
{
namespace driver
{

class context: public polymorphic_resource<CUcontext, hipCtx_t, host_context_t>{
protected:
  static std::string get_cache_path();

public:
  context(CUcontext cu, bool take_ownership);
  context(host_context_t hst, bool take_ownership);
  std::string const & cache_path() const;
  static context* create(driver::device *dev);

protected:
  std::string cache_path_;
};

// Host
class host_context: public context {
public:
  host_context(driver::host_device* dev);
};

// CUDA
class cu_context: public context {
public:
  cu_context(CUcontext cu, bool take_ownership = true);
  cu_context(driver::cu_device* dev);
};

// HIP
class hip_context: public context {
public:
  hip_context(hipCtx_t hip, bool take_ownership = true);
//  hip_context(driver::hip_device* dev);
};

}
}

#endif
