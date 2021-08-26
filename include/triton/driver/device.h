#pragma once

#ifndef _TRITON_DRIVER_DEVICE_H_
#define _TRITON_DRIVER_DEVICE_H_

#include "triton/driver/platform.h"
#include "triton/driver/handle.h"

namespace triton
{

namespace codegen
{
class target;
}

namespace driver
{

class context;

// Base device
class device: public polymorphic_resource<CUdevice, hipDeviceWrapper, host_device_t>{
public:
  using polymorphic_resource::polymorphic_resource;
  virtual std::unique_ptr<codegen::target> make_target() const = 0;
};

// Host device
class host_device: public device {
public:
  host_device(): device(host_device_t(), true){ }
  std::unique_ptr<codegen::target> make_target() const;
};

// CUDA device
class cu_device: public device {
private:
  //Metaprogramming elper to get cuda info from attribute
  template<CUdevice_attribute attr>
  int cuGetInfo() const;

public:
  cu_device(CUdevice cu = CUdevice(), bool take_ownership = true): device(cu, take_ownership){}
  // Compute Capability
  void interpret_as(int cc);
  int compute_capability() const;
  // attributes
  size_t max_shared_memory() const;
  void enable_peer_access(CUdeviceptr peer_mem_ptr) const;
  // Target
  std::unique_ptr<codegen::target> make_target() const;

private:
  std::shared_ptr<int> interpreted_as_;
};

}

}

#endif
