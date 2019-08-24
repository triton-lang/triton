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
class device: public polymorphic_resource<CUdevice, cl_device_id, host_device_t>{
public:
  using polymorphic_resource::polymorphic_resource;
  virtual size_t max_threads_per_block() const = 0;
  virtual size_t max_shared_memory() const = 0;
  virtual std::unique_ptr<codegen::target> make_target() const = 0;
};

// Host device
class host_device: public device {
public:
  host_device(): device(host_device_t(), true){ }
  size_t max_threads_per_block() const { return 1; }
  size_t max_shared_memory() const { return 0; }
  std::unique_ptr<codegen::target> make_target() const;
};

// OpenCL device
class ocl_device: public device {
public:
  ocl_device(cl_device_id cl, bool take_ownership = true): device(cl, take_ownership) { }
  size_t max_threads_per_block() const;
  size_t max_shared_memory() const;
  std::unique_ptr<codegen::target> make_target() const;
};

// CUDA device
class cu_device: public device {
public:
  //Supported architectures
  enum class Architecture{
    //NVidia
    SM_2_0,
    SM_2_1,
    SM_3_0,
    SM_3_5,
    SM_3_7,
    SM_5_0,
    SM_5_2,
    SM_6_0,
    SM_6_1,
    SM_7_0,
    UNKNOWN
  };

private:
  //Metaprogramming elper to get cuda info from attribute
  template<CUdevice_attribute attr>
  int cuGetInfo() const;

  inline Architecture nv_arch(std::pair<unsigned int, unsigned int> sm) const;
  inline nvmlDevice_t nvml_device() const;

public:
  cu_device(CUdevice cu = CUdevice(), bool take_ownership = true): device(cu, take_ownership){}
  // Accessors
  Architecture architecture() const;
  // Informations
  std::string infos() const;
  size_t address_bits() const;
  std::vector<size_t> max_block_dim() const;
  size_t warp_size() const;
  // Compute Capability
  void interpret_as(std::pair<size_t, size_t> cc);
  std::pair<size_t, size_t> compute_capability() const;
  // Identifier
  std::string name() const;
  std::string pci_bus_id() const;
  // Clocks
  size_t current_sm_clock() const;
  size_t current_mem_clock() const;
  size_t max_threads_per_block() const;
  size_t max_shared_memory() const;
  size_t max_sm_clock() const;
  size_t max_mem_clock() const;
  void set_max_clock();
  // Target
  std::unique_ptr<codegen::target> make_target() const;

private:
  std::shared_ptr<std::pair<size_t, size_t>> interpreted_as_;
};

}

}

#endif
