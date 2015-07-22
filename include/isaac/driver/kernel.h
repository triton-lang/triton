#ifndef ISAAC_DRIVER_KERNEL_H
#define ISAAC_DRIVER_KERNEL_H

#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/program.h"
#include "isaac/driver/handle.h"

#include <memory>

namespace isaac
{

namespace driver
{

class Buffer;

// Kernel
class ISAACAPI Kernel
{
  friend class CommandQueue;
public:
  Kernel(Program const & program, const char * name);
  void setArg(unsigned int index, std::size_t size, void* ptr);
  void setArg(unsigned int index, Buffer const &);
  void setSizeArg(unsigned int index, std::size_t N);
  template<class T> void setArg(unsigned int index, T value) { setArg(index, sizeof(T), (void*)&value); }

private:
  backend_type backend_;
  unsigned int address_bits_;
#ifdef ISAAC_WITH_CUDA
  std::vector<std::shared_ptr<void> >  cu_params_store_;
  std::vector<void*>  cu_params_;
#endif
  HANDLE_TYPE(cl::Kernel, CUfunction) h_;
};

}

}

#endif

