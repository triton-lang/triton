#ifndef ISAAC_DRIVER_BUFFER_H
#define ISAAC_DRIVER_BUFFER_H

#include "isaac/driver/common.h"
#include "isaac/driver/context.h"
#include "isaac/driver/handle.h"

namespace isaac
{

namespace driver
{

// Buffer
class Buffer
{
  friend class CommandQueue;
  friend class Kernel;
public:
  Buffer(cl::Buffer const & Buffer);
  Buffer(Context const & context, std::size_t size);
  Context const & context() const;
  bool operator<(Buffer const &) const;
  bool operator==(Buffer const &) const;
  HANDLE_TYPE(cl::Buffer, CUdeviceptr)&  handle();
private:
  backend_type backend_;
  Context context_;
  HANDLE_TYPE(cl::Buffer, CUdeviceptr) h_;
};

}

}

#endif
