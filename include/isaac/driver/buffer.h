#ifndef ISAAC_DRIVER_BUFFER_H
#define ISAAC_DRIVER_BUFFER_H

#include "isaac/types.h"
#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include "isaac/driver/context.h"
#include "isaac/driver/handle.h"

namespace isaac
{

namespace driver
{

// Buffer
class ISAACAPI Buffer
{
  friend class CommandQueue;
  friend class Kernel;
public:
  Buffer(cl_mem Buffer = 0, bool take_ownership = true);
  Buffer(Context const & context, size_t size);
  Context const & context() const;
  size_t size() const;
  bool operator<(Buffer const &) const;
  bool operator==(Buffer const &) const;
  HANDLE_TYPE(cl_mem, CUdeviceptr)&  handle();
  HANDLE_TYPE(cl_mem, CUdeviceptr) const &  handle() const;
private:
  backend_type backend_;
  Context context_;
  size_t size_;
  HANDLE_TYPE(cl_mem, CUdeviceptr) h_;
};

}

}

#endif
