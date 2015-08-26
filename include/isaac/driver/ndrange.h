#ifndef ISAAC_DRIVER_NDRANGE_H
#define ISAAC_DRIVER_NDRANGE_H

#include "isaac/defines.h"
#include "isaac/driver/common.h"

namespace isaac
{

namespace driver
{

// NDRange
class ISAACAPI NDRange
{
public:
  NDRange(size_t size0);
  NDRange(size_t size0, size_t size1);
  NDRange(size_t size0, size_t size1, size_t size2);
  int dimension() const;
  operator const size_t*() const;
private:
  size_t sizes_[3];
  int dimension_;
};

}

}

#endif
