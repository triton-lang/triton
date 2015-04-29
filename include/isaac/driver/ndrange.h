#ifndef ISAAC_DRIVER_NDRANGE_H
#define ISAAC_DRIVER_NDRANGE_H

#include "isaac/driver/common.h"

namespace isaac
{

namespace driver
{

// NDRange
class NDRange
{
public:
  NDRange(size_t size0 = 1, size_t size1 = 1, size_t size2 = 1);
  operator cl::NDRange() const;
  operator const size_t*() const;
private:
  size_t sizes_[3];
};

}

}

#endif
