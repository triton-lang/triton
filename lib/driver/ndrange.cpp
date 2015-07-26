#include "isaac/driver/ndrange.h"

namespace isaac
{

namespace driver
{

NDRange::NDRange(size_t size0, size_t size1, size_t size2)
{
    sizes_[0] = size0;
    sizes_[1] = size1;
    sizes_[2] = size2;
}

size_t NDRange::dimension() const
{
 return (int)(sizes_[0]>1) + (int)(sizes_[1]>1) + (int)(sizes_[2]>1);
}

NDRange::operator const size_t*() const
{
  return (const size_t*) sizes_;
}

}

}
