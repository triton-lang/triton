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

NDRange::operator cl::NDRange() const
{
  return cl::NDRange(sizes_[0], sizes_[1], sizes_[2]);
}

NDRange::operator const size_t*() const
{
  return (const size_t*) sizes_;
}

}

}
