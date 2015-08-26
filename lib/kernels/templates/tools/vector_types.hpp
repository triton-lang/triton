#include <string>
#include <cassert>

#include "isaac/driver/common.h"

#include "to_string.hpp"

namespace isaac
{
namespace templates
{

inline std::string append_simd_suffix(std::string const & str, unsigned int i)
{
  assert(i < 16);
  char suffixes[] = {'0','1','2','3','4','5','6','7','8','9',
                           'a','b','c','d','e','f'};
  return str + tools::to_string(suffixes[i]);
}


inline std::string access_vector_type(std::string const & v, int i)
{
    switch(i)
    {
      case 0: return v + ".x";
      case 1: return v + ".y";
      case 2: return v + ".z";
      case 3: return v + ".w";
      default: throw;
    }
}

inline std::string append_width(std::string const & str, unsigned int width)
{
  if (width==1)
    return str;
  return str + tools::to_string(width);
}


inline std::string vstore(unsigned int simd_width, std::string const & dtype, std::string const & value, std::string const & offset, std::string const & ptr, driver::backend_type backend)
{
    std::string vdtype = append_width(dtype,simd_width);
    if (simd_width==1)
      return "(" + ptr + ")[" + offset + "] = " + value;
    else
    {
      switch(backend)
      {
        case driver::CUDA:
          return "reinterpret_cast<" + vdtype + "*>(" + ptr + ")[" + offset + "] = " + value;
        case driver::OPENCL:
          return append_width("vstore", simd_width) + "(" + value + ", " + offset + ", " + ptr + ")";
        default:
          throw;
      }
    }
}

inline std::string vload(unsigned int simd_width, std::string const & dtype, std::string const & offset, std::string const & ptr, driver::backend_type backend)
{
    std::string vdtype = append_width(dtype,simd_width);
    if (simd_width==1)
      return "(" + ptr + ")[" + offset + "]";
    else
    {
      switch(backend)
      {
        case driver::CUDA:
          return "reinterpret_cast<" + vdtype + "*>(" + ptr + ")[" + offset + "]";
        case driver::OPENCL:
          return append_width("vload", simd_width) + "(" + offset + ", " + ptr + ")";
        default:
          throw;
      }
    }
}

}
}
