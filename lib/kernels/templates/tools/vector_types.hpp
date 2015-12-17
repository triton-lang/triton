#include <string>
#include <cassert>

#include "isaac/driver/common.h"

#include "to_string.hpp"

namespace isaac
{
namespace templates
{

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


inline std::string vstore(unsigned int simd_width, std::string const & dtype, std::string const & value, std::string const & offset, std::string const & ptr, std::string const & stride, driver::backend_type backend, bool aligned = true)
{
    std::string vdtype = append_width(dtype,simd_width);
    if (simd_width==1)
      return "(" + ptr + ")[" + offset + "] = " + value;
    else
    {
        if(backend == driver::CUDA && stride == "1" && aligned)
          return "reinterpret_cast<" + vdtype + "*>(" + ptr + ")[" + offset + "] = " + value;
        else if(backend == driver::OPENCL && stride == "1")
          return append_width("vstore", simd_width) + "(" + value + ", " + offset + ", " + ptr + ")";
        else
        {
          std::string stridestr = (stride=="1")?"":("*" + stride);
          std::string res;
          for(unsigned int s = 0 ; s < simd_width ; ++s)
              res +=  (s>0?";(":"(") + ptr + ")[" + offset + "*" + tools::to_string(simd_width) + " + " + tools::to_string(s) + stridestr + "] = " + access_vector_type(value, s);
          return res;
        }
    }
}


inline std::string vload(unsigned int simd_width, std::string const & dtype, std::string const & offset, std::string const & ptr, std::string const & stride, driver::backend_type backend, bool aligned = true)
{
    std::string vdtype = append_width(dtype,simd_width);
    if (simd_width==1)
      return "(" + ptr + ")[" + offset + "]";
    else
    {
      if(backend == driver::CUDA && stride == "1" && aligned)
          return "reinterpret_cast<" + vdtype + "*>(" + ptr + ")[" + offset + "]";
      else if(backend == driver::OPENCL && stride == "1")
          return append_width("vload", simd_width) + "(" + offset + ", " + ptr + ")";
      else
      {
        std::string stridestr = (stride=="1")?"":("*" + stride);
        std::string res;
        if(backend == driver::CUDA)
          res = "make_" + vdtype + "(";
        else
          res = "(" + vdtype + ")(";
        for(unsigned int s = 0 ; s < simd_width ; ++s)
            res += ((s>0)?",(":"(") + ptr + ")[" + offset + "*" + tools::to_string(simd_width) + " + " + tools::to_string(s) + stridestr  + "]";
        res += ")";
        return res;
      }
    }
}

}
}
