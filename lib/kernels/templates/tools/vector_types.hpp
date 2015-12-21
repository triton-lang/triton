/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

#include <string>
#include <cassert>

#include "isaac/driver/common.h"

#include "cpp/to_string.hpp"

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
