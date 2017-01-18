/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <string>
#include <cassert>

#include "isaac/driver/common.h"

#include "isaac/tools/cpp/string.hpp"

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

inline std::string access_vector_type(std::string const & v, int i, unsigned int vwidth)
{
    if(vwidth==1)
      return v;
    else
      return access_vector_type(v, i);
}

inline std::string append_width(std::string const & str, unsigned int width)
{
  if (width==1)
    return str;
  return str + tools::to_string(width);
}


inline std::string vstore(unsigned int vwidth, std::string const & dtype, std::string const & value, std::string const & offset, std::string const & ptr, std::string const & stride, driver::backend_type backend, bool aligned = true)
{
    std::string vdtype = append_width(dtype,vwidth);
    if (vwidth==1)
      return "(" + ptr + ")[" + offset + "] = " + value;
    else
    {
        if(backend == driver::CUDA && stride == "1" && aligned)
          return "reinterpret_cast<" + vdtype + "*>(" + ptr + ")[" + offset + "] = " + value;
        else if(backend == driver::OPENCL && stride == "1")
          return append_width("vstore", vwidth) + "(" + value + ", " + offset + ", " + ptr + ")";
        else
        {
          std::string stridestr = (stride=="1")?"":("*" + stride);
          std::string res;
          for(unsigned int s = 0 ; s < vwidth ; ++s)
              res +=  (s>0?";(":"(") + ptr + ")[" + offset + "*" + tools::to_string(vwidth) + " + " + tools::to_string(s) + stridestr + "] = " + access_vector_type(value, s);
          return res;
        }
    }
}


inline std::string vload(unsigned int vwidth, std::string const & dtype, std::string const & offset, std::string const & ptr, std::string const & stride, driver::backend_type backend, bool aligned = true)
{
    std::string vdtype = append_width(dtype,vwidth);
    if (vwidth==1)
      return "(" + ptr + ")[" + offset + "]";
    else
    {
      if(backend == driver::CUDA && stride == "1" && aligned)
          return "reinterpret_cast<" + vdtype + "*>(" + ptr + ")[" + offset + "]";
      else if(backend == driver::OPENCL && stride == "1")
          return append_width("vload", vwidth) + "(" + offset + ", " + ptr + ")";
      else
      {
        std::string stridestr = (stride=="1")?"":("*" + stride);
        std::string res;
        if(backend == driver::CUDA)
          res = "make_" + vdtype + "(";
        else
          res = "(" + vdtype + ")(";
        for(unsigned int s = 0 ; s < vwidth ; ++s)
            res += ((s>0)?",(":"(") + ptr + ")[" + offset + "*" + tools::to_string(vwidth) + " + " + tools::to_string(s) + stridestr  + "]";
        res += ")";
        return res;
      }
    }
}

}
}
