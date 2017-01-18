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

#ifndef ISAAC_PYTHON_COMMON_HPP
#define ISAAC_PYTHON_COMMON_HPP

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <boost/numpy/dtype.hpp>

#include "isaac/array.h"

#define MAP_ENUM(v, ns) .value(#v, ns::v)

namespace bp = boost::python;
namespace np = boost::numpy;

namespace sc = isaac;
namespace rt = sc::runtime;

namespace tools
{

    template<class IT>
    bp::list to_list(IT const & begin, IT const & end)
    {
      bp::list res;
      for (IT it = begin; it != end; ++it)
        res.append(*it);
      return res;
    }

    template<class T>
    std::vector<T> to_vector(bp::object const & iterable)
    {
      if(bp::extract<T>(iterable).check())
        return {bp::extract<T>(iterable)};
      std::size_t len = bp::len(iterable);
      std::vector<T> res; res.reserve(len);
      for(std::size_t i = 0 ; i < len ; ++i)
        res.push_back(boost::python::extract<T>(iterable[i]));
      return res;
    }


    inline sc::numeric_type extract_dtype(bp::object const & odtype)
    {
        std::string name = bp::extract<std::string>(odtype.attr("__class__").attr("__name__"))();
        if(name=="class")
          name = bp::extract<std::string>(odtype.attr("__name__"))();
        else
          name = bp::extract<std::string>(odtype.attr("__class__").attr("__name__"))();

        if(name=="int8") return sc::CHAR_TYPE;
        else if(name=="uint8") return sc::UCHAR_TYPE;
        else if(name=="int16") return sc::SHORT_TYPE;
        else if(name=="uint16") return sc::USHORT_TYPE;
        else if(name=="int32") return sc::INT_TYPE;
        else if(name=="uint32") return sc::UINT_TYPE;
        else if(name=="int64") return sc::LONG_TYPE;
        else if(name=="uint64") return sc::ULONG_TYPE;
        else if(name=="float32") return sc::FLOAT_TYPE;
        else if(name=="float64") return sc::DOUBLE_TYPE;
        else
        {
            PyErr_SetString(PyExc_TypeError, "Data type not understood");
            bp::throw_error_already_set();
            throw;
        }
    }
}
#endif
