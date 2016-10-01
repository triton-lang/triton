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

    inline sc::expression_type extract_template_type(bp::object const & odtype)
    {
        std::string name = bp::extract<std::string>(odtype.attr("__class__").attr("__name__"))();
        if(name=="class")
          name = bp::extract<std::string>(odtype.attr("__name__"))();
        else
          name = bp::extract<std::string>(odtype.attr("__class__").attr("__name__"))();

        if(name=="elementwise_1d") return sc::ELEMENTWISE_1D;
        else if(name=="elementwise_2d") return sc::ELEMENTWISE_2D;
        else if(name=="reduce_1d") return sc::REDUCE_1D;
        else if(name=="reduce_2d_rows") return sc::REDUCE_2D_ROWS;
        else if(name=="reduce_2d_cols") return sc::REDUCE_2D_COLS;
        else if(name=="gemm_nn") return sc::GEMM_NN;
        else if(name=="gemm_tn") return sc::GEMM_TN;
        else if(name=="gemm_nt") return sc::GEMM_NT;
        else if(name=="gemm_tt") return sc::GEMM_TT;
        else
        {
            PyErr_SetString(PyExc_TypeError, "Template type not understood");
            bp::throw_error_already_set();
            throw;
        }
    }

}
#endif
