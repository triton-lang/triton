#ifndef ISAAC_PYTHON_COMMON_HPP
#define ISAAC_PYTHON_COMMON_HPP

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <boost/numpy/dtype.hpp>

#include "isaac/array.h"

#define MAP_ENUM(v, ns) .value(#v, ns::v)

namespace bp = boost::python;
namespace sc = isaac;
namespace np = boost::numpy;

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
    std::vector<T> to_vector(bp::list const & list)
    {
      std::size_t len = bp::len(list);
      std::vector<T> res; res.reserve(len);
      for(std::size_t i = 0 ; i < len ; ++i)
        res.push_back(boost::python::extract<T>(list[i]));
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

        if(name=="elementwise_1d") return sc::AXPY_TYPE;
        else if(name=="elementwise_2d") return sc::GER_TYPE;
        else if(name=="dot") return sc::DOT_TYPE;
        else if(name=="reduce_2d_rows") return sc::GEMV_N_TYPE;
        else if(name=="reduce_2d_cols") return sc::GEMV_T_TYPE;
        else if(name=="matrix_product_nn") return sc::GEMM_NN_TYPE;
        else if(name=="matrix_product_tn") return sc::GEMM_TN_TYPE;
        else if(name=="matrix_product_nt") return sc::GEMM_NT_TYPE;
        else if(name=="matrix_product_tt") return sc::GEMM_TT_TYPE;
        else
        {
            PyErr_SetString(PyExc_TypeError, "Template type not understood");
            bp::throw_error_already_set();
            throw;
        }
    }

}
#endif
