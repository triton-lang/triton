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

#include "isaac/runtime/profiles.h"
#include "common.hpp"
#include "core.h"

namespace tpt = sc::templates;

namespace detail
{


sc::numeric_type to_sc_dtype(np::dtype const & T)
{
  if(T==np::detail::get_int_dtype<8, false>()) return sc::CHAR_TYPE;
  else if(T==np::detail::get_int_dtype<8, true>()) return sc::UCHAR_TYPE;
  else if(T==np::detail::get_int_dtype<16, false>()) return sc::SHORT_TYPE;
  else if(T==np::detail::get_int_dtype<16, true>()) return sc::USHORT_TYPE;
  else if(T==np::detail::get_int_dtype<32, false>()) return sc::INT_TYPE;
  else if(T==np::detail::get_int_dtype<32, true>()) return sc::UINT_TYPE;
  else if(T==np::detail::get_int_dtype<64, false>()) return sc::LONG_TYPE;
  else if(T==np::detail::get_int_dtype<64, true>()) return sc::ULONG_TYPE;
//  else if(T==np::detail::get_float_dtype<16>()) return sc::HALF_TYPE;
  else if(T==np::detail::get_float_dtype<32>()) return sc::FLOAT_TYPE;
  else if(T==np::detail::get_float_dtype<64>()) return sc::DOUBLE_TYPE;
  else{
    PyErr_SetString(PyExc_TypeError, "Unrecognized datatype");
    bp::throw_error_already_set();
    throw; // suppress warning; throw_error_already_set() never returns but isn't marked noreturn: https://svn.boost.org/trac/boost/ticket/1482
  }
}

np::dtype to_np_dtype(sc::numeric_type const & T) throw()
{
  if(T==sc::CHAR_TYPE) return np::detail::get_int_dtype<8, false>();
  else if(T==sc::UCHAR_TYPE) return np::detail::get_int_dtype<8, true>();
  else if(T==sc::SHORT_TYPE) return np::detail::get_int_dtype<16, false>();
  else if(T==sc::USHORT_TYPE) return np::detail::get_int_dtype<16, true>();
  else if(T==sc::INT_TYPE) return np::detail::get_int_dtype<32, false>();
  else if(T==sc::UINT_TYPE) return np::detail::get_int_dtype<32, true>();
  else if(T==sc::LONG_TYPE) return np::detail::get_int_dtype<64, false>();
  else if(T==sc::ULONG_TYPE) return np::detail::get_int_dtype<64, true>();
//  else if(T==sc::HALF_TYPE) return np::detail::get_float_dtype<16>();
  else if(T==sc::FLOAT_TYPE) return np::detail::get_float_dtype<32>();
  else if(T==sc::DOUBLE_TYPE) return np::detail::get_float_dtype<64>();
  else{
    PyErr_SetString(PyExc_TypeError, "Unrecognized datatype");
    bp::throw_error_already_set();
    throw; // suppress warning; throw_error_already_set() never returns but isn't marked noreturn: https://svn.boost.org/trac/boost/ticket/1482
  }
}

bp::tuple get_shape(sc::array_base const & x)
{
  return bp::make_tuple(x.shape()[0], x.shape()[1]);
}

template<class T>
struct datatype : public sc::value_scalar
{
  datatype(T t) : sc::value_scalar(t){ }

};

template<class T>
unsigned int size(datatype<T> const & dt)
{ return sc::size_of(dt.dtype()) ; }

#define INSTANTIATE(name, clname) \
  struct name : public detail::datatype<clname> {  name(clname value) : detail::datatype<clname>(value){} };
  INSTANTIATE(int8, cl_char)
  INSTANTIATE(uint8, cl_uchar)
  INSTANTIATE(int16, cl_short)
  INSTANTIATE(uint16, cl_ushort)
  INSTANTIATE(int32, cl_int)
  INSTANTIATE(uint32, cl_uint)
  INSTANTIATE(int64, cl_long)
  INSTANTIATE(uint64, cl_ulong)
  INSTANTIATE(float32, cl_float)
  INSTANTIATE(float64, cl_double)
#undef INSTANTIATE

}

namespace detail
{
  std::shared_ptr<rt::profiles::value_type> construct_model(bp::object const & tp, bp::object dtype, sc::driver::CommandQueue & queue)
  {
      tpt::base* raw =  bp::extract<tpt::base*>(tp);
      return std::make_shared<rt::profiles::value_type>(tools::extract_dtype(dtype), raw->getptr(), queue);
  }

  std::shared_ptr<sc::array>
  ndarray_to_scarray(const np::ndarray& array, sc::driver::Context const & ctx)
  {

    int d = array.get_nd();
    if (d > 2) {
      PyErr_SetString(PyExc_TypeError, "Only 1-D and 2-D arrays are supported!");
      bp::throw_error_already_set();
    }

    sc::numeric_type dtype = to_sc_dtype(array.get_dtype());
    sc::int_t size = (sc::int_t)array.shape(0);
    sc::array* v = new sc::array(size, dtype, ctx);

    void* data = (void*)array.get_data();
    sc::copy(data, *v);

    return std::shared_ptr<sc::array>(v);
  }

  sc::driver::Context const & extract_context(bp::object context)
  {
    if(context.is_none())
        return sc::driver::backend::contexts::get_default();
    sc::driver::Context const * ctx = bp::extract<sc::driver::Context const *>(context);
    if(ctx)
        return *ctx;
    PyErr_SetString(PyExc_TypeError, "Context type not understood");
    bp::throw_error_already_set();
    throw;
  }

  inline void check_sizes(std::vector<int> s)
  {
      if(s.size() < 1 || s.size() > 2)
      {
          PyErr_SetString(PyExc_TypeError, "Only 1-D and 2-D arrays are supported!");
          bp::throw_error_already_set();
      }
  }


  std::shared_ptr<sc::array> create_array(bp::object const & obj, bp::object odtype, bp::object pycontext)
  {
    return ndarray_to_scarray(np::from_object(obj, to_np_dtype(tools::extract_dtype(odtype))), extract_context(pycontext));
  }

  std::shared_ptr<sc::array> create_zeros_array(bp::object pysizes, bp::object pydtype, bp::object pycontext)
  {
      std::vector<int> sizes = tools::to_vector<int>(pysizes);
      sc::numeric_type dtype = tools::extract_dtype(pydtype);
      sc::driver::Context const & context = extract_context(pycontext);
      check_sizes(sizes);
      if(sizes.size()==1)
          return std::shared_ptr<sc::array>(new sc::array(sc::zeros({sizes[0]}, dtype, context)));
      return std::shared_ptr<sc::array> (new sc::array(sc::zeros({sizes[0], sizes[1]}, dtype, context)));
  }


  std::shared_ptr<sc::array> create_empty_array(bp::object pysizes, bp::object pydtype, bp::object pycontext)
  {
      std::vector<int> sizes = tools::to_vector<int>(pysizes);
      sc::numeric_type dtype = tools::extract_dtype(pydtype);
      sc::driver::Context const & context = extract_context(pycontext);
      check_sizes(sizes);
      if(sizes.size()==1)
          return std::shared_ptr<sc::array>(new sc::array(sizes[0], dtype, context));
      return std::shared_ptr<sc::array> (new sc::array(sizes[0], sizes[1], dtype, context));
  }

  std::string type_name(bp::object const & obj)
  {
    std::string name = bp::extract<std::string>(obj.attr("__class__").attr("__name__"))();
    if(name=="class")
      return bp::extract<std::string>(obj.attr("__name__"))();
    else
      return bp::extract<std::string>(obj.attr("__class__").attr("__name__"))();
  }

  std::shared_ptr<sc::scalar> construct_scalar(bp::object obj, bp::object pycontext)
  {
    typedef std::shared_ptr<sc::scalar> result_type;
    sc::driver::Context const & context = extract_context(pycontext);
    std::string name = type_name(obj);
    if(name=="int") return result_type(new sc::scalar(bp::extract<int>(obj)(), context));
    else if(name=="float") return result_type(new sc::scalar(bp::extract<double>(obj)(), context));
    else if(name=="long") return result_type(new sc::scalar(bp::extract<long>(obj)(), context));
    else if(name=="int") return result_type(new sc::scalar(bp::extract<int>(obj)(), context));

    else if(name=="int8") return result_type(new sc::scalar(sc::CHAR_TYPE, context));
    else if(name=="uint8") return result_type(new sc::scalar(sc::UCHAR_TYPE, context));
    else if(name=="int16") return result_type(new sc::scalar(sc::SHORT_TYPE, context));
    else if(name=="uint16") return result_type(new sc::scalar(sc::USHORT_TYPE, context));
    else if(name=="int32") return result_type(new sc::scalar(sc::INT_TYPE, context));
    else if(name=="uint32") return result_type(new sc::scalar(sc::UINT_TYPE, context));
    else if(name=="int64") return result_type(new sc::scalar(sc::LONG_TYPE, context));
    else if(name=="uint64") return result_type(new sc::scalar(sc::ULONG_TYPE, context));
    else if(name=="float32") return result_type(new sc::scalar(sc::FLOAT_TYPE, context));
    else if(name=="float64") return result_type(new sc::scalar(sc::DOUBLE_TYPE, context));
    else{
        PyErr_SetString(PyExc_TypeError, "Data type not understood");
        bp::throw_error_already_set();
        throw;
    }
  }

  struct model_map_indexing
  {
      static rt::profiles::value_type& get_item(rt::profiles::map_type& container, bp::tuple i_)
      {
          tpt::base* tpt =  bp::extract<tpt::base*>(i_[0]);
          sc::numeric_type dtype = tools::extract_dtype(i_[1]);
          rt::profiles::map_type::iterator i = container.find(std::make_pair(tpt->type(), dtype));
          if (i == container.end())
          {
              PyErr_SetString(PyExc_KeyError, "Invalid key");
              bp::throw_error_already_set();
          }
          return *i->second;
      }

      static void set_item(rt::profiles::map_type& container, bp::tuple i_, rt::profiles::value_type const & v)
      {
          tpt::base* tpt =  bp::extract<tpt::base*>(i_[0]);
          sc::numeric_type dtype = tools::extract_dtype(i_[1]);
          container[std::make_pair(tpt->type(), dtype)].reset(new rt::profiles::value_type(v));
      }
  };
}


//////////////
/// EXPORT
/////////////
void export_core()
{

    bp::class_<rt::profiles::value_type>("profile", bp::no_init)
                    .def("__init__", bp::make_constructor(detail::construct_model))
                    .def("execute", &rt::profiles::value_type::execute);

    bp::class_<sc::value_scalar>("value_scalar", bp::no_init)
              .add_property("dtype", &sc::value_scalar::dtype);

  #define INSTANTIATE(name, clname) \
    bp::class_<detail::datatype<clname>, bp::bases<sc::value_scalar> >(#name, bp::init<clname>());\
    bp::class_<detail::name, bp::bases<detail::datatype<clname> > >(#name, bp::init<clname>())\
      .add_property("size", &detail::size<clname>)\
      ;


    INSTANTIATE(int8, cl_char)
    INSTANTIATE(uint8, cl_uchar)
    INSTANTIATE(int16, cl_short)
    INSTANTIATE(uint16, cl_ushort)
    INSTANTIATE(int32, cl_int)
    INSTANTIATE(uint32, cl_uint)
    INSTANTIATE(int64, cl_long)
    INSTANTIATE(uint64, cl_ulong)
    INSTANTIATE(float32, cl_float)
    INSTANTIATE(float64, cl_double)
    #undef INSTANTIATE

    bp::enum_<sc::expression_type>("operations")
      MAP_ENUM(ELEMENTWISE_1D, sc)
      MAP_ENUM(ELEMENTWISE_2D, sc)
      MAP_ENUM(REDUCE_1D, sc)
      MAP_ENUM(REDUCE_2D_ROWS, sc)
      MAP_ENUM(REDUCE_2D_COLS, sc)
      MAP_ENUM(GEMM_NN, sc)
      MAP_ENUM(GEMM_TN, sc)
      MAP_ENUM(GEMM_NT, sc)
      MAP_ENUM(GEMM_TT, sc);

#define ADD_SCALAR_HANDLING(OP)\
  .def(bp::self                                    OP int())\
  .def(bp::self                                    OP long())\
  .def(bp::self                                    OP double())\
  .def(bp::self                                    OP bp::other<sc::value_scalar>())\
  .def(int()                                       OP bp::self)\
  .def(long()                                      OP bp::self)\
  .def(double()                                     OP bp::self)\
  .def(bp::other<sc::value_scalar>()              OP bp::self)

#define ADD_ARRAY_OPERATOR(OP)\
  .def(bp::self OP bp::self)\
  ADD_SCALAR_HANDLING(OP)

  bp::class_<sc::expression_tree >("expression_tree", bp::no_init)
      ADD_ARRAY_OPERATOR(+)
      ADD_ARRAY_OPERATOR(-)
      ADD_ARRAY_OPERATOR(*)
      ADD_ARRAY_OPERATOR(/)
      ADD_ARRAY_OPERATOR(>)
      ADD_ARRAY_OPERATOR(>=)
      ADD_ARRAY_OPERATOR(<)
      ADD_ARRAY_OPERATOR(<=)
      ADD_ARRAY_OPERATOR(==)
      ADD_ARRAY_OPERATOR(!=)
      .add_property("context", bp::make_function(&sc::expression_tree::context, bp::return_internal_reference<>()))
      .add_property("dtype", &sc::expression_tree::dtype)
      .def(bp::self_ns::abs(bp::self))
//      .def(bp::self_ns::pow(bp::self))
  ;
#undef ADD_ARRAY_OPERATOR

#define ADD_ARRAY_OPERATOR(OP) \
  .def(bp::self                            OP bp::self)\
  .def(bp::self                            OP bp::other<sc::expression_tree>())\
  .def(bp::other<sc::expression_tree>() OP bp::self) \
  ADD_SCALAR_HANDLING(OP)

  bp::class_<sc::array_base, boost::noncopyable>("array_base", bp::no_init)
      .add_property("dtype", &sc::array_base::dtype)
      .add_property("context", bp::make_function(&sc::array_base::context, bp::return_internal_reference<>()))
      .add_property("T", &sc::array_base::T)
      .add_property("shape", &detail::get_shape)
      ADD_ARRAY_OPERATOR(+)
      ADD_ARRAY_OPERATOR(-)
      ADD_ARRAY_OPERATOR(*)
      ADD_ARRAY_OPERATOR(/)
      ADD_ARRAY_OPERATOR(>)
      ADD_ARRAY_OPERATOR(>=)
      ADD_ARRAY_OPERATOR(<)
      ADD_ARRAY_OPERATOR(<=)
      ADD_ARRAY_OPERATOR(==)
      ADD_ARRAY_OPERATOR(!=)
      .def(bp::self_ns::abs(bp::self))
//      .def(bp::self_ns::pow(bp::self))
      .def(bp::self_ns::str(bp::self_ns::self))
  ;

  bp::class_<sc::array,std::shared_ptr<sc::array>, bp::bases<sc::array_base> >
          ( "array", bp::no_init)
          .def("__init__", bp::make_constructor(detail::create_array, bp::default_call_policies(), (bp::arg("obj"), bp::arg("dtype") = bp::scope().attr("float32"), bp::arg("context")= bp::object())))
          .def(bp::init<sc::expression_tree>())
  ;

  bp::class_<sc::view, bp::bases<sc::array_base> >
      ("view", bp::no_init)
  ;

  bp::class_<sc::scalar, bp::bases<sc::array_base> >
      ("scalar", bp::no_init)
      .def("__init__", bp::make_constructor(detail::construct_scalar, bp::default_call_policies(), (bp::arg(""), bp::arg("context")=bp::object())))
  ;

//Other numpy-like initializers
  bp::def("empty", &detail::create_empty_array, (bp::arg("shape"), bp::arg("dtype") = bp::scope().attr("float32"), bp::arg("context")=bp::object()));

//Assign
    bp::def("assign", static_cast<sc::expression_tree (*)(sc::array_base const &, sc::array_base const &)>(&sc::assign));\
    bp::def("assign", static_cast<sc::expression_tree (*)(sc::array_base const &, sc::expression_tree const &)>(&sc::assign));\

//Binary
#define MAP_FUNCTION(name) \
      bp::def(#name, static_cast<sc::expression_tree (*)(sc::array_base const &, sc::array_base const &)>(&sc::name));\
      bp::def(#name, static_cast<sc::expression_tree (*)(sc::expression_tree const &, sc::array_base const &)>(&sc::name));\
      bp::def(#name, static_cast<sc::expression_tree (*)(sc::array_base const &, sc::expression_tree const &)>(&sc::name));\
      bp::def(#name, static_cast<sc::expression_tree (*)(sc::expression_tree const &, sc::expression_tree const &)>(&sc::name));

  MAP_FUNCTION(maximum)
  MAP_FUNCTION(minimum)
  MAP_FUNCTION(pow)
  MAP_FUNCTION(dot)
#undef MAP_FUNCTION

//Unary
#define MAP_FUNCTION(name) \
      bp::def(#name, static_cast<sc::expression_tree (*)(sc::array_base const &)>(&sc::name));\
      bp::def(#name, static_cast<sc::expression_tree (*)(sc::expression_tree const &)>(&sc::name));

      bp::def("zeros", &detail::create_zeros_array, (bp::arg("shape"), bp::arg("dtype") = bp::scope().attr("float32"), bp::arg("context")=bp::object()));

  MAP_FUNCTION(abs)
  MAP_FUNCTION(acos)
  MAP_FUNCTION(asin)
  MAP_FUNCTION(atan)
  MAP_FUNCTION(ceil)
  MAP_FUNCTION(cos)
  MAP_FUNCTION(cosh)
  MAP_FUNCTION(exp)
  MAP_FUNCTION(floor)
  MAP_FUNCTION(log)
  MAP_FUNCTION(log10)
  MAP_FUNCTION(sin)
  MAP_FUNCTION(sinh)
  MAP_FUNCTION(sqrt)
  MAP_FUNCTION(tan)
  MAP_FUNCTION(tanh)
#undef MAP_FUNCTION

  /*--- Reduction operators----*/
  //---------------------------------------
#define MAP_FUNCTION(name) \
      bp::def(#name, static_cast<sc::expression_tree (*)(sc::array_base const &, sc::int_t)>(&sc::name));\
      bp::def(#name, static_cast<sc::expression_tree (*)(sc::expression_tree const &, sc::int_t)>(&sc::name));

  MAP_FUNCTION(sum)
  MAP_FUNCTION(max)
  MAP_FUNCTION(min)
  MAP_FUNCTION(argmax)
  MAP_FUNCTION(argmin)
#undef MAP_FUNCTION

  /*--- Profiles----*/
  //---------------------------------------
  bp::class_<rt::profiles::map_type>("profiles")
      .def("__getitem__", &detail::model_map_indexing::get_item, bp::return_internal_reference<>())
      .def("__setitem__", &detail::model_map_indexing::set_item, bp::with_custodian_and_ward<1,2>())
      ;
}
