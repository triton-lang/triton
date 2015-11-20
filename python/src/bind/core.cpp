#include "isaac/profiles/profiles.h"
#include "common.hpp"
#include "core.h"

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
  std::shared_ptr<sc::profiles::value_type> construct_model(bp::object const & tp, bp::object dtype, sc::driver::CommandQueue & queue)
  {
      return std::shared_ptr<sc::profiles::value_type>(new sc::profiles::value_type(tools::extract_template_type(tp), tools::extract_dtype(dtype), (isaac::templates::base const &)bp::extract<isaac::templates::base>(tp), queue));
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

  isaac::driver::Context const & extract_context(bp::object context)
  {
    if(context.is_none())
        return isaac::driver::backend::contexts::get_default();
    isaac::driver::Context const * ctx = bp::extract<isaac::driver::Context const *>(context);
    if(ctx)
        return *ctx;
    PyErr_SetString(PyExc_TypeError, "Context type not understood");
    bp::throw_error_already_set();
    throw;
  }


  std::shared_ptr<sc::array> create_array(bp::object const & obj, bp::object odtype, bp::object pycontext)
  {
    return ndarray_to_scarray(np::from_object(obj, to_np_dtype(tools::extract_dtype(odtype))), extract_context(pycontext));
  }

  std::shared_ptr<sc::array> create_zeros_array(sc::int_t M, sc::int_t N, bp::object odtype, bp::object pycontext)
  {
   return std::shared_ptr<sc::array>(new sc::array(sc::zeros(M, N, tools::extract_dtype(odtype), extract_context(pycontext))));
  }

  std::shared_ptr<sc::array> create_empty_array(bp::object sizes, bp::object odtype, bp::object pycontext)
  {
      typedef std::shared_ptr<sc::array> result_type;

      std::size_t len;
      int size1;
      int size2;
      try{
        len = bp::len(sizes);
        size1 = bp::extract<int>(sizes[0])();
        size2 = bp::extract<int>(sizes[1])();
      }catch(bp::error_already_set const &){
        PyErr_Clear();
        len = 1;
        size1 = bp::extract<int>(sizes)();
      }

      sc::numeric_type dtype = tools::extract_dtype(odtype);
      if(len < 1 || len > 2)
      {
          PyErr_SetString(PyExc_TypeError, "Only 1-D and 2-D arrays are supported!");
          bp::throw_error_already_set();
      }

      sc::driver::Context const & context = extract_context(pycontext);
      if(len==1)
          return result_type(new sc::array(size1, dtype, context));
      return result_type(new sc::array(size1, size2, dtype, context));
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
      static sc::profiles::value_type& get_item(sc::profiles::map_type& container, bp::tuple i_)
      {
          sc::expression_type expression = tools::extract_template_type(i_[0]);
          sc::numeric_type dtype = tools::extract_dtype(i_[1]);
          sc::profiles::map_type::iterator i = container.find(std::make_pair(expression, dtype));
          if (i == container.end())
          {
              PyErr_SetString(PyExc_KeyError, "Invalid key");
              bp::throw_error_already_set();
          }
          return *i->second;
      }

      static void set_item(sc::profiles::map_type& container, bp::tuple i_, sc::profiles::value_type const & v)
      {
          sc::expression_type expression = tools::extract_template_type(i_[0]);
          sc::numeric_type dtype = tools::extract_dtype(i_[1]);
          container[std::make_pair(expression, dtype)].reset(new sc::profiles::value_type(v));
      }
  };
}


//////////////
/// EXPORT
/////////////
void export_core()
{

    bp::class_<isaac::profiles::value_type>("profile", bp::no_init)
                    .def("__init__", bp::make_constructor(detail::construct_model))
                    .def("execute", &sc::profiles::value_type::execute);

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
      MAP_ENUM(AXPY_TYPE, sc)
      MAP_ENUM(GER_TYPE, sc)
      MAP_ENUM(DOT_TYPE, sc)
      MAP_ENUM(GEMV_N_TYPE, sc)
      MAP_ENUM(GEMV_T_TYPE, sc)
      MAP_ENUM(GEMM_NN_TYPE, sc)
      MAP_ENUM(GEMM_TN_TYPE, sc)
      MAP_ENUM(GEMM_NT_TYPE, sc)
      MAP_ENUM(GEMM_TT_TYPE, sc);

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

  bp::class_<sc::math_expression>
      ("math_expression_container", bp::init<sc::math_expression const &>())
  ;

  bp::class_<sc::math_expression >("math_expression", bp::no_init)
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
      .add_property("context", bp::make_function(&sc::math_expression::context, bp::return_internal_reference<>()))
      .def(bp::self_ns::abs(bp::self))
//      .def(bp::self_ns::pow(bp::self))
  ;
#undef ADD_ARRAY_OPERATOR

#define ADD_ARRAY_OPERATOR(OP) \
  .def(bp::self                            OP bp::self)\
  .def(bp::self                            OP bp::other<sc::math_expression>())\
  .def(bp::other<sc::math_expression>() OP bp::self) \
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
          .def(bp::init<sc::math_expression>())
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
    bp::def("assign", static_cast<sc::math_expression (*)(sc::array_base const &, sc::array_base const &)>(&sc::assign));\
    bp::def("assign", static_cast<sc::math_expression (*)(sc::array_base const &, sc::math_expression const &)>(&sc::assign));\

//Binary
#define MAP_FUNCTION(name) \
      bp::def(#name, static_cast<sc::math_expression (*)(sc::array_base const &, sc::array_base const &)>(&sc::name));\
      bp::def(#name, static_cast<sc::math_expression (*)(sc::math_expression const &, sc::array_base const &)>(&sc::name));\
      bp::def(#name, static_cast<sc::math_expression (*)(sc::array_base const &, sc::math_expression const &)>(&sc::name));\
      bp::def(#name, static_cast<sc::math_expression (*)(sc::math_expression const &, sc::math_expression const &)>(&sc::name));

  MAP_FUNCTION(maximum)
  MAP_FUNCTION(minimum)
  MAP_FUNCTION(pow)
  MAP_FUNCTION(dot)
#undef MAP_FUNCTION

//Unary
#define MAP_FUNCTION(name) \
      bp::def(#name, static_cast<sc::math_expression (*)(sc::array_base const &)>(&sc::name));\
      bp::def(#name, static_cast<sc::math_expression (*)(sc::math_expression const &)>(&sc::name));

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
      bp::def(#name, static_cast<sc::math_expression (*)(sc::array_base const &, sc::int_t)>(&sc::name));\
      bp::def(#name, static_cast<sc::math_expression (*)(sc::math_expression const &, sc::int_t)>(&sc::name));

  MAP_FUNCTION(sum)
  MAP_FUNCTION(max)
  MAP_FUNCTION(min)
  MAP_FUNCTION(argmax)
  MAP_FUNCTION(argmin)
#undef MAP_FUNCTION

  /*--- Profiles----*/
  //---------------------------------------
  bp::class_<sc::profiles::map_type>("profiles")
      .def("__getitem__", &detail::model_map_indexing::get_item, bp::return_internal_reference<>())
      .def("__setitem__", &detail::model_map_indexing::set_item, bp::with_custodian_and_ward<1,2>())
      ;
}
