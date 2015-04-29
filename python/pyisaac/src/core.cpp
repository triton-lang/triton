#include "core.h"
#include "common.hpp"

std::shared_ptr<atd::array>
ndarray_to_atdarray(const np::ndarray& array, const atd::driver::Context& ctx)
{

  int d = array.get_nd();
  if (d > 2) {
    PyErr_SetString(PyExc_TypeError, "Only 1-D and 2-D arrays are supported!");
    bp::throw_error_already_set();
  }

  atd::numeric_type dtype = detail::to_atd_dtype(array.get_dtype());
  atd::int_t size = (atd::int_t)array.shape(0);
  atd::array* v = new atd::array(size, dtype, ctx);

  void* data = (void*)array.get_data();
  atd::copy(data, *v);

  return std::shared_ptr<atd::array>(v);
}



std::shared_ptr<atd::array> create_array(bp::object const & obj, bp::object odtype, atd::driver::Context context)
{
  return ndarray_to_atdarray(np::from_object(obj, detail::to_np_dtype(detail::extract_dtype(odtype))), context);
}

std::shared_ptr<atd::array> create_empty_array(bp::object sizes, bp::object odtype, atd::driver::Context context)
{
    typedef std::shared_ptr<atd::array> result_type;

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

    atd::numeric_type dtype = detail::extract_dtype(odtype);
    if(len < 1 || len > 2)
    {
        PyErr_SetString(PyExc_TypeError, "Only 1-D and 2-D arrays are supported!");
        bp::throw_error_already_set();
    }
    if(len==1)
        return result_type(new atd::array(size1, dtype, context));
    return result_type(new atd::array(size1, size2, dtype, context));
}

std::string type_name(bp::object const & obj)
{
  std::string name = bp::extract<std::string>(obj.attr("__class__").attr("__name__"))();
  if(name=="class")
    return bp::extract<std::string>(obj.attr("__name__"))();
  else
    return bp::extract<std::string>(obj.attr("__class__").attr("__name__"))();
}

std::shared_ptr<atd::scalar> construct_scalar(bp::object obj, atd::driver::Context const & context)
{
  typedef std::shared_ptr<atd::scalar> result_type;
  std::string name = type_name(obj);
  if(name=="int") return result_type(new atd::scalar(bp::extract<int>(obj)(), context));
  else if(name=="float") return result_type(new atd::scalar(bp::extract<double>(obj)(), context));
  else if(name=="long") return result_type(new atd::scalar(bp::extract<long>(obj)(), context));
  else if(name=="int") return result_type(new atd::scalar(bp::extract<int>(obj)(), context));

  else if(name=="int8") return result_type(new atd::scalar(atd::CHAR_TYPE, context));
  else if(name=="uint8") return result_type(new atd::scalar(atd::UCHAR_TYPE, context));
  else if(name=="int16") return result_type(new atd::scalar(atd::SHORT_TYPE, context));
  else if(name=="uint16") return result_type(new atd::scalar(atd::USHORT_TYPE, context));
  else if(name=="int32") return result_type(new atd::scalar(atd::INT_TYPE, context));
  else if(name=="uint32") return result_type(new atd::scalar(atd::UINT_TYPE, context));
  else if(name=="int64") return result_type(new atd::scalar(atd::LONG_TYPE, context));
  else if(name=="uint64") return result_type(new atd::scalar(atd::ULONG_TYPE, context));
  else if(name=="float32") return result_type(new atd::scalar(atd::FLOAT_TYPE, context));
  else if(name=="float64") return result_type(new atd::scalar(atd::DOUBLE_TYPE, context));
  else{
      PyErr_SetString(PyExc_TypeError, "Data type not understood");
      bp::throw_error_already_set();
      throw;
  }

}

bp::tuple get_shape(atd::array const & x)
{
  return bp::make_tuple(x.shape()[0], x.shape()[1]);
}

template<class T>
struct datatype : public atd::value_scalar
{
  datatype(T t) : atd::value_scalar(t){ }

};

template<class T>
unsigned int size(datatype<T> const & dt)
{ return atd::size_of(dt.dtype()) ; }

#define INSTANTIATE(name, clname) \
  struct name : public datatype<clname> {  name(clname value) : datatype<clname>(value){} };
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

void export_core()
{
  /*-------------------
   * TYPES
   *------------------*/
  #define INSTANTIATE(name, clname) \
    bp::class_<datatype<clname>, bp::bases<atd::value_scalar> >(#name, bp::init<clname>());\
    bp::class_<name, bp::bases<datatype<clname> > >(#name, bp::init<clname>())\
      .add_property("size", &size<clname>)\
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

    bp::enum_<atd::expression_type>("operations")
      MAP_ENUM(VECTOR_AXPY_TYPE, atd)
      MAP_ENUM(MATRIX_AXPY_TYPE, atd)
      MAP_ENUM(REDUCTION_TYPE, atd)
      MAP_ENUM(ROW_WISE_REDUCTION_TYPE, atd)
      MAP_ENUM(COL_WISE_REDUCTION_TYPE, atd)
      MAP_ENUM(VECTOR_AXPY_TYPE, atd)
      MAP_ENUM(VECTOR_AXPY_TYPE, atd)
      MAP_ENUM(VECTOR_AXPY_TYPE, atd)
      MAP_ENUM(VECTOR_AXPY_TYPE, atd)
      ;

  /*-------------------
   * ARRAY
   *------------------*/
  #define ADD_SCALAR_HANDLING(OP)\
    .def(bp::self                                    OP int())\
    .def(bp::self                                    OP long())\
    .def(bp::self                                    OP double())\
    .def(bp::self                                    OP bp::other<atd::value_scalar>())\
    .def(int()                                       OP bp::self)\
    .def(long()                                      OP bp::self)\
    .def(double()                                     OP bp::self)\
    .def(bp::other<atd::value_scalar>()              OP bp::self)

  #define ADD_ARRAY_OPERATOR(OP)\
    .def(bp::self OP bp::self)\
    ADD_SCALAR_HANDLING(OP)

    bp::class_<atd::expressions_tuple>
        ("array_expression_container", bp::init<atd::array_expression const &>())
    ;

    bp::class_<atd::array_expression >("array_expression", bp::no_init)
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
        .add_property("context", bp::make_function(&atd::array_expression::context, bp::return_internal_reference<>()))
        .def(bp::self_ns::abs(bp::self))
  //      .def(bp::self_ns::pow(bp::self))
    ;
  #undef ADD_ARRAY_OPERATOR

  #define ADD_ARRAY_OPERATOR(OP) \
    .def(bp::self                            OP bp::self)\
    .def(bp::self                            OP bp::other<atd::array_expression>())\
    .def(bp::other<atd::array_expression>() OP bp::self) \
    ADD_SCALAR_HANDLING(OP)

    bp::class_<atd::array,
            std::shared_ptr<atd::array> >
    ( "array", bp::no_init)
        .def("__init__", bp::make_constructor(create_array, bp::default_call_policies(), (bp::arg("obj"), bp::arg("dtype") = bp::scope().attr("float32"), bp::arg("context")=atd::driver::queues.default_context())))
        .def(bp::init<atd::array_expression>())
        .add_property("dtype", &atd::array::dtype)
        .add_property("context", bp::make_function(&atd::array::context, bp::return_internal_reference<>()))
        .add_property("T", &atd::array::T)
        .add_property("shape", &get_shape)
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

    bp::class_<atd::scalar, bp::bases<atd::array> >
        ("scalar", bp::no_init)
        .def("__init__", bp::make_constructor(construct_scalar, bp::default_call_policies(), (bp::arg(""), bp::arg("context")=atd::driver::queues.default_context())))
        ;

    //Other numpy-like initializers
    bp::def("empty", &create_empty_array, (bp::arg("shape"), bp::arg("dtype") = bp::scope().attr("float32"), bp::arg("context")=atd::driver::queues.default_context()));

  //Binary
  #define MAP_FUNCTION(name) \
        bp::def(#name, static_cast<atd::array_expression (*)(atd::array const &, atd::array const &)>(&atd::name));\
        bp::def(#name, static_cast<atd::array_expression (*)(atd::array_expression const &, atd::array const &)>(&atd::name));\
        bp::def(#name, static_cast<atd::array_expression (*)(atd::array const &, atd::array_expression const &)>(&atd::name));\
        bp::def(#name, static_cast<atd::array_expression (*)(atd::array_expression const &, atd::array_expression const &)>(&atd::name));

    MAP_FUNCTION(maximum)
    MAP_FUNCTION(minimum)
    MAP_FUNCTION(pow)
    MAP_FUNCTION(dot)
  #undef MAP_FUNCTION

  //Unary
  #define MAP_FUNCTION(name) \
        bp::def(#name, static_cast<atd::array_expression (*)(atd::array const &)>(&atd::name));\
        bp::def(#name, static_cast<atd::array_expression (*)(atd::array_expression const &)>(&atd::name));

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
        bp::def(#name, static_cast<atd::array_expression (*)(atd::array const &, atd::int_t)>(&atd::name));\
        bp::def(#name, static_cast<atd::array_expression (*)(atd::array_expression const &, atd::int_t)>(&atd::name));

    MAP_FUNCTION(sum)
    MAP_FUNCTION(max)
    MAP_FUNCTION(min)
    MAP_FUNCTION(argmax)
    MAP_FUNCTION(argmin)
  #undef MAP_FUNCTION

  /*-------------------
   * SCALAR
   *------------------*/
  bp::class_<atd::value_scalar>("value_scalar", bp::no_init)
          .add_property("dtype", &atd::value_scalar::dtype);


}
