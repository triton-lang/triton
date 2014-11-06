#include <list>
#include <boost/python.hpp>

#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

#include "atidlas/templates/vector_axpy.hpp"
#include "atidlas/templates/matrix_axpy.hpp"
#include "atidlas/templates/reduction.hpp"
#include "atidlas/templates/row_wise_reduction.hpp"
#include "atidlas/templates/matrix_product.hpp"

#include "atidlas/model/model.hpp"

#define ENUM_VALUE(NS, V) .value( #V, NS :: V )

namespace bp = boost::python;
namespace vcl = viennacl;
namespace atd = atidlas;

void export_atidlas()
{


  bp::class_<atidlas::model>("model", bp::init<atd::template_base const &, vcl::ocl::context &, vcl::ocl::device const & >())
                  .def("execute", &atd::model::execute)
                  ;
  
  bp::enum_<atidlas::fetching_policy_type>
      ("fetching_policy_type")
      ENUM_VALUE(atidlas, FETCH_FROM_LOCAL)
      ENUM_VALUE(atidlas, FETCH_FROM_GLOBAL_STRIDED)
      ENUM_VALUE(atidlas, FETCH_FROM_GLOBAL_CONTIGUOUS)
      ;

  //Base
  {
    #define __PROP(name) .def_readonly(#name, &atidlas::template_base::parameters_type::name)
    bp::scope outer = bp::class_<atidlas::template_base, boost::noncopyable>("template_base", bp::no_init)
            .def("lmem_usage", &atidlas::template_base::lmem_usage)
            .def("registers_usage", &atidlas::template_base::registers_usage)
            .def("check_invalid", &atidlas::template_base::check_invalid);
    bp::class_<atidlas::template_base::parameters_type>("parameters_type", bp::no_init)
              __PROP(simd_width)
              __PROP(local_size_0)
              __PROP(local_size_1);

    #undef __PROP
  }

  #define WRAP_TEMPLATE(name, ...) bp::class_<atidlas::template_base_impl<atidlas::name, atidlas::name::parameters_type>, bp::bases<atidlas::template_base>, boost::noncopyable>(#name "_base_impl", bp::no_init)\
                                  .def("parameters", bp::make_function(&atidlas::name::parameters, bp::return_internal_reference<>()));\
                                  bp::scope outer = bp::class_<atidlas::name, bp::bases<atidlas::template_base_impl<atidlas::name, atidlas::name::parameters_type> > >(#name, bp::init<atidlas::name::parameters_type, ## __VA_ARGS__>())

  #define WRAP_PARAMETERS(name, ...) bp::class_<atidlas::name::parameters_type, bp::bases<atidlas::template_base::parameters_type> >("parameters_type", bp::init< __VA_ARGS__ >())
  #define __PROP_BASE(name, tpname) .def_readonly(#name, &atidlas::tpname::parameters_type::name)
  //Vector AXPY
  {
    #define __PROP(name) __PROP_BASE(name, vector_axpy_template)
    WRAP_TEMPLATE(vector_axpy_template);
    WRAP_PARAMETERS(vector_axpy_template, uint, uint, uint, atidlas::fetching_policy_type)
        __PROP(num_groups) __PROP(fetching_policy);
    #undef __PROP
  }

  //Matrix AXPY
  {
    #define __PROP(name) __PROP_BASE(name, matrix_axpy_template)
    WRAP_TEMPLATE(matrix_axpy_template);
    WRAP_PARAMETERS(matrix_axpy_template, uint, uint, uint, uint, uint, atidlas::fetching_policy_type)
        __PROP(num_groups_0) __PROP(num_groups_1)  __PROP(fetching_policy);
    #undef __PROP
  }

  //Reduction
  {
    #define __PROP(name) __PROP_BASE(name, reduction_template)
    WRAP_TEMPLATE(reduction_template);
    WRAP_PARAMETERS(reduction_template, uint, uint, uint, atidlas::fetching_policy_type)
        __PROP(num_groups)  __PROP(fetching_policy);
    #undef __PROP
  }

  //Row-wise reduction
  {
    #define __PROP(name) __PROP_BASE(name, row_wise_reduction_template)
    WRAP_TEMPLATE(row_wise_reduction_template);
    WRAP_PARAMETERS(row_wise_reduction_template, uint, uint, uint, uint, atidlas::fetching_policy_type)
        __PROP(num_groups_0);
    #undef __PROP
  }

  //Matrix product
  {
    #define __PROP(name) __PROP_BASE(name, matrix_product_template)
    WRAP_TEMPLATE(matrix_product_template, char, char);
    WRAP_PARAMETERS(matrix_product_template, uint, uint, uint, uint, uint, uint, uint, atidlas::fetching_policy_type, atidlas::fetching_policy_type, uint, uint)
        __PROP(kL) __PROP(mS) __PROP(kS) __PROP(nS)
        __PROP(A_fetching_policy) __PROP(B_fetching_policy)
        __PROP(local_fetch_0) __PROP(local_fetch_1)
        __PROP(mL) __PROP(nL);

    #undef __PROP
  }

}

BOOST_PYTHON_MODULE(_atidlas)
{
// specify that this module is actually a package
  bp::object package = bp::scope();
  package.attr("__path__") = "_atidlas";
  export_atidlas();
}
