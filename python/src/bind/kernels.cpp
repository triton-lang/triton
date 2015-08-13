#include "isaac/kernels/templates/axpy.h"
#include "isaac/kernels/templates/ger.h"
#include "isaac/kernels/templates/dot.h"
#include "isaac/kernels/templates/gemv.h"
#include "isaac/kernels/templates/gemm.h"

#include "common.hpp"
#include "kernels.h"


namespace tpt = isaac::templates;


namespace detail
{
  bp::list input_sizes(tpt::base & temp, sc::expressions_tuple const & tree)
  {
      std::vector<int> tmp = temp.input_sizes(tree);
      return tools::to_list(tmp.begin(), tmp.end());
  }
}

void export_templates()
{

  bp::object templates_module(bp::handle<>(bp::borrowed(PyImport_AddModule("isaac.templates"))));
  bp::scope().attr("templates") = templates_module;
  bp::scope template_scope = templates_module;


  bp::enum_<tpt::fetching_policy_type>
      ("fetching_policy_type");

  bp::scope().attr("FETCH_FROM_LOCAL") = tpt::FETCH_FROM_LOCAL;
  bp::scope().attr("FETCH_FROM_GLOBAL_STRIDED") = tpt::FETCH_FROM_GLOBAL_CONTIGUOUS;
  bp::scope().attr("FETCH_FROM_GLOBAL_CONTIGUOUS") = tpt::FETCH_FROM_GLOBAL_STRIDED;

  //Base
  {
    #define __PROP(name) .def_readonly(#name, &tpt::base::parameters_type::name)
    bp::class_<tpt::base, boost::noncopyable>("base", bp::no_init)
            .def("lmem_usage", &tpt::base::lmem_usage)
            .def("registers_usage", &tpt::base::registers_usage)
            .def("is_invalid", &tpt::base::is_invalid)
            .def("input_sizes", &detail::input_sizes)
        ;
    #undef __PROP
  }

  #define WRAP_BASE(name) bp::class_<tpt::base_impl<tpt::name, tpt::name::parameters_type>, bp::bases<tpt::base>, boost::noncopyable>(#name, bp::no_init)\
                                      .add_property("local_size_0", &tpt::base_impl<tpt::name, tpt::name::parameters_type>::local_size_0)\
                                      .add_property("local_size_1", &tpt::base_impl<tpt::name, tpt::name::parameters_type>::local_size_1);

  #define WRAP_TEMPLATE(name, basename, ...) bp::class_<tpt::name, bp::bases<tpt::base_impl<tpt::basename, tpt::basename::parameters_type> > >(#name, bp::init<__VA_ARGS__>())\
                                      ;
  #define WRAP_SINGLE_TEMPLATE(name, ...) WRAP_BASE(name) WRAP_TEMPLATE(name, name, __VA_ARGS__)

  //Vector AXPY
  WRAP_SINGLE_TEMPLATE(axpy, uint, uint, uint, tpt::fetching_policy_type)
  WRAP_SINGLE_TEMPLATE(ger, uint, uint, uint, uint, uint, tpt::fetching_policy_type)
  WRAP_SINGLE_TEMPLATE(dot, uint, uint, uint, tpt::fetching_policy_type)
  WRAP_BASE(gemv)
  WRAP_TEMPLATE(gemv_n, gemv, uint, uint, uint, uint, uint, tpt::fetching_policy_type)
  WRAP_TEMPLATE(gemv_t, gemv, uint, uint, uint, uint, uint, tpt::fetching_policy_type)
  WRAP_BASE(gemm)
  WRAP_TEMPLATE(gemm_nn, gemm, uint, uint, uint, uint, uint, uint, uint, uint, tpt::fetching_policy_type, tpt::fetching_policy_type, uint, uint)
  WRAP_TEMPLATE(gemm_tn, gemm, uint, uint, uint, uint, uint, uint, uint, uint, tpt::fetching_policy_type, tpt::fetching_policy_type, uint, uint)
  WRAP_TEMPLATE(gemm_nt, gemm, uint, uint, uint, uint, uint, uint, uint, uint, tpt::fetching_policy_type, tpt::fetching_policy_type, uint, uint)
  WRAP_TEMPLATE(gemm_tt, gemm, uint, uint, uint, uint, uint, uint, uint, uint, tpt::fetching_policy_type, tpt::fetching_policy_type, uint, uint)


}
