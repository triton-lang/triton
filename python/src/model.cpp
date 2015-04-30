#include "isaac/backend/templates/vaxpy.h"
#include "isaac/backend/templates/maxpy.h"
#include "isaac/backend/templates/reduction.h"
#include "isaac/backend/templates/mreduction.h"
#include "isaac/backend/templates/mproduct.h"
#include "isaac/model/model.h"

#include "model.h"
#include "common.hpp"

bp::list input_sizes(isaac::base & temp, isc::expressions_tuple const & tree)
{
    std::vector<int> tmp = temp.input_sizes(tree);
    return detail::to_list(tmp.begin(), tmp.end());
}

std::shared_ptr<isc::model> construct_model(bp::object dtype, bp::object const & tp, isc::driver::CommandQueue & queue)
{
    return std::shared_ptr<isc::model>(new isc::model(detail::extract_template_type(tp), detail::extract_dtype(dtype), (isc::base const &)bp::extract<isc::base>(tp), queue));
}

void export_model()
{

  bp::class_<isaac::model>("model", bp::no_init)
                  .def("__init__", bp::make_constructor(construct_model))
                  .def("execute", &isc::model::execute);

  bp::enum_<isaac::fetching_policy_type>
      ("fetching_policy_type")
      .value("FETCH_FROM_LOCAL", isc::FETCH_FROM_LOCAL)
      .value("FETCH_FROM_GLOBAL_STRIDED", isc::FETCH_FROM_GLOBAL_STRIDED)
      .value("FETCH_FROM_GLOBAL_CONTIGUOUS", isc::FETCH_FROM_GLOBAL_CONTIGUOUS)
      ;

  //Base
  {
    #define __PROP(name) .def_readonly(#name, &isaac::base::parameters_type::name)
    bp::class_<isaac::base, boost::noncopyable>("base", bp::no_init)
            .def("lmem_usage", &isaac::base::lmem_usage)
            .def("registers_usage", &isaac::base::registers_usage)
            .def("is_invalid", &isaac::base::is_invalid)
            .def("input_sizes", &input_sizes)
        ;
    #undef __PROP
  }

  #define WRAP_BASE(name) bp::class_<isaac::base_impl<isaac::name, isaac::name::parameters_type>, bp::bases<isaac::base>, boost::noncopyable>(#name "_base_impl", bp::no_init);
  #define WRAP_TEMPLATE(name, basename, ...) bp::class_<isaac::name, bp::bases<isaac::base_impl<isaac::basename, isaac::basename::parameters_type> > >(#name, bp::init<__VA_ARGS__>())\
                                      .add_property("local_size_0", &isc::name::local_size_0)\
                                      .add_property("local_size_1", &isc::name::local_size_1);
  #define WRAP_SINGLE_TEMPLATE(name, ...) WRAP_BASE(name) WRAP_TEMPLATE(name, name, __VA_ARGS__)

  //Vector AXPY
  WRAP_SINGLE_TEMPLATE(vaxpy, uint, uint, uint, isaac::fetching_policy_type)
  WRAP_SINGLE_TEMPLATE(maxpy, uint, uint, uint, uint, uint, isaac::fetching_policy_type)
  WRAP_SINGLE_TEMPLATE(reduction, uint, uint, uint, isaac::fetching_policy_type)
  WRAP_BASE(mreduction)
  WRAP_TEMPLATE(mreduction_rows, mreduction, uint, uint, uint, uint, uint, isaac::fetching_policy_type)
  WRAP_TEMPLATE(mreduction_cols, mreduction, uint, uint, uint, uint, uint, isaac::fetching_policy_type)
  WRAP_BASE(mproduct)
  WRAP_TEMPLATE(mproduct_nn, mproduct, uint, uint, uint, uint, uint, uint, uint, uint, isaac::fetching_policy_type, isaac::fetching_policy_type, uint, uint)
  WRAP_TEMPLATE(mproduct_tn, mproduct, uint, uint, uint, uint, uint, uint, uint, uint, isaac::fetching_policy_type, isaac::fetching_policy_type, uint, uint)
  WRAP_TEMPLATE(mproduct_nt, mproduct, uint, uint, uint, uint, uint, uint, uint, uint, isaac::fetching_policy_type, isaac::fetching_policy_type, uint, uint)
  WRAP_TEMPLATE(mproduct_tt, mproduct, uint, uint, uint, uint, uint, uint, uint, uint, isaac::fetching_policy_type, isaac::fetching_policy_type, uint, uint)
}
