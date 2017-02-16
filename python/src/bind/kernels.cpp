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

#include "isaac/jit/generation/elementwise_1d.h"
#include "isaac/jit/generation/elementwise_2d.h"
#include "isaac/jit/generation/reduce_1d.h"
#include "isaac/jit/generation/reduce_2d.h"
#include "isaac/jit/generation/gemm.h"

#include "common.hpp"
#include "kernels.h"


namespace tpt = isaac::templates;


namespace detail
{
  bp::list input_sizes(tpt::base & temp, sc::expression_tree const & tree)
  {
      std::vector<isaac::int_t> tmp = temp.input_sizes(tree);
      return tools::to_list(tmp.begin(), tmp.end());
  }
}

void export_templates()
{

  bp::object templates_module(bp::handle<>(bp::borrowed(PyImport_AddModule("isaac.templates"))));
  bp::scope().attr("templates") = templates_module;
  bp::scope template_scope = templates_module;

  //Base
  {
    #define __PROP(name) .def_readonly(#name, &tpt::base::name)
    bp::class_<tpt::base, std::shared_ptr<tpt::base>, boost::noncopyable>("base", bp::no_init)
            .def("lmem_usage", &tpt::base::lmem_usage)
            .def("registers_usage", &tpt::base::registers_usage)
            .def("is_invalid", &tpt::base::is_invalid)
            .def("input_sizes", &detail::input_sizes)
        ;
    #undef __PROP
  }

  bp::class_<tpt::parameterized_base, bp::bases<tpt::base>, boost::noncopyable>("parameterized_base", bp::no_init)
                                        .add_property("ls0", &tpt::parameterized_base::ls0)
                                        .add_property("ls1", &tpt::parameterized_base::ls1);

  bp::class_<tpt::external_base, bp::bases<tpt::base>, boost::noncopyable>("external_base", bp::no_init);

#define WRAP_BASE(name) bp::class_<tpt::name, bp::bases<tpt::parameterized_base>, boost::noncopyable>(#name, bp::no_init);

  #define WRAP_TEMPLATE(name, basename, ...) bp::class_<tpt::name, std::shared_ptr<tpt::name>, bp::bases<basename>>(#name, bp::init<__VA_ARGS__>())\
                                      ;
  WRAP_TEMPLATE(elementwise_1d, tpt::parameterized_base, uint, uint, uint)
  WRAP_TEMPLATE(elementwise_2d, tpt::parameterized_base, uint, uint, uint, uint, uint)
  WRAP_TEMPLATE(reduce_1d, tpt::parameterized_base, uint, uint, uint)
  WRAP_BASE(reduce_2d)
  WRAP_TEMPLATE(reduce_2d_rows, tpt::reduce_2d, uint, uint, uint, uint, uint)
  WRAP_TEMPLATE(reduce_2d_cols, tpt::reduce_2d, uint, uint, uint, uint, uint)
  WRAP_BASE(gemm)
  WRAP_TEMPLATE(gemm_nn, tpt::gemm, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint)
  WRAP_TEMPLATE(gemm_tn, tpt::gemm, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint)
  WRAP_TEMPLATE(gemm_nt, tpt::gemm, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint)
  WRAP_TEMPLATE(gemm_tt, tpt::gemm, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint)
  WRAP_TEMPLATE(cublas_gemm, tpt::external_base, char, char)
  WRAP_TEMPLATE(intelblas_gemm, tpt::external_base, char, char)
  WRAP_TEMPLATE(intelblas_gemm_image, tpt::external_base, char, char)

}
