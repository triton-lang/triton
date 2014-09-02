#ifndef VIENNACL_DEVICE_SPECIFIC_EXECUTE_HPP
#define VIENNACL_DEVICE_SPECIFIC_EXECUTE_HPP

#include <cstring>
#include <vector>
#include <typeinfo>

#include "viennacl/tools/tools.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/scheduler/forwards.h"

#include "atidlas/forwards.h"
#include "atidlas/templates/template_base.hpp"
#include "atidlas/tools/tree_parsing.hpp"
#include "atidlas/tools/execution_handler.hpp"

namespace atidlas
{

//viennacl::scheduler::statement morph_layout(viennacl::scheduler::statement const & statement)
//{
//  std::vector<viennacl::scheduler::lhs_rhs_element> matrices = tools::filter_elements(viennacl::scheduler::DENSE_MATRIX_TYPE, *it);
//  for(std::vector<viennacl::scheduler::lhs_rhs_element>::iterator it = matrices.begin() ; it != matrices.end() ; ++it)
//}

inline void execute(template_base const & T, statements_container const & statements, viennacl::ocl::context & ctx = viennacl::ocl::current_context(), bool force_compilation = false)
{
  //Generate program name
  std::string program_name = tools::statements_representation(statements, BIND_TO_HANDLE);
  execution_handler handler(program_name, ctx, ctx.current_device(), force_compilation);
  handler.add(program_name, T, statements);
  handler.execute(program_name, statements);
}

}

#endif
