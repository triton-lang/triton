#ifndef ATIDLAS_TEMPLATES_VECTOR_AXPY_HPP
#define ATIDLAS_TEMPLATES_VECTOR_AXPY_HPP

#include <vector>
#include <cmath>

#include "viennacl/scheduler/forwards.h"

#include "atidlas/mapped_objects.hpp"
#include "atidlas/tree_parsing.hpp"
#include "atidlas/forwards.h"
#include "atidlas/utils.hpp"

#include "atidlas/templates/template_base.hpp"
#include "atidlas/templates/utils.hpp"

#include "viennacl/tools/tools.hpp"

namespace atidlas
{

class vector_axpy_parameters : public template_base::parameters_type
{
public:
  vector_axpy_parameters(unsigned int _simd_width,
                         unsigned int _group_size, unsigned int _num_groups,
                         fetching_policy_type _fetching_policy) : template_base::parameters_type(_simd_width, _group_size, 1, 1), num_groups(_num_groups), fetching_policy(_fetching_policy){ }



  unsigned int num_groups;
  fetching_policy_type fetching_policy;
};

class vector_axpy_template : public template_base_impl<vector_axpy_template, vector_axpy_parameters>
{
private:
  virtual int check_invalid_impl(viennacl::ocl::device const & /*dev*/) const
  {
    if (p_.fetching_policy==FETCH_FROM_LOCAL)
      return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
    return TEMPLATE_VALID;
  }

  std::vector<std::string> generate_impl(std::string const & kernel_prefix, statements_container const & statements, std::vector<mapping_type> const & mappings) const
  {
    std::vector<std::string> result;
    for (unsigned int i = 0; i < 2; ++i)
    {
      utils::kernel_generation_stream stream;
      unsigned int simd_width = (i==0)?1:p_.simd_width;
      std::string str_simd_width = tools::to_string(simd_width);
      std::string data_type = utils::append_width("#scalartype",simd_width);

      stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << ",1,1)))" << std::endl;
      stream << "__kernel void " << kernel_prefix << i << "(unsigned int N," << generate_arguments(data_type, mappings, statements) << ")" << std::endl;
      stream << "{" << std::endl;
      stream.inc_tab();

      tree_parsing::process(stream, PARENT_NODE_TYPE,
                            utils::create_process_accessors("scalar", "#scalartype #namereg = *#pointer;")
                                                                       ("matrix", "#pointer += #start1 + #start2*#ld;")
                                                                       ("vector", "#pointer += #start;")
                                                                       ("vector", "#start/=" + str_simd_width + ";"), statements, mappings);

      std::string init, upper_bound, inc;
      fetching_loop_info(p_.fetching_policy, "N/"+str_simd_width, stream, init, upper_bound, inc, "get_global_id(0)", "get_global_size(0)");
      stream << "for(unsigned int i = " << init << "; i < " << upper_bound << "; i += " << inc << ")" << std::endl;
      stream << "{" << std::endl;
      stream.inc_tab();
      tree_parsing::process(stream, PARENT_NODE_TYPE,
                            utils::create_process_accessors("vector", data_type + " #namereg = #pointer[i*#stride];")
                                                                       ("matrix_row", "#scalartype #namereg = #pointer[$OFFSET{#row*#stride1, i*#stride2}];")
                                                                       ("matrix_column", "#scalartype #namereg = #pointer[$OFFSET{i*#stride1,#column*#stride2}];")
                                                                       ("matrix_diag", "#scalartype #namereg = #pointer[#diag_offset<0?$OFFSET{(i - #diag_offset)*#stride1, i*#stride2}:$OFFSET{i*#stride1, (i + #diag_offset)*#stride2}];")
                                                                       , statements, mappings);

      tree_parsing::evaluate(stream, PARENT_NODE_TYPE, utils::create_evaluate_accessors("vector", "#namereg")
                                                                                                  ("matrix_row", "#namereg")
                                                                                                  ("matrix_column", "#namereg")
                                                                                                  ("matrix_diag", "#namereg")
                                                                                                  ("scalar", "#namereg"), statements, mappings);

      tree_parsing::process(stream, LHS_NODE_TYPE, utils::create_process_accessors("vector", "#pointer[i*#stride] = #namereg;")
                                                                                             ("matrix_row", "#pointer[$OFFSET{#row, i}] = #namereg;")
                                                                                             ("matrix_column", "#pointer[$OFFSET{i, #column}] = #namereg;")
                                                                                             ("matrix_diag", "#pointer[#diag_offset<0?$OFFSET{i - #diag_offset, i}:$OFFSET{i, i + #diag_offset}] = #namereg;")
                                                                                             ,statements, mappings);

      stream.dec_tab();
      stream << "}" << std::endl;

      stream.dec_tab();
      stream << "}" << std::endl;
      result.push_back(stream.str());
    }

    return result;
  }

public:
  vector_axpy_template(vector_axpy_template::parameters_type const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base_impl<vector_axpy_template, vector_axpy_parameters>(parameters, binding_policy), up_to_internal_size_(false){ }

  void up_to_internal_size(bool v) { up_to_internal_size_ = v; }

  void enqueue(std::string const & kernel_prefix, std::vector<lazy_program_compiler> & programs,  statements_container const & statements)
  {
    viennacl::scheduler::statement const & statement = statements.data().front();
    atidlas_int_t size = vector_size(lhs_most(statement.array(), statement.root()), up_to_internal_size_);

    viennacl::ocl::kernel * kernel;
    if(p_.simd_width > 1 && (has_strided_access(statements) || (size%p_.simd_width>0) || has_misaligned_offset(statements)))
      kernel = &programs[0].program().get_kernel(kernel_prefix+"0");
    else
      kernel = &programs[1].program().get_kernel(kernel_prefix+"1");
    kernel->local_work_size(0, p_.local_size_0);
    kernel->global_work_size(0, p_.local_size_0*p_.num_groups);
    unsigned int current_arg = 0;
    kernel->arg(current_arg++, static_cast<cl_uint>(size));
    set_arguments(statements, *kernel, current_arg);
    viennacl::ocl::enqueue(*kernel);
  }

private:
  bool up_to_internal_size_;
};

}

#endif
