#ifndef ATIDLAS_TEMPLATES_MATRIX_AXPY_HPP
#define ATIDLAS_TEMPLATES_MATRIX_AXPY_HPP


#include <vector>

#include "atidlas/backend/templates/template_base.hpp"

#include "viennacl/scheduler/forwards.h"
#include "viennacl/tools/tools.hpp"

namespace atidlas
{

class matrix_axpy_parameters : public template_base::parameters_type
{
public:
  matrix_axpy_parameters(unsigned int _simd_width,
                              unsigned int _local_size_0, unsigned int _local_size_1,
                              unsigned int _num_groups_0, unsigned int _num_groups_1,
                              fetching_policy_type _fetching_policy) : template_base::parameters_type(_simd_width, _local_size_0, _local_size_1, 1), num_groups_0(_num_groups_0), num_groups_1(_num_groups_1), fetching_policy(_fetching_policy){ }

  unsigned int num_groups_0;
  unsigned int num_groups_1;
  fetching_policy_type fetching_policy;
};

class matrix_axpy_template : public template_base_impl<matrix_axpy_template, matrix_axpy_parameters>
{
private:
  int check_invalid_impl(viennacl::ocl::device const &, statements_container const &) const
  {
    if (p_.simd_width>1)
      return TEMPLATE_INVALID_SIMD_WIDTH;
    if(p_.fetching_policy==FETCH_FROM_LOCAL)
      return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
    return TEMPLATE_VALID;
  }

  std::string generate_impl(std::string const & kernel_prefix, statements_container const & statements, std::vector<mapping_type> const & mappings, unsigned int simd_width) const
  {
    tools::kernel_generation_stream stream;

    std::string init0, upper_bound0, inc0, init1, upper_bound1, inc1;

    stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl;
    stream << "__kernel void " << kernel_prefix << "(unsigned int M, unsigned int N, " << generate_arguments("#scalartype", mappings, statements) << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    tools::process(stream, PARENT_NODE_TYPE, tools::create_process_accessors("scalar", "#scalartype #namereg = *#pointer;")
                                                                                               ("matrix", "#pointer += $OFFSET{#start1, #start2};")
                                                                                               ("vector", "#pointer += #start;"), statements, mappings);

    fetching_loop_info(p_.fetching_policy, "M", stream, init0, upper_bound0, inc0, "get_global_id(0)", "get_global_size(0)");
    stream << "for(unsigned int i = " << init0 << "; i < " << upper_bound0 << "; i += " << inc0 << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    fetching_loop_info(p_.fetching_policy, "N", stream, init1, upper_bound1, inc1, "get_global_id(1)", "get_global_size(1)");
    stream << "for(unsigned int j = " << init1 << "; j < " << upper_bound1 << "; j += " << inc1 << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    tools::process(stream, PARENT_NODE_TYPE, tools::create_process_accessors("matrix", tools::append_width("#scalartype",simd_width) + " #namereg = #pointer[$OFFSET{i*#stride1,j*#stride2}];")
                                                                                  ("vector_diag", "#scalartype #namereg = ((i + ((#diag_offset<0)?#diag_offset:0))!=(j-((#diag_offset>0)?#diag_offset:0)))?0:#pointer[min(i*#stride, j*#stride)];")
                                                                                              , statements, mappings);

    tools::evaluate(stream, PARENT_NODE_TYPE, tools::create_evaluate_accessors("matrix", "#namereg")
                                                                                      ("vector_diag", "#namereg")
                                                                                      ("scalar", "#namereg")
                                                    , statements, mappings);

    tools::process(stream, LHS_NODE_TYPE, tools::create_process_accessors("matrix", "#pointer[$OFFSET{i*#stride1,j*#stride2}] = #namereg;")
                                               , statements, mappings);

    stream.dec_tab();
    stream << "}" << std::endl;
    stream.dec_tab();
    stream << "}" << std::endl;

    stream.dec_tab();
    stream << "}" << std::endl;

    return stream.str();
  }

  std::vector<std::string> generate_impl(std::string const & kernel_prefix, statements_container const & statements, std::vector<mapping_type> const & mappings) const
  {
    std::vector<std::string> res;
    res.push_back(generate_impl(kernel_prefix, statements, mappings, 1));
    return res;
  }

public:
  matrix_axpy_template(parameters_type const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base_impl<matrix_axpy_template, matrix_axpy_parameters>(parameters, binding_policy), up_to_internal_size_(false){ }

  void up_to_internal_size(bool v)
  { up_to_internal_size_ = v; }

  std::vector<atidlas_int_t> input_sizes(statements_container const & statements)
  {
    viennacl::scheduler::statement const & statement = statements.data().front();
    std::pair<atidlas_int_t, atidlas_int_t> size = matrix_size(lhs_most(statement.array(), statement.root()), up_to_internal_size_);
    return tools::make_vector<atidlas_int_t>() << size.first << size.second;
  }

  void enqueue(std::string const & kernel_prefix, std::vector<lazy_program_compiler> & programs, statements_container const & statements)
  {
    viennacl::ocl::kernel & kernel = programs[0].program().get_kernel(kernel_prefix);

    kernel.local_work_size(0, p_.local_size_0);
    kernel.local_work_size(1, p_.local_size_1);
    kernel.global_work_size(0,p_.local_size_0*p_.num_groups_0);
    kernel.global_work_size(1,p_.local_size_1*p_.num_groups_1);

    unsigned int current_arg = 0;
    std::vector<atidlas_int_t> MN = input_sizes(statements);
    kernel.arg(current_arg++, cl_uint(MN[0]));
    kernel.arg(current_arg++, cl_uint(MN[1]));
    set_arguments(statements, kernel, current_arg);

    viennacl::ocl::enqueue(kernel);
  }


private:
  bool up_to_internal_size_;
};

}

#endif
