#ifndef ATIDLAS_TEMPLATES_ROW_WISE_REDUCTION_HPP
#define ATIDLAS_TEMPLATES_ROW_WISE_REDUCTION_HPP


#include <vector>

#include "viennacl/scheduler/forwards.h"

#include "atidlas/mapped_objects.hpp"
#include "atidlas/tree_parsing.hpp"
#include "atidlas/utils.hpp"

#include "atidlas/templates/template_base.hpp"
#include "atidlas/templates/utils.hpp"

#include "viennacl/tools/tools.hpp"

#include "viennacl/scheduler/io.hpp"

namespace atidlas
{

struct row_wise_reduction_parameters : public template_base::parameters_type
{
  row_wise_reduction_parameters(unsigned int _simd_width,
                                unsigned int _local_size_0, unsigned int _local_size_1,
                                unsigned int _num_groups_0, fetching_policy_type _fetch_policy): template_base::parameters_type(_simd_width, _local_size_0, _local_size_1, 1),
    num_groups_0(_num_groups_0), fetch_policy(_fetch_policy) { }

  unsigned int num_groups_0;
  fetching_policy_type fetch_policy;
};

class row_wise_reduction_template : public template_base_impl<row_wise_reduction_template, row_wise_reduction_parameters>
{
private:
  virtual int check_invalid_impl(viennacl::ocl::device const & /*dev*/) const
  {
    if (p_.fetch_policy==FETCH_FROM_LOCAL)
      return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
    return TEMPLATE_VALID;
  }

  unsigned int n_lmem_elements() const
  {
    return p_.local_size_0*(p_.local_size_1+1);
  }

  static void parse(viennacl::scheduler::statement const & statement, std::vector<size_t> & idx, bool & is_trans, viennacl::scheduler::lhs_rhs_element & matrix)
  {
    idx = tree_parsing::filter_nodes(&utils::is_reduction, statement, false);
    is_trans = is_node_trans(statement.array(), idx[0], LHS_NODE_TYPE);
    matrix = lhs_most(statement.array(), idx[0]).lhs;
  }

  std::string generate_impl(std::string const & kernel_prefix, statements_container const & statements, std::vector<mapping_type> const & mappings, unsigned int simd_width, bool is_trans, std::vector<mapped_row_wise_reduction*> const & exprs) const
  {
    using tools::to_string;

    unsigned int lsize0 = p_.local_size_0;
    unsigned int lsize1 = p_.local_size_1+1;
    std::string lsize1str = to_string(lsize1);

    utils::kernel_generation_stream stream;

    stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl;
    stream << "__kernel void " << kernel_prefix << "(unsigned int M, unsigned int N, " << generate_arguments("#scalartype", mappings, statements) << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    tree_parsing::process(stream, PARENT_NODE_TYPE,
                          utils::create_process_accessors("scalar", "#scalartype #namereg = *#pointer;")
                                                          ("matrix", "#pointer += #start1 + #start2*#ld;")
                                                          ("matrix", "#ld *= #nldstride;")
                                                          ("vector", "#pointer += #start;"), statements, mappings);

    for (std::vector<mapped_row_wise_reduction*>::const_iterator it = exprs.begin(); it != exprs.end(); ++it)
      stream << (*it)->process("__local #scalartype #name_buf[" + to_string(lsize0*lsize1) + "];") << std::endl;

    stream << "unsigned int lid0 = get_local_id(0);" << std::endl;
    stream << "unsigned int lid1 = get_local_id(1);" << std::endl;
    stream << "unsigned int upper_bound_0 = ( M +" << p_.local_size_0 - 1 << ")/" << p_.local_size_0 << "*" << p_.local_size_0 << ";" << std::endl;
    stream << "for(unsigned int r = get_global_id(0); r < upper_bound_0; r += get_global_size(0)){" << std::endl;
    stream.inc_tab();

    for (std::vector<mapped_row_wise_reduction*>::const_iterator it = exprs.begin(); it != exprs.end(); ++it)
      stream << (*it)->process("#scalartype #name_acc = " + neutral_element((*it)->root_op()) + ";") << std::endl;

    stream << "if (r < M)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    class loop_body : public loop_body_base
    {
    public:
      loop_body(std::vector<mapped_row_wise_reduction*> const & _exprs, bool _is_trans) : exprs(_exprs), is_trans(_is_trans){ }

      void operator()(utils::kernel_generation_stream & stream, unsigned int simd_width) const
      {
        std::string data_type = utils::append_width("#scalartype",simd_width);

        for (std::vector<mapped_row_wise_reduction*>::const_iterator it = exprs.begin(); it != exprs.end(); ++it)
        {
          std::multimap<std::string, std::string> accessors;
          if (is_trans)
            accessors.insert(std::make_pair("matrix_trans", data_type + " #namereg = " + vload(simd_width, "c*#stride1", "#pointer + r*#ld")+";"));
          else
            accessors.insert(std::make_pair("matrix","#scalartype #namereg = #pointer[r*#stride1 + c*#ld];"));
          accessors.insert(std::make_pair("vector", data_type + " #namereg = " + vload(simd_width, "c*#stride", "#pointer")+";"));
          (*it)->process_recursive(stream, PARENT_NODE_TYPE, accessors);
        }


        //Update accumulators
        std::vector<std::string> str(simd_width);
        if (simd_width==1)
          str[0] = "#namereg";
        else
          for (unsigned int a = 0; a < simd_width; ++a)
            str[a] = "#namereg.s" + to_string(a);


        for (unsigned int k = 0; k < exprs.size(); ++k)
        {
          for (unsigned int a = 0; a < simd_width; ++a)
          {
            std::map<std::string, std::string> accessors;
            if (is_trans)
              accessors["matrix_trans"] = str[a];
            else
              accessors["matrix"] = str[a];
            accessors["vector"] = str[a];
            accessors["scalar"] = "#namereg";
            std::string value = exprs[k]->evaluate_recursive(LHS_NODE_TYPE, accessors);
            if (exprs[k]->root_node().op.type==viennacl::scheduler::OPERATION_BINARY_MAT_VEC_PROD_TYPE)
              value+= "*" + exprs[k]->evaluate_recursive(RHS_NODE_TYPE, accessors);

            if (exprs[k]->is_index_reduction())
              compute_index_reduction(stream, exprs[k]->process("#name_acc"), "c*"+to_string(simd_width) + to_string(a), exprs[k]->process("#name_acc_value"), value,exprs[k]->root_op());
            else
              compute_reduction(stream, exprs[k]->process("#name_acc"), value,exprs[k]->root_op());
          }
        }
      }
    private:
      std::vector<mapped_row_wise_reduction*> exprs;
      bool is_trans;
    };

    element_wise_loop_1D(stream, loop_body(exprs, is_trans), p_.fetch_policy, simd_width, "c", "N", "get_local_id(1)", "get_local_size(1)");
    stream.dec_tab();
    stream << "}" << std::endl;

    for (unsigned int k = 0; k < exprs.size(); ++k)
      stream << exprs[k]->process("#name_buf[lid0*" + lsize1str + "+ lid1] = #name_acc;") << std::endl;

    stream << "#pragma unroll" << std::endl;
    stream << "for(unsigned int stride = " << p_.local_size_1/2 << "; stride >0; stride /=2)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
    stream <<  "if (lid1 < stride)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    for (unsigned int k = 0; k < exprs.size(); k++)
      if (exprs[k]->is_index_reduction())
        compute_index_reduction(stream, exprs[k]->process("#name_buf[lid0*" + lsize1str + " + lid1]"), exprs[k]->process("#name_buf[lid0*" + lsize1str + " + lid1 + stride]")
                                , exprs[k]->process("#name_buf_value[lid0*" + lsize1str + " + lid1]"), exprs[k]->process("#name_buf_value[lid0*" + lsize1str + " + lid1 + stride]"),
                                exprs[k]->root_op());
      else
        compute_reduction(stream,exprs[k]->process("#name_buf[lid0*" + lsize1str + " + lid1]"), exprs[k]->process("#name_buf[lid0*" + lsize1str + " + lid1 + stride]"), exprs[k]->root_op());

    stream.dec_tab();
    stream << "}" << std::endl;

    stream.dec_tab();
    stream << "}" << std::endl;


    stream <<  "if (lid1 == 0 && r < M)";
    stream << "{" << std::endl;
    stream.inc_tab();
    std::map<std::string, std::string> accessors;
    accessors["row_wise_reduction"] = "#name_buf[lid0*" + lsize1str + "]";
    accessors["vector"] = "#pointer[r*#stride]";
    tree_parsing::evaluate(stream, PARENT_NODE_TYPE, accessors, statements, mappings);
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
    std::vector<mapped_row_wise_reduction*> exprs;
    bool is_trans = false;
    bool row_major = false;
    statements_container::data_type::const_iterator sit;
    std::vector<mapping_type>::const_iterator mit;
    for (mit = mappings.begin(), sit = statements.data().begin(); mit != mappings.end(); ++mit, ++sit)
    {
      std::vector<size_t> idx;
      viennacl::scheduler::lhs_rhs_element A;
      parse(*sit, idx, is_trans, A);
      row_major = utils::call_on_matrix(A, utils::row_major_fun());
      for (unsigned int j = 0; j < idx.size(); ++j)
        exprs.push_back((mapped_row_wise_reduction*)(mit->at(mapping_key(idx[j], PARENT_NODE_TYPE)).get()));
    }
    is_trans = is_trans ^ row_major;

    std::vector<std::string> res;
    if (is_trans && p_.simd_width>1)
    {
      res.push_back(generate_impl(kernel_prefix, statements, mappings, p_.simd_width, is_trans, exprs));
      res.push_back(generate_impl(kernel_prefix, statements, mappings, 1, is_trans, exprs));
    }
    else
      res.push_back(generate_impl(kernel_prefix, statements, mappings, 1, is_trans, exprs));

    return res;
  }
public:
  row_wise_reduction_template(row_wise_reduction_template::parameters_type const & parameters, char A_trans, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base_impl<row_wise_reduction_template, row_wise_reduction_parameters>(parameters, binding_policy), A_trans_(A_trans){ }

  void enqueue(std::string const & kernel_prefix, std::vector<lazy_program_compiler> & programs, statements_container const & statements)
  {
    std::vector<size_t> idx;
    viennacl::scheduler::lhs_rhs_element A;
    bool is_trans;
    parse(statements.data().front(), idx, is_trans, A);
    bool row_major = utils::call_on_matrix(A, utils::row_major_fun());

    viennacl::ocl::kernel * kernel;
    if((is_trans  ^ row_major)&& p_.simd_width>1)
    {
      if (has_strided_access(statements))
        kernel = &programs[1].program().get_kernel(kernel_prefix);
      else
        kernel = &programs[0].program().get_kernel(kernel_prefix);
    }
    else
      kernel = &programs[0].program().get_kernel(kernel_prefix);

    kernel->local_work_size(0,p_.local_size_0);
    kernel->local_work_size(1,p_.local_size_1);
    kernel->global_work_size(0,p_.local_size_0*p_.num_groups_0);
    kernel->global_work_size(1,p_.local_size_1);

    unsigned int current_arg = 0;
    if (is_trans)
    {
      kernel->arg(current_arg++, cl_uint(utils::call_on_matrix(A, utils::size2_fun())));
      kernel->arg(current_arg++, cl_uint(utils::call_on_matrix(A, utils::size1_fun())));
    }
    else
    {
      kernel->arg(current_arg++, cl_uint(utils::call_on_matrix(A, utils::size1_fun())));
      kernel->arg(current_arg++, cl_uint(utils::call_on_matrix(A, utils::size2_fun())));
    }


    set_arguments(statements, *kernel, current_arg);
    viennacl::ocl::enqueue(*kernel);
  }

private:
  const char A_trans_;
};

}

#endif
