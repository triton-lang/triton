#ifndef ATIDLAS_TEMPLATES_ROW_WISE_REDUCTION_HPP
#define ATIDLAS_TEMPLATES_ROW_WISE_REDUCTION_HPP


#include <vector>

#include "atidlas/scheduler/forwards.h"
#include "atidlas/traits/size.hpp"
#include "atidlas/backend/templates/template_base.hpp"

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
  virtual int check_invalid_impl(cl::Device const &, statements_container const &) const
  {
    if (p_.fetch_policy==FETCH_FROM_LOCAL)
      return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
    return TEMPLATE_VALID;
  }

  unsigned int lmem_usage() const
  {
    return p_.local_size_0*(p_.local_size_1+1);
  }

  static void parse(scheduler::statement const & statement, std::vector<size_t> & idx, bool & is_trans, scheduler::lhs_rhs_element & matrix)
  {
    idx = tools::filter_nodes(&tools::is_reduction, statement, false);
    is_trans = is_node_trans(statement.array(), idx[0], LHS_NODE_TYPE);
    matrix = lhs_most(statement.array(), idx[0]).lhs;
  }

  std::string generate_impl(std::string const & kernel_prefix, statements_container const & statements, std::vector<mapping_type> const & mappings, unsigned int simd_width, bool is_trans, std::vector<mapped_row_wise_reduction*> const & exprs) const
  {
    using tools::to_string;

    unsigned int lsize0 = p_.local_size_0;
    unsigned int lsize1 = p_.local_size_1+1;
    std::string lsize1str = to_string(lsize1);

    tools::kernel_generation_stream stream;

    stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl;
    stream << "__kernel void " << kernel_prefix << "(unsigned int M, unsigned int N, " << generate_arguments("#scalartype", mappings, statements) << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    tools::process(stream, PARENT_NODE_TYPE,
                          tools::create_process_accessors("scalar", "#scalartype #namereg = *#pointer;")
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

      void operator()(tools::kernel_generation_stream & stream, unsigned int simd_width) const
      {
        std::string data_type = tools::append_width("#scalartype",simd_width);

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
            str[a] = append_simd_suffix("#namereg.s",a);


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
            if (exprs[k]->root_node().op.type==scheduler::OPERATION_BINARY_MAT_VEC_PROD_TYPE)
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
    tools::evaluate(stream, PARENT_NODE_TYPE, accessors, statements, mappings);
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
    statements_container::data_type::const_iterator sit;
    std::vector<mapping_type>::const_iterator mit;
    for (mit = mappings.begin(), sit = statements.data().begin(); mit != mappings.end(); ++mit, ++sit)
    {
      std::vector<size_t> idx;
      scheduler::lhs_rhs_element A;
      parse(*sit, idx, is_trans, A);
      for (unsigned int j = 0; j < idx.size(); ++j)
        exprs.push_back((mapped_row_wise_reduction*)(mit->at(mapping_key(idx[j], PARENT_NODE_TYPE)).get()));
    }

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

  std::vector<atidlas_int_t> infos(statements_container const & statements, bool & is_trans)
  {
    std::vector<size_t> idx;
    scheduler::lhs_rhs_element A;
    parse(statements.data().front(), idx, is_trans, A);
    atidlas_int_t M = traits::size1(*A.matrix);
    atidlas_int_t N = traits::size2(*A.matrix);
    if(is_trans)
      std::swap(M,N);
    return tools::make_vector<atidlas_int_t>() << M << N;
  }

public:
  row_wise_reduction_template(row_wise_reduction_template::parameters_type const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base_impl<row_wise_reduction_template, row_wise_reduction_parameters>(parameters, binding_policy){ }

  virtual std::vector<atidlas_int_t> input_sizes(statements_container const & statements)
  {
    bool dummy;
    return infos(statements, dummy);
  }

  void enqueue(std::string const & kernel_prefix, std::vector<lazy_program_compiler> & programs, statements_container const & statements)
  {
    bool is_trans;
    std::vector<atidlas_int_t> MN = infos(statements, is_trans);

    cl::Kernel * kernel;
    if(is_trans  && p_.simd_width>1)
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
    kernel->arg(current_arg++, cl_uint(MN[0]));
    kernel->arg(current_arg++, cl_uint(MN[1]));
    set_arguments(statements, *kernel, current_arg);
//    cl::CommandQueue().enqueue()
  }
};

}

#endif
