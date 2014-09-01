#ifndef ATIDLAS_REDUCTION_TEMPLATE_HPP
#define ATIDLAS_REDUCTION_TEMPLATE_HPP


#include <vector>

#include "viennacl/backend/opencl.hpp"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/tools/tools.hpp"

#include "atidlas/tree_parsing.hpp"
#include "atidlas/utils.hpp"
#include "atidlas/templates/template_base.hpp"

namespace atidlas
{

struct reduction_parameters : public template_base::parameters_type
{
  reduction_parameters(unsigned int _simd_width,
                       unsigned int _group_size, unsigned int _num_groups,
                       fetching_policy_type _fetching_policy) : template_base::parameters_type(_simd_width, _group_size, 1, 2), num_groups(_num_groups), fetching_policy(_fetching_policy){ }

  unsigned int num_groups;
  fetching_policy_type fetching_policy;
};

class reduction_template : public template_base_impl<reduction_template, reduction_parameters>
{

private:
  unsigned int n_lmem_elements() const
  {
    return p_.local_size_0;
  }

  int check_invalid_impl(viennacl::ocl::device const & /*dev*/) const
  {
    if (p_.fetching_policy==FETCH_FROM_LOCAL)
      return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
    return TEMPLATE_VALID;
  }

  inline void reduce_1d_local_memory(utils::kernel_generation_stream & stream, unsigned int size, std::vector<mapped_scalar_reduction*> exprs,
                                     std::string const & buf_str, std::string const & buf_value_str) const
  {
    stream << "#pragma unroll" << std::endl;
    stream << "for(unsigned int stride = " << size/2 << "; stride >0; stride /=2)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
    stream << "if (lid <  stride)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    for (unsigned int k = 0; k < exprs.size(); k++)
      if (exprs[k]->is_index_reduction())
        compute_index_reduction(stream, exprs[k]->process(buf_str+"[lid]"), exprs[k]->process(buf_str+"[lid+stride]")
                                , exprs[k]->process(buf_value_str+"[lid]"), exprs[k]->process(buf_value_str+"[lid+stride]"),
                                exprs[k]->root_op());
      else
        compute_reduction(stream, exprs[k]->process(buf_str+"[lid]"), exprs[k]->process(buf_str+"[lid+stride]"), exprs[k]->root_op());
    stream.dec_tab();
    stream << "}" << std::endl;
    stream.dec_tab();
    stream << "}" << std::endl;
  }

  std::string generate_impl(std::string const & kernel_prefix,  statements_container const & statements, std::vector<mapping_type> const & mappings, unsigned int simd_width) const
  {
    utils::kernel_generation_stream stream;

    std::vector<mapped_scalar_reduction*> exprs;
    for (std::vector<mapping_type>::const_iterator it = mappings.begin(); it != mappings.end(); ++it)
      for (mapping_type::const_iterator iit = it->begin(); iit != it->end(); ++iit)
        if (mapped_scalar_reduction * p = dynamic_cast<mapped_scalar_reduction*>(iit->second.get()))
          exprs.push_back(p);
    std::size_t N = exprs.size();

    std::string arguments = generate_value_kernel_argument("unsigned int", "N");
    for (unsigned int k = 0; k < N; ++k)
    {
      std::string numeric_type = utils::numeric_type_to_string(lhs_most(exprs[k]->statement().array(),
                                                                        exprs[k]->statement().root()).lhs.numeric_type);
      if (exprs[k]->is_index_reduction())
      {
        arguments += generate_pointer_kernel_argument("__global", "unsigned int",  exprs[k]->process("#name_temp"));
        arguments += generate_pointer_kernel_argument("__global", numeric_type,  exprs[k]->process("#name_temp_value"));
      }
      else
        arguments += generate_pointer_kernel_argument("__global", numeric_type,  exprs[k]->process("#name_temp"));
    }


    /* ------------------------
     * First Kernel
     * -----------------------*/
    stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << ",1,1)))" << std::endl;
    stream << "__kernel void " << kernel_prefix << "_0" << "(" << arguments << generate_arguments("#scalartype", mappings, statements) << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    stream << "unsigned int lid = get_local_id(0);" << std::endl;
    tree_parsing::process(stream, PARENT_NODE_TYPE, utils::create_process_accessors("scalar", "#scalartype #namereg = *#pointer;")
                                                                                                ("vector", "#pointer += #start;"), statements, mappings);

    for (unsigned int k = 0; k < N; ++k)
    {
      if (exprs[k]->is_index_reduction())
      {
        stream << exprs[k]->process("__local #scalartype #name_buf_value[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
        stream << exprs[k]->process("#scalartype #name_acc_value = " + neutral_element(exprs[k]->root_op()) + ";") << std::endl;
        stream << exprs[k]->process("__local unsigned int #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
        stream << exprs[k]->process("unsigned int #name_acc = 0;") << std::endl;
      }
      else
      {
        stream << exprs[k]->process("__local #scalartype #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
        stream << exprs[k]->process("#scalartype #name_acc = " + neutral_element(exprs[k]->root_op()) + ";") << std::endl;
      }
    }

    class loop_body : public loop_body_base
    {
    public:
      loop_body(std::vector<mapped_scalar_reduction*> const & _exprs) : exprs(_exprs){ }

      void operator()(utils::kernel_generation_stream & stream, unsigned int simd_width) const
      {
        std::string i = (simd_width==1)?"i*#stride":"i";
        //Fetch vector entry
        for (std::vector<mapped_scalar_reduction*>::const_iterator it = exprs.begin(); it != exprs.end(); ++it)
          (*it)->process_recursive(stream, PARENT_NODE_TYPE, utils::create_process_accessors("vector",  utils::append_width("#scalartype",simd_width) + " #namereg = " + vload(simd_width,i,"#pointer")+";")
                                                                                           ("matrix_row",  "#scalartype #namereg = #pointer[$OFFSET{#row*#stride1, i*#stride2}];")
                                                                                           ("matrix_column", "#scalartype #namereg = #pointer[$OFFSET{i*#stride1,#column*#stride2}];")
                                                                                           ("matrix_diag", "#scalartype #namereg = #pointer[#diag_offset<0?$OFFSET{(i - #diag_offset)*#stride1, i*#stride2}:$OFFSET{i*#stride1, (i + #diag_offset)*#stride2}];"));


        //Update accumulators
        std::vector<std::string> str(simd_width);
        if (simd_width==1)
          str[0] = "#namereg";
        else
          for (unsigned int a = 0; a < simd_width; ++a)
            str[a] = "#namereg.s" + tools::to_string(a);

        for (unsigned int k = 0; k < exprs.size(); ++k)
        {
          for (unsigned int a = 0; a < simd_width; ++a)
          {
            std::map<std::string, std::string> accessors;
            accessors["vector"] = str[a];
            accessors["matrix_row"] = str[a];
            accessors["matrix_column"] = str[a];
            accessors["matrix_diag"] = str[a];
            accessors["scalar"] = "#namereg";
            std::string value = exprs[k]->evaluate_recursive(LHS_NODE_TYPE, accessors);
            if (exprs[k]->root_node().op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE)
              value+= "*" + exprs[k]->evaluate_recursive(RHS_NODE_TYPE, accessors);

            if (exprs[k]->is_index_reduction())
              compute_index_reduction(stream, exprs[k]->process("#name_acc"),  "i*"+tools::to_string(simd_width) + "+" + tools::to_string(a), exprs[k]->process("#name_acc_value"), value,exprs[k]->root_op());
            else
              compute_reduction(stream, exprs[k]->process("#name_acc"), value,exprs[k]->root_op());
          }
        }
      }

    private:
      std::vector<mapped_scalar_reduction*> exprs;
    };

    element_wise_loop_1D(stream, loop_body(exprs), p_.fetching_policy, simd_width, "i", "N", "get_global_id(0)", "get_global_size(0)");

    //Fills local memory
    for (unsigned int k = 0; k < N; ++k)
    {
      if (exprs[k]->is_index_reduction())
        stream << exprs[k]->process("#name_buf_value[lid] = #name_acc_value;") << std::endl;
      stream << exprs[k]->process("#name_buf[lid] = #name_acc;") << std::endl;
    }

    //Reduce local memory
    reduce_1d_local_memory(stream, p_.local_size_0, exprs, "#name_buf", "#name_buf_value");

    //Write to temporary buffers
    stream << "if (lid==0)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    for (unsigned int k = 0; k < N; ++k)
    {
      if (exprs[k]->is_index_reduction())
        stream << exprs[k]->process("#name_temp_value[get_group_id(0)] = #name_buf_value[0];") << std::endl;
      stream << exprs[k]->process("#name_temp[get_group_id(0)] = #name_buf[0];") << std::endl;
    }
    stream.dec_tab();
    stream << "}" << std::endl;

    stream.dec_tab();
    stream << "}" << std::endl;

    /* ------------------------
     * Second kernel
     * -----------------------*/
    stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << ",1,1)))" << std::endl;
    stream << "__kernel void " << kernel_prefix << "_1" << "(" << arguments << generate_arguments("#scalartype", mappings, statements) << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    stream << "unsigned int lid = get_local_id(0);" << std::endl;

    for (unsigned int k = 0; k < N; ++k)
    {
      if (exprs[k]->is_index_reduction())
      {
        stream << exprs[k]->process("__local unsigned int #name_buf[" + tools::to_string(p_.local_size_0) + "];");
        stream << exprs[k]->process("unsigned int #name_acc = 0;") << std::endl;
        stream << exprs[k]->process("__local #scalartype #name_buf_value[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
        stream << exprs[k]->process("#scalartype #name_acc_value = " + neutral_element(exprs[k]->root_op()) + ";");
      }
      else
      {
        stream << exprs[k]->process("__local #scalartype #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
        stream << exprs[k]->process("#scalartype #name_acc = " + neutral_element(exprs[k]->root_op()) + ";");
      }
    }

    stream << "for(unsigned int i = lid; i < " << p_.num_groups << "; i += get_local_size(0))" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    for (unsigned int k = 0; k < N; ++k)
      if (exprs[k]->is_index_reduction())
        compute_index_reduction(stream, exprs[k]->process("#name_acc"), exprs[k]->process("#name_temp[i]"),
                                exprs[k]->process("#name_acc_value"),exprs[k]->process("#name_temp_value[i]"),exprs[k]->root_op());
      else
        compute_reduction(stream, exprs[k]->process("#name_acc"), exprs[k]->process("#name_temp[i]"), exprs[k]->root_op());

    stream.dec_tab();
    stream << "}" << std::endl;

    for (unsigned int k = 0; k < N; ++k)
    {
      if (exprs[k]->is_index_reduction())
        stream << exprs[k]->process("#name_buf_value[lid] = #name_acc_value;") << std::endl;
      stream << exprs[k]->process("#name_buf[lid] = #name_acc;") << std::endl;
    }


    //Reduce and write final result
    reduce_1d_local_memory(stream, p_.local_size_0, exprs, "#name_buf", "#name_buf_value");

    stream << "if (lid==0)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    std::map<std::string, std::string> accessors;
    accessors["scalar_reduction"] = "#name_buf[0]";
    accessors["scalar"] = "*#pointer";
    accessors["vector"] = "#pointer[#start]";
    tree_parsing::evaluate(stream, PARENT_NODE_TYPE, accessors, statements, mappings);
    stream.dec_tab();
    stream << "}" << std::endl;

    stream.dec_tab();
    stream << "}" << std::endl;

    return stream.str();
  }

  std::vector<std::string> generate_impl(std::string const & kernel_prefix,  statements_container const & statements, std::vector<mapping_type> const & mappings) const
  {
    std::vector<std::string> result;
    result.push_back(generate_impl(kernel_prefix + "_strided", statements, mappings, 1));
    result.push_back(generate_impl(kernel_prefix, statements, mappings, p_.simd_width));
    return result;
  }
public:
  reduction_template(reduction_template::parameters_type const & parameters, binding_policy_t binding_policy = BIND_ALL_UNIQUE) : template_base_impl<reduction_template, reduction_parameters>(parameters, binding_policy) { }

  void enqueue(std::string const & kernel_prefix, std::vector<lazy_program_compiler> & programs, statements_container const & statements)
  {
    std::vector<scheduler::statement_node const *> reductions;
    cl_uint size = 0;
    for (statements_container::data_type::const_iterator it = statements.data().begin(); it != statements.data().end(); ++it)
    {
      std::vector<size_t> reductions_idx = tree_parsing::filter_nodes(&utils::is_reduction, *it, false);
      size = static_cast<cl_uint>(vector_size(lhs_most(it->array(), reductions_idx[0]), false));
      for (std::vector<size_t>::iterator itt = reductions_idx.begin(); itt != reductions_idx.end(); ++itt)
        reductions.push_back(&it->array()[*itt]);
    }

    scheduler::statement const & statement = statements.data().front();
    unsigned int scalartype_size = utils::size_of(lhs_most(statement.array(), statement.root()).lhs.numeric_type);

    viennacl::ocl::kernel * kernels[2];
    if (has_strided_access(statements) && p_.simd_width > 1)
    {
      kernels[0] = &programs[0].program().get_kernel(kernel_prefix+"_strided_0");
      kernels[1] = &programs[0].program().get_kernel(kernel_prefix+"_strided_1");
    }
    else
    {
      kernels[0] = &programs[1].program().get_kernel(kernel_prefix+"_0");
      kernels[1] = &programs[1].program().get_kernel(kernel_prefix+"_1");
    }

    kernels[0]->local_work_size(0, p_.local_size_0);
    kernels[0]->global_work_size(0,p_.local_size_0*p_.num_groups);

    kernels[1]->local_work_size(0, p_.local_size_0);
    kernels[1]->global_work_size(0,p_.local_size_0);

    for (unsigned int k = 0; k < 2; k++)
    {
      unsigned int n_arg = 0;
      kernels[k]->arg(n_arg++, size);
      unsigned int i = 0;
      unsigned int j = 0;
      for (std::vector<scheduler::statement_node const *>::const_iterator it = reductions.begin(); it != reductions.end(); ++it)
      {
        if (utils::is_index_reduction((*it)->op))
        {
          if (tmpidx_.size() <= j)
            tmpidx_.push_back(kernels[k]->context().create_memory(CL_MEM_READ_WRITE, p_.num_groups*4));
          kernels[k]->arg(n_arg++, tmpidx_[j]);
          j++;
        }

        if (tmp_.size() <= i)
          tmp_.push_back(kernels[k]->context().create_memory(CL_MEM_READ_WRITE, p_.num_groups*scalartype_size));
        kernels[k]->arg(n_arg++, tmp_[i]);
        i++;
      }
      set_arguments(statements, *kernels[k], n_arg);
    }

    for (unsigned int k = 0; k < 2; k++)
      viennacl::ocl::enqueue(*kernels[k]);

  }

private:
  std::vector< viennacl::ocl::handle<cl_mem> > tmp_;
  std::vector< viennacl::ocl::handle<cl_mem> > tmpidx_;
};

}

#endif
