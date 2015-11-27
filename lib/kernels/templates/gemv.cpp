#include <cstring>
#include <iostream>
#include "isaac/kernels/stream.h"
#include "isaac/kernels/keywords.h"
#include "isaac/kernels/templates/gemv.h"

#include "tools/arguments.hpp"
#include "tools/loop.hpp"
#include "tools/reductions.hpp"
#include "tools/vector_types.hpp"

#include <string>

namespace isaac
{
namespace templates
{

gemv_parameters::gemv_parameters(unsigned int _simd_width,
                              unsigned int _local_size_0, unsigned int _local_size_1,
                              unsigned int _num_groups_0, unsigned int _num_groups_1, fetching_policy_type _fetch_policy): base::parameters_type(_simd_width, _local_size_0, _local_size_1, 1),
num_groups_0(_num_groups_0), num_groups_1(_num_groups_1), fetch_policy(_fetch_policy) { }


int gemv::is_invalid_impl(driver::Device const &, math_expression const &) const
{
  if (p_.fetch_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

unsigned int gemv::lmem_usage(const math_expression&) const
{
  return (p_.local_size_0+1)*p_.local_size_1;
}

std::string gemv::generate_impl(std::string const & suffix, math_expression const & expression, driver::Device const & device, mapping_type const & mapping) const
{
  using tools::to_string;


  std::vector<mapped_gemv*> dots;
  std::vector<size_t> idx = filter_nodes(&is_dot, expression, expression.root(), false);
  for (auto & elem : idx)
    dots.push_back((mapped_gemv*)(mapping.at(mapping_key(elem, PARENT_NODE_TYPE)).get()));

  kernel_generation_stream stream;
  driver::backend_type backend = device.backend();
  std::string _size_t = size_type(device);

  std::string name[2] = {"prod", "reduce"};
  name[0] += suffix;
  name[1] += suffix;

  std::string arguments = _size_t + " M, " + _size_t + " N, " ;
  for (const auto & e : dots)
  {
    std::string numeric_type = to_string(lhs_most(e->math_expression().tree(), e->math_expression().root()).lhs.dtype);
    if (e->is_index_dot())
    {
      arguments += e->process(Global(backend).get() + " unsigned int* #name_temp, ");
      arguments += e->process(Global(backend).get() + " " + numeric_type + "* #name_temp_value,");
    }
    else
      arguments += e->process(Global(backend).get() + " " + numeric_type + "* #name_temp, ");
  }

  int col_simd_width = (dot_type_ == REDUCE_COLUMNS) ? 1 : p_.simd_width;
  switch(backend)
  {
    case driver::CUDA:
      stream << "#include  \"helper_math.h\"" << std::endl; break;
    case driver::OPENCL:
      stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl; break;
  }

  stream << KernelPrefix(backend) << " void " << name[0] << "(" << arguments << generate_arguments("#scalartype", device, mapping, expression) << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  process(stream, PARENT_NODE_TYPE,
                        {{"array1", "#scalartype #namereg = #pointer[#start];"},
                         {"arrayn", "#pointer += #start;"},
                         {"arraynn", "#pointer += #start;"}}, expression, mapping);

  unsigned int local_size_0_ld = p_.local_size_0;
  std::string local_size_0_ld_str = to_string(local_size_0_ld);

  for (const auto & e : dots)
    stream << e->process(Local(backend).get() + " " + append_width("#scalartype", col_simd_width) + " #name_buf[" + to_string(p_.local_size_1*local_size_0_ld) + "];") << std::endl;

  stream << "for(" << _size_t << " r = " << GlobalIdx1(backend) << "*" << col_simd_width << "; r < (M +" << p_.local_size_1 - 1 << ")/" << p_.local_size_1 << "*" << p_.local_size_1*col_simd_width << "; r += " << GlobalSize1(backend) << "*" << col_simd_width << ")" << std::endl;
  stream << "{" << std::endl;

  stream.inc_tab();
  stream << "" << _size_t << " lidx = " << LocalIdx0(backend) << ";" << std::endl;
  stream << "" << _size_t << " lidy = " << LocalIdx1(backend) <<";" << std::endl;

  for (const auto & e : dots){
    std::string data_type = append_width("#scalartype",col_simd_width);

    stream << e->process(data_type + " #name_acc = " + InitPrefix(backend, data_type).get()  + "(" + neutral_element((e)->root_op(), backend, "#scalartype") + ");") << std::endl;
  }

  stream << "if (r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  element_wise_loop_1D(stream, p_.fetch_policy, (dot_type_==REDUCE_COLUMNS)?p_.simd_width:1, "c", "N", GlobalIdx0(backend).get(), GlobalSize0(backend).get(), device, [&](unsigned int row_simd_width)
  {

    std::set<std::string> already_fetched;
    for (const auto & e : dots)
    {
      std::map<std::string, std::string> accessors;
      if(dot_type_==REDUCE_COLUMNS)
      {
        std::string data_type = append_width("#scalartype",row_simd_width);
        accessors["arraynn"] = data_type + " #namereg = " + vload(row_simd_width, "#scalartype", "c*#stride", "#pointer + r*#ld", "1", backend,false)+";";
        accessors["repeat"] = data_type + " #namereg = " + vload(row_simd_width, "#scalartype", "(c%#sub0)*#stride", "#pointer + (r%#sub1)*#stride ", "1", backend,false)+";";
      }
      else
      {
        std::string data_type = append_width("#scalartype",col_simd_width);
        accessors["arraynn"] = data_type + " #namereg = " + vload(col_simd_width, "#scalartype", "0", "#pointer + r*#stride + c*#ld", "1", backend,false) + ";";
        accessors["repeat"] = "#scalartype #namereg = $VALUE{(r%#sub0)*#stride, (c%#sub1)*#stride};";
      }
      e->process_recursive(stream, PARENT_NODE_TYPE, accessors, already_fetched);
    }

    //Update accumulators
    std::vector<std::string> str(row_simd_width);
    if (row_simd_width==1)
      str[0] = "#namereg";
    else
      for (unsigned int a = 0; a < row_simd_width; ++a)
        str[a] = access_vector_type("#namereg",a);


    for (auto & elem : dots)
      for (unsigned int a = 0; a < row_simd_width; ++a)
      {
        std::string value = elem->evaluate_recursive(LHS_NODE_TYPE, {{"arraynn", str[a]}, {"repeat", str[a]}, {"array1", "#namereg"}});
        if (elem->is_index_dot())
          compute_index_dot(stream, elem->process("#name_acc"), "c*"+to_string(row_simd_width) + to_string(a), elem->process("#name_acc_value"), value, elem->root_op());
        else
          compute_dot(stream, elem->process("#name_acc"), value,elem->root_op());
      }
  });
  stream.dec_tab();
  stream << "}" << std::endl;

  for (auto & expr : dots)
    stream << expr->process("#name_buf[lidy*" + local_size_0_ld_str + "+ lidx] = #name_acc;") << std::endl;

  stream << "#pragma unroll" << std::endl;
  stream << "for(" << _size_t << " stride = " << p_.local_size_0/2 << "; stride >0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << LocalBarrier(backend) << ";" << std::endl;
  stream <<  "if (lidx < stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  for (auto & e : dots)
    if (e->is_index_dot())
      compute_index_dot(stream, e->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx]"), e->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx + stride]")
                                    , e->process("#name_buf_value[lidy*" + local_size_0_ld_str + " + lidx]"), e->process("#name_buf_value[lidy*" + local_size_0_ld_str + " + lidx + stride]")
                                    , e->root_op());
    else
      compute_dot(stream,e->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx]"), e->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx + stride]"), e->root_op());

  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;


  stream <<  "if (lidx == 0 && r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  if(p_.num_groups_0==1)
  {
    std::map<std::string, std::string> accessors;
    for(int s = 0 ; s < col_simd_width ; ++s)
    {
        accessors["gemv"] = "#name_buf[lidy*" + local_size_0_ld_str + "]";
        if(col_simd_width > 1)
            accessors["gemv"] = access_vector_type(accessors["gemv"], s);
        accessors["arrayn"] = "#pointer[(r +" + to_string(s) + ")*#stride]";
        accessors["array1n"] = "#pointer[(r +" + to_string(s) + ")*#stride]";
        accessors["arrayn1"] = "#pointer[(r +" + to_string(s) + ")*#stride]";
        stream << evaluate(PARENT_NODE_TYPE, accessors, expression, expression.root(), mapping) << ";" << std::endl;
    }
  }
  else
  {
    for (mapped_dot const * e : dots)
    {
      if(col_simd_width > 1)
          stream << "if(M - r > " << col_simd_width << "){" << std::endl;
      if (e->is_index_dot())
          stream << e->process(vstore(col_simd_width,"uint", "#name_buf_value[lidy*" + local_size_0_ld_str + "]", "0", "#name_temp_value + r + M*" + GroupIdx0(backend).get(), "1", backend, false)) << ";" << std::endl;
      stream << e->process(vstore(col_simd_width,"#scalartype", "#name_buf[lidy*" + local_size_0_ld_str + "]", "0", "#name_temp + r + M*" + GroupIdx0(backend).get(), "1", backend, false)) << ";" << std::endl;
      if(col_simd_width > 1)
      {
          stream << "}" << std::endl;
          stream << "else{" << std::endl;
          stream.inc_tab();
          for(int s = 0 ; s < col_simd_width ; ++s){
              if (e->is_index_dot())
                  stream << "if(r + " << s << "< M) " << e->process("#name_temp_value[r + " + to_string(s) + " + M*" + GroupIdx0(backend).get() + "] = " + access_vector_type("#name_buf_value[lidy*" + local_size_0_ld_str + "]", s)) << ";" << std::endl;
              stream << "if(r + " << s << "< M) " << e->process("#name_temp[r + " + to_string(s) + " + M*" + GroupIdx0(backend).get() + "] = " + access_vector_type("#name_buf[lidy*" + local_size_0_ld_str + "]", s)) << ";" << std::endl;
          }
          stream.dec_tab();
          stream << "}" << std::endl;
      }
    }
  }
  stream.dec_tab();
  stream << "}" << std::endl;


  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;


  if(p_.num_groups_0>1)
  {
  /////////////////////////////////////////
  ////////////// Kernel 2
  ////////////////////////////////////////

  if(backend==driver::OPENCL)
    stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl;

  stream << KernelPrefix(backend) << " void " << name[1] << "(" << arguments << generate_arguments("#scalartype", device, mapping, expression) << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  process(stream, PARENT_NODE_TYPE,
                        {{"array1", "#scalartype #namereg = #pointer[#start];"},
                         {"arrayn", "#pointer += #start;"},
                         {"array1n", "#pointer += #start;"},
                         {"arrayn1", "#pointer += #start;"},
                         {"arraynn", "#pointer += #start; "}}, expression, mapping);

  for (const auto & e : dots)
    stream << e->process(Local(backend).get() + " #scalartype #name_buf[" + to_string(p_.local_size_1*local_size_0_ld) + "];") << std::endl;

  stream << "for(" << _size_t << " r = " << GlobalIdx1(backend) << "; r < (M +" << p_.local_size_1 - 1 << ")/" << p_.local_size_1 << "*" << p_.local_size_1 << "; r += " << GlobalSize1(backend) << "){" << std::endl;
  stream.inc_tab();
  stream << _size_t << " lidx = " << LocalIdx0(backend) << ";" << std::endl;
  stream << _size_t << " lidy = " << LocalIdx1(backend) <<";" << std::endl;

  for (const auto & e : dots)
    stream << e->process("#scalartype #name_acc = " + neutral_element((e)->root_op(), backend, "#scalartype") + ";") << std::endl;

  stream << "if (r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << "for(" << _size_t << " c = lidx; c < " << p_.num_groups_0 << "; c += " << LocalSize0(backend) << "){" << std::endl;
  stream.inc_tab();

  for (mapped_dot* e: dots)
    compute_dot(stream, e->process("#name_acc"), e->process("#name_temp[r + M*c]"), e->root_op());

  stream.dec_tab();
  stream << "}" << std::endl;


  stream.dec_tab();
  stream << "}" << std::endl;

  for (auto & expr : dots)
    stream << expr->process("#name_buf[lidy*" + local_size_0_ld_str + "+ lidx] = #name_acc;") << std::endl;

  stream << "#pragma unroll" << std::endl;
  stream << "for(" << _size_t << " stride = " << p_.local_size_0/2 << "; stride >0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << LocalBarrier(backend) << ";" << std::endl;
  stream <<  "if (lidx < stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  for (auto & e : dots)
    if (e->is_index_dot())
      compute_index_dot(stream, e->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx]"), e->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx + stride]")
                                    , e->process("#name_buf_value[lidy*" + local_size_0_ld_str + " + lidx]"), e->process("#name_buf_value[lidy*" + local_size_0_ld_str + " + lidx + stride]")
                                    , e->root_op());
    else
      compute_dot(stream,e->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx]"), e->process("#name_buf[lidy*" + local_size_0_ld_str + " + lidx + stride]"), e->root_op());

  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;


  stream <<  "if (lidx == 0 && r < M)";
  stream << "{" << std::endl;
  stream.inc_tab();

  std::map<std::string, std::string> accessors;
  accessors["gemv"] = "#name_buf[lidy*" + local_size_0_ld_str + "]";
  accessors["arrayn"] = "#pointer[r*#stride]";
  accessors["array1n"] = "#pointer[r*#stride]";
  accessors["arrayn1"] = "#pointer[r*#stride]";
  stream << evaluate(PARENT_NODE_TYPE, accessors, expression, expression.root(), mapping) << ";" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;


  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;
  }

//  std::cout << stream.str() << std::endl;
  return stream.str();
}

gemv::gemv(gemv::parameters_type const & parameters,
                                         gemv::dot_type rtype,
                                         binding_policy_t binding_policy) :
  base_impl<gemv, gemv_parameters>(parameters, binding_policy),
  dot_type_(rtype){ }

std::vector<int_t> gemv::input_sizes(math_expression const & expression) const
{
  std::vector<std::size_t> idx = filter_nodes(&is_dot, expression, expression.root(), false);
  std::pair<int_t, int_t> MN = matrix_size(expression.tree(), lhs_most(expression.tree(), idx[0]));
  if(dot_type_==REDUCE_COLUMNS)
    std::swap(MN.first,MN.second);
  return {MN.first, MN.second};
}

void gemv::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, base & fallback, execution_handler const & control)
{
  math_expression const & expression = control.x();

  std::vector<int_t> MN = input_sizes(expression);
  std::vector<math_expression::node const *> dots;
  std::vector<size_t> dots_idx = filter_nodes(&is_dot, expression, expression.root(), false);
  for (size_t idx : dots_idx)
    dots.push_back(&expression.tree()[idx]);

  //Fallback
  if(p_.simd_width>1 && requires_fallback(expression))
  {
      fallback.enqueue(queue, program, "fallback", fallback, control);
      return;
  }

  //Kernel
  std::string name[2] = {"prod", "reduce"};
  name[0] += suffix;
  name[1] += suffix;

  unsigned int nk = (p_.num_groups_0==1)?1:2;

  std::vector<driver::Kernel> kernels;
  for(unsigned int k = 0 ; k < nk ; ++k)
    kernels.push_back(driver::Kernel(program, name[k].c_str()));

  for(unsigned int k = 0 ; k < nk ; ++k)
  {
    driver::Kernel & kernel = kernels[k];
    unsigned int n_arg = 0;
    int_t M = MN[0];
    int_t N = MN[1];
    kernel.setSizeArg(n_arg++, M);
    kernel.setSizeArg(n_arg++, N);
    kernel.setArg(n_arg++, driver::backend::workspaces::get(queue)); //Temporary buffers
    set_arguments(expression, kernel, n_arg, binding_policy_);
  }

  //NDRange
  driver::NDRange global[2] = { driver::NDRange(p_.local_size_0*p_.num_groups_0, p_.local_size_1*p_.num_groups_1), driver::NDRange(p_.local_size_0, p_.local_size_1*p_.num_groups_1) };
  driver::NDRange local[2] = { driver::NDRange(p_.local_size_0, p_.local_size_1), driver::NDRange(p_.local_size_0, p_.local_size_1) };
  for(unsigned int i = 0 ; i < nk ; ++i)
    control.execution_options().enqueue(program.context(), kernels[i], global[i], local[i]);
}

gemv_n::gemv_n(gemv_parameters  const & parameters,binding_policy_t binding_policy): gemv(parameters, REDUCE_ROWS, binding_policy){}

gemv_n::gemv_n(unsigned int simd, unsigned int ls1, unsigned int ls2,  unsigned int ng1, unsigned int ng2,
               fetching_policy_type fetch, binding_policy_t bind): gemv(gemv_parameters(simd, ls1, ls2, ng1, ng2, fetch), REDUCE_ROWS, bind) {}

gemv_t::gemv_t(gemv::parameters_type  const & parameters, binding_policy_t binding_policy): gemv(parameters, REDUCE_COLUMNS, binding_policy){}

gemv_t::gemv_t(unsigned int simd, unsigned int ls1, unsigned int ls2, unsigned int ng1, unsigned int ng2,
               fetching_policy_type fetch, binding_policy_t bind): gemv(gemv_parameters(simd, ls1, ls2, ng1, ng2, fetch), REDUCE_COLUMNS, bind) {}


}
}
