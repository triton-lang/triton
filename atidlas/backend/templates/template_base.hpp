#ifndef ATIDLAS_TEMPLATES_TEMPLATE_BASE_
#define ATIDLAS_TEMPLATES_TEMPLATE_BASE_


#include <list>
#include <set>

#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/device_utils.hpp"

#include "viennacl/scheduler/forwards.h"
#include "viennacl/scheduler/io.hpp"

#include "atidlas/tools/lazy_program_compiler.hpp"
#include "atidlas/backend/templates/template_base.hpp"
#include "atidlas/backend/tools/misc.hpp"
#include "atidlas/backend/tools/tree_parsing.hpp"
#include "atidlas/backend/mapped_objects.hpp"

namespace atidlas
{

enum fetching_policy_type
{
  FETCH_FROM_LOCAL,
  FETCH_FROM_GLOBAL_STRIDED,
  FETCH_FROM_GLOBAL_CONTIGUOUS
};

class template_base
{
public:
  struct parameters_type
  {
    parameters_type(unsigned int _simd_width, unsigned int _local_size_1, unsigned int _local_size_2, unsigned int _num_kernels) : simd_width(_simd_width), local_size_0(_local_size_1), local_size_1(_local_size_2), num_kernels(_num_kernels){ }

    unsigned int simd_width;
    unsigned int local_size_0;
    unsigned int local_size_1;
    unsigned int num_kernels;
  };

private:
  /** @brief Functor to map the statements to the types defined in mapped_objects.hpp */
  class map_functor : public tools::traversal_functor
  {

    viennacl::scheduler::statement_node_numeric_type numeric_type(viennacl::scheduler::statement const * statement, atidlas_int_t root_idx) const
    {
      viennacl::scheduler::statement_node const * root_node = &statement->array()[root_idx];
      while (root_node->lhs.numeric_type==viennacl::scheduler::INVALID_NUMERIC_TYPE)
        root_node = &statement->array()[root_node->lhs.node_index];
      return root_node->lhs.numeric_type;
    }

  public:
    typedef tools::shared_ptr<mapped_object> result_type;

    map_functor(symbolic_binder & binder, mapping_type & mapping) : binder_(binder), mapping_(mapping){ }

    /** @brief Binary leaf */
    template<class T>
    result_type binary_leaf(viennacl::scheduler::statement const * statement, atidlas_int_t root_idx, mapping_type const * mapping) const
    {
      return result_type(new T(tools::numeric_type_to_string(numeric_type(statement,root_idx)), binder_.get(NULL), mapped_object::node_info(mapping, statement, root_idx)));
    }

    template<class NumericT>
    result_type operator()(NumericT const & /*scalar*/) const
    {
      return result_type(new mapped_host_scalar(tools::type_to_string<NumericT>::value(), binder_.get(NULL)));
    }

    /** @brief Scalar mapping */
    template<class NumericT>
    result_type operator()(viennacl::scalar<NumericT> const & scal) const
    {
      return result_type(new mapped_scalar(tools::type_to_string<NumericT>::value(), binder_.get(&viennacl::traits::handle(scal))));
    }

    /** @brief Vector mapping */
    template<class NumericT>
    result_type operator()(viennacl::vector_base<NumericT> const & vec) const
    {
      return result_type(new mapped_vector(tools::type_to_string<NumericT>::value(), binder_.get(&viennacl::traits::handle(vec))));
    }

    /** @brief Implicit vector mapping */
    template<class NumericT>
    result_type operator()(viennacl::implicit_vector_base<NumericT> const & /*vec*/) const
    {
      return result_type(new mapped_implicit_vector(tools::type_to_string<NumericT>::value(), binder_.get(NULL)));
    }

    /** @brief Matrix mapping */
    template<class NumericT>
    result_type operator()(viennacl::matrix_base<NumericT> const & mat) const
    {
      return result_type(new mapped_matrix(tools::type_to_string<NumericT>::value(), binder_.get(&viennacl::traits::handle(mat))));
    }

    /** @brief Implicit matrix mapping */
    template<class NumericT>
    result_type operator()(viennacl::implicit_matrix_base<NumericT> const & /*mat*/) const
    {
      return result_type(new mapped_implicit_matrix(tools::type_to_string<NumericT>::value(), binder_.get(NULL)));
    }

    /** @brief Traversal functor */
    void operator()(viennacl::scheduler::statement const & statement, atidlas_int_t root_idx, leaf_t leaf_t) const {
      mapping_type::key_type key(root_idx, leaf_t);
      viennacl::scheduler::statement_node const & root_node = statement.array()[root_idx];

      if (leaf_t == LHS_NODE_TYPE && root_node.lhs.type_family != viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
        mapping_.insert(mapping_type::value_type(key, tools::call_on_element(root_node.lhs, *this)));
      else if (leaf_t == RHS_NODE_TYPE && root_node.rhs.type_family != viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
        mapping_.insert(mapping_type::value_type(key,  tools::call_on_element(root_node.rhs, *this)));
      else if ( leaf_t== PARENT_NODE_TYPE)
      {
        if (root_node.op.type==viennacl::scheduler::OPERATION_BINARY_VECTOR_DIAG_TYPE)
          mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_vector_diag>(&statement, root_idx, &mapping_)));
        else if (root_node.op.type==viennacl::scheduler::OPERATION_BINARY_MATRIX_DIAG_TYPE)
          mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_diag>(&statement, root_idx, &mapping_)));
        else if (root_node.op.type==viennacl::scheduler::OPERATION_BINARY_MATRIX_ROW_TYPE)
          mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_row>(&statement, root_idx, &mapping_)));
        else if (root_node.op.type==viennacl::scheduler::OPERATION_BINARY_MATRIX_COLUMN_TYPE)
          mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_column>(&statement, root_idx, &mapping_)));
        else if (is_scalar_reduction(root_node))
          mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_scalar_reduction>(&statement, root_idx, &mapping_)));
        else if (is_vector_reduction(root_node))
          mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_row_wise_reduction>(&statement, root_idx, &mapping_)));
        else if (root_node.op.type == viennacl::scheduler::OPERATION_BINARY_MAT_MAT_PROD_TYPE)
          mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_product>(&statement, root_idx, &mapping_)));
        else if (root_node.op.type == viennacl::scheduler::OPERATION_UNARY_TRANS_TYPE)
          mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_trans>(&statement, root_idx, &mapping_)));
      }
    }

  private:
    symbolic_binder & binder_;
    mapping_type & mapping_;
  };

  /** @brief functor for setting the arguments of a kernel */
  class set_arguments_functor : public tools::traversal_functor
  {
  public:
    typedef void result_type;

    set_arguments_functor(symbolic_binder & binder, unsigned int & current_arg, viennacl::ocl::kernel & kernel) : binder_(binder), current_arg_(current_arg), kernel_(kernel){ }

    template<class NumericT>
    result_type operator()(NumericT const & scal) const
    {
      typedef typename viennacl::result_of::cl_type<NumericT>::type cl_scalartype;
      kernel_.arg(current_arg_++, cl_scalartype(scal));
    }

    /** @brief Scalar mapping */
    template<class NumericT>
    result_type operator()(viennacl::scalar<NumericT> const & scal) const
    {
      if (binder_.bind(&viennacl::traits::handle(scal)))
        kernel_.arg(current_arg_++, scal.handle().opencl_handle());
    }

    /** @brief Vector mapping */
    template<class NumericT>
    result_type operator()(viennacl::vector_base<NumericT> const & vec) const
    {
      if (binder_.bind(&viennacl::traits::handle(vec)))
      {
        kernel_.arg(current_arg_++, vec.handle().opencl_handle());
        kernel_.arg(current_arg_++, cl_uint(viennacl::traits::start(vec)));
        kernel_.arg(current_arg_++, cl_uint(viennacl::traits::stride(vec)));
      }
    }

    /** @brief Implicit vector mapping */
    template<class NumericT>
    result_type operator()(viennacl::implicit_vector_base<NumericT> const & vec) const
    {
      typedef typename viennacl::result_of::cl_type<NumericT>::type cl_scalartype;
      kernel_.arg(current_arg_++, cl_scalartype(vec.value()));
      if (vec.has_index())
        kernel_.arg(current_arg_++, cl_uint(vec.index()));
    }

    /** @brief Matrix mapping */
    template<class NumericT>
    result_type operator()(viennacl::matrix_base<NumericT> const & mat) const
    {
      if (binder_.bind(&viennacl::traits::handle(mat)))
      {
        kernel_.arg(current_arg_++, mat.handle().opencl_handle());
        kernel_.arg(current_arg_++, cl_uint(viennacl::traits::ld(mat)));
        kernel_.arg(current_arg_++, cl_uint(viennacl::traits::start1(mat)));
        kernel_.arg(current_arg_++, cl_uint(viennacl::traits::start2(mat)));
        kernel_.arg(current_arg_++, cl_uint(viennacl::traits::stride1(mat)));
        kernel_.arg(current_arg_++, cl_uint(viennacl::traits::stride2(mat)));
      }
    }

    /** @brief Implicit matrix mapping */
    template<class NumericT>
    result_type operator()(viennacl::implicit_matrix_base<NumericT> const & mat) const
    {
      kernel_.arg(current_arg_++, typename viennacl::result_of::cl_type<NumericT>::type(mat.value()));
    }

    /** @brief Traversal functor: */
    void operator()(viennacl::scheduler::statement const & statement, atidlas_int_t root_idx, leaf_t leaf_t) const
    {
      viennacl::scheduler::statement_node const & root_node = statement.array()[root_idx];
      if (leaf_t==LHS_NODE_TYPE && root_node.lhs.type_family != viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
        tools::call_on_element(root_node.lhs, *this);
      else if (leaf_t==RHS_NODE_TYPE && root_node.rhs.type_family != viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
        tools::call_on_element(root_node.rhs, *this);
    }

  private:
    symbolic_binder & binder_;
    unsigned int & current_arg_;
    viennacl::ocl::kernel & kernel_;
  };

protected:

  static inline void compute_reduction(tools::kernel_generation_stream & os, std::string acc, std::string cur, viennacl::scheduler::op_element const & op)
  {
    if (tools::elementwise_function(op))
      os << acc << "=" << tools::evaluate(op.type) << "(" << acc << "," << cur << ");" << std::endl;
    else
      os << acc << "= (" << acc << ")" << tools::evaluate(op.type)  << "(" << cur << ");" << std::endl;
  }

  static inline void compute_index_reduction(tools::kernel_generation_stream & os, std::string acc, std::string cur, std::string const & acc_value, std::string const & cur_value, viennacl::scheduler::op_element const & op)
  {
    //        os << acc << " = " << cur_value << ">" << acc_value  << "?" << cur << ":" << acc << ";" << std::endl;
    os << acc << "= select(" << acc << "," << cur << "," << cur_value << ">" << acc_value << ");" << std::endl;
    os << acc_value << "=";
    if (op.type==viennacl::scheduler::OPERATION_BINARY_ELEMENT_ARGFMAX_TYPE) os << "fmax";
    if (op.type==viennacl::scheduler::OPERATION_BINARY_ELEMENT_ARGMAX_TYPE) os << "max";
    if (op.type==viennacl::scheduler::OPERATION_BINARY_ELEMENT_ARGFMIN_TYPE) os << "fmin";
    if (op.type==viennacl::scheduler::OPERATION_BINARY_ELEMENT_ARGMIN_TYPE) os << "min";
    os << "(" << acc_value << "," << cur_value << ");"<< std::endl;
  }

  static inline void process_all(std::string const & type_key, std::string const & str,
                          tools::kernel_generation_stream & stream, std::vector<mapping_type> const & mappings)
  {
    for (std::vector<mapping_type>::const_iterator mit = mappings.begin(); mit != mappings.end(); ++mit)
      for (mapping_type::const_iterator mmit = mit->begin(); mmit != mit->end(); ++mmit)
        if (mmit->second->type_key()==type_key)
          stream << mmit->second->process(str) << std::endl;
  }


  static inline void process_all_at(std::string const & type_key, std::string const & str,
                             tools::kernel_generation_stream & stream, std::vector<mapping_type> const & mappings,
                             size_t root_idx, leaf_t leaf)
  {
    for (std::vector<mapping_type>::const_iterator mit = mappings.begin(); mit != mappings.end(); ++mit)
    {
      mapped_object * obj = mit->at(mapping_key(root_idx, leaf)).get();
      if (obj->type_key()==type_key)
        stream << obj->process(str) << std::endl;
    }
  }

  static inline std::string neutral_element(viennacl::scheduler::op_element const & op)
  {
    switch (op.type)
    {
    case viennacl::scheduler::OPERATION_BINARY_ADD_TYPE : return "0";
    case viennacl::scheduler::OPERATION_BINARY_MULT_TYPE : return "1";
    case viennacl::scheduler::OPERATION_BINARY_DIV_TYPE : return "1";
    case viennacl::scheduler::OPERATION_BINARY_ELEMENT_FMAX_TYPE : return "-INFINITY";
    case viennacl::scheduler::OPERATION_BINARY_ELEMENT_ARGFMAX_TYPE : return "-INFINITY";
    case viennacl::scheduler::OPERATION_BINARY_ELEMENT_MAX_TYPE : return "-INFINITY";
    case viennacl::scheduler::OPERATION_BINARY_ELEMENT_ARGMAX_TYPE : return "-INFINITY";
    case viennacl::scheduler::OPERATION_BINARY_ELEMENT_FMIN_TYPE : return "INFINITY";
    case viennacl::scheduler::OPERATION_BINARY_ELEMENT_ARGFMIN_TYPE : return "INFINITY";
    case viennacl::scheduler::OPERATION_BINARY_ELEMENT_MIN_TYPE : return "INFINITY";
    case viennacl::scheduler::OPERATION_BINARY_ELEMENT_ARGMIN_TYPE : return "INFINITY";

    default: throw generator_not_supported_exception("Unsupported reduction operator : no neutral element known");
    }
  }

  static std::string generate_arguments(std::vector<mapping_type> const & mappings, std::multimap<std::string, std::string> const & accessors, statements_container const & statements)
  {
    tools::kernel_generation_stream stream;
    tools::process(stream, PARENT_NODE_TYPE, accessors, statements, mappings);
    std::string res = stream.str();
    res.erase(res.rfind(','));
    return res;
  }

  static std::string matrix_arguments(std::string const & data_type)
  {
    return "__global " + data_type + "* #pointer, uint #ld, uint #start1, uint #start2, uint #stride1, uint #stride2,";
  }

  static std::string vector_arguments(std::string const & data_type)
  {
    return "__global " + data_type + "* #pointer, uint #start, uint #stride,";
  }

  static std::string generate_arguments(std::string const & data_type, std::vector<mapping_type> const & mappings, statements_container const & statements)
  {
    return generate_arguments(mappings, tools::create_process_accessors("scalar", "__global #scalartype* #pointer,")
                                                                      ("host_scalar", "#scalartype #name,")
                                                                      ("matrix", matrix_arguments(data_type))
                                                                      ("vector", vector_arguments(data_type))
                                                                      ("implicit_vector", "#scalartype #name,")
                                                                      ("implicit_matrix", "#scalartype #name,"), statements);
  }



  void set_arguments(statements_container const & statements, viennacl::ocl::kernel & kernel, unsigned int & current_arg)
  {
    tools::shared_ptr<symbolic_binder> binder = make_binder(binding_policy_);
    for (statements_container::data_type::const_iterator itt = statements.data().begin(); itt != statements.data().end(); ++itt)
      tools::traverse(*itt, itt->root(), set_arguments_functor(*binder,current_arg,kernel), true);
  }

  class invalid_template_exception : public std::exception
  {
  public:
    invalid_template_exception() : message_() {}
    invalid_template_exception(std::string message) :
      message_("ViennaCL: Internal error: The generator cannot apply the given template to the given statement: " + message + "\n"
               "If you are using a builtin template, please report on viennacl-support@lists.sourceforge.net! We will provide a fix as soon as possible\n"
               "If you are using your own template, please try using other parameters") {}
    virtual const char* what() const throw() { return message_.c_str(); }
    virtual ~invalid_template_exception() throw() {}
  private:
    std::string message_;
  };

  static void fetching_loop_info(fetching_policy_type policy, std::string const & bound, tools::kernel_generation_stream & stream, std::string & init, std::string & upper_bound, std::string & inc, std::string const & domain_id, std::string const & domain_size)
  {
    if (policy==FETCH_FROM_GLOBAL_STRIDED)
    {
      init = domain_id;
      upper_bound = bound;
      inc = domain_size;
    }
    else if (policy==FETCH_FROM_GLOBAL_CONTIGUOUS)
    {
      std::string chunk_size = "chunk_size";
      std::string chunk_start = "chunk_start";
      std::string chunk_end = "chunk_end";

      stream << "unsigned int " << chunk_size << " = (" << bound << "+" << domain_size << "-1)/" << domain_size << ";" << std::endl;
      stream << "unsigned int " << chunk_start << " =" << domain_id << "*" << chunk_size << ";" << std::endl;
      stream << "unsigned int " << chunk_end << " = min(" << chunk_start << "+" << chunk_size << ", " << bound << ");" << std::endl;
      init = chunk_start;
      upper_bound = chunk_end;
      inc = "1";
    }
  }

  static bool is_node_trans(viennacl::scheduler::statement::container_type const & array, size_t root_idx, leaf_t leaf_type)
  {
    bool res = false;
    viennacl::scheduler::lhs_rhs_element viennacl::scheduler::statement_node::*ptr;
    if (leaf_type==LHS_NODE_TYPE)
      ptr = &viennacl::scheduler::statement_node::lhs;
    else
      ptr = &viennacl::scheduler::statement_node::rhs;
    viennacl::scheduler::statement_node const * node = &array[root_idx];
    while ((node->*ptr).type_family==viennacl::scheduler::COMPOSITE_OPERATION_FAMILY)
    {
      if (array[(node->*ptr).node_index].op.type==viennacl::scheduler::OPERATION_UNARY_TRANS_TYPE)
        res = !res;
      node = &array[(node->*ptr).node_index];
    }
    return res;
  }

protected:

  static std::string append_simd_suffix(std::string const & str, unsigned int i)
  {
    assert(i < 16);
    static char suffixes[] = {'0','1','2','3','4','5','6','7','8','9',
                             'a','b','c','d','e','f'};
    return str + tools::to_string(suffixes[i]);
  }
  
  static bool is_offset_modifier(viennacl::scheduler::statement_node const & node)
  {
    return node.op.type==viennacl::scheduler::OPERATION_BINARY_VECTOR_DIAG_TYPE
        || node.op.type==viennacl::scheduler::OPERATION_BINARY_MATRIX_DIAG_TYPE
        || node.op.type==viennacl::scheduler::OPERATION_BINARY_MATRIX_ROW_TYPE
        || node.op.type==viennacl::scheduler::OPERATION_BINARY_MATRIX_COLUMN_TYPE;
  }

  static bool has_strided_access(statements_container const & statements)
  {
    for (statements_container::data_type::const_iterator it = statements.data().begin(); it != statements.data().end(); ++it)
    {
      //checks for vectors
      std::vector<viennacl::scheduler::lhs_rhs_element> vectors = tools::filter_elements(viennacl::scheduler::DENSE_VECTOR_TYPE, *it);
      for (std::vector<viennacl::scheduler::lhs_rhs_element>::iterator itt = vectors.begin(); itt != vectors.end(); ++itt)
        if (tools::call_on_vector(*itt, tools::stride_fun())>1)
          return true;

      //checks for matrix
      std::vector<viennacl::scheduler::lhs_rhs_element> matrices = tools::filter_elements(viennacl::scheduler::DENSE_MATRIX_TYPE, *it);
      for (std::vector<viennacl::scheduler::lhs_rhs_element>::iterator itt = matrices.begin(); itt != matrices.end(); ++itt)
        if (tools::call_on_matrix(*itt, tools::stride1_fun())>1 || tools::call_on_matrix(*itt, tools::stride2_fun())>2)
          return true;

      if(tools::filter_nodes(&is_offset_modifier, *it, true).empty()==false)
        return true;
    }
    return false;
  }

  static atidlas_int_t vector_size(viennacl::scheduler::statement_node const & node, bool up_to_internal_size)
  {
    using namespace viennacl::scheduler;
    using namespace tools;
    if (node.op.type==OPERATION_BINARY_MATRIX_DIAG_TYPE)
    {
      atidlas_int_t size1 = up_to_internal_size?call_on_matrix(node.lhs, internal_size1_fun()):call_on_matrix(node.lhs, size1_fun());
      atidlas_int_t size2 = up_to_internal_size?call_on_matrix(node.lhs, internal_size2_fun()):call_on_matrix(node.lhs, size2_fun());
      return std::min<atidlas_int_t>(size1, size2);
    }
    else if (node.op.type==OPERATION_BINARY_MATRIX_ROW_TYPE)
      return up_to_internal_size?call_on_matrix(node.lhs, internal_size2_fun()):call_on_matrix(node.lhs, size2_fun());
    else if (node.op.type==OPERATION_BINARY_MATRIX_COLUMN_TYPE)
      return up_to_internal_size?call_on_matrix(node.lhs, internal_size1_fun()):call_on_matrix(node.lhs, size1_fun());
    else
      return up_to_internal_size?call_on_vector(node.lhs, internal_size_fun()):call_on_vector(node.lhs, size_fun());
  }

  static std::pair<atidlas_int_t, atidlas_int_t> matrix_size(viennacl::scheduler::statement_node const & node, bool up_to_internal_size)
  {
    using namespace tools;
    if (node.op.type==viennacl::scheduler::OPERATION_BINARY_VECTOR_DIAG_TYPE)
    {
      atidlas_int_t is = call_on_vector(node.lhs, internal_size_fun());
      atidlas_int_t s = call_on_vector(node.lhs, size_fun());
      return up_to_internal_size?std::make_pair(is,is):std::make_pair(s,s);
    }
    else
    {
      atidlas_int_t size1 = up_to_internal_size?call_on_matrix(node.lhs, internal_size1_fun()):call_on_matrix(node.lhs, size1_fun());
      atidlas_int_t size2 = up_to_internal_size?call_on_matrix(node.lhs, internal_size2_fun()):call_on_matrix(node.lhs, size2_fun());
      return std::make_pair(size1, size2);
    }
  }

  //NB : templates are not used here because declaring a functor out of the generate() functions would be harder to read
  struct loop_body_base
  {
    virtual void operator()(tools::kernel_generation_stream & stream, unsigned int simd_width) const = 0;
  };

  static void element_wise_loop_1D(tools::kernel_generation_stream & stream, loop_body_base const & loop_body,
                                   fetching_policy_type fetch, unsigned int simd_width, std::string const & i, std::string const & bound, std::string const & domain_id, std::string const & domain_size)
  {
    std::string strwidth = tools::to_string(simd_width);
    std::string boundround = bound + "/" + strwidth;

    std::string init, upper_bound, inc;
    fetching_loop_info(fetch, boundround, stream, init, upper_bound, inc, domain_id, domain_size);
    stream << "for(unsigned int " << i << " = " << init << "; " << i << " < " << upper_bound << "; " << i << " += " << inc << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    loop_body(stream, simd_width);
    stream.dec_tab();
    stream << "}" << std::endl;

    if (simd_width>1)
    {
      stream << "for(unsigned int " << i << " = " << boundround << "*" << strwidth << " + " << domain_id << "; " << i << " < " << bound << "; " << i << " += " + domain_size + ")" << std::endl;
      stream << "{" << std::endl;
      stream.inc_tab();
      loop_body(stream, 1);
      stream.dec_tab();
      stream << "}" << std::endl;
    }
  }

  static std::string vstore(unsigned int simd_width, std::string const & value, std::string const & offset, std::string const & ptr)
  {
    if (simd_width==1)
      return "(" + ptr + ")[" + offset + "] = " + value;
    else
      return tools::append_width("vstore", simd_width) + "(" + value + ", " + offset + ", " + ptr + ")";
  }

  static std::string vload(unsigned int simd_width, std::string const & offset, std::string const & ptr)
  {
    if (simd_width==1)
      return "(" + ptr + ")[" + offset + "]";
    else
      return tools::append_width("vload", simd_width) + "(" + offset + ", " + ptr + ")";
  }

private:
  /** @brief Generates the body of the associated kernel function */
  virtual std::vector<std::string> generate_impl(std::string const & kernel_prefix, statements_container const & statements, std::vector<mapping_type> const & mapping) const = 0;

public:
  template_base(binding_policy_t binding_policy) : binding_policy_(binding_policy) {}

  virtual unsigned int lmem_usage(statements_container const &) const  { return 0; }

  virtual unsigned int registers_usage(statements_container const &) const { return 0; }

  virtual std::vector<atidlas_int_t> input_sizes(statements_container const & statements) = 0;

  virtual ~template_base(){ }

  std::vector<std::string> generate(std::string const & kernel_prefix, statements_container const & statements, viennacl::ocl::device const & device)
  {
    statements_container::data_type::const_iterator sit;
    std::vector<mapping_type>::iterator mit;

    if(int err = check_invalid(statements, device))
      throw generator_not_supported_exception("The supplied parameters for this template are invalid : err " + tools::to_string(err));

    //Create mapping
    std::vector<mapping_type> mappings(statements.data().size());
    tools::shared_ptr<symbolic_binder> binder = make_binder(binding_policy_);
    for (mit = mappings.begin(), sit = statements.data().begin(); sit != statements.data().end(); ++sit, ++mit)
      tools::traverse(*sit, sit->root(), map_functor(*binder,*mit), true);

    return generate_impl(kernel_prefix, statements, mappings);
  }

  /** @brief returns whether or not the profile has undefined behavior on particular device */
  virtual int check_invalid(statements_container const & statements, viennacl::ocl::device const & device) const = 0;

  virtual void enqueue(std::string const & kernel_prefix, std::vector<lazy_program_compiler> & programs, statements_container const & statements) = 0;

  virtual tools::shared_ptr<template_base> clone() const = 0;

private:
  binding_policy_t binding_policy_;
};


template<class TemplateType, class ParametersType>
class template_base_impl : public template_base
{
private:
  virtual int check_invalid_impl(viennacl::ocl::device const &, statements_container const &) const { return TEMPLATE_VALID; }

protected:
  bool has_misaligned_offset(statements_container const & statements)
  {
    for (statements_container::data_type::const_iterator it = statements.data().begin(); it != statements.data().end(); ++it)
    {
      //checks for vectors
      std::vector<viennacl::scheduler::lhs_rhs_element> vectors = tools::filter_elements(viennacl::scheduler::DENSE_VECTOR_TYPE, *it);
      for (std::vector<viennacl::scheduler::lhs_rhs_element>::iterator itt = vectors.begin(); itt != vectors.end(); ++itt)
        if (tools::call_on_vector(*itt, tools::stride_fun())>1)
          return true;

      //checks for matrix
      std::vector<viennacl::scheduler::lhs_rhs_element> matrices = tools::filter_elements(viennacl::scheduler::DENSE_MATRIX_TYPE, *it);
      for (std::vector<viennacl::scheduler::lhs_rhs_element>::iterator itt = matrices.begin(); itt != matrices.end(); ++itt)
        if (tools::call_on_matrix(*itt, tools::stride1_fun())>1 || tools::call_on_matrix(*itt, tools::stride2_fun())>2)
          return true;
    }
    return false;
  }

public:
  typedef ParametersType parameters_type;

  /** @brief The constructor */
  template_base_impl(parameters_type const & parameters, binding_policy_t binding_policy) : template_base(binding_policy), p_(parameters){ }

  parameters_type const & parameters() const
  { return p_; }

  tools::shared_ptr<template_base> clone() const
  { return tools::shared_ptr<template_base>(new TemplateType(*dynamic_cast<TemplateType const *>(this))); }

  /** @brief returns whether or not the profile has undefined behavior on particular device */
  int check_invalid(statements_container const & statements, viennacl::ocl::device const & device) const
  {
    using namespace viennacl::tools;

    //Query device informations
    size_t lmem_available = static_cast<size_t>(device.local_mem_size());
    size_t lmem_used = lmem_usage(statements);
    if (lmem_used>lmem_available)
      return TEMPLATE_LOCAL_MEMORY_OVERFLOW;

    //Invalid work group size
    size_t max_workgroup_size = device.max_work_group_size();
    std::vector<size_t> max_work_item_sizes = device.max_work_item_sizes();
    if (p_.local_size_0*p_.local_size_1 > max_workgroup_size)
      return TEMPLATE_WORK_GROUP_SIZE_OVERFLOW;
    if (p_.local_size_0 > max_work_item_sizes[0])
      return TEMPLATE_LOCAL_SIZE_0_OVERFLOW;

    if (p_.local_size_1 > max_work_item_sizes[1])
      return TEMPLATE_LOCAL_SIZE_1_OVERFLOW;

    //Advice from the Intel guide
    unsigned int warp_size = 8;
    if (device.type()==CL_DEVICE_TYPE_GPU)
    {
      //Advice from the nvidia guide
      warp_size = 32;
      //Advice from the AMD guide
      if (device.vendor_id()==4098)
        warp_size = 64;
    }
    if (((p_.local_size_0*p_.local_size_1)%warp_size)>0)
      return TEMPLATE_LOCAL_SIZE_NOT_WARP_MULTIPLE;

    //Invalid SIMD Width
    if (p_.simd_width!=1 && p_.simd_width!=2 &&
        p_.simd_width!=4 && p_.simd_width!=8 &&
        p_.simd_width!=16)
      return TEMPLATE_INVALID_SIMD_WIDTH;

    return check_invalid_impl(device, statements);
  }

protected:
  parameters_type p_;
  binding_policy_t binding_policy_;
};

}

#endif
