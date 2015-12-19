#include <string>
#include <vector>
#include "isaac/kernels/mapped_object.h"
#include "isaac/kernels/parse.h"

namespace isaac
{

namespace templates
{

class map_functor : public traversal_functor
{

  numeric_type get_numeric_type(isaac::expression_tree const * expression_tree, size_t root_idx) const
{
  expression_tree::node const * root_node = &expression_tree->tree()[root_idx];
  while (root_node->lhs.dtype==INVALID_NUMERIC_TYPE)
    root_node = &expression_tree->tree()[root_node->lhs.node_index];
  return root_node->lhs.dtype;
}

  template<class T>
  std::shared_ptr<mapped_object> binary_leaf(isaac::expression_tree const * expression_tree, size_t root_idx, mapping_type const * mapping) const
  {
    return std::shared_ptr<mapped_object>(new T(to_string(expression_tree->dtype()), binder_.get(), mapped_object::node_info(mapping, expression_tree, root_idx)));
  }

  std::shared_ptr<mapped_object> create(numeric_type dtype, values_holder) const
  {
    std::string strdtype = to_string(dtype);
    return std::shared_ptr<mapped_object>(new mapped_host_scalar(strdtype, binder_.get()));
  }

  std::shared_ptr<mapped_object> create(array_base const * a, bool is_assigned)  const
  {
    std::string dtype = to_string(a->dtype());
    unsigned int id = binder_.get(a, is_assigned);
    std::string type = "array";
    for(int_t i = 0 ; i < a->dim() ; ++i)
      type += (a->shape()[i]==1)?'1':'n';
    return std::shared_ptr<mapped_object>(new mapped_array(dtype, id, type));
  }

  std::shared_ptr<mapped_object> create(tree_node const & lhs_rhs, bool is_assigned = false) const
  {
    switch(lhs_rhs.subtype)
    {
      case VALUE_SCALAR_TYPE: return create(lhs_rhs.dtype, lhs_rhs.vscalar);
      case DENSE_ARRAY_TYPE: return create(lhs_rhs.array, is_assigned);
      case FOR_LOOP_INDEX_TYPE: return std::shared_ptr<mapped_object>(new mapped_placeholder(lhs_rhs.for_idx.level));
      default: throw "";
    }
  }


public:
  map_functor(symbolic_binder & binder, mapping_type & mapping, const driver::Device &device)
      : binder_(binder), mapping_(mapping), device_(device)
  {
  }

  void operator()(isaac::expression_tree const & expression_tree, size_t root_idx, leaf_t leaf_t) const
  {
    mapping_type::key_type key(root_idx, leaf_t);
    expression_tree::node const & root_node = expression_tree.tree()[root_idx];

    if (leaf_t == LHS_NODE_TYPE && root_node.lhs.subtype != COMPOSITE_OPERATOR_TYPE)
      mapping_.insert(mapping_type::value_type(key, create(root_node.lhs, detail::is_assignment(root_node.op))));
    else if (leaf_t == RHS_NODE_TYPE && root_node.rhs.subtype != COMPOSITE_OPERATOR_TYPE)
      mapping_.insert(mapping_type::value_type(key, create(root_node.rhs)));
    else if ( leaf_t== PARENT_NODE_TYPE)
    {
      if (root_node.op.type==VDIAG_TYPE)
        mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_vdiag>(&expression_tree, root_idx, &mapping_)));
      else if (root_node.op.type==MATRIX_DIAG_TYPE)
        mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_diag>(&expression_tree, root_idx, &mapping_)));
      else if (root_node.op.type==MATRIX_ROW_TYPE)
        mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_row>(&expression_tree, root_idx, &mapping_)));
      else if (root_node.op.type==MATRIX_COLUMN_TYPE)
        mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_column>(&expression_tree, root_idx, &mapping_)));
      else if(root_node.op.type==ACCESS_INDEX_TYPE)
        mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_array_access>(&expression_tree, root_idx, &mapping_)));
      else if (detail::is_scalar_reduce_1d(root_node))
        mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_reduce_1d>(&expression_tree, root_idx, &mapping_)));
      else if (detail::is_vector_reduce_1d(root_node))
        mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_reduce_2d>(&expression_tree, root_idx, &mapping_)));
      else if (root_node.op.type_family == MATRIX_PRODUCT_TYPE_FAMILY)
        mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_matrix_product>(&expression_tree, root_idx, &mapping_)));
      else if (root_node.op.type == REPEAT_TYPE)
        mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_repeat>(&expression_tree, root_idx, &mapping_)));
      else if (root_node.op.type == OUTER_PROD_TYPE)
        mapping_.insert(mapping_type::value_type(key, binary_leaf<mapped_outer>(&expression_tree, root_idx, &mapping_)));
      else if (detail::is_cast(root_node.op))
        mapping_.insert(mapping_type::value_type(key, std::shared_ptr<mapped_object>(new mapped_cast(root_node.op.type, binder_.get()))));
    }
  }
private:
  symbolic_binder & binder_;
  mapping_type & mapping_;
  driver::Device const & device_;
};


}

}
