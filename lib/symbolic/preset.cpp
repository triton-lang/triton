#include "isaac/symbolic/preset.h"

namespace isaac
{

namespace symbolic
{

namespace preset
{

void gemm::handle_node(math_expression::container_type const & tree, size_t rootidx, args & a)
{
    //Matrix-Matrix product node
    if(tree[rootidx].op.type_family==OPERATOR_GEMM_TYPE_FAMILY)
    {
        if(tree[rootidx].lhs.type_family==ARRAY_TYPE_FAMILY) a.A = &tree[rootidx].lhs;
        if(tree[rootidx].rhs.type_family==ARRAY_TYPE_FAMILY) a.B = &tree[rootidx].rhs;
        switch(tree[rootidx].op.type)
        {
          case OPERATOR_GEMM_NN_TYPE: a.type = GEMM_NN_TYPE; break;
          case OPERATOR_GEMM_NT_TYPE: a.type = GEMM_NT_TYPE; break;
          case OPERATOR_GEMM_TN_TYPE: a.type = GEMM_TN_TYPE; break;
          case OPERATOR_GEMM_TT_TYPE: a.type = GEMM_TT_TYPE; break;
          default: break;
        }
    }

    //Scalar multiplication node
    if(tree[rootidx].op.type==OPERATOR_MULT_TYPE)
    {
        //alpha*PROD
        if(tree[rootidx].lhs.type_family==VALUE_TYPE_FAMILY  && tree[rootidx].rhs.type_family==COMPOSITE_OPERATOR_FAMILY
           && tree[tree[rootidx].rhs.node_index].op.type_family==OPERATOR_GEMM_TYPE_FAMILY)
        {
            a.alpha = value_scalar(tree[rootidx].lhs.vscalar, tree[rootidx].lhs.dtype);
            handle_node(tree, tree[rootidx].rhs.node_index, a);
        }

        //beta*C
        if(tree[rootidx].lhs.type_family==VALUE_TYPE_FAMILY  && tree[rootidx].rhs.type_family==ARRAY_TYPE_FAMILY)
        {
            a.beta = value_scalar(tree[rootidx].lhs.vscalar, tree[rootidx].lhs.dtype);
            a.C = &tree[rootidx].rhs;
        }
    }
}

gemm::args gemm::check(math_expression::container_type const & tree, size_t rootidx)
{
    lhs_rhs_element const * assigned = &tree[rootidx].lhs;
    numeric_type dtype = assigned->dtype;
    gemm::args result ;
    if(dtype==INVALID_NUMERIC_TYPE)
      return result;
    result.alpha = value_scalar(1, dtype);
    result.beta = value_scalar(0, dtype);
    if(tree[rootidx].rhs.type_family==COMPOSITE_OPERATOR_FAMILY)
    {
        rootidx = tree[rootidx].rhs.node_index;
        bool is_add = tree[rootidx].op.type==OPERATOR_ADD_TYPE;
        bool is_sub = tree[rootidx].op.type==OPERATOR_SUB_TYPE;
        //Form X +- Y"
        if(is_add || is_sub)
        {
            if(tree[rootidx].lhs.type_family==COMPOSITE_OPERATOR_FAMILY)
                handle_node(tree, tree[rootidx].lhs.node_index, result);
            else if(tree[rootidx].lhs.type_family==ARRAY_TYPE_FAMILY)
            {
                result.C = &tree[rootidx].lhs;
                result.beta = value_scalar(1, dtype);
                result.alpha = value_scalar(is_add?1:-1,  dtype);
            }

            if(tree[rootidx].rhs.type_family==COMPOSITE_OPERATOR_FAMILY)
                handle_node(tree, tree[rootidx].rhs.node_index, result);
            else if(tree[rootidx].rhs.type_family==ARRAY_TYPE_FAMILY)
            {
                result.C = &tree[rootidx].rhs;
                result.alpha = value_scalar(1, dtype);
                result.beta = value_scalar(is_add?1:-1, dtype);
            }
        }
        else{
            handle_node(tree, rootidx, result);
        }
    }
    if(result.C == NULL)
        result.C = assigned;
    else if(result.C->array != assigned->array)
        result.C = NULL;
    return result;
}

}

}

}
