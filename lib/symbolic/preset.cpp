#include "isaac/symbolic/preset.h"

namespace isaac
{

namespace symbolic
{

namespace preset
{

void gemm::handle_node(array_expression::container_type &tree, size_t rootidx, args & a)
{
    //Matrix-Matrix product node
    if(tree[rootidx].op.type_family==OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY)
    {
        if(tree[rootidx].lhs.type_family==ARRAY_TYPE_FAMILY) a.A = &tree[rootidx].lhs;
        if(tree[rootidx].rhs.type_family==ARRAY_TYPE_FAMILY) a.B = &tree[rootidx].rhs;
        switch(tree[rootidx].op.type)
        {
          case OPERATOR_MATRIX_PRODUCT_NN_TYPE: a.type = MATRIX_PRODUCT_NN_TYPE; break;
          case OPERATOR_MATRIX_PRODUCT_NT_TYPE: a.type = MATRIX_PRODUCT_NT_TYPE; break;
          case OPERATOR_MATRIX_PRODUCT_TN_TYPE: a.type = MATRIX_PRODUCT_TN_TYPE; break;
          case OPERATOR_MATRIX_PRODUCT_TT_TYPE: a.type = MATRIX_PRODUCT_TT_TYPE; break;
          default: break;
        }
    }

    //Scalar multiplication node
    if(tree[rootidx].op.type==OPERATOR_MULT_TYPE)
    {
        //alpha*PROD
        if(tree[rootidx].lhs.type_family==VALUE_TYPE_FAMILY  && tree[rootidx].rhs.type_family==COMPOSITE_OPERATOR_FAMILY
           && tree[tree[rootidx].rhs.node_index].op.type_family==OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY)
        {
            a.alpha = &tree[rootidx].lhs;
            handle_node(tree, tree[rootidx].rhs.node_index, a);
        }

        //beta*C
        if(tree[rootidx].lhs.type_family==VALUE_TYPE_FAMILY  && tree[rootidx].rhs.type_family==ARRAY_TYPE_FAMILY)
        {
            a.beta = &tree[rootidx].lhs;
            a.C = &tree[rootidx].rhs;
        }
    }
}

gemm::args gemm::check(array_expression::container_type & tree, size_t rootidx)
{
    lhs_rhs_element * assigned = &tree[rootidx].lhs;
    gemm::args result ;
    if(tree[rootidx].rhs.type_family==COMPOSITE_OPERATOR_FAMILY)
    {
        rootidx = tree[rootidx].rhs.node_index;
        //Form X + Y
        if(tree[rootidx].op.type==OPERATOR_ADD_TYPE || tree[rootidx].op.type==OPERATOR_SUB_TYPE)
        {
            if(tree[rootidx].lhs.type_family==COMPOSITE_OPERATOR_FAMILY)
                handle_node(tree, tree[rootidx].lhs.node_index, result);
            if(tree[rootidx].rhs.type_family==COMPOSITE_OPERATOR_FAMILY)
                handle_node(tree, tree[rootidx].rhs.node_index, result);
        }
        else
            handle_node(tree, rootidx, result);
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
