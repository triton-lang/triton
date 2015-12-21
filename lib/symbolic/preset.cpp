/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

#include "isaac/symbolic/preset.h"

namespace isaac
{

namespace symbolic
{

namespace preset
{

void matrix_product::handle_node(expression_tree::container_type const & tree, size_t rootidx, args & a)
{
    //Matrix-Matrix product node
    if(tree[rootidx].op.type_family==MATRIX_PRODUCT_TYPE_FAMILY)
    {
        if(tree[rootidx].lhs.subtype==DENSE_ARRAY_TYPE) a.A = &tree[rootidx].lhs;
        if(tree[rootidx].rhs.subtype==DENSE_ARRAY_TYPE) a.B = &tree[rootidx].rhs;
        switch(tree[rootidx].op.type)
        {
          case MATRIX_PRODUCT_NN_TYPE: a.type = MATRIX_PRODUCT_NN; break;
          case MATRIX_PRODUCT_NT_TYPE: a.type = MATRIX_PRODUCT_NT; break;
          case MATRIX_PRODUCT_TN_TYPE: a.type = MATRIX_PRODUCT_TN; break;
          case MATRIX_PRODUCT_TT_TYPE: a.type = MATRIX_PRODUCT_TT; break;
          default: break;
        }
    }

    //Scalar multiplication node
    if(tree[rootidx].op.type==MULT_TYPE)
    {
        //alpha*PROD
        if(tree[rootidx].lhs.subtype==VALUE_SCALAR_TYPE  && tree[rootidx].rhs.subtype==COMPOSITE_OPERATOR_TYPE
           && tree[tree[rootidx].rhs.node_index].op.type_family==MATRIX_PRODUCT_TYPE_FAMILY)
        {
            a.alpha = value_scalar(tree[rootidx].lhs.vscalar, tree[rootidx].lhs.dtype);
            handle_node(tree, tree[rootidx].rhs.node_index, a);
        }

        //beta*C
        if(tree[rootidx].lhs.subtype==VALUE_SCALAR_TYPE  && tree[rootidx].rhs.subtype==DENSE_ARRAY_TYPE)
        {
            a.beta = value_scalar(tree[rootidx].lhs.vscalar, tree[rootidx].lhs.dtype);
            a.C = &tree[rootidx].rhs;
        }
    }
}

matrix_product::args matrix_product::check(expression_tree::container_type const & tree, size_t rootidx)
{
    tree_node const * assigned = &tree[rootidx].lhs;
    numeric_type dtype = assigned->dtype;
    matrix_product::args result ;
    if(dtype==INVALID_NUMERIC_TYPE)
      return result;
    result.alpha = value_scalar(1, dtype);
    result.beta = value_scalar(0, dtype);
    if(tree[rootidx].rhs.subtype==COMPOSITE_OPERATOR_TYPE)
    {
        rootidx = tree[rootidx].rhs.node_index;
        bool is_add = tree[rootidx].op.type==ADD_TYPE;
        bool is_sub = tree[rootidx].op.type==SUB_TYPE;
        //Form X +- Y"
        if(is_add || is_sub)
        {
            if(tree[rootidx].lhs.subtype==COMPOSITE_OPERATOR_TYPE)
                handle_node(tree, tree[rootidx].lhs.node_index, result);
            else if(tree[rootidx].lhs.subtype==DENSE_ARRAY_TYPE)
            {
                result.C = &tree[rootidx].lhs;
                result.beta = value_scalar(1, dtype);
                result.alpha = value_scalar(is_add?1:-1,  dtype);
            }

            if(tree[rootidx].rhs.subtype==COMPOSITE_OPERATOR_TYPE)
                handle_node(tree, tree[rootidx].rhs.node_index, result);
            else if(tree[rootidx].rhs.subtype==DENSE_ARRAY_TYPE)
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
