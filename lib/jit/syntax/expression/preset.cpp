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

#include "isaac/jit/syntax/expression/preset.h"

namespace isaac
{

namespace symbolic
{

namespace preset
{

void matrix_product::handle_node(expression_tree::data_type const & tree, size_t root, args & a)
{
    expression_tree::node const & node = tree[root];
    if(node.type != COMPOSITE_OPERATOR_TYPE)
      return;

    expression_tree::node const & left = tree[node.binary_operator.lhs];
    expression_tree::node const & right = tree[node.binary_operator.rhs];

    //Matrix-Matrix product node
    if(node.binary_operator.op.type_family==MATRIX_PRODUCT)
    {
        if(left.type==DENSE_ARRAY_TYPE) a.A = &left;
        if(right.type==DENSE_ARRAY_TYPE) a.B = &right;
        switch(node.binary_operator.op.type)
        {
          case MATRIX_PRODUCT_NN_TYPE: a.type = MATRIX_PRODUCT_NN; break;
          case MATRIX_PRODUCT_NT_TYPE: a.type = MATRIX_PRODUCT_NT; break;
          case MATRIX_PRODUCT_TN_TYPE: a.type = MATRIX_PRODUCT_TN; break;
          case MATRIX_PRODUCT_TT_TYPE: a.type = MATRIX_PRODUCT_TT; break;
          default: break;
        }
    }

    //Scalar multiplication node
    if(node.binary_operator.op.type==MULT_TYPE)
    {
        //alpha*PROD
        if(left.type==VALUE_SCALAR_TYPE  && right.type==COMPOSITE_OPERATOR_TYPE
           && right.binary_operator.op.type_family==MATRIX_PRODUCT)
        {
            a.alpha = cast(value_scalar(left.scalar, left.dtype), node.dtype);
            handle_node(tree, node.binary_operator.rhs, a);
        }

        //beta*C
        if(left.type==VALUE_SCALAR_TYPE  && right.type==DENSE_ARRAY_TYPE)
        {
            a.beta = cast(value_scalar(left.scalar, left.dtype), node.dtype);
            a.C = &right;
        }
    }
}

matrix_product::args matrix_product::check(expression_tree::data_type const & tree, size_t root)
{
    expression_tree::node const & node = tree[root];
    expression_tree::node const & left = tree[node.binary_operator.lhs];
    expression_tree::node const & right = tree[node.binary_operator.rhs];
    numeric_type dtype = node.dtype;
    matrix_product::args result ;
    if(dtype==INVALID_NUMERIC_TYPE)
      return result;
    result.alpha = value_scalar(1, dtype);
    result.beta = value_scalar(0, dtype);
    if(right.type==COMPOSITE_OPERATOR_TYPE)
    {
        bool is_add = right.binary_operator.op.type==ADD_TYPE;
        bool is_sub = right.binary_operator.op.type==SUB_TYPE;
        //Form X +- Y"
        if(is_add || is_sub)
        {
            expression_tree::node const & rleft = tree[right.binary_operator.lhs];
            expression_tree::node const & rright = tree[right.binary_operator.rhs];

            if(rleft.type==COMPOSITE_OPERATOR_TYPE)
                handle_node(tree, right.binary_operator.lhs, result);
            else if(rleft.type==DENSE_ARRAY_TYPE)
            {
                result.C = &rleft;
                result.beta = value_scalar(1, dtype);
                result.alpha = value_scalar(is_add?1:-1,  dtype);
            }

            if(rright.type==COMPOSITE_OPERATOR_TYPE)
                handle_node(tree, right.binary_operator.rhs, result);
            else if(rright.type==DENSE_ARRAY_TYPE)
            {
                result.C = &rright;
                result.alpha = value_scalar(1, dtype);
                result.beta = value_scalar(is_add?1:-1, dtype);
            }
        }
        else{
            handle_node(tree, node.binary_operator.rhs, result);
        }
    }
    if(result.C == NULL)
        result.C = &left;
    else if(result.C->array.base != left.array.base)
        result.C = NULL;
    return result;
}

}

}

}
