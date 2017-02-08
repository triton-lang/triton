/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "isaac/jit/syntax/expression/preset.h"

namespace isaac
{

namespace symbolic
{

namespace preset
{

void gemm::handle_node(expression_tree::data_type const & tree, size_t root, args & a)
{
    expression_tree::node const & node = tree[root];
    if(node.type != COMPOSITE_OPERATOR_TYPE)
      return;

    expression_tree::node const & left = tree[node.binary_operator.lhs];
    expression_tree::node const & right = tree[node.binary_operator.rhs];

    //Matrix-Matrix product node
    if(node.binary_operator.op.type_family==GEMM)
    {
        if(left.type==DENSE_ARRAY_TYPE) a.A = &left;
        if(right.type==DENSE_ARRAY_TYPE) a.B = &right;
        switch(node.binary_operator.op.type)
        {
          case GEMM_NN_TYPE: a.type = GEMM_NN; break;
          case GEMM_NT_TYPE: a.type = GEMM_NT; break;
          case GEMM_TN_TYPE: a.type = GEMM_TN; break;
          case GEMM_TT_TYPE: a.type = GEMM_TT; break;
          default: break;
        }
    }
    numeric_type abtype = (node.dtype == HALF_TYPE) ? FLOAT_TYPE : node.dtype;
    //Scalar multiplication node
    if(node.binary_operator.op.type==MULT_TYPE)
    {
        //alpha*PROD
        if(left.type==VALUE_SCALAR_TYPE  && right.type==COMPOSITE_OPERATOR_TYPE
           && right.binary_operator.op.type_family==GEMM)
        {
            a.alpha = cast(value_scalar(left.scalar, left.dtype), abtype);
            handle_node(tree, node.binary_operator.rhs, a);
        }

        //beta*C
        if(left.type==VALUE_SCALAR_TYPE  && right.type==DENSE_ARRAY_TYPE)
        {
            a.beta = cast(value_scalar(left.scalar, left.dtype), abtype);
            a.C = &right;
        }
    }
}

gemm::args gemm::check(expression_tree::data_type const & tree, size_t root)
{
    expression_tree::node const & node = tree[root];
    expression_tree::node const & left = tree[node.binary_operator.lhs];
    expression_tree::node const & right = tree[node.binary_operator.rhs];
    numeric_type dtype = node.dtype;
    numeric_type abtype = (dtype == HALF_TYPE) ? FLOAT_TYPE : dtype;
    gemm::args result ;
    if(dtype==INVALID_NUMERIC_TYPE)
      return result;
    result.alpha = value_scalar(1, abtype);
    result.beta = value_scalar(0, abtype);
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
                result.beta = value_scalar(1, abtype);
                result.alpha = value_scalar(is_add?1:-1, abtype);
            }

            if(rright.type==COMPOSITE_OPERATOR_TYPE)
                handle_node(tree, right.binary_operator.rhs, result);
            else if(rright.type==DENSE_ARRAY_TYPE)
            {
                result.C = &rright;
                result.alpha = value_scalar(1, abtype);
                result.beta = value_scalar(is_add?1:-1, abtype);
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
