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

#ifndef ISAAC_SYMBOLIC_PRESET_H_
#define ISAAC_SYMBOLIC_PRESET_H_

#include "isaac/jit/syntax/expression/expression.h"
#include "isaac/common/numeric_type.h"

namespace isaac
{

namespace symbolic
{

namespace preset
{


class gemm
{

public:
    struct args
    {
        args(): A(NULL), B(NULL), C(NULL), type(INVALID_EXPRESSION_TYPE){ }
        value_scalar alpha;
        expression_tree::node const * A;
        expression_tree::node const * B;
        value_scalar beta;
        expression_tree::node const * C;
        expression_type type;

        operator bool() const
        {
            return type!=INVALID_EXPRESSION_TYPE && A!=NULL && B!=NULL && C!=NULL;
        }
    };
private:
    static void handle_node( expression_tree::data_type const &tree, size_t rootidx, args & a);

public:
    static args check(expression_tree::data_type const &tree, size_t rootidx);
};

}

}

}

#endif
