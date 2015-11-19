#ifndef ISAAC_SYMBOLIC_PRESET_H_
#define ISAAC_SYMBOLIC_PRESET_H_

#include "isaac/symbolic/expression.h"

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
        lhs_rhs_element const * A;
        lhs_rhs_element const * B;
        value_scalar beta;
        lhs_rhs_element const * C;
        expression_type type;

        operator bool() const
        {
            return type!=INVALID_EXPRESSION_TYPE && C!=NULL;
        }
    };

private:
    static void handle_node( math_expression::container_type const &tree, size_t rootidx, args & a);

public:
    static args check(math_expression::container_type const &tree, size_t rootidx);
};

}

}

}

#endif
