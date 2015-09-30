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
        args(): alpha(NULL), A(NULL), B(NULL), beta(NULL), C(NULL), type(INVALID_EXPRESSION_TYPE){ }
        lhs_rhs_element const * alpha;
        lhs_rhs_element const * A;
        lhs_rhs_element const * B;
        lhs_rhs_element const * beta;
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
