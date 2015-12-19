#ifndef ISAAC_SYMBOLIC_PRESET_H_
#define ISAAC_SYMBOLIC_PRESET_H_

#include "isaac/symbolic/expression.h"

namespace isaac
{

namespace symbolic
{

namespace preset
{


class matrix_product
{

public:
    struct args
    {
        args(): A(NULL), B(NULL), C(NULL), type(INVALID_EXPRESSION_TYPE){ }
        value_scalar alpha;
        tree_node const * A;
        tree_node const * B;
        value_scalar beta;
        tree_node const * C;
        expression_type type;

        operator bool() const
        {
            return type!=INVALID_EXPRESSION_TYPE && C!=NULL;
        }
    };

private:
    static void handle_node( expression_tree::container_type const &tree, size_t rootidx, args & a);

public:
    static args check(expression_tree::container_type const &tree, size_t rootidx);
};

}

}

}

#endif
