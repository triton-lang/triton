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
