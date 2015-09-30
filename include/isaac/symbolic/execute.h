#ifndef _ISAAC_SCHEDULER_EXECUTE_H
#define _ISAAC_SCHEDULER_EXECUTE_H

#include "isaac/profiles/profiles.h"
#include "isaac/symbolic/expression.h"

namespace isaac
{

/** @brief Executes a math_expression on the given queue for the given models map*/
void execute(execution_handler const & , profiles::map_type &);

/** @brief Executes a math_expression on the default models map*/
void execute(execution_handler const &);

}

#endif
