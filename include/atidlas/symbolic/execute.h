#ifndef _ATIDLAS_SCHEDULER_EXECUTE_H
#define _ATIDLAS_SCHEDULER_EXECUTE_H

#include "atidlas/cl/cl.hpp"
#include "atidlas/model/model.h"
#include "atidlas/symbolic/expression.h"

namespace atidlas
{

/** @brief Executes a symbolic_expression on the given queue for the given models map*/
void execute(symbolic_expression &, model_map_t &);

}

#endif
