#ifndef _ATIDLAS_SCHEDULER_EXECUTE_H
#define _ATIDLAS_SCHEDULER_EXECUTE_H

#include <CL/cl.hpp>
#include "atidlas/model/model.h"
#include "atidlas/symbolic/expression.h"

namespace atidlas
{

/** @brief Executes a array_expression on the given queue for the given models map*/
void execute(array_expression &, model_map_t &);

}

#endif
