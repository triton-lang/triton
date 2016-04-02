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

#ifndef _ISAAC_SYMBOLIC_SCHEDULER_STRATEGIES_STRATEGY_H
#define _ISAAC_SYMBOLIC_SCHEDULER_STRATEGIES_STRATEGY_H

#include <vector>
#include <map>
#include <functional>

namespace isaac
{
namespace symbolic
{
namespace scheduler
{

typedef float cost_t;
typedef float time_t;
typedef size_t job_t;
typedef size_t proc_t;
typedef std::vector<size_t> procs_t;

typedef std::map<job_t,std::vector<job_t> > adjacency_t;

typedef std::function<cost_t (job_t, proc_t)> compcostfun_t;
typedef std::function<cost_t (job_t, job_t, proc_t, proc_t)> commcostfun_t;

struct order_t
{
    job_t job;
    time_t start;
    time_t end;
};
typedef std::map<proc_t, std::vector<order_t> > orders_t;
typedef std::map<job_t, proc_t> jobson_t;

}
}
}

#endif
