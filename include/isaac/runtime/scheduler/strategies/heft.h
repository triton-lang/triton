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

#ifndef _ISAAC_SYMBOLIC_SCHEDULER_STRATEGIES_HEFT_H
#define _ISAAC_SYMBOLIC_SCHEDULER_STRATEGIES_HEFT_H

#include "strategy.h"

namespace isaac
{
namespace runtime
{
namespace scheduler
{

class heft
{
private:
  static cost_t wbar(job_t job, procs_t const & procs, compcostfun_t const & compcost);
  static cost_t cbar(job_t jobi, job_t jobj, procs_t const & procs, commcostfun_t const & commcost);
  static cost_t ranku(job_t ni, adjacency_t const & dag, procs_t const & procs,
                      compcostfun_t const & compcost, commcostfun_t const & commcost);
  static time_t end_time(job_t job, std::vector<order_t> const & events);
  static time_t find_first_gap(typename orders_t::mapped_type const & proc_orders, time_t desired_start, time_t duration);
  static time_t start_time(job_t job, proc_t proc, orders_t const & orders, adjacency_t const & prec, jobson_t const & jobson,
                    compcostfun_t const & compcost, commcostfun_t const & commcost);
  static void allocate(job_t job, orders_t & orders, jobson_t & jobson, adjacency_t const & prec,
                       compcostfun_t const & compcost, commcostfun_t const & commcost);

public:
  static time_t makespan(orders_t const & orders);
  static void schedule(adjacency_t const & succ, procs_t const & procs,
                       compcostfun_t const & compcost, commcostfun_t const & commcost,
                       orders_t & orders, jobson_t & jobson);
};


}
}
}

#endif
