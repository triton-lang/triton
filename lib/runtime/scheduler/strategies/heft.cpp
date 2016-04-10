#include <set>
#include <algorithm>
#include <iostream>

#include "isaac/runtime/scheduler/strategies/heft.h"

namespace isaac
{
namespace runtime
{
namespace scheduler
{

cost_t heft::wbar(job_t job, procs_t const & procs, compcostfun_t const & compcost)
{
    cost_t res = 0;
    for(proc_t const & a: procs){ res += compcost(job, a); }
    return res/procs.size();
}

cost_t heft::cbar(job_t jobi, job_t jobj, procs_t const & procs, commcostfun_t const & commcost)
{
    cost_t res = 0;
    size_t N = procs.size();
    for(size_t i = 0 ; i < N ; ++i)
        for(size_t j = i + 1 ; j < N ; ++j)
            res += commcost(jobi, jobj, i, j);
    size_t npairs = N*(N-1)/2;
    return res/npairs;
}

cost_t heft::ranku(job_t ni, adjacency_t const & dag, procs_t const & procs, compcostfun_t const & compcost, commcostfun_t const & commcost)
{
    auto rank = [&](job_t n){ return ranku(n, dag, procs, compcost, commcost); };
    auto w = [&](job_t n){ return wbar(n, procs, compcost); };
    auto c = [&](job_t n1, job_t n2){ return cbar(n1, n2, procs, commcost); };
    if(dag.find(ni)==dag.end())
        return w(ni);
    else
    {
        cost_t res = 0;
        for(job_t nj: dag.at(ni))
            res = std::max(res, c(ni, nj) + rank(nj));
        return w(ni) + res;
    }
}

time_t heft::end_time(job_t job, std::vector<order_t> const & events)
{
    for(order_t e: events)
        if(e.job==job)
            return e.end;
    return INFINITY;
}

time_t heft::find_first_gap(typename orders_t::mapped_type const & proc_orders, time_t desired_start, time_t duration)
{
    if(proc_orders.empty())
        return desired_start;
    for(size_t i = 0 ; i < proc_orders.size() ; ++i)
    {
        time_t earliest = std::max(desired_start, (i==0)?0:proc_orders[i-1].end);
        if(proc_orders[i].start - earliest > duration)
            return earliest;
    }
    return std::max(proc_orders.back().end, desired_start);
}

time_t heft::start_time(job_t job, proc_t proc, orders_t const & orders, adjacency_t const & prec, jobson_t const & jobson,
                    compcostfun_t const & compcost, commcostfun_t const & commcost)
{
    time_t duration = compcost(job, proc);
    time_t comm_ready = 0;
    if(prec.find(job)!=prec.end()){
        for(job_t other_job: prec.at(job)){
            proc_t other_proc = jobson.at(other_job);
            comm_ready = std::max(comm_ready, end_time(other_job, orders.at(other_proc)) + commcost(other_job, job, proc, other_proc));
        }
    }
    return find_first_gap(orders.at(proc), comm_ready, duration);
}

void heft::allocate(job_t job, orders_t & orders, jobson_t & jobson, adjacency_t const & prec, compcostfun_t const & compcost, commcostfun_t const & commcost)
{
    auto start = [&](proc_t proc){ return start_time(job, proc, orders, prec, jobson, compcost, commcost);};
    auto finish = [&](proc_t proc) { return start(proc) + compcost(job, proc); };
    proc_t proc = orders.begin()->first;
    for(auto const & pair: orders){
        if(finish(pair.first) < finish(proc))
            proc = pair.first;
    }
    typename orders_t::mapped_type & orders_list = orders[proc];
    orders_list.push_back({job, start(proc), finish(proc)});
    //Update
    std::sort(orders_list.begin(), orders_list.end(), [](order_t const & o1, order_t const & o2){ return o1.start < o2.start;});
    jobson[job] = proc;
}

time_t heft::makespan(orders_t const & orders)
{
    time_t res = 0;
    for(auto const & x: orders)
        res = std::max(res, x.second.back().end);
    return res;
}

void heft::schedule(adjacency_t const & succ, procs_t const & procs, compcostfun_t const & compcost, commcostfun_t const & commcost, orders_t & orders, jobson_t & jobson)
{
    //Get precedence
    adjacency_t prec;
    for(auto const & pair: succ)
        for(job_t next_job: pair.second)
          prec[next_job].push_back(pair.first);

    //Prioritize jobs
    auto rank = [&](job_t const & job) { return ranku(job, succ, procs, compcost, commcost); };
    auto rank_compare = [&](job_t const & t1, job_t const & t2){ return rank(t1) < rank(t2); };
    std::set<job_t, std::function<bool (job_t, job_t)> > jobs(rank_compare);
    for(auto const & pair: succ){
        jobs.insert(pair.first);
        for(job_t next_job: pair.second)
          jobs.insert(next_job);
    }

    //Assign job to processor
    orders = orders_t();
    jobson = jobson_t();
    for(proc_t proc: procs) orders.insert({proc, {}});
    for(auto it = jobs.rbegin(); it != jobs.rend() ; ++it)
        allocate(*it, orders, jobson, prec, compcost, commcost);
}


}
}
}
