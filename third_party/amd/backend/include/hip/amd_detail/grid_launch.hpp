#pragma once

#include "grid_launch.h"
#include "hc.hpp"

class grid_launch_parm_cxx : public grid_launch_parm
{
public:
  grid_launch_parm_cxx() = default;

  // customized serialization: don't need av and cf in kernel
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize& s) const {
    s.Append(sizeof(int), &grid_dim.x);
    s.Append(sizeof(int), &grid_dim.y);
    s.Append(sizeof(int), &grid_dim.z);
    s.Append(sizeof(int), &group_dim.x);
    s.Append(sizeof(int), &group_dim.y);
    s.Append(sizeof(int), &group_dim.z);
  }

  __attribute__((annotate("user_deserialize")))
  grid_launch_parm_cxx(int grid_dim_x,  int grid_dim_y,  int grid_dim_z,
                   int group_dim_x, int group_dim_y, int group_dim_z) {
    grid_dim.x  = grid_dim_x;
    grid_dim.y  = grid_dim_y;
    grid_dim.z  = grid_dim_z;
    group_dim.x = group_dim_x;
    group_dim.y = group_dim_y;
    group_dim.z = group_dim_z;
  }
};


extern inline void grid_launch_init(grid_launch_parm *lp) {
  lp->grid_dim.x = lp->grid_dim.y = lp->grid_dim.z = 1;

  lp->group_dim.x = lp->group_dim.y = lp->group_dim.z = 1;

  lp->dynamic_group_mem_bytes = 0;

  lp->barrier_bit = barrier_bit_queue_default;
  lp->launch_fence = -1;

  // TODO - set to NULL?
  static hc::accelerator_view av = hc::accelerator().get_default_view();
  lp->av = &av;
  lp->cf = NULL;
}

