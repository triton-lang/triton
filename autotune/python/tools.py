from __future__ import division
import pyopencl
import time
from pyviennacl.atidlas import StatementsTuple

class PhysicalLimits:
    def __init__(self, dev):
      self.compute_capability = pyopencl.characterize.nv_compute_capability(dev)
      if self.compute_capability[0]==1:
        if self.compute_capability[1]<=1:
          self.warps_per_mp = 24
          self.threads_per_mp = 768
          self.num_32b_reg_per_mp = 8192
          self.reg_alloc_unit_size = 256
        else:
          self.warps_per_mp = 32
          self.threads_per_mp = 1024
          self.num_32b_reg_per_mp = 16384
          self.reg_alloc_unit_size = 512
        self.threads_per_warp = 32
        self.thread_blocks_per_mp = 8
        self.reg_alloc_granularity = 'block'
        self.reg_per_thread = 124
        self.shared_mem_per_mp = 16384
        self.shared_mem_alloc_unit_size = 512
        self.warp_alloc_granularity = 2
        self.max_thread_block_size = 512
        
      elif self.compute_capability[0]==2:
        self.threads_per_warp = 32
        self.warps_per_mp = 48
        self.threads_per_mp = 1536
        self.thread_blocks_per_mp = 8
        self.num_32b_reg_per_mp = 32768
        self.reg_alloc_unit_size = 64
        self.reg_alloc_granularity = 'warp'
        self.reg_per_thread = 63
        self.shared_mem_per_mp = 49152
        self.shared_mem_alloc_unit_size = 128
        self.warp_alloc_granularity = 2
        self.max_thread_block_size = 1024
      
      elif self.compute_capability[0]==3:
        self.threads_per_warp = 32
        self.warps_per_mp = 64
        self.threads_per_mp = 2048
        self.thread_blocks_per_mp = 16
        self.num_32b_reg_per_mp = 65536
        self.reg_alloc_unit_size = 256
        self.reg_alloc_granularity = 'warp'
        if(self.compute_capability[1]==5):
          self.reg_per_thread = 255
        else:
          self.reg_per_thread = 63
        self.shared_mem_per_mp = 49152
        self.shared_mem_alloc_unit_size = 256
        self.warp_alloc_granularity = 4
        self.max_thread_block_size = 1024
        
      else:
        raise Exception('Compute capability not supported!')
        
def _int_floor(value, multiple_of=1):
  """Round C{value} down to be a C{multiple_of} something."""
  # Mimicks the Excel "floor" function (for code stolen from occupancy calculator)

  from math import floor
  return int(floor(value/multiple_of))*multiple_of
  
def _int_ceiling(value, multiple_of=1):
  """Round C{value} up to be a C{multiple_of} something."""
  # Mimicks the Excel "floor" function (for code stolen from occupancy calculator)

  from math import ceil
  return int(ceil(value/multiple_of))*multiple_of
      
class OccupancyRecord:
      
    def __init__(self, dev, threads, shared_mem=0, registers=0):
      physical_limits = PhysicalLimits(dev)
      limits = [];
      allocated_warps =  max(1,_int_ceiling(threads/physical_limits.threads_per_warp))
      max_warps_per_mp = physical_limits.warps_per_mp;
      limits.append((min(physical_limits.thread_blocks_per_mp, _int_floor(max_warps_per_mp/allocated_warps)), 'warps'))
      
      if registers>0:
        if registers > physical_limits.reg_per_thread:
          limits.append((0, 'registers'))
        else:
          allocated_regs = {'warp': allocated_warps,
                            'block': _int_ceiling(_int_ceiling(allocated_warps, physical_limits.warp_alloc_granularity)*registers*physical_limits.threads_per_warp,allocated_warps)}[physical_limits.reg_alloc_granularity]
          max_reg_per_mp = {'warp': _int_floor(physical_limits.num_32b_reg_per_mp/_int_ceiling(registers*physical_limits.threads_per_warp, physical_limits.reg_alloc_unit_size), physical_limits.warp_alloc_granularity),
                            'block':physical_limits.num_32b_reg_per_mp}[physical_limits.reg_alloc_granularity]
          limits.append((_int_floor(max_reg_per_mp/allocated_regs), 'registers'))
      
      if shared_mem>0:
        allocated_shared_mem = _int_ceiling(shared_mem, physical_limits.shared_mem_alloc_unit_size)
        max_shared_mem_per_mp = physical_limits.shared_mem_per_mp
        limits.append((_int_floor(max_shared_mem_per_mp/allocated_shared_mem), 'shared memory'))
      
      self.limit, self.limited_by = min(limits)
      self.warps_per_mp = self.limit*allocated_warps
      self.occupancy = 100*self.warps_per_mp/physical_limits.warps_per_mp
        

def skip(template, statement, device):
      statements = StatementsTuple(statement)
      registers_usage = template.registers_usage(statements)/4
      lmem_usage = template.lmem_usage(statements)
      local_size = template.parameters.local_size_0*template.parameters.local_size_1
      occupancy_record = OccupancyRecord(device, local_size, lmem_usage, registers_usage)
      if template.check(statement) or occupancy_record.occupancy < 15:
        return True
      return False
      
def benchmark(template, statement, device):
      statements = StatementsTuple(statement)
      registers_usage = template.registers_usage(statements)/4
      lmem_usage = template.lmem_usage(statements)
      local_size = template.parameters.local_size_0*template.parameters.local_size_1
      occupancy_record = OccupancyRecord(device, local_size, lmem_usage, registers_usage)
      if occupancy_record.occupancy < 15 :
        raise ValueError("Template has too low occupancy")
      else:
        #~ try:
        template.execute(statement, True)
        statement.result.context.finish_all_queues()
        N = 0
        current_time = 0
        while current_time < 1e-2:
          time_before = time.time()
          template.execute(statement,False)
          statement.result.context.finish_all_queues()
          current_time += time.time() - time_before
          N+=1
        return current_time/N
        #~ except:
          #~ raise ValueError("Invalid template")
