from __future__ import division

import time
import os
import sys
import pyatidlas as atd
import numpy as np

class PhysicalLimitsNV:
    def __init__(self, dev):
        self.compute_capability = dev.nv_compute_capability
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

        elif self.compute_capability[0]==5:  #[KR]: copy-pasted from Kepler and adjusted according to http://en.wikipedia.org/wiki/CUDA
            self.threads_per_warp = 32
            self.warps_per_mp = 64
            self.threads_per_mp = 2048
            self.thread_blocks_per_mp = 32
            self.num_32b_reg_per_mp = 65536
            self.reg_alloc_unit_size = 256
            self.reg_alloc_granularity = 'warp'
            self.reg_per_thread = 255
            self.shared_mem_per_mp = 65536
            self.shared_mem_alloc_unit_size = 256
            self.warp_alloc_granularity = 4
            self.max_thread_block_size = 1024


        else:
            raise Exception('Compute capability not supported!')

class PhysicalLimitsAMD:
    def __init__(self, dev):

        infos =\
        {
            #APU:
            'Devastator': {'arch': 'VLIW', 'WFmax_cu': 96, 'LDS_cu': 32768, 'GPR_cu': 8192},
            'Scrapper': {'arch': 'VLIW', 'WFmax_cu': 96, 'LDS_cu': 32768, 'GPR_cu': 8192},
            
            #HD5000
            'Cedar': {'arch': 'VLIW', 'WFmax_cu': 96, 'LDS_cu': 32768, 'GPR_cu': 8192},
            'Redwood': {'arch': 'VLIW', 'WFmax_cu': 62, 'LDS_cu': 32768, 'GPR_cu': 16384},
            'Juniper': {'arch': 'VLIW', 'WFmax_cu': 24.8, 'LDS_cu': 32768, 'GPR_cu': 16384},
            'Cypress': {'arch': 'VLIW', 'WFmax_cu': 27.6, 'LDS_cu': 32768, 'GPR_cu': 16384},
            'Hemlock': {'arch': 'VLIW', 'WFmax_cu': 24.8, 'LDS_cu': 32768, 'GPR_cu': 16384},

            #HD6000
            'Seymour': {'arch': 'VLIW', 'WFmax_cu': 96, 'LDS_cu': 32768, 'GPR_cu': 16384},
            'Caicos': {'arch': 'VLIW', 'WFmax_cu': 96, 'LDS_cu': 32768, 'GPR_cu': 16384},
            'Turks': {'arch': 'VLIW', 'WFmax_cu': 41.3, 'LDS_cu': 32768, 'GPR_cu': 16384},
            'Whistler': {'arch': 'VLIW', 'WFmax_cu': 41.3, 'LDS_cu': 32768, 'GPR_cu': 16384},
            'Barts': {'arch': 'VLIW', 'WFmax_cu': 49.6, 'LDS_cu': 32768, 'GPR_cu': 16384},

            #HD7000
            'Capeverde': {'arch': 'GCN', 'WFmax_cu': 40, 'LDS_cu': 65536, 'GPR_cu': 65536},
            'Pitcairn': {'arch': 'GCN', 'WFmax_cu': 40, 'LDS_cu': 65536, 'GPR_cu': 65536},
            'Bonaire': {'arch': 'GCN', 'WFmax_cu': 40, 'LDS_cu': 65536, 'GPR_cu': 65536},
            'Tahiti': {'arch': 'GCN', 'WFmax_cu': 40, 'LDS_cu': 65536, 'GPR_cu': 65536},

            #Rx 200
            'Oland': {'arch': 'GCN', 'WFmax_cu': 40, 'LDS_cu': 65536, 'GPR_cu': 65536},
            'Tonga': {'arch': 'GCN', 'WFmax_cu': 40, 'LDS_cu': 65536, 'GPR_cu': 65536},
            'Hawaii': {'arch': 'GCN', 'WFmax_cu': 40, 'LDS_cu': 65536, 'GPR_cu': 65536}
        }

        self.WFsize = 64
        self.WFmax_cu = infos[dev.name]['WFmax_cu']
        self.LDS_cu = infos[dev.name]['LDS_cu']
        self.GPR_cu = infos[dev.name]['GPR_cu']
        self.arch =  infos[dev.name]['arch']
        pass

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

    def init_nvidia(self, dev, threads, shared_mem, registers):
        pl = PhysicalLimitsNV(dev)
        limits = []
        allocated_warps =  max(1,_int_ceiling(threads/pl.threads_per_warp))
        max_warps_per_mp = pl.warps_per_mp
        limits.append((min(pl.thread_blocks_per_mp, _int_floor(max_warps_per_mp/allocated_warps)), 'warps'))

        if registers>0:
            if registers > pl.reg_per_thread:
                limits.append((0, 'registers'))
            else:
                allocated_regs = {'warp': allocated_warps,
                                  'block': _int_ceiling(_int_ceiling(allocated_warps, pl.warp_alloc_granularity)*registers*pl.threads_per_warp,allocated_warps)}[pl.reg_alloc_granularity]
                max_reg_per_mp = {'warp': _int_floor(pl.num_32b_reg_per_mp/_int_ceiling(registers*pl.threads_per_warp, pl.reg_alloc_unit_size), pl.warp_alloc_granularity),
                                  'block':pl.num_32b_reg_per_mp}[pl.reg_alloc_granularity]
                limits.append((_int_floor(max_reg_per_mp/allocated_regs), 'registers'))

        if shared_mem>0:
            allocated_shared_mem = _int_ceiling(shared_mem, pl.shared_mem_alloc_unit_size)
            max_shared_mem_per_mp = pl.shared_mem_per_mp
            limits.append((_int_floor(max_shared_mem_per_mp/allocated_shared_mem), 'shared memory'))

        limit, limited_by = min(limits)
        warps_per_mp = limit*allocated_warps
        self.occupancy = 100*warps_per_mp/pl.warps_per_mp

    def init_amd(self, dev, threads, shared_mem, NReg):
        pl = PhysicalLimitsAMD(dev)
        limits = {}

        WFwg = _int_ceiling(threads/pl.WFsize)
        #WFmax without constraint
        if pl.arch=='VLIW':
            limits['wg'] = pl.WFmax_cu if WFwg > pl.WFmax_cu else _int_floor(pl.WFmax_cu,WFwg)
        else:
            limits['wg'] = min(16*WFwg, pl.WFmax_cu)
        #WFmax with LDS constraints
        if shared_mem > 0:
            WGmax = _int_floor(pl.LDS_cu/shared_mem)
            limits['lds'] = WGmax*WFwg
        #WFmax with GPR constraints
        if NReg > 0:
            #Amount of work group per CU
            NRegWG = NReg*pl.WFsize*WFwg
            WGmax = _int_floor(pl.GPR_cu/NRegWG)
            limits['gpr'] = WFwg*WGmax

        self.occupancy = 100.0*min(list(limits.values()))/pl.WFmax_cu


    def __init__(self, dev, threads, shared_mem=0, registers=0):
        vendor = dev.vendor.lower()
        if any(X in vendor for X in ['advanced micro devices', 'amd']):
            self.init_amd(dev, threads, shared_mem, registers)
        elif 'nvidia' in vendor:
            self.init_nvidia(dev, threads, shared_mem, registers)
        elif 'intel' in vendor:
            self.occupancy = 100



def skip(template, symbolic):
    device = symbolic.context.queues[0].device
    symbolic_expressions = atd.symbolic_expression_container(symbolic)
    registers_usage = template.registers_usage(symbolic_expressions)/4
    lmem_usage = template.lmem_usage(symbolic_expressions)
    local_size = template.local_size_0*template.local_size_1
    occupancy_record = OccupancyRecord(device, local_size, lmem_usage, registers_usage)
    if template.check_invalid(symbolic_expressions, device) or occupancy_record.occupancy < 15:
        return True
    return False

def benchmark(template, symbolic):
    queue = symbolic.context.queues[0]
    device = queue.device
    symbolic_expressions = atd.symbolic_expression_container(symbolic)
    registers_usage = template.registers_usage(symbolic_expressions)/4
    lmem_usage = template.lmem_usage(symbolic_expressions)
    local_size = template.local_size_0*template.local_size_1
    occupancy_record = OccupancyRecord(device, local_size, lmem_usage, registers_usage)
    if occupancy_record.occupancy < 15 :
        raise ValueError("Template has too low occupancy")
    else:
        queue.models[template, atd.float32] = atd.model(template, queue)
        x = atd.array(symbolic)
        atd.synchronize(symbolic.context)
        current_time = 0
        timings = []
        while current_time < 1e-1:
            time_before = time.time()
            x = atd.array(symbolic)
            atd.synchronize(symbolic.context)
            timings.append(time.time() - time_before)
            current_time = current_time + timings[-1]
        return np.median(timings)


def sanitize_string(string, keep_chars = ['_']):
    string = string.replace(' ', '_').replace('-', '_').lower()
    string = "".join(c for c in string if c.isalnum() or c in keep_chars).rstrip()
    return string
