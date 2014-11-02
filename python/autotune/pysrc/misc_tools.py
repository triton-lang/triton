from __future__ import division

import pyopencl
import time
import os
import sys

import pyopencl as cl
import pyviennacl as vcl
import numpy as np

class PhysicalLimitsNV:
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
            'Bart': {'arch': 'VLIW', 'WFmax_cu': 49.6, 'LDS_cu': 32768, 'GPR_cu': 16384},

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
        if 'advanced micro devices' in dev.vendor.lower():
            self.init_amd(dev, threads, shared_mem, registers)
        elif 'nvidia' in dev.vendor.lower():
            self.init_nvidia(dev, threads, shared_mem, registers)



def skip(template, statement, device):
    statements = vcl.pycore.StatementsTuple(statement)
    registers_usage = template.registers_usage(statements)/4
    lmem_usage = template.lmem_usage(statements)
    local_size = template.parameters.local_size_0*template.parameters.local_size_1
    occupancy_record = OccupancyRecord(device, local_size, lmem_usage, registers_usage)
    if template.check(statement) or occupancy_record.occupancy < 15:
        return True
    return False

def benchmark(template, statement, device):
    statements = vcl.pycore.StatementsTuple(statement)
    registers_usage = template.registers_usage(statements)/4
    lmem_usage = template.lmem_usage(statements)
    local_size = template.parameters.local_size_0*template.parameters.local_size_1
    occupancy_record = OccupancyRecord(device, local_size, lmem_usage, registers_usage)
    if occupancy_record.occupancy < 15 :
        raise ValueError("Template has too low occupancy")
    else:
        template.execute(statement, True)
        statement.result.context.finish_all_queues()
        current_time = 0
        timings = []
        while current_time < 1e-1:
            time_before = time.time()
            template.execute(statement,False)
            statement.result.context.finish_all_queues()
            timings.append(time.time() - time_before)
            current_time = current_time + timings[-1]
        return np.median(timings)


def sanitize_string(string, keep_chars = ['_']):
    string = string.replace(' ', '_').replace('-', '_').lower()
    string = "".join(c for c in string if c.isalnum() or c in keep_chars).rstrip()
    return string

def update_viennacl_headers(viennacl_root, device, datatype, operation, additional_parameters, parameters):
    
    def append_include(data, path):
        include_name = '#include "' + path +'"\n'
        already_included = data.find(include_name)
        if already_included == -1:
            insert_index = data.index('\n', data.index('#define')) + 1
            return data[:insert_index] + '\n' + include_name + data[insert_index:]
        return data


    builtin_database_dir = os.path.join(viennacl_root, "device_specific", "builtin_database")
    if not os.path.isdir(builtin_database_dir):
        raise EnvironmentError('ViennaCL root path is incorrect. Cannot access ' + builtin_database_dir + '!\n'
                                'Your version of ViennaCL may be too old and/or corrupted.')

    function_name_dict = { vcl.float32: 'add_4B',
                           vcl.float64: 'add_8B' }

    additional_parameters_dict = {'N':  "char_to_type<'N'>",
                                  'T':  "char_to_type<'T'>"}

    #Create the device-specific headers
    cpp_device_name = sanitize_string(device.name)
    function_name = function_name_dict[datatype]
    operation = operation.replace('-','_')

    cpp_class_name = operation + '_template'
    header_name = cpp_device_name + ".hpp"
    function_declaration = 'inline void ' + function_name + '(' + ', '.join(['database_type<' + cpp_class_name + '::parameters_type> & db'] + \
                                                                          [additional_parameters_dict[x] for x in additional_parameters]) + ')'


    device_type_prefix = {  
                            cl.device_type.GPU: 'gpu',
                            cl.device_type.CPU: 'cpu',
                            cl.device_type.ACCELERATOR: 'accelerator'
                         }[device.type]
    vendor_prefix = {   
                        vcl.opencl.VendorId.beignet_id: 'beignet',
                        vcl.opencl.VendorId.nvidia_id: 'nvidia',
                        vcl.opencl.VendorId.amd_id: 'amd',
                        vcl.opencl.VendorId.intel_id: 'intel'
                    }[device.vendor_id]
    architecture_family = vcl.opencl.architecture_family(device.vendor_id, device.name)

    header_hierarchy = ["devices", device_type_prefix, vendor_prefix, architecture_family]
    header_directory = os.path.join(builtin_database_dir, *header_hierarchy)
    header_path = os.path.join(header_directory, header_name)

    if not os.path.exists(header_directory):
        os.makedirs(header_directory)

    if os.path.exists(header_path):
        with open (header_path, "r") as myfile:
            data=myfile.read()
    else:
        data = ''

    if not data:
        ifndef_suffix = ('_'.join(header_hierarchy + [cpp_device_name]) + '_hpp_').upper()
        data =  ('#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_' + ifndef_suffix + '\n'
            '#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_' + ifndef_suffix + '\n'
            '\n'
            '#include "viennacl/device_specific/forwards.h"\n'
            '#include "viennacl/device_specific/builtin_database/common.hpp"\n'
            '\n'
            'namespace viennacl{\n'
            'namespace device_specific{\n'
            'namespace builtin_database{\n'
            'namespace devices{\n'
            'namespace '  + device_type_prefix + '{\n'
            'namespace '  + vendor_prefix + '{\n'
            'namespace '  + architecture_family + '{\n'
            'namespace '  + cpp_device_name + '{\n'
            '\n'
            '}\n'
            '}\n'
            '}\n'
            '}\n'
            '}\n'
            '}\n'
            '}\n'
            '}\n'
            '#endif\n'
            '')

    data = append_include(data, 'viennacl/device_specific/templates/' + cpp_class_name + '.hpp')
    device_type = { 
                    cl.device_type.GPU: 'CL_DEVICE_TYPE_GPU',
                    cl.device_type.CPU: 'CL_DEVICE_TYPE_CPU',
                    cl.device_type.ACCELERATOR: 'CL_DEVICE_TYPE_ACCELERATOR'
                  }[device.type]
    add_to_database_arguments = [vendor_prefix + '_id', device_type, 'ocl::'+architecture_family,
                  '"' + device.name + '"',  cpp_class_name + '::parameters_type(' + ','.join(map(str,parameters)) + ')']
    core = '  db.' + function_name + '(' + ', '.join(add_to_database_arguments) + ');'

    already_declared = data.find(function_declaration)
    if already_declared==-1:
        substr = 'namespace '  + cpp_device_name + '{\n'
        insert_index = data.index(substr) + len(substr)
        data = data[:insert_index] + '\n' + function_declaration + '\n{\n' + core + '\n}\n' + data[insert_index:]
    else:
        i1 = data.find('{', already_declared)
        if data[i1-1]=='\n':
            i1 = i1 - 1
        i2 = data.find('}', already_declared) + 1
        data = data[:i1]  + '\n{\n' + core + '\n}' + data[i2:]

    #Write the header file
    with open(header_path, "w+") as myfile:
        myfile.write(data)

    #Updates the global ViennaCL headers
    with open(os.path.join(builtin_database_dir, operation + '.hpp'), 'r+') as operation_header:
        data = operation_header.read()
        data = append_include(data, os.path.relpath(header_path, os.path.join(viennacl_root, os.pardir)))

        scope_name = '_'.join(('init', operation) + additional_parameters)
        scope = data.index(scope_name)
        function_call = '  ' + '::'.join(header_hierarchy + [cpp_device_name, function_name]) + '(' + ', '.join(['result'] + [additional_parameters_dict[k] + '()' for k in additional_parameters]) + ')'
        if function_call not in data:
            insert_index = data.rindex('\n', 0, data.index('return result', scope))
            data = data[:insert_index] + function_call + ';\n' + data[insert_index:]

        operation_header.seek(0)
        operation_header.truncate()
        operation_header.write(data)
