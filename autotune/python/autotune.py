from __future__ import division

import argparse
import itertools
import os

from external.configobj import ConfigObj

import pyopencl as cl
import pyviennacl as vcl
from pyviennacl import backend
from pyviennacl import opencl
from pyviennacl import atidlas

import utils
import vclio
import optimize
import sys

DATATYPES = { 'single' : vcl.float32,
              'double' : vcl.float64 }

TYPES = { 'vector-axpy': {'template':vcl.atidlas.VectorAxpyTemplate,
                          'parameter-names':['simd-width', 'local-size-0', 'num-groups-0', 'fetch'],
                          'perf-index':lambda x: 3*x[0]*x[1][0]/x[2]*1e-9,
                          'perf-measure':'GB/s'},
                          
          'matrix-axpy': {'template':vcl.atidlas.MatrixAxpyTemplate,
                          'parameter-names':['simd-width', 'local-size-0', 'local-size-1', 'num-groups-0', 'num-groups-1', 'fetch'],
                          'perf-index':lambda x: 3*x[0]*x[1][0]*x[1][1]/x[2]*1e-9,
                          'perf-measure':'GB/s'},
                          
          'reduction': {'template':vcl.atidlas.ReductionTemplate,
                        'parameter-names':['simd-width', 'local-size-0', 'num-groups-0', 'fetch'],
                        'perf-index':lambda x: 2*x[0]*x[1][0]*x[1][1]/x[2]*1e-9,
                        'perf-measure':'GB/s'},
          
          'row-wise-reduction': {'template':vcl.atidlas.RowWiseReductionTemplate,
                                'parameter-names':['simd-width', 'local-size-0', 'local-size-1', 'num-groups-0', 'fetch'],
                                'perf-index':lambda x: x[0]*x[1][0]*x[1][1]/x[2]*1e-9,
                                'perf-measure':'GB/s'},
          
          'matrix-product': {'template':vcl.atidlas.MatrixProductTemplate,
                            'parameter-names':['simd-width', 'local-size-0', 'kL', 'local-size-1', 'mS', 'kS', 'nS', 'A-fetch-policy', 'B-fetch-policy', 'local-fetch-size-0', 'local-fetch-size-1'],
                            'perf-index': lambda x: 2*x[1][0]*x[1][1]*x[1][2]/x[2]*1e-9,
                            'perf-measure': 'GFLOP/s'} }
    
def parameter_space(operation):
  simd = [1, 2, 4, 8]
  pow2_1D = [2**k for k in range(12)]
  pow2_2D = [2**k for k in range(10)]
  pow2_2D_unrolled = [2**k for k in range(6)]
  FetchingPolicy = vcl.atidlas.FetchingPolicy
  fetch = [FetchingPolicy.FETCH_FROM_LOCAL, FetchingPolicy.FETCH_FROM_GLOBAL_STRIDED, FetchingPolicy.FETCH_FROM_GLOBAL_CONTIGUOUS]
  if operation == 'vector-axpy': return [simd, pow2_1D, pow2_1D, fetch]
  if operation == 'reduction': return [simd, pow2_1D, pow2_1D, fetch]
  if operation == 'matrix-axpy': return [simd, pow2_2D, pow2_2D, pow2_2D, pow2_2D, fetch]
  if operation == 'row-wise-reduction': return [simd, pow2_2D, pow2_2D, pow2_1D, fetch]
  if operation == 'matrix-product': return [simd, pow2_2D, pow2_2D, pow2_2D, pow2_2D_unrolled,  pow2_2D_unrolled,  pow2_2D_unrolled, fetch, fetch, [0] + pow2_2D, [0] + pow2_2D]
  
def do_tuning(config_fname, spec_fname, viennacl_root):    

  config = ConfigObj(config_fname, configspec=spec_fname)
  map_to_list = lambda T: list(map(T[0], T[1] if isinstance(T[1], list) else [T[1]])) 
 
  for operation in ['vector-axpy', 'matrix-axpy', 'row-wise-reduction', 'matrix-product']:
    
    tmp_folder = config['tmp-folder'] if 'tmp-folder' in config else ""
    
    if operation in config:
      p = config[operation]        
      confdevices = p['devices']
      devices = utils.DEVICES_PRESETS[confdevices] if confdevices in utils.DEVICES_PRESETS else [utils.all_devices[int(i)] for i in confdevices]
      precisions =  map_to_list((str, p['precision']))
      datatypes = [DATATYPES[k] for k in precisions]
      s = map_to_list((int, p['size']))
      
      for datatype, device in itertools.product(datatypes, devices):
        ctx = cl.Context([device])
        ctx = vcl.backend.Context(ctx)
        device = ctx.current_device

        if datatype is vcl.float64 and not device.double_fp_config:
          sys.stderr.write('Warning : The device ' + device.name + ' does not support double precision! Skipping ...')
          continue

        pairs = []
        
        def execute(node, other_params):
          print('-----')
          print(' '.join(map(str, ("Now tuning:", datatype.__name__, '-', operation, '-'.join(other_params), '[' + device.name, '(' + device.platform.name + ')]'))))
          tmp_file = os.path.join(tmp_folder, utils.sanitize_string(device.name) + "-" + datatype.__name__ + "-" + operation + '-'.join(other_params) + ".dat")
          if tmp_folder:
            print('Saving history to ' + tmp_file)
            fname = tmp_file
          else:
            fname = os.devnull
          with open(fname, "w+") as archive:
            with vcl.Statement(node) as statement:
              result = optimize.genetic(statement, ctx, TYPES[operation]['template'], lambda p: TYPES[operation]['template'](p, *other_params),
                                    TYPES[operation]['parameter-names'], parameter_space(operation), lambda t: TYPES[operation]['perf-index']([datatype().itemsize, s, t]), TYPES[operation]['perf-measure'], archive)
            if result and viennacl_root:
              vclio.generate_viennacl_headers(viennacl_root, device, datatype, operation, other_params, result[1])
        
        if operation=='vector-axpy':
          x = vcl.Vector(s[0], context=ctx, dtype=datatype)
          y = vcl.Vector(s[0], context=ctx, dtype=datatype)
          execute(vcl.ElementProd(vcl.exp(x + y),vcl.cos(x + y)), ())
        
        if operation=='matrix-axpy':
          A = vcl.Matrix(s, context=ctx, dtype=datatype)
          B = vcl.Matrix(s, context=ctx, dtype=datatype)
          execute(A+B, ())
        
        if operation=='row-wise-reduction':
          layouts = map_to_list((str,p['layout']))
          if 'all' in layouts:
            layouts = ['N', 'T']
          for A_trans in layouts:
            A = vcl.Matrix(s if A_trans=='N' else s[::-1], context=ctx, dtype=datatype, layout=vcl.COL_MAJOR)
            x = vcl.Vector(s[1] if A_trans=='N' else s[0], context=ctx, dtype=datatype)
            LHS = A if A_trans=='N' else A.T
            execute(LHS*x, ())
          
        if operation=='matrix-product':
          layouts = map_to_list((str,p['layout']))
          if 'all' in layouts:
            layouts = ['NN', 'NT', 'TN', 'TT']
          for layout in layouts:
            A_trans = layout[0]
            B_trans = layout[1]
            
            A = vcl.Matrix((s[0], s[1]) if A_trans=='N' else (s[1],s[0]), context=ctx, dtype=datatype, layout=vcl.COL_MAJOR);
            B = vcl.Matrix((s[1], s[2]) if B_trans=='N' else (s[2],s[1]), context=ctx, dtype=datatype, layout=vcl.COL_MAJOR);
            LHS = A if A_trans=='N' else A.T
            RHS = B if B_trans=='N' else B.T
            alpha = vcl.HostScalar(1.0,  context=ctx, dtype = datatype)
            beta = vcl.HostScalar(1.0, context=ctx, dtype = datatype)
            C = vcl.Matrix((s[0], s[2]), context=ctx, dtype = datatype, layout=vcl.COL_MAJOR)
            execute(vcl.Assign(C,LHS*RHS*alpha + C*beta),(A_trans, B_trans))
            

if __name__ == "__main__":
  parser = argparse.ArgumentParser();
  
  subparsers = parser.add_subparsers(dest='action')

  print_devices_parser = subparsers.add_parser('list-devices', help='list the devices available')
  
  tune_parser = subparsers.add_parser('tune', help='tune using a specific configuration file')
  tune_parser.add_argument("--config", default="config.ini", required=False, type=str)
  tune_parser.add_argument("--viennacl-root", default='', required=False, type=str)
  args = parser.parse_args()
  
  if(args.action=='list-devices'):
      print("----------------")
      print("Devices available:")
      print("----------------")
      devices = [d for platform in cl.get_platforms() for d in platform.get_devices()]
      for (i, d) in enumerate(devices):
          print('Device', i, ':', utils.DEVICE_TYPE_PREFIX[d.type].upper() + ':', d.name, 'on', d.platform.name)
      print("----------------")
  else:
      print("------")
      print("Auto-tuning")
      print("------")
      do_tuning(args.config, 'config_spec.ini', args.viennacl_root)
