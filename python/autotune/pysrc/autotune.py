from __future__ import division

import argparse, itertools, os, sys, json
import misc_tools, optimize

import pyopencl as cl
import pyviennacl as vcl
import pyatidlas as atd
import numpy as np

from configobj import ConfigObj
from numpy import random
from dataset import generate_dataset
from model import train_model


TYPES = { 'vector-axpy': {'template':atd.VectorAxpyTemplate,
                          'perf-index':lambda x: 3*x[0]*x[1][0]/x[2]*1e-9,
                          'perf-measure':'GB/s'},

          'matrix-axpy': {'template':atd.MatrixAxpyTemplate,
                          'perf-index':lambda x: 3*x[0]*x[1][0]*x[1][1]/x[2]*1e-9,
                          'perf-measure':'GB/s'},

          'reduction': {'template':atd.ReductionTemplate,
                        'perf-index':lambda x: 2*x[0]*x[1][0]/x[2]*1e-9,
                        'perf-measure':'GB/s'},

          'row-wise-reduction': {'template':atd.RowWiseReductionTemplate,
                                'perf-index':lambda x: x[0]*x[1][0]*x[1][1]/x[2]*1e-9,
                                'perf-measure':'GB/s'},

          'matrix-product': {'template': atd.MatrixProductTemplate,
                            'perf-index': lambda x: 2*x[1][0]*x[1][1]*x[1][2]/x[2]*1e-9,
                            'perf-measure': 'GFLOP/s'} }


def do_tuning(args, devices):
    json_out = {}
    device = devices[args.device]

    def map_to_list(T, x):
        return list(map(T, x if isinstance(x, list) else [x]))

    if(args.method=='unique'):
        default_tuning_sizes = {'vector-axpy': tuple(args.sizes[:1]), 'reduction': tuple(args.sizes[:1]),
                                'matrix-axpy' : tuple(args.sizes[1:3]), 'row-wise-reduction' : tuple(args.sizes[1:3]),
                                'matrix-product': tuple(args.sizes[3:])}
    for operation in ['vector-axpy', 'matrix-axpy', 'reduction', 'row-wise-reduction', 'matrix-product']:

          #Iterate through the datatypes
          for datatype in [vcl.float32, vcl.float64]:

              ctx = cl.Context([device])
              ctx = vcl.backend.Context(ctx)

              #Check data-type
              if datatype is vcl.float64 and not device.double_fp_config:
                  sys.stderr.write('Warning : The device ' + device.name + ' does not support double precision! Skipping ...')
                  continue

              #Helper for execution
              def execute(device, node, other_params, sizes, fname = os.devnull, parameters = None):
                  with vcl.Statement(node) as statement:
                      if parameters:
                          TemplateType = TYPES[operation]['template']
                          return misc_tools.benchmark(TemplateType(TemplateType.Parameters(*parameters),*other_params), statement, device)
                      print('-----')
                      print(' '.join(map(str, ("Now tuning:", datatype.__name__, '-', operation, '-'.join(other_params), '[' + device.name, '(' + device.platform.name + ')] for sizes', sizes))))
                      with open(fname, "w+") as archive:
                          return optimize.genetic(statement, device, TYPES[operation]['template'], lambda p: TYPES[operation]['template'](p, *other_params),
                                                  lambda t: TYPES[operation]['perf-index']([datatype().itemsize, sizes, t]), TYPES[operation]['perf-measure'], archive)

              #Helper for tuning
              def tune(execution_handler, nTuning, nDataPoints, draw, additional_parameters):
                  #Update JSON
                  full_operation = operation + ''.join(additional_parameters)
                  if full_operation not in json_out:
                      json_out[full_operation] = {}
                  json_out[full_operation][datatype.__name__] = {}
                  D = json_out[full_operation][datatype.__name__]

                  if args.method == 'unique':
                      profiles = [execution_handler(map(int,default_tuning_sizes[operation]))]
                      if args.viennacl_src_path:
                          misc_tools.update_viennacl_headers(args.viennacl_src_path,device,datatype,operation,additional_parameters,profiles[0])
                  else:
                      def compute_perf(x, t):
                          return TYPES[operation]['perf-index']([datatype().itemsize, x, t])
                      X, Y, profiles = generate_dataset(TYPES[operation]['template'], execution_handler, nTuning, nDataPoints, draw)
                      clf = train_model(X, Y, profiles, TYPES[operation]['perf-measure'])
                      D['predictor'] = [{'children_left': e.tree_.children_left.tolist(),
                                     'children_right': e.tree_.children_right.tolist(),
                                     'threshold': e.tree_.threshold.astype('float32').tolist(),
                                     'feature': e.tree_.feature.astype('float32').tolist(),
                                     'value': e.tree_.value[:,:,0].astype('float32').tolist()} for e in clf.estimators_]
                  D['profiles'] = [ prof.astype('int').tolist() for prof in profiles]


              #Vector AXPY
              if operation=='vector-axpy':
                  def execution_handler(sizes, fname=os.devnull, parameters=None):
                      x = vcl.Vector(sizes[0], context=ctx, dtype=datatype)
                      y = vcl.Vector(sizes[0], context=ctx, dtype=datatype)
                      z = vcl.Vector(sizes[0], context=ctx, dtype=datatype)
                      return execute(device, vcl.Assign(z, vcl.ElementProd(vcl.exp(x + y),vcl.cos(x + y))), (), sizes, fname, parameters)
                  tune(execution_handler, 30, 1000, lambda : 64*np.random.randint(low=10, high=100000, size=1), ())
              #Reduction
              if operation=='reduction':
                  def execution_handler(sizes, fname=os.devnull, parameters=None):
                      x = vcl.Vector(sizes[0], context=ctx, dtype=datatype)
                      y = vcl.Vector(sizes[0], context=ctx, dtype=datatype)
                      s = vcl.Scalar(0, context=ctx, dtype=datatype)
                      return execute(device, vcl.Assign(s, vcl.Dot(x,y)), (), sizes, fname, parameters)
                  tune(execution_handler, 30, 1000, lambda : 64*np.random.randint(low=10, high=100000, size=1), ())
              #Matrix AXPY
              if operation=='matrix-axpy':
                  def execution_handler(sizes, fname=os.devnull, parameters=None):
                      A = vcl.Matrix(sizes, context=ctx, dtype=datatype)
                      B = vcl.Matrix(sizes, context=ctx, dtype=datatype)
                      C = vcl.Matrix(sizes, context=ctx, dtype=datatype)
                      return execute(device, vcl.Assign(C,A+B), (), sizes, fname, parameters)
                  tune(execution_handler, 30, 1000, lambda : 64*np.random.randint(low=5, high=100, size=2), ())
              #Row-wise reduction
              if operation=='row-wise-reduction':
                  layouts = ['N', 'T']
                  for A_trans in layouts:
                      def execution_handler(sizes, fname=os.devnull, parameters=None):
                          A = vcl.Matrix(sizes if A_trans=='N' else sizes[::-1], context=ctx, dtype=datatype, layout=vcl.COL_MAJOR)
                          x = vcl.Vector(sizes[1] if A_trans=='N' else sizes[0], context=ctx, dtype=datatype)
                          y = vcl.Vector(sizes[0] if A_trans=='N' else sizes[1], context=ctx, dtype=datatype)
                          LHS = A if A_trans=='N' else A.T
                          return execute(device, vcl.Assign(y, LHS*x), (), sizes, fname, parameters)
                      tune(execution_handler, 30, 1000, lambda : 64*np.random.randint(low=5, high=100, size=2), (A_trans,))
              #Matrix Product
              if operation=='matrix-product':
                  layouts = ['NN', 'NT', 'TN', 'TT']
                  for layout in layouts:
                      def execution_handler(sizes, fname=os.devnull, parameters=None):
                          A_trans = layout[0]
                          B_trans = layout[1]
                          A = vcl.Matrix((sizes[0], sizes[1]) if A_trans=='N' else (sizes[1],sizes[0]), context=ctx, dtype=datatype, layout=vcl.COL_MAJOR);
                          B = vcl.Matrix((sizes[1], sizes[2]) if B_trans=='N' else (sizes[2],sizes[1]), context=ctx, dtype=datatype, layout=vcl.COL_MAJOR);
                          LHS = A if A_trans=='N' else A.T
                          RHS = B if B_trans=='N' else B.T
                          alpha = vcl.HostScalar(1.0,  context=ctx, dtype = datatype)
                          beta = vcl.HostScalar(1.0, context=ctx, dtype = datatype)
                          C = vcl.Matrix((sizes[0], sizes[2]), context=ctx, dtype = datatype, layout=vcl.COL_MAJOR)
                          return execute(device, vcl.Assign(C,LHS*RHS*alpha + C*beta),(A_trans, B_trans), sizes, fname, parameters)
                      tune(execution_handler, 30, 1000, lambda : 64*np.random.randint(low=1, high=40, size=3),(layout[0], layout[1]))

    dname = misc_tools.sanitize_string(device.name)
    json_out["version"] = "1.0"
    json.dump(json_out, open(dname + '.json','w'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='action')
    print_devices_parser = subparsers.add_parser('list-devices', help='list the devices available')
    tune_parser = subparsers.add_parser('tune', help='tune using a specific configuration file')
    tune_parser.add_argument("--device", default=0, required=False, type=str)

    tune_subparsers = tune_parser.add_subparsers(dest='method')
    big_sizes_parser = tune_subparsers.add_parser('unique', help = 'Tune each operation for unique sizes')
    big_sizes_parser.add_argument("--sizes", nargs='+', default=[10e6,2560,2560,1536,1536,1536], required=False, type=int, help = '6 = 1 + 2 + 3 sizes for respectively BLAS1, BLAS2, BLAS3')
    big_sizes_parser.add_argument("--viennacl-src-path", default='', required=False, type=str)

    model_parser = tune_subparsers.add_parser('build-model', help = 'Build an input-dependent model')


    args = parser.parse_args()

    devices = [d for platform in cl.get_platforms() for d in platform.get_devices()]
    if(args.action=='list-devices'):
        print("----------------")
        print("Devices available:")
        print("----------------")
        for (i, d) in enumerate(devices):
            print 'Device', i, '|',  cl.device_type.to_string(d.type), '|', d.name, 'on', d.platform.name
        print("----------------")
    else:
        print("------")
        print("Auto-tuning")
        print("------")
        do_tuning(args, devices)
