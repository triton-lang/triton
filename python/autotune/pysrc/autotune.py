from __future__ import division

import argparse, itertools, os, sys, json
import misc_tools, optimize, dataset
import pyopencl as cl
import pyviennacl as vcl
import pyatidlas as atd
import numpy as np

from numpy import random
from model import train_model


TYPES = { 'vector-axpy': {'template':atd.VectorAxpyTemplate,
                          'perf-index':lambda x: 2*x[0]*x[1][0]/x[2]*1e-9,
                          'perf-measure':'GB/s'},

          'matrix-axpy': {'template':atd.MatrixAxpyTemplate,
                          'perf-index':lambda x: 2*x[0]*x[1][0]*x[1][1]/x[2]*1e-9,
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


def do_tuning(args):
    device = args.device

    if os.path.isfile(args.json_file):
        json_out = json.load(open(args.json_file, 'r'))
    else:
        json_out = {}
        json_out["version"] = "1.0"

    def map_to_list(T, x):
        return list(map(T, x if isinstance(x, list) else [x]))

    if(args.method=='simple'):
        default_tuning_sizes = {'vector-axpy': args.blas1_size, 'reduction': args.blas1_size,
                                'matrix-axpy' : args.blas2_size, 'row-wise-reduction' : args.blas2_size,
                                'matrix-product': args.blas3_size}

    for operation in ['vector-axpy', 'reduction', 'matrix-axpy', 'row-wise-reduction', 'matrix-product']:

          for datatype in [vcl.float32, vcl.float64]:

              if any(x in args.exclude_operations for x in [operation, operation + '-' + datatype.__name__]):
                  continue

              ctx = cl.Context([device])
              ctx = vcl.backend.Context(ctx)


              #Check data-type
              if datatype is vcl.float64 and not device.double_fp_config:
                  sys.stderr.write('Warning : The device ' + device.name + ' does not support double precision! Skipping ...')
                  continue

              #Helper for execution
              def execute(device, node, other_params, sizes, fname = os.devnull, parameters = None):
                  with vcl.Statement(node) as statement:
                      if parameters is not None:
                          TemplateType = TYPES[operation]['template']
                          return misc_tools.benchmark(TemplateType(TemplateType.Parameters(*parameters),*other_params), statement, device)
                      with open(fname, "w+") as archive:
                          return optimize.genetic(statement, device, TYPES[operation]['template'], lambda p: TYPES[operation]['template'](p, *other_params),
                                                  lambda t: TYPES[operation]['perf-index']([datatype().itemsize, sizes, t]), TYPES[operation]['perf-measure'], archive)

              def log_uniform_sample(a,b):
                  return np.exp(np.random.uniform(low=np.log(a), high=np.log(b), size=1)).astype(int)

              def log_space_gen_product(a,b,N,dim):
                  N = int(N**(1.0/dim))
                  def log_space_gen(a,b):
                      for i in range(N):
                          v = int(np.exp(np.log(a) + (np.log(b) - np.log(a))*(i+1)/N))
                          yield (v//64 + 1)*64

                  return tuple(itertools.product(*[log_space_gen(a,b) for i in range(dim)]))


              #Helper for tuning
              def tune(execution_handler, a, b, dimsample, additional_parameters):
                  print('-----')
                  print(' '.join(map(str, ("Now tuning:", datatype.__name__, '-', operation, '-'.join(additional_parameters), '[' + device.name, '(' + device.platform.name + ')]'))))
                  #Update JSON
                  full_operation = operation + ''.join(additional_parameters)
                  if full_operation not in json_out:
                      json_out[full_operation] = {}
                  json_out[full_operation][datatype.__name__] = {}
                  D = json_out[full_operation][datatype.__name__]

                  if args.method == 'simple':
                      print default_tuning_sizes[operation]
                      profiles = [execution_handler(map(int,default_tuning_sizes[operation]))]
                  else:
                      def compute_perf(x, t):
                          return TYPES[operation]['perf-index']([datatype().itemsize, x, t])
                      profiles_generator = log_space_gen_product(a, b, args.sample_size, dimsample)
                      profiles = dataset.sample_profiles(execution_handler, profiles_generator)
                      if args.build_model:
                        dataset_generator = log_space_gen_product(a, b, 1000, dimsample)
                        X, Y, profiles = dataset.sample_dataset(os.path.join(full_operation,datatype.__name__), profiles, execution_handler, dataset_generator)
                        clf = train_model(X, Y, profiles, TYPES[operation]['perf-measure'])
                        D['predictor'] = [{'children_left': e.tree_.children_left.tolist(),
                                       'children_right': e.tree_.children_right.tolist(),
                                       'threshold': e.tree_.threshold.astype('float64').tolist(),
                                       'feature': e.tree_.feature.astype('float64').tolist(),
                                       'value': e.tree_.value[:,:,0].astype('float64').tolist()} for e in clf.estimators_]
                  if args.viennacl_src_path:
                    misc_tools.update_viennacl_headers(args.viennacl_src_path, device,datatype,operation,additional_parameters,profiles[0])
                  D['profiles'] = [map(int, x) for x in profiles]


              #Vector AXPY
              if operation=='vector-axpy':
                  def execution_handler(sizes, fname=os.devnull, parameters=None):
                      x = vcl.Vector(sizes[0], context=ctx, dtype=datatype)
                      z = vcl.Vector(sizes[0], context=ctx, dtype=datatype)
                      return execute(device, vcl.Assign(z, x), (), sizes, fname, parameters)
                  tune(execution_handler, 1e4, 1e7, 1, ())
              #Reduction
              if operation=='reduction':
                  def execution_handler(sizes, fname=os.devnull, parameters=None):
                      x = vcl.Vector(sizes[0], context=ctx, dtype=datatype)
                      y = vcl.Vector(sizes[0], context=ctx, dtype=datatype)
                      s = vcl.Scalar(0, context=ctx, dtype=datatype)
                      return execute(device, vcl.Assign(s, vcl.Dot(x,y)), (), sizes, fname, parameters)
                  tune(execution_handler, 1e4, 1e7, 1, ())
              #Matrix AXPY
              if operation=='matrix-axpy':
                  def execution_handler(sizes, fname=os.devnull, parameters=None):
                      A = vcl.Matrix(sizes, context=ctx, dtype=datatype, layout=vcl.COL_MAJOR)
                      C = vcl.Matrix(sizes, context=ctx, dtype=datatype, layout=vcl.COL_MAJOR)
                      return execute(device, vcl.Assign(C,A), (), sizes, fname, parameters)
                  tune(execution_handler, 100, 4000, 2, ())
              #Row-wise reduction
              if operation=='row-wise-reduction':
                  for A_trans in  args.gemv_layouts:
                      def execution_handler(sizes, fname=os.devnull, parameters=None):
                          A = vcl.Matrix(sizes if A_trans=='N' else sizes[::-1], context=ctx, dtype=datatype, layout=vcl.COL_MAJOR)
                          x = vcl.Vector(sizes[1], context=ctx, dtype=datatype)
                          y = vcl.Vector(sizes[0], context=ctx, dtype=datatype)
                          LHS = A if A_trans=='N' else A.T
                          return execute(device, vcl.Assign(y, LHS*x), (), sizes, fname, parameters)
                      tune(execution_handler, 100, 4000, 2, (A_trans,))
              #Matrix Product
              if operation=='matrix-product':
                  for L in args.gemm_layouts:
                      A_trans = L[0]
                      B_trans = L[1]
                      def execution_handler(sizes, fname=os.devnull, parameters=None):
                          A = vcl.Matrix((sizes[0], sizes[2]) if A_trans=='N' else (sizes[2],sizes[0]), context=ctx, dtype=datatype, layout=vcl.COL_MAJOR)
                          B = vcl.Matrix((sizes[2], sizes[1]) if B_trans=='N' else (sizes[1],sizes[2]), context=ctx, dtype=datatype, layout=vcl.COL_MAJOR)
                          LHS = A if A_trans=='N' else A.T
                          RHS = B if B_trans=='N' else B.T
                          alpha = vcl.HostScalar(1.0,  context=ctx, dtype = datatype)
                          beta = vcl.HostScalar(1.0, context=ctx, dtype = datatype)
                          C = vcl.Matrix((sizes[0], sizes[1]), context=ctx, dtype = datatype, layout=vcl.COL_MAJOR)
                          return execute(device, vcl.Assign(C,LHS*RHS*alpha + C*beta),(A_trans,B_trans), sizes, fname, parameters)
                      tune(execution_handler, 100, 2000, 3,(A_trans,B_trans))

              json.dump(json_out, open(args.json_file,'w'))




class ArgumentsHandler:

    def __init__(self):

        #Command line arguments
        parent_parser = argparse.ArgumentParser('parent', add_help=False)
        parent_parser.add_argument('--version', action='version', version='%(prog)s 2.0')

        parser = argparse.ArgumentParser(parents=[parent_parser])
        subparsers = parser.add_subparsers(dest='action')
        print_devices_parser = subparsers.add_parser('list-devices', help='List the devices available', parents=[parent_parser])
        tune_parser = subparsers.add_parser('tune', help='Auto-tuning', parents=[parent_parser])
        tune_parser.add_argument("--device", default=0, type=int)
        tune_parser.add_argument("--exclude-operations", default = '', type=str)
        tune_parser.add_argument("--gemm-layouts", default='NN,NT,TN,TT', type=str)
        tune_parser.add_argument("--gemv-layouts", default='N,T', type=str)
        tune_parser.add_argument("--json-file", default='', type=str)
        tune_parser.add_argument("--viennacl-src-path", default='', type=str)

        tune_subparsers = tune_parser.add_subparsers(dest='method')
        simple_parser = tune_subparsers.add_parser('simple', help = 'Tune each operation for unique sizes')

        simple_parser.add_argument("--blas1-size", default = 10e6, type=int)
        simple_parser.add_argument("--blas2-size", nargs=2, default=[2560,2560], type=int)
        simple_parser.add_argument("--blas3-size", nargs=3, default=[1536,1536,1536],type=int)

        full_parser = tune_subparsers.add_parser('full', help = 'Tune each operation for randomly chosen sizes')
        full_parser.add_argument("--build-model", default=False, type=bool)
        full_parser.add_argument("--sample-size", default=30, type=int)

        args = parent_parser.parse_args()
        self.__dict__ = args.__dict__.copy()

        #No action argument -> interactive tuning
        if 'action' not in vars(args):
                def add_input(help, default):
                    return raw_input(help + "[" + default + "] : ") or default

                self.device = add_input('Device to tune for','0')
                self.exclude_operations = add_input('Operations to exclude','vector-axpy,matrix-axpy,reduction,row-wise-reduction,matrix-product-float64').split(',')
                self.gemm_layouts = '' if 'matrix-product' in self.exclude_operations else add_input('GEMV Layouts', 'NN,NT,TN,TT')
                self.gemv_layouts = '' if 'row-wise-reduction' in self.exclude_operations else add_input('GEMV Layouts', 'N,T')
                self.json_file = add_input('JSON File', misc_tools.sanitize_string(devices[int(self.device)].name) + '.json')
                self.method = add_input('Tuning type', 'simple')
                if self.method == 'simple':
                    self.blas1_size = add_input('BLAS1 size', '10e6')
                    self.blas2_size = add_input('BLAS2 sizes (M,N)', '2560,2560').split(',')
                    self.blas3_size = add_input('BLAS3 sizes (M,N,K)', '1024,1024,1024').split(',')
                self.build_model = True
                self.sample_size = 30

        #Retypes
        self.device = devices[int(self.device)]
        self.gemm_layouts = self.gemm_layouts.split(',')
        self.gemv_layouts = self.gemv_layouts.split(',')
        if self.method == 'simple':
            self.blas1_size = [int(float(self.blas1_size))]
            self.blas2_size = map(int, self.blas2_size)
            self.blas3_size = map(int, self.blas3_size)

if __name__ == "__main__":

    devices = [d for platform in cl.get_platforms() for d in platform.get_devices()]
    print("----------------")
    print("Devices available:")
    print("----------------")
    for (i, d) in enumerate(devices):
        print 'Device', i, '|',  cl.device_type.to_string(d.type), '|', d.name, 'on', d.platform.name
    print("----------------")

    args = ArgumentsHandler()

    print("------")
    print("Auto-tuning")
    print("------")

    do_tuning(args)
