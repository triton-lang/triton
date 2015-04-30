from __future__ import division

import argparse, itertools, os, sys, json
import misc_tools, optimize, dataset
import isaac as isc
import numpy as np

from numpy import random
from model import train_model


TYPES = { 'vaxpy': {'template':isc.vaxpy,
                          'perf-index':lambda x: 3*x[0]*x[1][0]/x[2]*1e-9,
                          'perf-measure':'GB/s'},

          'maxpy': {'template':isc.maxpy,
                          'perf-index':lambda x: 3*x[0]*x[1][0]*x[1][1]/x[2]*1e-9,
                          'perf-measure':'GB/s'},

          'dot': {'template':isc.reduction,
                        'perf-index':lambda x: 2*x[0]*x[1][0]/x[2]*1e-9,
                        'perf-measure':'GB/s'},

          'gemv': {'template': {'N': isc.mreduction_rows, 'T': isc.mreduction_cols},
                                'perf-index':lambda x: x[0]*x[1][0]*x[1][1]/x[2]*1e-9,
                                'perf-measure':'GB/s'},

          'gemm': {'template': {('N','N'): isc.mproduct_nn, ('T','N'): isc.mproduct_tn, 
                                          ('N','T'): isc.mproduct_nt, ('T','T'): isc.mproduct_tt},
                            'perf-index': lambda x: 2*x[1][0]*x[1][1]*x[1][2]/x[2]*1e-9,
                            'perf-measure': 'GFLOP/s'} }


def do_tuning(args):
    device = args.device
    context = isc.context(device)
    context.queues.append(isc.command_queue(context, device))
    if os.path.isfile(args.out):
        json_out = json.load(open(args.out, 'r'))
    else:
        json_out = {}
        json_out["version"] = "1.0"

    def map_to_list(T, x):
        return list(map(T, x if isinstance(x, list) else [x]))

    if(args.method=='simple'):
        default_tuning_sizes = {'vaxpy': args.blas1_size, 'dot': args.blas1_size,
                                'maxpy' : args.blas2_size, 'gemv' : args.blas2_size,
                                'gemm': args.blas3_size}

    for operation in ['vaxpy', 'dot', 'maxpy', 'gemv', 'gemm']:

          for datatype in [isc.float32, isc.float64]:
              
              dtypestr = datatype.__name__
              
              if operation not in args.operations and operation + '-' + dtypestr not in args.operations:
                  continue

              #Check data-type
              if datatype is isc.float64 and not device.double_fp_config:
                  sys.stderr.write('Warning : The device ' + device.name + ' does not support double precision! Skipping ...')
                  continue

              #~ #Helper for execution
              def execute(symbolic, sizes, Template, parameters = None, fname = os.devnull):
                  if parameters is not None:
                    return misc_tools.benchmark(Template(*parameters), symbolic)
                  with open(fname, "w+") as archive:
                    return optimize.genetic(symbolic, Template, lambda t: TYPES[operation]['perf-index']([datatype(0).size, sizes, t]), 
                                             TYPES[operation]['perf-measure'], archive)
             
              def log_spaced_points(a,b,N,r=128):
                  t = np.ceil(np.exp(np.linspace(np.log(a), np.log(b), N))/r)*r
                  return t.reshape(t.size,1).astype(int)


              #Helper for tuning
              def tune(execution_handler, layouts, tuning_sizes, training_sizes):
                  print('-----')
                  print(' '.join(map(str, ("Now tuning:", dtypestr, '-', operation, '-'.join(layouts), '[' + device.name, '(' + device.platform.name + ')]'))))
                  #Update JSON
                  full_operation = operation + ''.join(layouts)
                  prefix = os.path.join('data',os.path.join(full_operation,dtypestr))
                  if not os.path.exists(prefix):
                      os.makedirs(prefix)
                  if full_operation not in json_out:
                      json_out[full_operation] = {}
                  json_out[full_operation][dtypestr] = {}
                  D = json_out[full_operation][dtypestr]

                  if args.method == 'simple':
                      print 'Size : ', ','.join(map(str, default_tuning_sizes[operation]))
                      profiles = [execution_handler(map(int,default_tuning_sizes[operation]))]
                  else:
                      def compute_perf(x, t):
                          return TYPES[operation]['perf-index']([datatype(0).size, x, t])
                      #profiles = dataset.sample_profiles(execution_handler, tuning_sizes)
                      if args.build_model:
                        #X, Y, profiles = dataset.sample_dataset(prefix, profiles, execution_handler, training_sizes)
                        profiles = np.loadtxt(prefix+'/profiles.csv')
                        X = np.loadtxt(prefix+'/X.csv',ndmin=2)
                        Y = np.loadtxt(prefix+'/Y.csv',ndmin=2)
                        clf = train_model(X, Y, profiles, compute_perf, TYPES[operation]['perf-measure'])
                        D['predictor'] = [{'children_left': e.tree_.children_left.tolist(),
                                       'children_right': e.tree_.children_right.tolist(),
                                       'threshold': e.tree_.threshold.astype('float64').tolist(),
                                       'feature': e.tree_.feature.astype('float64').tolist(),
                                       'value': e.tree_.value[:,:,0].astype('float64').tolist()} for e in clf.estimators_]
                  D['profiles'] = [map(int, x) for x in profiles]


              Template = TYPES[operation]['template']
              
              #Vector AXPY
              if operation=='vaxpy':
                  def execution_handler(sizes, fname=os.devnull, parameters=None):
                      x = isc.empty(sizes[0], datatype, context=context)
                      y = isc.empty(sizes[0], datatype, context=context)
                      return execute(x + y, sizes, Template, parameters, fname)
                  tune(execution_handler, (), log_spaced_points(1e4, 1e7, 20), log_spaced_points(1e4, 1e7, 1000))
              #Dot
              if operation=='dot':
                  def execution_handler(sizes, fname=os.devnull, parameters=None):
                      x = isc.empty(sizes[0], datatype, context=context)
                      y = isc.empty(sizes[0], datatype, context=context)
                      s = isc.scalar(datatype)
                      return execute(isc.dot(x, y), sizes, Template, parameters, fname)
                  tune(execution_handler, (), log_spaced_points(1e4, 1e7, 50), log_spaced_points(1e4, 1e7, 1000))
              #Matrix AXPY
              if operation=='maxpy':
                  def execution_handler(sizes, fname=os.devnull, parameters=None):
                      A = isc.empty(sizes, datatype, context=context)
                      C = isc.empty(sizes, datatype, context=context)
                      return execute(A + C, sizes, Template, parameters, fname)
                  tune(execution_handler, 64, 5000, 2, (),'log', 'log')
              #Row-wise dot
              if operation=='gemv':
                  for A_trans in  args.gemv_layouts:
                      def execution_handler(sizes, fname=os.devnull, parameters=None):
                          A = isc.empty(sizes if A_trans=='N' else sizes[::-1], datatype, context=context)
                          x = isc.empty(sizes[1], datatype, context=context)
                          LHS = A if A_trans=='N' else A.T
                          return execute(isc.dot(LHS, x), sizes, Template[A_trans], parameters, fname)
                      tuning_sizes = itertools.chain( itertools.product([128, 512, 2048, 8192], [128, 512, 2048, 8192]),
                                                     itertools.product([128, 512, 2048, 8192], [16384, 32768, 65536]),
                                                     itertools.product([16384, 32768, 65536], [128, 512, 2048, 8192]))
                      
                      training_sizes = itertools.chain( itertools.product([2**k for k in range(4, 13)], [2**k for k in range(4, 13)]),
                                                        itertools.product([2**k for k in range(4, 13)], [2**k for k in range(13, 17)]),
                                                        itertools.product([2**k for k in range(13, 17)], [2**k for k in range(4, 13)]))
                      tune(execution_handler, (A_trans,), tuning_sizes, training_sizes)
              #Matrix Product
              if operation=='gemm':
                  for L in args.gemm_layouts:
                      A_trans = L[0]
                      B_trans = L[1]
                      def execution_handler(sizes, fname=os.devnull, parameters=None):
                          A = isc.empty((sizes[0], sizes[2]) if A_trans=='N' else (sizes[2], sizes[0]), datatype, context=context)
                          B = isc.empty((sizes[2], sizes[1]) if B_trans=='N' else (sizes[1], sizes[2]), datatype, context=context)
                          LHS = A if A_trans=='N' else A.T
                          RHS = B if B_trans=='N' else B.T
                          return execute(isc.dot(LHS, RHS), sizes, Template[(A_trans, B_trans)], parameters, fname)
                      
                      tuning_sizes = itertools.product([64, 256, 1024, 2560], [64, 256, 1024, 2560], [256, 2560, 32768, 65536])
                      training_sizes = itertools.product([2**k for k in range(6, 13)], [2**k for k in range(6, 13)], [2**k for k in range(6, 17)])
                      tune(execution_handler,(A_trans,B_trans), tuning_sizes, training_sizes)
                      
              json.dump(json_out, open(args.out,'w'))




class ArgumentsHandler:
    def __init__(self, devices):
        #Command line arguments
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='action')
        print_devices_parser = subparsers.add_parser('list-devices', help='List the devices available')
        tune_parser = subparsers.add_parser('tune', help='Auto-tuning')
        tune_parser.add_argument("--device", default=0, type=int)
        tune_parser.add_argument("--operations", default = 'vaxpy,maxpy,dot,gemv,gemm-float32', type=str)
        tune_parser.add_argument("--gemm-layouts", default='NN,NT,TN,TT', type=str)
        tune_parser.add_argument("--gemv-layouts", default='N,T', type=str)
        tune_parser.add_argument("--out", default='', type=str)
        tune_parser.add_argument("--viennacl-src-path", default='', type=str)

        tune_subparsers = tune_parser.add_subparsers(dest='method')
        simple_parser = tune_subparsers.add_parser('simple', help = 'Tune each operation for unique sizes')

        simple_parser.add_argument("--blas1-size", default = 10e6, type=int)
        simple_parser.add_argument("--blas2-size", nargs=2, default=[2560,2560], type=int)
        simple_parser.add_argument("--blas3-size", nargs=3, default=[1536,1536,1536],type=int)

        full_parser = tune_subparsers.add_parser('full', help = 'Tune each operation for randomly chosen sizes')
        full_parser.add_argument("--build-model", default=True, type=bool)
        full_parser.add_argument("--sample-size", default=64, type=int)

        args = parser.parse_args()
        self.__dict__ = args.__dict__.copy()
        
        if self.action == 'tune':
            #Retypes
            self.device = devices[int(self.device)]
            if not self.out:
                self.out = misc_tools.sanitize_string(self.device.name) + '.json'
            self.operations = self.operations.split(',')
            self.gemm_layouts = self.gemm_layouts.split(',')
            self.gemv_layouts = self.gemv_layouts.split(',')
            if self.method == 'simple':
                self.blas1_size = [int(float(self.blas1_size))]
                self.blas2_size = map(int, self.blas2_size)
                self.blas3_size = map(int, self.blas3_size)

if __name__ == "__main__":
    isc.state.queue_properties = isc.CL_QUEUE_PROFILING_ENABLE
    
    platforms = isc.get_platforms()
    devices = [d for platform in platforms for d in platform.get_devices()]
    
    args = ArgumentsHandler(devices)

    print("----------------")
    print("Devices available:")
    print("----------------")
    for (i, d) in enumerate(devices):
        print 'Device', i, '|',  isc.device_type_to_string(d.type), '|', d.name, 'on', d.platform.name
    print("----------------")
    
    if args.action=='tune':
        print("------")
        print("Auto-tuning")
        print("------")
        do_tuning(args)
