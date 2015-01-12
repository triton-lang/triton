from __future__ import division

import argparse, itertools, os, sys, json
import misc_tools, optimize, dataset
import pyatidlas as atd
import pyopencl as cl
import numpy as np

from numpy import random
from model import train_model


TYPES = { 'vaxpy': {'template':atd.vaxpy,
                          'perf-index':lambda x: 3*x[0]*x[1][0]/x[2]*1e-9,
                          'perf-measure':'GB/s'},

          'maxpy': {'template':atd.maxpy,
                          'perf-index':lambda x: 3*x[0]*x[1][0]*x[1][1]/x[2]*1e-9,
                          'perf-measure':'GB/s'},

          'dot': {'template':atd.reduction,
                        'perf-index':lambda x: 2*x[0]*x[1][0]/x[2]*1e-9,
                        'perf-measure':'GB/s'},

          'gemv': {'template': {'N': atd.mreduction_rows, 'T': atd.mreduction_cols},
                                'perf-index':lambda x: x[0]*x[1][0]*x[1][1]/x[2]*1e-9,
                                'perf-measure':'GB/s'},

          'gemm': {'template': {('N','N'): atd.mproduct_nn, ('T','N'): atd.mproduct_tn, 
                                          ('N','T'): atd.mproduct_nt, ('T','T'): atd.mproduct_tt},
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
        default_tuning_sizes = {'vaxpy': args.blas1_size, 'dot': args.blas1_size,
                                'maxpy' : args.blas2_size, 'gemv' : args.blas2_size,
                                'gemm': args.blas3_size}

    for operation in ['vaxpy', 'dot', 'maxpy', 'gemv', 'gemm']:

          for datatype in [atd.float32, atd.float64]:
              
              dtypestr = datatype.__name__
              
              if operation not in args.operations and operation + '-' + dtypestr not in args.operations:
                  continue

              #Check data-type
              if datatype is atd.float64 and not device.double_fp_config:
                  sys.stderr.write('Warning : The device ' + device.name + ' does not support double precision! Skipping ...')
                  continue

              #~ #Helper for execution
              def execute(symbolic, sizes, Template, parameters = None, fname = os.devnull):
                  if parameters is not None:
                    return misc_tools.benchmark(Template(*parameters), symbolic)
                  with open(fname, "w+") as archive:
                    return optimize.genetic(symbolic, Template, lambda t: TYPES[operation]['perf-index']([datatype(0).size, sizes, t]), 
                                             TYPES[operation]['perf-measure'], archive)
             
              def log_uniform_sample(a,b):
                  return np.exp(np.random.uniform(low=np.log(a), high=np.log(b), size=1)).astype(int)

              def space_gen_product(a,b,N,dim,method):
                  N = int(N**(1.0/dim))
                  def space_gen(a,b,method):
                      for i in range(N):
                          if method == 'linear':
                              v = int(a + (b-a)*i/N)
                          if method == 'log':
                              v = int(np.exp(np.log(a) + (np.log(b) - np.log(a))*i/N))
                          yield (v//64 + 1)*64
                  return tuple(itertools.product(*[space_gen(a,b,method) for i in range(dim)]))


              #Helper for tuning
              def tune(execution_handler, a, b, dimsample, layouts, sample_method_profiles, sample_method_dataset):
                  print('-----')
                  print(' '.join(map(str, ("Now tuning:", dtypestr, '-', operation, '-'.join(layouts), '[' + device.name, '(' + device.platform.name + ')]'))))
                  #Update JSON
                  full_operation = operation + ''.join(layouts)
                  if full_operation not in json_out:
                      json_out[full_operation] = {}
                  json_out[full_operation][dtypestr] = {}
                  D = json_out[full_operation][dtypestr]

                  if args.method == 'simple':
                      print default_tuning_sizes[operation]
                      profiles = [execution_handler(map(int,default_tuning_sizes[operation]))]
                  else:
                      def compute_perf(x, t):
                          return TYPES[operation]['perf-index']([datatype(0).size, x, t])
                      profiles_generator = space_gen_product(a, b, args.sample_size, dimsample, sample_method_profiles)
                      profiles = dataset.sample_profiles(execution_handler, profiles_generator)
                      if args.build_model:
                        dataset_generator = space_gen_product(a, b, 1000, dimsample, sample_method_dataset)
                        X, Y, profiles = dataset.sample_dataset(os.path.join(full_operation,dtypestr), profiles, execution_handler, dataset_generator)
                        # profiles = np.loadtxt('data/'+full_operation+'/'+datatype+'/profiles.csv')
                        # X = np.loadtxt('data/'+full_operation+'/'+datatype+'/X.csv',ndmin=2)
                        # Y = np.loadtxt('data/'+full_operation+'/'+datatype+'/Y.csv',ndmin=2)
                        clf = train_model(X, Y, profiles, TYPES[operation]['perf-measure'])
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
                      x = atd.empty(sizes[0], datatype)
                      y = atd.empty(sizes[0], datatype)
                      return execute(x + y, sizes, Template, parameters, fname)
                  tune(execution_handler, 1e3, 2e7, 1, (),'log', 'log')
              #dot
              if operation=='dot':
                  def execution_handler(sizes, fname=os.devnull, parameters=None):
                      x = atd.empty(sizes[0], datatype)
                      y = atd.empty(sizes[0], datatype)
                      s = atd.scalar(datatype)
                      return execute(atd.dot(x, y), sizes, Template, parameters, fname)
                  tune(execution_handler, 1e3, 2e7, 1, (),'log', 'log')
              #Matrix AXPY
              if operation=='maxpy':
                  def execution_handler(sizes, fname=os.devnull, parameters=None):
                      A = atd.empty(sizes, datatype)
                      C = atd.empty(sizes, datatype)
                      return execute(A + C, sizes, Template, parameters, fname)
                  tune(execution_handler, 100, 5000, 2, (),'log', 'log')
              #Row-wise dot
              if operation=='gemv':
                  for A_trans in  args.gemv_layouts:
                      def execution_handler(sizes, fname=os.devnull, parameters=None):
                          Template = Template[A_trans]
                          A = atd.empty(sizes if A_trans=='N' else sizes[::-1], datatype)
                          x = atd.empty(sizes[1], datatype)
                          LHS = A if A_trans=='N' else A.T
                          return execute(device, atd.dot(LHS, x), sizes, Template, parameters, fname)
                      tune(execution_handler, 100, 5000, 2, (A_trans,),'log', 'log')
              #Matrix Product
              if operation=='gemm':
                  for L in args.gemm_layouts:
                      A_trans = L[0]
                      B_trans = L[1]
                      def execution_handler(sizes, fname=os.devnull, parameters=None):
                          Template = Template[A_trans, B_trans]
                          A = atd.empty((sizes[0], sizes[2]) if A_trans=='N' else (sizes[2], sizes[0]), datatype)
                          B = atd.empty((sizes[2], sizes[1]) if B_trans=='N' else (sizes[1], sizes[2]), datatype)
                          LHS = A if A_trans=='N' else A.T
                          RHS = B if B_trans=='N' else B.T
                          return execute(device, atd.dot(LHS, RHS),(A_trans,B_trans), sizes, fname, parameters)
                      tune(execution_handler, 100, 2000, 3,(A_trans,B_trans), 'linear')

              json.dump(json_out, open(args.json_file,'w'))




class ArgumentsHandler:

    def __init__(self):

        #No action argument -> interactive tuning
        if len(sys.argv)==1:
            def add_input(help, default):
                return raw_input(help + "[" + default + "] : ") or default

            self.device = add_input('Device to tune for','0')
            self.operations = add_input('Operations to tune for','vaxpy,maxpy,dot,gemv,gemm-float32')
            self.gemm_layouts = add_input('GEMV Layouts', 'NN,NT,TN,TT')
            self.gemv_layouts =  add_input('GEMV Layouts', 'N,T')
            self.json_file = add_input('JSON File', misc_tools.sanitize_string(devices[int(self.device)].name) + '.json')
            self.method = add_input('Tuning type', 'simple')
            if self.method == 'simple':
                self.blas1_size = add_input('BLAS1 size', '10e6')
                self.blas2_size = add_input('BLAS2 sizes (M,N)', '2560,2560').split(',')
                self.blas3_size = add_input('BLAS3 sizes (M,N,K)', '1024,1024,1024').split(',')
            else:
              self.build_model = True
              self.sample_size = 30
        else:
            #Command line arguments
            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers(dest='action')
            print_devices_parser = subparsers.add_parser('list-devices', help='List the devices available')
            tune_parser = subparsers.add_parser('tune', help='Auto-tuning')
            tune_parser.add_argument("--device", default=0, type=int)
            tune_parser.add_argument("--operations", default = 'vaxpy,maxpy,dot,gemv,gemm-float32', type=str)
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
            full_parser.add_argument("--build-model", default=True, type=bool)
            full_parser.add_argument("--sample-size", default=30, type=int)

            args = parser.parse_args()
            self.__dict__ = args.__dict__.copy()
            
        #Retypes
        self.device = devices[int(self.device)]
        if not self.json_file:
            self.json_file = misc_tools.sanitize_string(self.device.name) + '.json'
        self.operations = self.operations.split(',')
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
