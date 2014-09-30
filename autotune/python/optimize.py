import array
import numpy as np
import random
import sys

import itertools
import tools
import deap.tools

from genetic import GeneticOperators

#~ def parameter_space(operation):
  #~ simd = [1, 2, 4, 8]
  #~ pow2_1D = [2**k for k in range(12)]
  #~ pow2_2D = [2**i for i in range(8)]
  #~ pow2_2D_unrolled = [2**i for i in range(8)]
  #~ FetchingPolicy = vcl.atidlas.FetchingPolicy
  #~ fetch = [FetchingPolicy.FETCH_FROM_LOCAL, FetchingPolicy.FETCH_FROM_GLOBAL_CONTIGUOUS, FetchingPolicy.FETCH_FROM_GLOBAL_STRIDED]
  #~ if operation == 'vector-axpy': return [simd, pow2_1D, pow2_1D, fetch]
  #~ if operation == 'reduction': return [simd, pow2_1D, pow2_1D, fetch]
  #~ if operation == 'matrix-axpy': return [simd, pow2_2D, pow2_2D, pow2_2D, pow2_2D, fetch]
  #~ if operation == 'row-wise-reduction': return [simd, pow2_2D, pow2_2D, pow2_1D, fetch]
  #~ if operation == 'matrix-product': return [simd, pow2_2D, pow2_2D, pow2_2D, pow2_2D_unrolled,  pow2_2D_unrolled,  pow2_2D_unrolled, fetch, fetch, [0] + pow2_2D, [0] + pow2_2D]
  #~

#~ def exhaustive(statement, context, TemplateType, build_template, parameter_names, all_parameters, compute_perf, perf_metric, out):
  #~ device = context.devices[0]
  #~ nvalid = 0
  #~ current = 0
  #~ minT = float('inf')
  #~ for individual in itertools.product(*all_parameters):
    #~ template = build_template(TemplateType.Parameters(*individual))
    #~ if not tools.skip(template, statement, device):
      #~ nvalid = nvalid + 1
  #~ for individual in itertools.product(*all_parameters):
    #~ template = build_template(TemplateType.Parameters(*individual))
    #~ try:
      #~ T = tools.benchmark(template,statement,device)
      #~ current = current + 1
      #~ if T < minT:
        #~ minT = T
        #~ best = individual
      #~ sys.stdout.write('%d / %d , Best is %d %s for %s\r'%(current, nvalid, compute_perf(minT), perf_metric, best))
      #~ sys.stdout.flush()
    #~ except:
      #~ pass
  #~ sys.stdout.write('\n')
  #~ sys.stdout.flush()
    #~

def genetic(statement, context, TemplateType, build_template, parameter_names, compute_perf, perf_metric, out):
    GA = GeneticOperators(context.devices[0], statement, parameter_names, TemplateType, build_template, out)
    return GA.optimize(maxtime='2m30s', maxgen=1000, compute_perf=compute_perf, perf_metric=perf_metric)
