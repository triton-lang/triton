import array
import numpy as np
import random
import sys

import itertools
import tools
import deap.tools

from genetic import GeneticOperators

def exhaustive(statement, context, TemplateType, build_template, parameter_names, all_parameters, compute_perf, perf_metric, out):
  device = context.devices[0]
  nvalid = 0
  current = 0
  minT = float('inf')
  for individual in itertools.product(*all_parameters):
    template = build_template(TemplateType.Parameters(*individual))
    if not tools.skip(template, statement, device):
      nvalid = nvalid + 1
  for individual in itertools.product(*all_parameters):
    template = build_template(TemplateType.Parameters(*individual))
    try:
      T = tools.benchmark(template,statement,device)
      current = current + 1
      if T < minT:
        minT = T
        best = individual
      sys.stdout.write('%d / %d , Best is %d %s for %s\r'%(current, nvalid, compute_perf(minT), perf_metric, best))
      sys.stdout.flush()
    except:
      pass
  sys.stdout.write('\n')
  sys.stdout.flush()
    
  
def genetic(statement, context, TemplateType, build_template, parameter_names, all_parameters, compute_perf, perf_metric, out):
  GA = GeneticOperators(context.devices[0], statement, all_parameters, parameter_names, TemplateType, build_template)
  GA.optimize(maxtime='5m0s', maxgen=1000, compute_perf=compute_perf, perf_metric=perf_metric)
