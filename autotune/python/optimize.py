import array
import numpy as np
import random
import sys

import itertools
import tools
import deap.tools

from deap import base
from deap import creator
from genetic import GeneticOperators
from genetic import eaMuPlusLambda

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
  creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMin)

  toolbox = base.Toolbox()
  toolbox.register("individual", deap.tools.initIterate, creator.Individual, GA.init)
  toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
  toolbox.register("evaluate", GA.evaluate)
  toolbox.register("mate", deap.tools.cxTwoPoint)
  toolbox.register("mutate", GA.mutate)
  toolbox.register("select", deap.tools.selBest)
    
  pop = toolbox.population(n=50)
  hof = deap.tools.HallOfFame(1)

  best_performer = lambda x: max([compute_perf(hof[0].fitness.values[0]) for t in x])
  best_profile = lambda x: '(%s)'%','.join(map(str,hof[0]))
  
  stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("max (" + perf_metric + ")", lambda x: max([compute_perf(hof[0].fitness.values[0]) for t in x]))
  stats.register("profile ", lambda x: '(%s)'%','.join(map(str,hof[0])))

  pop = eaMuPlusLambda(pop, toolbox, 50, 70, cxpb=0.2, mutpb=0.3, maxtime='5m0s', maxgen=1000, halloffame=hof, compute_perf=compute_perf, perf_metric=perf_metric)
