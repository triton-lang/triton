import array
import numpy as np
import random

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
    except:
      pass
    
  
def genetic(statement, context, TemplateType, build_template, parameter_names, all_parameters, compute_perf, perf_metric, out):
  gen = GeneticOperators(context.devices[0], statement, all_parameters, parameter_names, TemplateType, build_template)
  creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMin)

  toolbox = base.Toolbox()
  toolbox.register("individual", tools.initIterate, creator.Individual, gen.init)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.register("evaluate", gen.evaluate)
  toolbox.register("mate", tools.cxTwoPoint)
  toolbox.decorate("mate", gen.repair)
  toolbox.register("mutate", gen.mutate)
  toolbox.decorate("mutate", gen.repair)
  toolbox.register("select", tools.selBest)
    
  pop = toolbox.population(n=30)
  hof = deap.tools.HallOfFame(1)

  best_performer = lambda x: max([compute_perf(hof[0].fitness.values[0]) for t in x])
  best_profile = lambda x: '(%s)'%','.join(map(str,hof[0]))
  
  stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("max (" + perf_metric + ")", lambda x: max([compute_perf(hof[0].fitness.values[0]) for t in x]))
  stats.register("profile ", lambda x: '(%s)'%','.join(map(str,hof[0])))

  pop = eaMuPlusLambda(pop, toolbox, 30, 50, cxpb=0.2, mutpb=0.3, maxtime='3m0s', maxgen=200, halloffame=hof, compute_perf=compute_perf, perf_metric=perf_metric)
