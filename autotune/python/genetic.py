import random
import time
import sys
import tools
import pyviennacl as vcl
import numpy as np
import copy
from deap import algorithms

from collections import OrderedDict as odict

def closest_divisor(N, x):
  x_low=x_high=max(1,min(round(x),N))
  while N % x_low > 0 and x_low>0:
    x_low = x_low - 1
  while N % x_high > 0 and x_high < N:
    x_high = x_high + 1
  return x_low if x - x_low < x_high - x else x_high
    
class GeneticOperators(object):
  
  def __init__(self, device, statement, parameters, parameter_names, TemplateType, build_template):
      self.device = device
      self.statement = statement
      self.parameters = parameters
      self.parameter_names = parameter_names
      self.TemplateType = TemplateType
      self.ParameterType = TemplateType.Parameters
      self.build_template = build_template
      self.cache = {}
      self.indpb = 0.1
  
  def decode(self):
    simd = 2**int(s[0:2], 2)
    ls0 = 2**int(s[2:5],2)
    ls1 = 2**int(s[5:8],2)
    kL = 2**int(s[8:11],2)
    mS = 2**int(s[11:14],2)
    kS = 2**int(s[14:17],2)
    nS = 2**int(s[17:20],2)
    fetchA = s[20:22].count(1)
    fetchB = s[22:24].count(1)
    lf0 = 2**int(s[24:7],2)
    return [simd, ls0, kL, ls1, mS, kS, nS, fetchA, fetchB, lf0, ls0*ls1/lf0]
    
  def init(self):
    while True:
      result = [random.choice(L) for L in self.parameters]
      template = self.build_template(self.TemplateType.Parameters(*result))
      registers_usage = template.registers_usage(vcl.atidlas.StatementsTuple(self.statement))/4
      lmem_usage = template.lmem_usage(vcl.atidlas.StatementsTuple(self.statement))
      local_size = template.parameters.local_size_0*template.parameters.local_size_1
      occupancy_record = tools.OccupancyRecord(self.device, local_size, lmem_usage, registers_usage)
      if template.check(self.statement)==0 and occupancy_record.occupancy >= 10 :
        return result
  
  def crossover(self, ind1, ind2):
    ind1[1], ind2[1] = ind2[1], ind1[1]
    ind1[3], ind2[3] = ind2[3], ind1[3]
    ind1[7], ind2[7] = ind2[7], ind1[7]
    ind1[8], ind2[8] = ind2[8], ind1[8]
    ind1[9], ind2[9] = ind2[9], ind1[9]
    ind1[10], ind2[10] = ind2[10], ind1[10]
    return ind1, ind2
    
  def mutate(self, individual):
    FetchingPolicy = vcl.atidlas.FetchingPolicy
    while True:
      new_individual = copy.deepcopy(individual)
      for i in new_individual:
        if random.random() < self.indpb:
          coef = 2**np.random.geometric(0.8)
          funs = [lambda x:max(1, x/coef), lambda x:x*coef]
          F = random.choice(funs)
          nF = funs[1] if F==funs[0] else funs[0]
          def m0():
            new_individual[1], new_individual[3] = new_individual[3], new_individual[1]
          def m1():
            new_individual[4], new_individual[6] = new_individual[6], new_individual[4]
          def m2():
            new_individual[9], new_individual[10] = new_individual[10], new_individual[9]
          def m3():
            new_individual[0] = F(new_individual[0])
          def m4():
            new_individual[1], new_individual[9] = F(new_individual[1]), F(new_individual[9])
          def m5():
            new_individual[2] = F(new_individual[2])
          def m6():
            new_individual[3], new_individual[10] = F(new_individual[3]), F(new_individual[10])
          def m7():
            new_individual[4] = F(new_individual[4])
          def m8():
            new_individual[5] = F(new_individual[5])
          def m9():
            new_individual[6] = F(new_individual[6])
          def m10():
            new_individual[7] = random.choice([x for x in self.parameters[7] if x!=individual[7]])
            if new_individual[7]!=FetchingPolicy.FETCH_FROM_LOCAL and new_individual[8]!=FetchingPolicy.FETCH_FROM_LOCAL:
              new_individual[9], new_individual[10] = 0, 0
            else:
              new_individual[9], new_individual[10] = new_individual[1], new_individual[3]
          def m11():
            new_individual[8] = random.choice([x for x in self.parameters[8] if x!=individual[8]])
            if new_individual[7]!=FetchingPolicy.FETCH_FROM_LOCAL and new_individual[8]!=FetchingPolicy.FETCH_FROM_LOCAL:
              new_individual[9], new_individual[10] = 0, 0
            else:
              new_individual[9], new_individual[10] = new_individual[1], new_individual[3]
          def m12():
            new_individual[9], new_individual[10] = F(new_individual[9]), nF(new_individual[10])
          def m13():
            new_individual[10], new_individual[9] = F(new_individual[10]), nF(new_individual[9])
          def m14():
            new_individual[1], new_individual[3] = F(new_individual[1]), nF(new_individual[3])
          def m15():
            new_individual[3], new_individual[1] = F(new_individual[3]), nF(new_individual[1])
          random.choice([m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13])()
      template = self.build_template(self.TemplateType.Parameters(*new_individual))
      if not tools.skip(template, self.statement, self.device):
        break
    return new_individual,
      
  def evaluate(self, individual):
    if tuple(individual) not in self.cache:
      template = self.build_template(self.TemplateType.Parameters(*individual))
      try:
        self.cache[tuple(individual)] = tools.benchmark(template, self.statement, self.device)
      except:
        self.cache[tuple(individual)] = 10
    return self.cache[tuple(individual)],

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, maxtime, maxgen, halloffame, compute_perf, perf_metric):
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    # Begin the generational process
    gen = 0
    maxtime = time.strptime(maxtime, '%Mm%Ss')
    maxtime = maxtime.tm_min*60 + maxtime.tm_sec
    start_time = time.time()
    while time.time() - start_time < maxtime and gen < maxgen:
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        gen = gen + 1
        
        best_profile = '(%s)'%','.join(map(str,halloffame[0]));
        best_performance = compute_perf(halloffame[0].fitness.values[0])
        sys.stdout.write('Generation %d | Time %d | Best %d %s [ for %s ]\r'%(gen, time.time() - start_time, best_performance, perf_metric, best_profile))
        sys.stdout.flush()
    sys.stdout.write('\n')
    return population
