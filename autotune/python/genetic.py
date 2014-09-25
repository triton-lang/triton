import random
import time
import sys
import tools
import pyviennacl as vcl
import numpy as np
import copy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools as deap_tools

from collections import OrderedDict as odict

def hamming_distance(ind1, ind2):
  res = 0
  for x,y in enumerate(ind1, ind2):
    if x==y:
      res = res + 1
  return res
  
def closest_divisor(N, x):
  x_low=x_high=max(1,min(round(x),N))
  while N % x_low > 0 and x_low>0:
    x_low = x_low - 1
  while N % x_high > 0 and x_high < N:
    x_high = x_high + 1
  return x_low if x - x_low < x_high - x else x_high
  
def b_gray_to_bin(A='00000000', endian='big'):
    assert type(endian) is str
    assert endian == 'little' or endian == 'big'
    if endian == 'little': A = A[::-1] # Make sure endianness is big before conversion
    b = A[0]
    for i in range(1, len(A)): b += str( int(b[i-1] != A[i]) )
    if endian == 'little': b = b[::-1] # Convert back to little endian if necessary
    return b
      
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
      self.indpb = 0.05
      
      creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
      creator.create("Individual", list, fitness=creator.FitnessMin)
    
      self.toolbox = base.Toolbox()
      self.toolbox.register("population", self.init)
      self.toolbox.register("evaluate", self.evaluate)
      self.toolbox.register("mate", deap_tools.cxTwoPoint)
      self.toolbox.register("mutate", self.mutate)
      self.toolbox.register("select", deap_tools.selNSGA2)

  @staticmethod
  def decode(s):
    FetchingPolicy = vcl.atidlas.FetchingPolicy
    fetch = [FetchingPolicy.FETCH_FROM_LOCAL, FetchingPolicy.FETCH_FROM_GLOBAL_CONTIGUOUS, FetchingPolicy.FETCH_FROM_GLOBAL_STRIDED]
    fetchA = fetch[s[0]]
    fetchB = fetch[s[1]]
    bincode = ''.join(s[2:])
    decode_element = lambda x:2**int(b_gray_to_bin(x), 2)
    simd = decode_element(bincode[0:3])
    ls0 = decode_element(bincode[2:5])
    ls1 = decode_element(bincode[5:8])
    kL = decode_element(bincode[8:11])
    mS = decode_element(bincode[11:14])
    kS = decode_element(bincode[14:17])
    nS = decode_element(bincode[17:20])
    if fetchA==FetchingPolicy.FETCH_FROM_LOCAL or fetchB==FetchingPolicy.FETCH_FROM_LOCAL:
      lf0 = decode_element(bincode[20:23])
      lf1 = ls0*ls1/lf0
    else:
      lf0, lf1 = 0, 0
    return [simd, ls0, kL, ls1, mS, kS, nS, fetchA, fetchB, lf0, lf1]
    
  def init(self, N):
    result = []
    fetchcount = [0, 0, 0]
    while len(result) < N:
      while True:
        fetch = random.randint(0,2)
        bincode = [fetch, fetch] + [str(random.randint(0,1)) for i in range(23)]
        parameters = self.decode(bincode)
        template = self.build_template(self.TemplateType.Parameters(*parameters))
        registers_usage = template.registers_usage(vcl.atidlas.StatementsTuple(self.statement))/4
        lmem_usage = template.lmem_usage(vcl.atidlas.StatementsTuple(self.statement))
        local_size = template.parameters.local_size_0*template.parameters.local_size_1
        occupancy_record = tools.OccupancyRecord(self.device, local_size, lmem_usage, registers_usage)
        if not tools.skip(template, self.statement, self.device):
          fetchcount[fetch] = fetchcount[fetch] + 1
          if max(fetchcount) - min(fetchcount) <= 1:
            result.append(creator.Individual(bincode))
            break
          else:
            fetchcount[fetch] = fetchcount[fetch] - 1
    return result

  def mutate(self, individual):
    while True:
      new_individual = copy.deepcopy(individual)
      for i in range(len(new_individual)):
        if i < 2 and random.random() < 0.1:
          while new_individual[i] == individual[i]:
            new_individual[i] = random.randint(0, 2)
        elif i >= 2 and random.random() < self.indpb:
          new_individual[i] = '1' if new_individual[i]=='0' else '0'
      parameters = self.decode(new_individual)
      template = self.build_template(self.TemplateType.Parameters(*parameters))
      #print tools.skip(template, self.statement, self.device), parameters
      if not tools.skip(template, self.statement, self.device):
        break
    return new_individual,
      
  def evaluate(self, individual):
    if tuple(individual) not in self.cache:
      parameters = self.decode(individual)
      template = self.build_template(self.TemplateType.Parameters(*parameters))
      try:
        self.cache[tuple(individual)] = tools.benchmark(template, self.statement, self.device)
      except:
        self.cache[tuple(individual)] = 10
    return self.cache[tuple(individual)],
        
  def optimize(self, maxtime, maxgen, compute_perf, perf_metric):
      hof = deap_tools.HallOfFame(1)
      # Begin the generational process
      gen = 0
      maxtime = time.strptime(maxtime, '%Mm%Ss')
      maxtime = maxtime.tm_min*60 + maxtime.tm_sec
      start_time = time.time()
      
      mu = 30
      cxpb = 0.2
      mutpb = 0.7
      
      population = self.init(mu)
      invalid_ind = [ind for ind in population if not ind.fitness.valid]
      fitnesses = self.toolbox.map(self.evaluate, invalid_ind)
      for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
      hof.update(population)
      
      while time.time() - start_time < maxtime:
        # Vary the population        
        offspring = []
        for _ in xrange(mu):
            op_choice = random.random()
            if op_choice < cxpb:            # Apply crossover
                ind1, ind2 = map(self.toolbox.clone, random.sample(population, 2))
                ind1, ind2 = self.toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                offspring.append(ind1)
            elif op_choice < cxpb + mutpb:  # Apply mutation
                ind = self.toolbox.clone(random.choice(population))
                ind, = self.toolbox.mutate(ind)
                del ind.fitness.values
                offspring.append(ind)
            else:                           # Apply reproduction
                offspring.append(random.choice(population))
                
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # Update the hall of fame with the generated individuals
        hof.update(offspring)
        # Select the next generation population
        population[:] = self.toolbox.select(population + offspring, mu)
        #Update
        gen = gen + 1
        best_profile = '(%s)'%','.join(map(str,GeneticOperators.decode(hof[0])));
        best_performance = compute_perf(hof[0].fitness.values[0])
        sys.stdout.write('Time %d | Best %d %s [ for %s ]\r'%(time.time() - start_time, best_performance, perf_metric, best_profile))
        sys.stdout.flush()
      sys.stdout.write('\n')
      return population
    
          
