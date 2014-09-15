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
  
def b_gray_to_bin(A='00000000', endian='big'):
    assert type(endian) is str
    assert endian == 'little' or endian == 'big'
    if endian == 'little': A = A[::-1] # Make sure endianness is big before conversion
    b = A[0]
    for i in range(1, len(A)): b += str( int(b[i-1] != A[i]) )
    assert len(A) == len(b), 'Error in this function! len(A) must equal len(b). Oh dear.'
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
      self.indpb = 0.1
  
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
    
  def init(self):
    while True:
      result = [random.randint(0,2), random.randint(0,2)] + [str(random.randint(0,1)) for i in range(23)]
      template = self.build_template(self.TemplateType.Parameters(*self.decode(result)))
      registers_usage = template.registers_usage(vcl.atidlas.StatementsTuple(self.statement))/4
      lmem_usage = template.lmem_usage(vcl.atidlas.StatementsTuple(self.statement))
      local_size = template.parameters.local_size_0*template.parameters.local_size_1
      occupancy_record = tools.OccupancyRecord(self.device, local_size, lmem_usage, registers_usage)
      if not tools.skip(template, self.statement, self.device):
        return result

  def mutate(self, individual):
    while True:
      new_individual = copy.deepcopy(individual)
      for i in range(len(new_individual)):
        if random.random() < self.indpb:
          if i < 2:
            while new_individual[i] == individual[i]:
              new_individual[i] = random.randint(0, 2)
          else:
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
        
        best_profile = '(%s)'%','.join(map(str,GeneticOperators.decode(halloffame[0])));
        best_performance = compute_perf(halloffame[0].fitness.values[0])
        sys.stdout.write('Generation %d | Time %d | Best %d %s [ for %s ]\n'%(gen, time.time() - start_time, best_performance, perf_metric, best_profile))
        sys.stdout.flush()
    sys.stdout.write('\n')
    return population
