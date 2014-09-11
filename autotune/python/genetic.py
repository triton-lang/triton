import random
import time
import sys
import tools
import pyviennacl as vcl
import numpy

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

  @staticmethod
  def min_to_hyperbol(a, tup):
    x = 1
    for i in range(100):
      dx = 2*(-a**2/x**3 + a*tup[1]/x**2 - tup[0] + x);
      ddx = 6*a**2/x**4 - 4*a*tup[1]/x**3 + 2;
      if abs(dx) < 1e-7 or abs(ddx) < 1e-7:
        break
      x-=dx/ddx; 
      if x<1 or x>a:
        x = max(1, min(x, a))
        break
    new_x = int(closest_divisor(a, x))
    new_y = int(a / new_x)
    return (new_x, new_y)
      
  def repair(self,func):
    
    def repair_impl(child):
      D = odict(zip(self.parameter_names, child))
      dummy_template = self.build_template(self.ParameterType(*D.values()))
      FetchingPolicy = vcl.atidlas.FetchingPolicy;
      D['local-size-0'] = max(1, D['local-size-0'])
      D['local-size-1'] = max(1, D['local-size-1'])
      if 'local-size-1' not in D:
        D['local-size-0'] = min(D['local-size-0'], self.device.max_work_group_size)
      elif D['local-size-0']*D['local-size-1'] > self.device.max_work_group_size:
        res = GeneticOperators.min_to_hyperbol(self.device.max_work_group_size, (D['local-size-0'], D['local-size-1']))
        D['local-size-0'] = res[0]
        D['local-size-1'] = res[1]
      
      if self.ParameterType is vcl.atidlas.MatrixProductTemplate.Parameters:
        if dummy_template.A_trans != 'N' and dummy_template.B_trans != 'T':
          D['simd-width'] = 1
        
        D['kL'] = max(1, D['kL'])
        D['kS'] = max(1, D['kS'])
        
        D['mS'] = max(D['mS'], D['simd-width'])
        D['nS'] = max(D['nS'], D['simd-width'])
        D['mS'] = D['mS'] - D['mS']%D['simd-width']
        D['nS'] = D['nS'] - D['nS']%D['simd-width']
        
        
        if D['A-fetch-policy']!=FetchingPolicy.FETCH_FROM_LOCAL and D['B-fetch-policy']!=FetchingPolicy.FETCH_FROM_LOCAL:
          D['local-fetch-size-0']=D['local-fetch-size-1']=0
        
        else:
          res = GeneticOperators.min_to_hyperbol(D['local-size-0']*D['local-size-1'], (D['local-fetch-size-0'], D['local-fetch-size-1']))
          D['local-fetch-size-0'] = res[0]
          D['local-fetch-size-1'] = res[1]      
        
        if D['A-fetch-policy']==FetchingPolicy.FETCH_FROM_LOCAL and dummy_template.A_trans=='N' and D['kL'] % D['local-fetch-size-1'] > 0:
          D['kL'] = max(1,round(D['kL']/D['local-fetch-size-1']))*D['local-fetch-size-1']
        
        if D['B-fetch-policy']==FetchingPolicy.FETCH_FROM_LOCAL and dummy_template.B_trans=='T' and D['kL'] % D['local-fetch-size-1'] > 0:
          D['kL'] = max(1,round(D['kL']/D['local-fetch-size-1']))*D['local-fetch-size-1']
          
        D['kS'] = min(D['kL'], D['kS'])
      
      return D.values()
      
    def wrappper(*args, **kargs):
      offspring = func(*args, **kargs)
      for child in offspring:
        new_child = repair_impl(child)
        for i in range(len(child)):
          if child[i] != new_child[i]:
            child[i] = new_child[i]
      return offspring
    return wrappper

  def mutate(self, individual, indpb = 0.15):
    for i in individual:
      if random.random() < indpb:
        coef = 2**(1 + numpy.random.poisson())
        funs = [lambda x:x/coef, lambda x:x*coef]
        F = random.choice(funs)
        nF = funs[1] if F==funs[0] else funs[0]
        #swapping-based mutations
        def m0():
          individual[1], individual[3] = individual[3], individual[1]
        def m1():
          individual[4], individual[6] = individual[6], individual[4]
        def m2():
          individual[9], individual[10] = individual[10], individual[9]
        #value modification mutations
        def m3():
          individual[0] = random.choice(self.parameters[0])
        def m4():
          individual[1] = F(individual[1])
          individual[9] = F(individual[9])
        def m5():
          individual[2] = F(individual[2])
        def m6():
          individual[3] = F(individual[3])
          individual[10] = F(individual[10])
        def m7():
          individual[4] = F(individual[4])
        def m8():
          individual[5] = F(individual[5])
        def m9():
          individual[6] = F(individual[6])
        def m10():
          individual[7] = random.choice([x for x in self.parameters[7] if x!=individual[7]])
        def m11():
          individual[8] = random.choice([x for x in self.parameters[8] if x!=individual[8]])
        def m12():
          individual[9] = F(individual[9])
          individual[10] = nF(individual[10])
        def m13():
          individual[10] = F(individual[10])
          individual[9] = nF(individual[9])
        random.choice([m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13])()
    return individual,
      
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
        sys.stdout.write('Generation %d | Time %d | Best %d %s [ for %s ]\n'%(gen, time.time() - start_time, best_performance, perf_metric, best_profile))
    sys.stdout.write('\n')
    return population
