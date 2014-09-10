import random
import time
import tools
import pyviennacl as vcl

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

  def mutate(self, individual):
    p = random.random()
    coef = random.choice([2, 4])
    multiply_or_divide = random.choice([lambda x:x/coef, lambda x:x*coef])
    if p < 0.1 :
      individual[0] = random.choice(self.parameters[0])
    elif p < 0.2:
      idx = random.choice([1, 4])
      nidx = 4 if idx==1 else 1
      individual[idx] = individual[idx]*coef
      individual[nidx] = individual[nidx]/coef
    elif p < 0.3:
      idx = random.choice([1, 4])
      individual[idx] = multiply_or_divide(individual[idx])
      if idx==1:
        individual[9] = multiply_or_divide(individual[9])
    elif p < 0.4:
      idx = random.choice([3, 6])
      nidx = 6 if idx==3 else 3
      individual[idx] = individual[idx]*coef
      individual[nidx] = individual[nidx]/coef
    elif p < 0.5:
      idx = random.choice([3, 6])
      individual[idx] = multiply_or_divide(individual[idx])
      if idx==3:
        individual[10] = multiply_or_divide(individual[10])
    elif p < 0.6:
      individual[2] = multiply_or_divide(individual[2])
    elif p < 0.7:
      individual[4] = multiply_or_divide(individual[4])
    elif p < 0.8:
      individual[7] = random.choice([x for x in self.parameters[7] if x!=individual[7]])
    elif p < 0.9:
      individual[8] = random.choice([x for x in self.parameters[8] if x!=individual[8]])
    elif p < 1:
      idx = random.choice([9, 10])
      nidx = 10 if idx==9 else 9
      individual[idx] = individual[idx]*coef
      individual[nidx] = individual[nidx]/coef
    return individual,
      
  def evaluate(self, individual):
    tupindividual = tuple(individual)
    if tupindividual not in self.cache:
      template = self.build_template(self.TemplateType.Parameters(*individual))
      registers_usage = template.registers_usage(vcl.atidlas.StatementsTuple(self.statement))/4
      lmem_usage = template.lmem_usage(vcl.atidlas.StatementsTuple(self.statement))
      local_size = template.parameters.local_size_0*template.parameters.local_size_1
      occupancy_record = tools.OccupancyRecord(self.device, local_size, lmem_usage, registers_usage)
      if occupancy_record.occupancy < 15 :
        self.cache[tupindividual] = 10
      else:
        try:
          template.execute(self.statement, True)
          self.statement.result.context.finish_all_queues()
          N = 0
          current_time = 0
          while current_time < 1e-2:
            time_before = time.time()
            template.execute(self.statement,False)
            self.statement.result.context.finish_all_queues()
            current_time += time.time() - time_before
            N+=1
          self.cache[tupindividual] = current_time/N
        except:
          self.cache[tupindividual] = 10
    return self.cache[tupindividual],
