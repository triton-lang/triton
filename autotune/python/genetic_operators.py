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
        if template.check(self.statement) and occupancy_record.occupancy >= 10 :
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
      if 'local-size-1' not in D:
        D['local-size-0'] = min(D['local-size-0'], self.device.max_work_group_size)
      elif D['local-size-0']*D['local-size-1'] > self.device.max_work_group_size:
        res = GeneticOperators.min_to_hyperbol(self.device.max_work_group_size, (D['local-size-0'], D['local-size-1']))
        D['local-size-0'] = res[0]
        D['local-size-1'] = res[1]
      
      if self.ParameterType is vcl.atidlas.MatrixProductTemplate.Parameters:
        if dummy_template.A_trans != 'N' and dummy_template.B_trans != 'T':
          D['simd-width'] = 1
          
        D['mS'] = max(D['mS'], D['simd-width'])
        D['mS'] = D['mS'] - D['mS']%D['simd-width']
        
        D['nS'] = max(D['nS'], D['simd-width'])
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

  def mutate(self, individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            j = self.parameters[i].index(individual[i])
            j = max(0,min(random.randint(j-1, j+1),len(self.parameters[i])-1))
            individual[i] = self.parameters[i][j]
    return individual,
      
  def evaluate(self, individual):
    tupindividual = tuple(individual)
    if tupindividual not in self.cache:
      template = self.build_template(self.TemplateType.Parameters(*individual))
      registers_usage = template.registers_usage(vcl.atidlas.StatementsTuple(self.statement))/4
      lmem_usage = template.lmem_usage(vcl.atidlas.StatementsTuple(self.statement))
      local_size = template.parameters.local_size_0*template.parameters.local_size_1
      occupancy_record = tools.OccupancyRecord(self.device, local_size, lmem_usage, registers_usage)
      if occupancy_record.occupancy < 10 :
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
