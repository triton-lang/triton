import isaac as sc
import random

from copy import deepcopy
from sys import stdout
from itertools import product

from external.deap import base
from external.deap import creator
from external.deap import tools as deap_tools

from numpy import cumsum

import tools
from tools import profile_execution_failure
from time import sleep

fetch_types = [sc.templates.fetching_policy_type.FETCH_FROM_GLOBAL_CONTIGUOUS,
               sc.templates.fetching_policy_type.FETCH_FROM_GLOBAL_STRIDED,
               sc.templates.fetching_policy_type.FETCH_FROM_LOCAL,
               sc.templates.fetching_policy_type.FETCH_FROM_LOCAL]

def exhaustive(template, sizes, context):
    tree, _ = tools.tree_of(template, sizes, context)
    metric = tools.metric_of(template)
    nbits = tools.genetic_infos_of(template)['nbits']
    categorical = tools.genetic_infos_of(template)['categorical']
    ranges = [range(2**x) for x in nbits]
    ranges = list(product(*ranges))
    timings = {}
    best = None
    for idx, r in enumerate(ranges):
        parameters = tuple([fetch_types[x] if i in categorical else 2**x for i,x in enumerate(r)])
        try:
            time = tools.benchmark(template, parameters, tree)
            if not best or time < best[1]:
                best = parameters, time
        except profile_execution_failure:
            pass
        if best:
            stdout.write('%.2f %% | Best %.2f [ for %s ]\r'%(float(idx*100)/len(ranges),metric(sizes, best[1]), best[0]))
    return best[0]
        

class GeneticOptimizer:
    
    def __init__(self, logger, naccept=500, niter=1000, cxpb=.4, mutpb=.4, popsize=10, progress_bar = None):
        self.logger = logger
        self.naccept = naccept
        self.niter = niter
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.popsize = popsize
        self.progress_bar = progress_bar
        
    def run(self, template, sizes, context, initializer = None, prior = None):
        tree, _ = tools.tree_of(template, sizes, context)
        metric = tools.metric_of(template)
        genetic_infos = tools.genetic_infos_of(template)
        nbits = genetic_infos['nbits']
        offsets = cumsum([0] + nbits)

        def bin2gray(A):
            g = [int(A[0])]
            for i in range(1, len(A)): 
                g += [int(A[i-1] != A[i])]
            return g
        
        def gray2int(A):
            b = [A[0]]
            for i in range(1, len(A)):
                b += [int(b[i-1] != A[i])]
            return int(''.join(map(str,b)), 2)
        
        def encode(genome):
            encoded = [bin2gray(bin(x)[2:].zfill(nb)) for x, nb in zip(genome, nbits)]
            return sum(encoded, [])
            
        def decode(genome):
            result = []
            for off1,off2 in zip(offsets[:-1],offsets[1:]):
                result += [gray2int(genome[off1:off2])]
            result = [fetch_types[x] if i in genetic_infos['categorical'] else 2**x for i,x in enumerate(result)]
            return result

        def evaluate(genome):
            idx = tuple(genome)
            if idx not in cache:
                cache[idx] = tools.benchmark(template, decode(genome), tree)
            self.progress_bar.update(max(len(cache), it), self.niter, decode(min(cache, key=cache.get)), metric(sizes, min(cache.values())))
            return cache[idx],
            
        cache = {}
        hof = deap_tools.HallOfFame(1)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", deap_tools.cxTwoPoint)
        toolbox.register("mutate", deap_tools.mutFlipBit)
        toolbox.register("select", deap_tools.selNSGA2)

        x = []
        y = []
        it = 0
        
        population = [] 
        #Initialization
        if initializer is None:
            initializer = ([random.randint(0, 2**x) for x in nbits] for i in iter(int,1))
        genome = encode(prior if prior else list(initializer.next()))
        while len(population) < self.popsize:
            individual = creator.Individual(genome)
            try:
                individual.fitness.values = toolbox.evaluate(genome)
                population += [individual]
            except profile_execution_failure:
                pass
            genome = encode(list(initializer.next()))
        hof.update(population)
        
        #Main iteration
        while len(cache) < self.naccept and it<self.niter:
            
            #Generate offspring
            offspring = []
            while len(offspring) < self.popsize:
                try:
                    op_choice = random.random()
                    #Cross-over
                    if op_choice < self.cxpb: 
                        ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
                        ind1, ind2 = toolbox.mate(ind1, ind2)
                        ind = ind1
                        toolbox.evaluate(ind)
                        offspring += [ind]
                    #Mutation
                    elif op_choice < self.cxpb + self.mutpb: 
                        ind = toolbox.clone(random.choice(population))
                        ind, = toolbox.mutate(ind, 1.0/offsets[-1])
                        toolbox.evaluate(ind)
                        offspring += [ind]
                    #Reproduction
                    else: 
                        offspring += [random.choice(population)]
                except profile_execution_failure:
                    pass

            #Update fitnesses
            fitnesses = toolbox.map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit
                
            #Update population
            population[:] = toolbox.select(population + offspring, self.popsize)
            hof.update(population)
            
            it += 1
        return tuple(decode(hof[0])), x, y
        
def is_local_optimum(parameters, template, sizes, context):
    tree, _ = tools.tree_of(template, sizes, context)
    genetic_infos = tools.genetic_infos_of(template)
    
    if issubclass(template, sc.templates.elementwise_1d):
        sweep_over = [0,1,2]
    elif issubclass(template, sc.templates.reduce_1d):
        sweep_over = [0,1,2]
    elif issubclass(template, sc.templates.elementwise_2d):
        sweep_over = [0,1,2,3,4]
    elif issubclass(template, sc.templates.reduce_2d):
        sweep_over = [0,1,2,3,4]
    elif issubclass(template, sc.templates.matrix_product):
        sweep_over = [0,1,3,4,5,7]
    
    #Evaluate the provided parameters guess
    try:
        reference = tools.benchmark(template, parameters, tree)
    except profile_execution_failure:
        return False
        
    #Latency bound -- ignore
    if reference < 1e-5:
        return True
        
    timings = {}
    domain = [[v  for v in [x/2, x, x*2] if 1 <= v <= 2**2**genetic_infos['nbits'][i]] \
              if i in sweep_over else [x] for i, x in enumerate(parameters)]
    for x in product(*domain):
        if x==parameters:
            pass
        try:
            time = tools.benchmark(template, x, tree)
            if time/reference < .97:
                return False
        except profile_execution_failure:
            pass
    return True
    
    
