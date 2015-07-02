import isaac as isc
import random

from copy import deepcopy
from sys import stdout
from itertools import product

from deap import algorithms
from deap import base
from deap import creator
from deap import tools as deap_tools

from numpy import cumsum

import tools

fetch_types = [isc.fetching_policy_type.FETCH_FROM_LOCAL,
               isc.fetching_policy_type.FETCH_FROM_LOCAL,
               isc.fetching_policy_type.FETCH_FROM_LOCAL,
               isc.fetching_policy_type.FETCH_FROM_LOCAL]

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
        except (isc.OperationNotSupported, isc.LaunchOutOfResources, isc.MemObjectAllocationFailure):
            pass
        if best:
            stdout.write('%.2f %% | Best %.2f [ for %s ]\r'%(float(idx*100)/len(ranges),metric(sizes, best[1]), best[0]))
    return best[0]
        
       
def genetic(template, sizes, context, naccept=200, niter = 1000, cxpb=0.4, mutpb=0.4, popsize = 10, initializer = None, prior = None):
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

    #Initialization
    if initializer is None:
        initializer = ([random.randint(0, 2**x) for x in nbits] for i in iter(int,1))
    population = [] 

    genome = encode(prior if prior else list(initializer.next()))
    while len(population) < popsize:
        individual = creator.Individual(genome)
        try:
            individual.fitness.values = toolbox.evaluate(genome)
            population += [individual]
        except (isc.OperationNotSupported, isc.LaunchOutOfResources, isc.MemObjectAllocationFailure ):
            pass
        genome = encode(list(initializer.next()))
    hof.update(population)
    
    x = []
    y = []
    it = 0
    
    while len(cache) < naccept and it<niter:
        pad = len(cache) - len(x)
        x += [len(cache)]*pad
        y += [metric(sizes, hof[0].fitness.values[0])]*pad
        
        offspring = []
        while len(offspring) < popsize:
            try:
                op_choice = random.random()
                #Cross-over
                if op_choice < cxpb: 
                    ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
                    ind1, ind2 = toolbox.mate(ind1, ind2)
                    ind = ind1
                    toolbox.evaluate(ind)
                    offspring += [ind]
                #Mutation
                elif op_choice < cxpb + mutpb: 
                    ind = toolbox.clone(random.choice(population))
                    ind, = toolbox.mutate(ind, 1.0/offsets[-1])
                    toolbox.evaluate(ind)
                    offspring += [ind]
                #Reproduction
                else: 
                    offspring += [random.choice(population)]
            except (isc.OperationNotSupported, isc.LaunchOutOfResources, isc.MemObjectAllocationFailure):
                pass


        #Update fitnesses
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
            
        #Update population
        population[:] = toolbox.select(population + offspring, popsize)
        hof.update(population)
        
        optimal = '(%s)'%','.join(map(str,decode(hof[0])))
        stdout.write('Iter %d | %d evaluated | Best %.2f [ for %s ]\r'%(it, x[-1], y[-1], optimal))
        stdout.flush()
        it += 1
    stdout.write('\n')

    return tuple(decode(hof[0])), x, y
    
def is_local_optimum(parameters, template, sizes, context):
    tree, _ = tools.tree_of(template, sizes, context)
    genetic_infos = tools.genetic_infos_of(template)
    
    if issubclass(template, isc.vaxpy):
        sweep_over = [0,1,2]
    elif issubclass(template, isc.reduction):
        sweep_over = [0,1,2]
    elif issubclass(template, isc.maxpy):
        sweep_over = [0,1,2,3,4]
    elif issubclass(template, isc.mreduction):
        sweep_over = [0,1,2,3,4]
    elif issubclass(template, isc.mproduct):
        sweep_over = [1,2,3,4,5,7,10,11]
    
    #Evaluate the provided parameters guess
    try:
        reference = tools.benchmark(template, parameters, tree)
    except (isc.OperationNotSupported, isc.LaunchOutOfResources, isc.MemObjectAllocationFailure):
        return False
        
    #Latency bound -- ignore
    if reference < 2e-5:
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
        except (isc.OperationNotSupported, isc.LaunchOutOfResources, isc.MemObjectAllocationFailure):
            pass
    return True
    
    
