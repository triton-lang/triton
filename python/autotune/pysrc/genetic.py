import random, time, sys, copy
import misc_tools

import numpy as np
import pyisaac as atd
from deap import algorithms
from deap import base
from deap import creator
from deap import tools as deap_tools

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
    if endian == 'little': b = b[::-1] # Convert back to little endian if necessary
    return b

class GeneticOperators(object):

    class Pow2(object):
        def __init__(self, v):
            self.value = v
        
        @property
        def decoded():
            return 2**self.value
    
    def __init__(self, symbolic, Template, out):
        self.device = symbolic.context.queues[0].device
        self.symbolic = symbolic
        self.Template = Template
        self.cache = {}
        self.out = out

  
        self.genome_info = {
                            atd.vaxpy: [2,4,4,atd.fetching_policy_type],
                            atd.reduction: [2,4,4,atd.fetching_policy_type],
                            atd.maxpy: [2,3,3,3,3,atd.fetching_policy_type],
                            atd.mreduction_rows: [2,3,3,3,3,atd.fetching_policy_type],
                            atd.mreduction_cols: [2,3,3,3,3,atd.fetching_policy_type],
                            atd.mproduct_nn: [2,3,3,3,3,3,3,3,atd.fetching_policy_type,atd.fetching_policy_type,3],
                            atd.mproduct_nt: [2,3,3,3,3,3,3,3,atd.fetching_policy_type,atd.fetching_policy_type,3],
                            atd.mproduct_tn: [2,3,3,3,3,3,3,3,atd.fetching_policy_type,atd.fetching_policy_type,3],
                            atd.mproduct_tt: [2,3,3,3,3,3,3,3,atd.fetching_policy_type,atd.fetching_policy_type,3]
                           }[Template]
        self.indpb = 1.0/sum([1 if x==atd.fetching_policy_type else x for x in self.genome_info])

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("population", self.init)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", deap_tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", deap_tools.selNSGA2)

    def decode(self, genome):
        fetching_policy_type = atd.fetching_policy_type
        fetch = [fetching_policy_type.FETCH_FROM_LOCAL, fetching_policy_type.FETCH_FROM_GLOBAL_STRIDED, fetching_policy_type.FETCH_FROM_GLOBAL_CONTIGUOUS]
        is_gemm = self.Template in [atd.mproduct_nn, atd.mproduct_nt, atd.mproduct_tn, atd.mproduct_tt]
        result = []
        offset = 0
        for i, x in enumerate(self.genome_info):
            if x==atd.fetching_policy_type:
                result.append(fetch[genome[offset]])
                offset = offset + 1
            else:
                decoded = int(b_gray_to_bin(''.join(genome[offset:offset+x])), 2)
                result.append(decoded if is_gemm and  i in [11, 12] else 2**decoded)
                offset = offset + x
        #GEMM peculiarities
        if is_gemm:
            if fetching_policy_type.FETCH_FROM_LOCAL in result:
                lf1 = result[1]*result[3]/result[10]
            else:
                result[10] = 0
                lf1 = 0
            result.append(lf1)
        return result

    def init(self, N):
        result = []
        allowed_idx = [0] if self.Template in [atd.mproduct_nn, atd.mproduct_nt, atd.mproduct_tn, atd.mproduct_tt] else [1,2]
        for idx in allowed_idx:
            current = []
            while len(current) < N/len(allowed_idx):
                while True:
                    bincode = []
                    for i, x in enumerate(self.genome_info):
                        if x==atd.fetching_policy_type:
                            bincode = bincode + [idx]
                        else:
                            bincode = bincode + [str(random.randint(0,1)) for i in range(x)]
                    parameters = self.decode(bincode)
                    template = self.Template(*parameters)
                    array_expressions = atd.array_expression_container(self.symbolic)
                    registers_usage = template.registers_usage(array_expressions)/4
                    lmem_usage = template.lmem_usage(array_expressions)
                    local_size = parameters[1]*parameters[3]
                    occupancy_record = misc_tools.OccupancyRecord(self.device, local_size, lmem_usage, registers_usage)
                    if not misc_tools.skip(template, self.symbolic):
                        current.append(creator.Individual(bincode))
                        break
            result = result + current
        return result

    def mutate(self, individual):
        while True:
            new_individual = copy.deepcopy(individual)
            for i in range(len(new_individual)):
                if isinstance(individual[i], int) and random.random() < 0.1:
                    while new_individual[i] == individual[i]:
                        new_individual[i] = random.randint(0, 2)
                elif not isinstance(individual[i], int) and random.random() < self.indpb:
                    new_individual[i] = '1' if new_individual[i]=='0' else '0'
            parameters = self.decode(new_individual)
            template = self.Template(*parameters)
            if not misc_tools.skip(template, self.symbolic):
                break
        return new_individual,

    def evaluate(self, individual):
        if tuple(individual) not in self.cache:
            parameters = self.decode(individual)
            template = self.Template(*parameters)
            tt = misc_tools.benchmark(template, self.symbolic)
            self.out.write(','.join([str(tt)]+map(str,map(int,parameters)))+'\n')
            self.cache[tuple(individual)] = tt
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

        while time.time() - start_time < maxtime and gen < maxgen:
            # Vary the population
            offspring = []
            for _ in xrange(mu):
                op_choice = random.random()
                if op_choice < cxpb:            # Apply crossover
                    while True:
                        ind1, ind2 = map(self.toolbox.clone, random.sample(population, 2))
                        ind1, ind2 = self.toolbox.mate(ind1, ind2)
                        del ind1.fitness.values
                        parameters = self.decode(ind1)
                        template = self.Template(*parameters)
                        if not misc_tools.skip(template, self.symbolic):
                            break
                    offspring.append(ind1)
                elif op_choice < cxpb + mutpb:  # Apply mutation
                    ind = self.toolbox.clone(random.choice(population))
                    ind, = self.toolbox.mutate(ind)
                    del ind.fitness.values
                    offspring.append(ind)
                else:                           # Apply reproduction
                    offspring.append(random.choice(population))
            #for x in offspring:
                    #print self.decode(x)
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
            best_profile = '(%s)'%','.join(map(str,self.decode(hof[0])))
            best_performance = compute_perf(hof[0].fitness.values[0])
            sys.stdout.write('Generation %d | Time %d | Best %d %s [ for %s ]\r'%(gen, time.time() - start_time, best_performance, perf_metric, best_profile))
            sys.stdout.flush()
        sys.stdout.write('\n')
        return self.decode(hof[0])
