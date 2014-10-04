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

    def __init__(self, device, statement, TemplateType, build_template, out):
        self.device = device
        self.statement = statement
        self.TemplateType = TemplateType
        self.ParameterType = TemplateType.Parameters
        self.build_template = build_template
        self.cache = {}
        self.out = out

        self.genome_info = {
                            vcl.atidlas.VectorAxpyTemplate: [3,4,4,vcl.atidlas.FetchingPolicy],
                            vcl.atidlas.MatrixAxpyTemplate: [3,3,3,3,3,vcl.atidlas.FetchingPolicy],
                            vcl.atidlas.RowWiseReductionTemplate: [3,3,3,4,vcl.atidlas.FetchingPolicy],
                            vcl.atidlas.MatrixProductTemplate: [3,3,3,3,3,3,3,vcl.atidlas.FetchingPolicy,vcl.atidlas.FetchingPolicy,3]
                           }[TemplateType]
        self.indpb = 1.0/sum([1 if x==vcl.atidlas.FetchingPolicy else x for x in self.genome_info])

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("population", self.init)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", deap_tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", deap_tools.selNSGA2)

    def decode(self, genome):
        FetchingPolicy = vcl.atidlas.FetchingPolicy
        fetch = [FetchingPolicy.FETCH_FROM_LOCAL, FetchingPolicy.FETCH_FROM_GLOBAL_CONTIGUOUS, FetchingPolicy.FETCH_FROM_GLOBAL_STRIDED]
        decode_element = lambda x:2**int(b_gray_to_bin(''.join(x)), 2)
        result = []
        offset = 0
        for x in self.genome_info:
            if x==vcl.atidlas.FetchingPolicy:
                result.append(fetch[genome[offset]])
                offset = offset + 1
            else:
                result.append(decode_element(genome[offset:offset+x]))
                offset = offset + x
        #GEMM peculiarities
        if self.TemplateType==vcl.atidlas.MatrixProductTemplate:
            if FetchingPolicy.FETCH_FROM_LOCAL in result:
                lf1 = result[1]*result[3]/result[9]
            else:
                result[9] = 0
                lf1 = 0
            result.append(lf1)
        return result

    def init(self, N):
        result = []
        while len(result) < N:
            while True:
                bincode = []
                for x in self.genome_info:
                    if x==vcl.atidlas.FetchingPolicy:
                        bincode = bincode + [random.randint(0,2)]
                    else:
                        bincode = bincode + [str(random.randint(0,1)) for i in range(x)]
                parameters = self.decode(bincode)
                template = self.build_template(self.TemplateType.Parameters(*parameters))
                registers_usage = template.registers_usage(vcl.atidlas.StatementsTuple(self.statement))/4
                lmem_usage = template.lmem_usage(vcl.atidlas.StatementsTuple(self.statement))
                local_size = template.parameters.local_size_0*template.parameters.local_size_1
                occupancy_record = tools.OccupancyRecord(self.device, local_size, lmem_usage, registers_usage)
                if not tools.skip(template, self.statement, self.device):
                    result.append(creator.Individual(bincode))
                    break
        return result

    def mutate(self, individual):
        while True:
            new_individual = copy.deepcopy(individual)
            for i in range(len(new_individual)):
                if isinstance(individual[i], int) and random.random() < self.indpb:
                    while new_individual[i] == individual[i]:
                        new_individual[i] = random.randint(0, 2)
                elif not isinstance(individual[i], int) and random.random() < self.indpb:
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
                tt = tools.benchmark(template, self.statement, self.device)
                self.out.write(','.join([str(tt)]+map(str,map(int,parameters)))+'\n')
                self.cache[tuple(individual)] = tt
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

        while time.time() - start_time < maxtime and gen < maxgen:
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
