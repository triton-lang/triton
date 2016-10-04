# Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
# 
# This file is part of ISAAC.
# 
# ISAAC is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301  USA

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

fetch_types = [sc.templates.fetch_type.FETCH_FROM_GLOBAL_CONTIGUOUS,
               sc.templates.fetch_type.FETCH_FROM_GLOBAL_STRIDED,
               sc.templates.fetch_type.FETCH_FROM_LOCAL,
               sc.templates.fetch_type.FETCH_FROM_LOCAL]

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
                time = tools.benchmark(template(*decode(genome)), tree)
                if time == float('inf'):
                    return time, 
                cache[idx] = time
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
            individual.fitness.values = toolbox.evaluate(genome)
            if max(individual.fitness.values) != float('inf'):
                population += [individual]
            genome = encode(list(initializer.next()))
        hof.update(population)
        
        #Main iteration
        while len(cache) < self.naccept and it<self.niter:
            
            #Generate offspring
            offspring = []
            while len(offspring) < self.popsize:
                op_choice = random.random()
                #Cross-over
                if op_choice < self.cxpb: 
                    ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
                    ind1, ind2 = toolbox.mate(ind1, ind2)
                    ind = ind1
                    toolbox.evaluate(ind)
                    if max(ind.fitness.values) != float('inf'):
                        offspring += [ind]
                #Mutation
                elif op_choice < self.cxpb + self.mutpb: 
                    ind = toolbox.clone(random.choice(population))
                    ind, = toolbox.mutate(ind, 1.0/offsets[-1])
                    toolbox.evaluate(ind)
                    if max(ind.fitness.values) != float('inf'):
                        offspring += [ind]
                #Reproduction
                else: 
                    offspring += [random.choice(population)]

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
    elif issubclass(template, sc.templates.gemm):
        sweep_over = [0,1,2,3,4]
    
    #Evaluate the provided parameters guess
    reference = tools.benchmark(template(*parameters), tree)
    if reference==float('inf'):
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
        time = tools.benchmark(template(*x), tree)
        if time/reference < .98:
            return False
    return True
    
    
