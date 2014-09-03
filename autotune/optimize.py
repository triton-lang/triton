import array
import numpy as np
import random
import time

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from genetic_operators import GeneticOperators

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, maxtime,
                   stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print logbook.stream

    # Begin the generational process
    gen = 0
    maxtime = time.strptime(maxtime, '%Mm%Ss')
    maxtime = maxtime.tm_min*60 + maxtime.tm_sec
    start_time = time.time()
    while time.time() - start_time < maxtime:
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
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream

    return population, logbook
    
def genetic(statement, context, TemplateType, build_template, parameter_names, all_parameters, compute_perf, perf_metric, out):
  gen = GeneticOperators(context.devices[0], statement, all_parameters, parameter_names, TemplateType, build_template)
  creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMin)

  toolbox = base.Toolbox()
  toolbox.register("individual", tools.initIterate, creator.Individual, gen.init)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.decorate("population", gen.repair)
  toolbox.register("evaluate", gen.evaluate)
  toolbox.register("mate", tools.cxUniform, indpb=0.3)
  toolbox.decorate("mate", gen.repair)
  toolbox.register("mutate", gen.mutate, indpb=0.2)
  toolbox.decorate("mutate", gen.repair)
  toolbox.register("select", tools.selNSGA2)
    
  pop = toolbox.population(n=10)
  hof = tools.HallOfFame(1)

  best_performer = lambda x: max([compute_perf(hof[0].fitness.values[0]) for t in x])
  best_profile = lambda x: '(%s)'%','.join(map(str,hof[0]))

  cxpb = 0.5
  mutpb = 0.2
  
  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("max (" + perf_metric + ")", lambda x: max([compute_perf(hof[0].fitness.values[0]) for t in x]))
  stats.register("profile ", lambda x: '(%s)'%','.join(map(str,hof[0])))

  pop, log = eaMuPlusLambda(pop, toolbox, 10, 20, cxpb=0.2, mutpb=0.2, maxtime='5m0s', stats=stats, halloffame=hof, verbose=True)
