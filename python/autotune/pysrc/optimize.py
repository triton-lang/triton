import array, random, itertools
import deap.tools
import numpy as np

from genetic import GeneticOperators

def genetic(statement, device, TemplateType, build_template, compute_perf, perf_metric, out):
    GA = GeneticOperators(device, statement, TemplateType, build_template, out)
    return GA.optimize(maxtime='2m30s', maxgen=1000, compute_perf=compute_perf, perf_metric=perf_metric)
