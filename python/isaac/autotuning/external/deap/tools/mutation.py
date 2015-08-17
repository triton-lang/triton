from __future__ import division
import math
import random

from itertools import repeat
from collections import Sequence

######################################
# GA Mutations                       #
######################################

def mutGaussian(individual, mu, sigma, indpb):
    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.
    
    :param individual: Individual to be mutated.
    :param mu: Mean or :term:`python:sequence` of means for the
               gaussian addition mutation.
    :param sigma: Standard deviation or :term:`python:sequence` of 
                  standard deviations for the gaussian addition mutation.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    
    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
    """
    size = len(individual)
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))
    
    for i, m, s in zip(xrange(size), mu, sigma):
        if random.random() < indpb:
            individual[i] += random.gauss(m, s)
    
    return individual,

def mutPolynomialBounded(individual, eta, low, up, indpb):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.
    
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param low: A value or a :term:`python:sequence` of values that
                is the lower bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that
               is the upper bound of the search space.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))
    
    for i, xl, xu in zip(xrange(size), low, up):
        if random.random() <= indpb:
            x = individual[i]
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy**(eta + 1)
                delta_q = val**mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy**(eta + 1)
                delta_q = 1.0 - val**mut_pow

            x = x + delta_q * (xu - xl)
            x = min(max(x, xl), xu)
            individual[i] = x
    return individual,

def mutShuffleIndexes(individual, indpb):
    """Shuffle the attributes of the input individual and return the mutant.
    The *individual* is expected to be a :term:`sequence`. The *indpb* argument is the
    probability of each attribute to be moved. Usually this mutation is applied on 
    vector of indices.
    
    :param individual: Individual to be mutated.
    :param indpb: Independent probability for each attribute to be exchanged to
                  another position.
    :returns: A tuple of one individual.
    
    This function uses the :func:`~random.random` and :func:`~random.randint`
    functions from the python base :mod:`random` module.
    """
    size = len(individual)
    for i in xrange(size):
        if random.random() < indpb:
            swap_indx = random.randint(0, size - 2)
            if swap_indx >= i:
                swap_indx += 1
            individual[i], individual[swap_indx] = \
                individual[swap_indx], individual[i]
    
    return individual,

def mutFlipBit(individual, indpb):
    """Flip the value of the attributes of the input individual and return the
    mutant. The *individual* is expected to be a :term:`sequence` and the values of the
    attributes shall stay valid after the ``not`` operator is called on them.
    The *indpb* argument is the probability of each attribute to be
    flipped. This mutation is usually applied on boolean individuals.
    
    :param individual: Individual to be mutated.
    :param indpb: Independent probability for each attribute to be flipped.
    :returns: A tuple of one individual.
    
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i in xrange(len(individual)):
        if random.random() < indpb:
            individual[i] = type(individual[i])(not individual[i])
    
    return individual,

def mutUniformInt(individual, low, up, indpb):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.
    
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from wich to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from wich to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))
    
    for i, xl, xu in zip(xrange(size), low, up):
        if random.random() < indpb:
            individual[i] = random.randint(xl, xu)
    
    return individual,


######################################
# ES Mutations                       #
######################################

def mutESLogNormal(individual, c, indpb):
    """Mutate an evolution strategy according to its :attr:`strategy`
    attribute as described in [Beyer2002]_. First the strategy is mutated
    according to an extended log normal rule, :math:`\\boldsymbol{\sigma}_t =
    \\exp(\\tau_0 \mathcal{N}_0(0, 1)) \\left[ \\sigma_{t-1, 1}\\exp(\\tau
    \mathcal{N}_1(0, 1)), \ldots, \\sigma_{t-1, n} \\exp(\\tau
    \mathcal{N}_n(0, 1))\\right]`, with :math:`\\tau_0 =
    \\frac{c}{\\sqrt{2n}}` and :math:`\\tau = \\frac{c}{\\sqrt{2\\sqrt{n}}}`,
    the the individual is mutated by a normal distribution of mean 0 and
    standard deviation of :math:`\\boldsymbol{\sigma}_{t}` (its current
    strategy) then . A recommended choice is ``c=1`` when using a :math:`(10,
    100)` evolution strategy [Beyer2002]_ [Schwefel1995]_.
    
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param c: The learning parameter.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    
    .. [Beyer2002] Beyer and Schwefel, 2002, Evolution strategies - A
       Comprehensive Introduction
       
    .. [Schwefel1995] Schwefel, 1995, Evolution and Optimum Seeking.
       Wiley, New York, NY
    """
    size = len(individual)
    t = c / math.sqrt(2. * math.sqrt(size))
    t0 = c / math.sqrt(2. * size)
    n = random.gauss(0, 1)
    t0_n = t0 * n
    
    for indx in xrange(size):
        if random.random() < indpb:
            individual.strategy[indx] *= math.exp(t0_n + t * random.gauss(0, 1))
            individual[indx] += individual.strategy[indx] * random.gauss(0, 1)
    
    return individual,

__all__ = ['mutGaussian', 'mutPolynomialBounded', 'mutShuffleIndexes', 
           'mutFlipBit', 'mutUniformInt', 'mutESLogNormal']
