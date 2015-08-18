#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import math

def bin2float(min_, max_, nbits):
    """Convert a binary array into an array of float where each
    float is composed of *nbits* and is between *min_* and *max_*
    and return the result of the decorated function.

    .. note::
        This decorator requires the first argument of
        the evaluation function to be named *individual*.
    """
    def wrap(function):
        def wrapped_function(individual, *args, **kargs):
            nelem = len(individual)/nbits
            decoded = [0] * nelem
            for i in xrange(nelem):
                gene = int("".join(map(str, individual[i*nbits:i*nbits+nbits])), 2)
                div = 2**nbits - 1
                temp = float(gene)/float(div)
                decoded[i] = min_ + (temp * (max_ - min_))
            return function(decoded, *args, **kargs)
        return wrapped_function
    return wrap

def trap(individual):
    u = sum(individual)
    k = len(individual)
    if u == k:
        return k
    else:
        return k - 1 - u

def inv_trap(individual):
    u = sum(individual)
    k = len(individual)
    if u == 0:
        return k
    else:
        return u - 1

def chuang_f1(individual):
    """Binary deceptive function from : Multivariate Multi-Model Approach for
    Globally Multimodal Problems by Chung-Yao Chuang and Wen-Lian Hsu.
    
    The function takes individual of 40+1 dimensions and has two global optima
    in [1,1,...,1] and [0,0,...,0].
    """    
    total = 0
    if individual[-1] == 0:
        for i in xrange(0,len(individual)-1,4):
            total += inv_trap(individual[i:i+4])
    else:
        for i in xrange(0,len(individual)-1,4):
            total += trap(individual[i:i+4])
    return total,

def chuang_f2(individual):
    """Binary deceptive function from : Multivariate Multi-Model Approach for
    Globally Multimodal Problems by Chung-Yao Chuang and Wen-Lian Hsu.
    
    The function takes individual of 40+1 dimensions and has four global optima
    in [1,1,...,0,0], [0,0,...,1,1], [1,1,...,1] and [0,0,...,0].    
    """    
    total = 0
    if individual[-2] == 0 and individual[-1] == 0:
        for i in xrange(0,len(individual)-2,8):
            total += inv_trap(individual[i:i+4]) + inv_trap(individual[i+4:i+8])
    elif individual[-2] == 0 and individual[-1] == 1:
        for i in xrange(0,len(individual)-2,8):
            total += inv_trap(individual[i:i+4]) + trap(individual[i+4:i+8])
    elif individual[-2] == 1 and individual[-1] == 0:
        for i in xrange(0,len(individual)-2,8):
            total += trap(individual[i:i+4]) + inv_trap(individual[i+4:i+8])
    else:
        for i in xrange(0,len(individual)-2,8):
            total += trap(individual[i:i+4]) + trap(individual[i+4:i+8])
    return total,

def chuang_f3(individual):
    """Binary deceptive function from : Multivariate Multi-Model Approach for
    Globally Multimodal Problems by Chung-Yao Chuang and Wen-Lian Hsu.

    The function takes individual of 40+1 dimensions and has two global optima
    in [1,1,...,1] and [0,0,...,0].
    """
    total = 0
    if individual[-1] == 0:
        for i in xrange(0,len(individual)-1,4):
            total += inv_trap(individual[i:i+4])
    else:
        for i in xrange(2,len(individual)-3,4):
            total += inv_trap(individual[i:i+4])
        total += trap(individual[-2:]+individual[:2])
    return total,

# Royal Road Functions
def royal_road1(individual, order):
    """Royal Road Function R1 as presented by Melanie Mitchell in : 
    "An introduction to Genetic Algorithms".
    """
    nelem = len(individual) / order
    max_value = int(2**order - 1)
    total = 0
    for i in xrange(nelem):
        value = int("".join(map(str, individual[i*order:i*order+order])), 2)
        total += int(order) * int(value/max_value)
    return total,

def royal_road2(individual, order):
    """Royal Road Function R2 as presented by Melanie Mitchell in : 
    "An introduction to Genetic Algorithms".
    """
    total = 0
    norder = order
    while norder < order**2:
        total += royal_road1(norder, individual)[0]
        norder *= 2
    return total,

