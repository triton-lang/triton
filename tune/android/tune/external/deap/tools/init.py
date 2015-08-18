from __future__ import division

def initRepeat(container, func, n):
    """Call the function *container* with a generator function corresponding
    to the calling *n* times the function *func*.
    
    :param container: The type to put in the data from func.
    :param func: The function that will be called n times to fill the
                 container.
    :param n: The number of times to repeat func.
    :returns: An instance of the container filled with data from func.
    
    This helper function can can be used in conjunction with a Toolbox 
    to register a generator of filled containers, as individuals or 
    population.
    
        >>> initRepeat(list, random.random, 2) # doctest: +ELLIPSIS, 
        ...                                    # doctest: +NORMALIZE_WHITESPACE
        [0.4761..., 0.6302...]

    See the :ref:`list-of-floats` and :ref:`population` tutorials for more examples.
    """
    return container(func() for _ in xrange(n))

def initIterate(container, generator):
    """Call the function *container* with an iterable as 
    its only argument. The iterable must be returned by 
    the method or the object *generator*.
    
    :param container: The type to put in the data from func.
    :param generator: A function returning an iterable (list, tuple, ...),
                      the content of this iterable will fill the container.
    :returns: An instance of the container filled with data from the
              generator.
    
    This helper function can can be used in conjunction with a Toolbox 
    to register a generator of filled containers, as individuals or 
    population.

        >>> from random import sample
        >>> from functools import partial
        >>> gen_idx = partial(sample, range(10), 10)
        >>> initIterate(list, gen_idx)
        [4, 5, 3, 6, 0, 9, 2, 7, 1, 8]

    See the :ref:`permutation` and :ref:`arithmetic-expr` tutorials for
    more examples.
    """
    return container(generator())

def initCycle(container, seq_func, n=1):
    """Call the function *container* with a generator function corresponding
    to the calling *n* times the functions present in *seq_func*.
    
    :param container: The type to put in the data from func.
    :param seq_func: A list of function objects to be called in order to
                     fill the container.
    :param n: Number of times to iterate through the list of functions.
    :returns: An instance of the container filled with data from the
              returned by the functions.
    
    This helper function can can be used in conjunction with a Toolbox 
    to register a generator of filled containers, as individuals or 
    population.
    
        >>> func_seq = [lambda:1 , lambda:'a', lambda:3]
        >>> initCycle(list, func_seq, n=2)
        [1, 'a', 3, 1, 'a', 3]

    See the :ref:`funky` tutorial for an example.
    """
    return container(func() for _ in xrange(n) for func in seq_func)

__all__ = ['initRepeat', 'initIterate', 'initCycle']


if __name__ == "__main__":
    import doctest
    import random
    random.seed(64)
    doctest.run_docstring_examples(initRepeat, globals())
    
    random.seed(64)
    doctest.run_docstring_examples(initIterate, globals())
    doctest.run_docstring_examples(initCycle, globals())
    
