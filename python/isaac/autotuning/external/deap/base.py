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

"""The :mod:`~deap.base` module provides basic structures to build
evolutionary algorithms. It contains the :class:`~deap.base.Toolbox`, useful
to store evolutionary operators, and a virtual :class:`~deap.base.Fitness`
class used as base class, for the fitness member of any individual. """

import sys

from collections import Sequence
from copy import deepcopy
from functools import partial 
from operator import mul, truediv

class Toolbox(object):
    """A toolbox for evolution that contains the evolutionary operators. At
    first the toolbox contains a :meth:`~deap.toolbox.clone` method that
    duplicates any element it is passed as argument, this method defaults to
    the :func:`copy.deepcopy` function. and a :meth:`~deap.toolbox.map`
    method that applies the function given as first argument to every items
    of the iterables given as next arguments, this method defaults to the
    :func:`map` function. You may populate the toolbox with any other
    function by using the :meth:`~deap.base.Toolbox.register` method.

    Concrete usages of the toolbox are shown for initialization in the
    :ref:`creating-types` tutorial and for tools container in the
    :ref:`next-step` tutorial.
    """

    def __init__(self):
        self.register("clone", deepcopy)
        self.register("map", map)

    def register(self, alias, function, *args, **kargs):
        """Register a *function* in the toolbox under the name *alias*. You 
        may provide default arguments that will be passed automatically when 
        calling the registered function. Fixed arguments can then be overriden 
        at function call time.
        
        :param alias: The name the operator will take in the toolbox. If the
                      alias already exist it will overwrite the the operator
                      already present.
        :param function: The function to which refer the alias.
        :param argument: One or more argument (and keyword argument) to pass
                         automatically to the registered function when called,
                         optional.
        
        The following code block is an example of how the toolbox is used. ::

            >>> def func(a, b, c=3):
            ...     print a, b, c
            ... 
            >>> tools = Toolbox()
            >>> tools.register("myFunc", func, 2, c=4)
            >>> tools.myFunc(3)
            2 3 4
        
        The registered function will be given the attributes :attr:`__name__`
        set to the alias and :attr:`__doc__` set to the original function's
        documentation. The :attr:`__dict__` attribute will also be updated
        with the original function's instance dictionnary, if any.
        """
        pfunc = partial(function, *args, **kargs)
        pfunc.__name__ = alias
        pfunc.__doc__ = function.__doc__
        
        if hasattr(function, "__dict__") and not isinstance(function, type):
            # Some functions don't have a dictionary, in these cases 
            # simply don't copy it. Moreover, if the function is actually
            # a class, we do not want to copy the dictionary.
            pfunc.__dict__.update(function.__dict__.copy())
        
        setattr(self, alias, pfunc)

    def unregister(self, alias):
        """Unregister *alias* from the toolbox.
        
        :param alias: The name of the operator to remove from the toolbox.
        """
        delattr(self, alias)

    def decorate(self, alias, *decorators):
        """Decorate *alias* with the specified *decorators*, *alias*
        has to be a registered function in the current toolbox.
        
        :param alias: The name of the operator to decorate.
        :param decorator: One or more function decorator. If multiple
                          decorators are provided they will be applied in
                          order, with the last decorator decorating all the
                          others.
        
        .. note::
            Decorate a function using the toolbox makes it unpicklable, and
            will produce an error on pickling. Although this limitation is not
            relevant in most cases, it may have an impact on distributed
            environments like multiprocessing.
            A function can still be decorated manually before it is added to
            the toolbox (using the @ notation) in order to be picklable.
        """
        pfunc = getattr(self, alias)
        function, args, kargs = pfunc.func, pfunc.args, pfunc.keywords
        for decorator in decorators:
            function = decorator(function)
        self.register(alias, function, *args, **kargs)


class Fitness(object):
    """The fitness is a measure of quality of a solution. If *values* are
    provided as a tuple, the fitness is initalized using those values,
    otherwise it is empty (or invalid).
    
    :param values: The initial values of the fitness as a tuple, optional.

    Fitnesses may be compared using the ``>``, ``<``, ``>=``, ``<=``, ``==``,
    ``!=``. The comparison of those operators is made lexicographically.
    Maximization and minimization are taken care off by a multiplication
    between the :attr:`weights` and the fitness :attr:`values`. The comparison
    can be made between fitnesses of different size, if the fitnesses are
    equal until the extra elements, the longer fitness will be superior to the
    shorter.

    Different types of fitnesses are created in the :ref:`creating-types`
    tutorial.

    .. note::
       When comparing fitness values that are **minimized**, ``a > b`` will
       return :data:`True` if *a* is **smaller** than *b*.
    """
    
    weights = None
    """The weights are used in the fitness comparison. They are shared among
    all fitnesses of the same type. When subclassing :class:`Fitness`, the
    weights must be defined as a tuple where each element is associated to an
    objective. A negative weight element corresponds to the minimization of
    the associated objective and positive weight to the maximization.

    .. note::
        If weights is not defined during subclassing, the following error will 
        occur at instantiation of a subclass fitness object: 
        
        ``TypeError: Can't instantiate abstract <class Fitness[...]> with
        abstract attribute weights.``
    """
    
    wvalues = ()
    """Contains the weighted values of the fitness, the multiplication with the
    weights is made when the values are set via the property :attr:`values`.
    Multiplication is made on setting of the values for efficiency.
    
    Generally it is unnecessary to manipulate wvalues as it is an internal
    attribute of the fitness used in the comparison operators.
    """
    
    def __init__(self, values=()):
        if self.weights is None:
            raise TypeError("Can't instantiate abstract %r with abstract "
                "attribute weights." % (self.__class__))
        
        if not isinstance(self.weights, Sequence):
            raise TypeError("Attribute weights of %r must be a sequence." 
                % self.__class__)
        
        if len(values) > 0:
            self.values = values
        
    def getValues(self):
        return tuple(map(truediv, self.wvalues, self.weights))
            
    def setValues(self, values):
        try:
            self.wvalues = tuple(map(mul, values, self.weights))
        except TypeError:
            _, _, traceback = sys.exc_info()
            raise TypeError, ("Both weights and assigned values must be a "
            "sequence of numbers when assigning to values of "
            "%r. Currently assigning value(s) %r of %r to a fitness with "
            "weights %s."
            % (self.__class__, values, type(values), self.weights)), traceback
            
    def delValues(self):
        self.wvalues = ()

    values = property(getValues, setValues, delValues,
        ("Fitness values. Use directly ``individual.fitness.values = values`` "
         "in order to set the fitness and ``del individual.fitness.values`` "
         "in order to clear (invalidate) the fitness. The (unweighted) fitness "
         "can be directly accessed via ``individual.fitness.values``."))
    
    def dominates(self, other, obj=slice(None)):
        """Return true if each objective of *self* is not strictly worse than 
        the corresponding objective of *other* and at least one objective is 
        strictly better.

        :param obj: Slice indicating on which objectives the domination is 
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False
        for self_wvalue, other_wvalue in zip(self.wvalues[obj], other.wvalues[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False                
        return not_equal

    @property
    def valid(self):
        """Assess if a fitness is valid or not."""
        return len(self.wvalues) != 0
        
    def __hash__(self):
        return hash(self.wvalues)

    def __gt__(self, other):
        return not self.__le__(other)
        
    def __ge__(self, other):
        return not self.__lt__(other)

    def __le__(self, other):
        return self.wvalues <= other.wvalues

    def __lt__(self, other):
        return self.wvalues < other.wvalues

    def __eq__(self, other):
        return self.wvalues == other.wvalues
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __deepcopy__(self, memo):
        """Replace the basic deepcopy function with a faster one.
        
        It assumes that the elements in the :attr:`values` tuple are 
        immutable and the fitness does not contain any other object 
        than :attr:`values` and :attr:`weights`.
        """
        copy_ = self.__class__()
        copy_.wvalues = self.wvalues
        return copy_

    def __str__(self):
        """Return the values of the Fitness object."""
        return str(self.values if self.valid else tuple())

    def __repr__(self):
        """Return the Python code to build a copy of the object."""
        return "%s.%s(%r)" % (self.__module__, self.__class__.__name__,
            self.values if self.valid else tuple())

