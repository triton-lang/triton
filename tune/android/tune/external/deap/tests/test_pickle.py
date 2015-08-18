
import sys
import unittest
import array
import pickle
import operator
from test import test_support

sys.path.append("..")

import numpy

import creator
import base
import gp
import tools

def func():
    return "True"

class Pickling(unittest.TestCase):

    def setUp(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("IndList", list, fitness=creator.FitnessMax)
        creator.create("IndArray", array.array,  typecode='f', fitness=creator.FitnessMax)
        creator.create("IndTree", gp.PrimitiveTree, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("func", func)
        self.toolbox.register("lambda_func", lambda: "True")
    
    def test_pickle_fitness(self):
        fitness = creator.FitnessMax()
        fitness.values = (1.0,)
        fitness_s = pickle.dumps(fitness)
        fitness_l = pickle.loads(fitness_s)
        self.failUnlessEqual(fitness, fitness_l, "Unpickled fitness != pickled fitness")

    def test_pickle_ind_list(self):
        ind = creator.IndList([1.0, 2.0, 3.0])
        ind.fitness.values = (4.0,)
        ind_s = pickle.dumps(ind)
        ind_l = pickle.loads(ind_s)
        self.failUnlessEqual(ind, ind_l, "Unpickled individual list != pickled individual list")
        self.failUnlessEqual(ind.fitness, ind_l.fitness, "Unpickled individual fitness != pickled individual fitness")
         
    def test_pickle_ind_array(self):
        ind = creator.IndArray([1.0, 2.0, 3.0])
        ind.fitness.values = (4.0,)
        ind_s = pickle.dumps(ind)
        ind_l = pickle.loads(ind_s)
        self.failUnlessEqual(ind, ind_l, "Unpickled individual array != pickled individual array")
        self.failUnlessEqual(ind.fitness, ind_l.fitness, "Unpickled individual fitness != pickled individual fitness")
    
    def test_pickle_tree(self):
        ind = creator.IndTree([operator.add, 1, 2])
        ind.fitness.values = (1.0,)
        ind_s = pickle.dumps(ind)
        ind_l = pickle.loads(ind_s)
        msg =  "Unpickled individual %s != pickled individual %s" % (str(ind), str(ind_l))
        self.failUnlessEqual(ind, ind_l, msg)
        msg =  "Unpickled fitness %s != pickled fitness %s" % (str(ind.fitness), str(ind_l.fitness))
        self.failUnlessEqual(ind.fitness, ind_l.fitness, msg)
    
    def test_pickle_population(self):
        ind1 = creator.IndList([1.0,2.0,3.0])
        ind1.fitness.values = (1.0,)
        ind2 = creator.IndList([4.0,5.0,6.0])
        ind2.fitness.values = (2.0,)
        ind3 = creator.IndList([7.0,8.0,9.0])
        ind3.fitness.values = (3.0,)
        
        pop = [ind1, ind2, ind3]
        
        pop_s = pickle.dumps(pop)
        pop_l = pickle.loads(pop_s)
        
        self.failUnlessEqual(pop[0], pop_l[0], "Unpickled individual list != pickled individual list")
        self.failUnlessEqual(pop[0].fitness, pop_l[0].fitness, "Unpickled individual fitness != pickled individual fitness")
        self.failUnlessEqual(pop[1], pop_l[1], "Unpickled individual list != pickled individual list")
        self.failUnlessEqual(pop[1].fitness, pop_l[1].fitness, "Unpickled individual fitness != pickled individual fitness")
        self.failUnlessEqual(pop[2], pop_l[2], "Unpickled individual list != pickled individual list")
        self.failUnlessEqual(pop[2].fitness, pop_l[2].fitness, "Unpickled individual fitness != pickled individual fitness")
    
    def test_pickle_logbook(self):
        stats = tools.Statistics()
        logbook = tools.Logbook()

        stats.register("mean", numpy.mean)
        record = stats.compile([1,2,3,4,5,6,8,9,10])
        logbook.record(**record)

        stats_s = pickle.dumps(logbook)
        logbook_r = pickle.loads(stats_s)

        self.failUnlessEqual(logbook, logbook_r, "Unpickled logbook != pickled logbook")


    if not sys.version_info < (2, 7):
        def test_pickle_partial(self):
            func_s = pickle.dumps(self.toolbox.func)
            func_l = pickle.loads(func_s)

            self.failUnlessEqual(self.toolbox.func(), func_l())
        
        @unittest.expectedFailure
        def test_pickle_lambda(self):
            func_s = pickle.dumps(self.toolbox.lambda_func)
            func_l = pickle.loads(func_s)
        
            self.failUnlessEqual(self.toolbox.lambda_func(), func_l())

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(Pickling)
    unittest.TextTestRunner(verbosity=2).run(suite)
