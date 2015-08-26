import sys
import unittest

sys.path.append("..")

import tools

class LogbookTest(unittest.TestCase):

    def setUp(self):
        self.logbook = tools.Logbook()
        print

    def test_multi_chapters(self):
        self.logbook.record(gen=0, evals=100, fitness={'obj 1' : {'avg' : 1.0, 'max' : 10},
                                          'avg' : 1.0, 'max' : 10},
                            length={'avg' : 1.0, 'max' : 30},
                            test={'avg' : 1.0, 'max' : 20})
        self.logbook.record(gen=0, evals=100, fitness={'obj 1' : {'avg' : 1.0, 'max' : 10},
                                          'avg' : 1.0, 'max' : 10},
                            length={'avg' : 1.0, 'max' : 30},
                            test={'avg' : 1.0, 'max' : 20})
        print(self.logbook.stream)


    def test_one_chapter(self):
        self.logbook.record(gen=0, evals=100, fitness={'avg' : 1.0, 'max' : 10})
        self.logbook.record(gen=0, evals=100, fitness={'avg' : 1.0, 'max' : 10})
        print(self.logbook.stream)

    def test_one_big_chapter(self):
        self.logbook.record(gen=0, evals=100, fitness={'obj 1' : {'avg' : 1.0, 'max' : 10}, 'obj 2' : {'avg' : 1.0, 'max' : 10}})
        self.logbook.record(gen=0, evals=100, fitness={'obj 1' : {'avg' : 1.0, 'max' : 10}, 'obj 2' : {'avg' : 1.0, 'max' : 10}})
        print(self.logbook.stream)

    def test_no_chapters(self):
        self.logbook.record(gen=0, evals=100, **{'avg' : 1.0, 'max' : 10})
        self.logbook.record(gen=0, evals=100, **{'avg' : 1.0, 'max' : 10})        
        print(self.logbook.stream)



if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(LogbookTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
    