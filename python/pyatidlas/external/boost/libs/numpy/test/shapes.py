#!/usr/bin/env python

# Copyright Jim Bosch & Ankit Daftery 2010-2012.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import shapes_mod
import unittest
import numpy

class TestShapes(unittest.TestCase):
    
    def testShapes(self):
        a1 = numpy.array([(0,1),(2,3)])
        a1_shape = (1,4)
        a1 = shapes_mod.reshape(a1,a1_shape)
        self.assertEqual(a1_shape,a1.shape)

if __name__=="__main__":
    unittest.main()
