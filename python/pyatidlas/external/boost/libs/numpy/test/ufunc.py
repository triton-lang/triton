#!/usr/bin/env python

# Copyright Jim Bosch & Ankit Daftery 2010-2012.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import ufunc_mod
import unittest
import numpy
from numpy.testing.utils import assert_array_almost_equal

class TestUnary(unittest.TestCase):

    def testScalar(self):
        f = ufunc_mod.UnaryCallable()
        assert_array_almost_equal(f(1.0), 2.0)
        assert_array_almost_equal(f(3.0), 6.0)

    def testArray(self):
        f = ufunc_mod.UnaryCallable()
        a = numpy.arange(5, dtype=float)
        b = f(a)
        assert_array_almost_equal(b, a*2.0) 
        c = numpy.zeros(5, dtype=float)
        d = f(a,output=c)
        self.assert_(c is d)
        assert_array_almost_equal(d, a*2.0) 

    def testList(self):
        f = ufunc_mod.UnaryCallable()
        a = range(5)
        b = f(a)
        assert_array_almost_equal(b/2.0, a) 

class TestBinary(unittest.TestCase):

    def testScalar(self):
        f = ufunc_mod.BinaryCallable()
        assert_array_almost_equal(f(1.0, 3.0), 11.0) 
        assert_array_almost_equal(f(3.0, 2.0), 12.0) 

    def testArray(self):
        f = ufunc_mod.BinaryCallable()
        a = numpy.random.randn(5)
        b = numpy.random.randn(5)
        assert_array_almost_equal(f(a,b), (a*2+b*3)) 
        c = numpy.zeros(5, dtype=float)
        d = f(a,b,output=c)
        self.assert_(c is d)
        assert_array_almost_equal(d, a*2 + b*3) 
        assert_array_almost_equal(f(a, 2.0), a*2 + 6.0) 
        assert_array_almost_equal(f(1.0, b), 2.0 + b*3) 
        

if __name__=="__main__":
    unittest.main()
