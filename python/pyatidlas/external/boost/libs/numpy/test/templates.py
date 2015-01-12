#!/usr/bin/env python

# Copyright Jim Bosch & Ankit Daftery 2010-2012.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import templates_mod
import unittest
import numpy

class TestTemplates(unittest.TestCase):

    def testTemplates(self):
        for dtype in (numpy.int16, numpy.int32, numpy.float32, numpy.complex128):
            v = numpy.arange(12, dtype=dtype)
            for shape in ((12,), (4, 3), (2, 6)):
                a1 = numpy.zeros(shape, dtype=dtype)
                a2 = v.reshape(a1.shape)
                templates_mod.fill(a1)
                self.assert_((a1 == a2).all())
        a1 = numpy.zeros((12,), dtype=numpy.float64)
        self.assertRaises(TypeError, templates_mod.fill, a1)
        a1 = numpy.zeros((12,2,3), dtype=numpy.float32)
        self.assertRaises(TypeError, templates_mod.fill, a1)

if __name__=="__main__":
    unittest.main()
