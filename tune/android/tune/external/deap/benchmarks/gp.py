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

from math import exp, sin, cos

def kotanchek(data):
    """Kotanchek benchmark function.
    
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1
       
       * - Range
         - :math:`\mathbf{x} \in [-1, 7]^2`
       * - Function
         - :math:`f(\mathbf{x}) = \\frac{e^{-(x_1 - 1)^2}}{3.2 + (x_2 - 2.5)^2}`
    """
    return exp(-(data[0] - 1)**2) / (3.2 + (data[1] - 2.5)**2)

def salustowicz_1d(data):
    """Salustowicz benchmark function.
    
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`x \in [0, 10]`
       * - Function
         - :math:`f(x) = e^x x^3 \cos(x) \sin(x) (\cos(x) \sin^2(x) - 1)`
    """
    return exp(-data[0]) * data[0]**3 * cos(data[0]) * sin(data[0]) * (cos(data[0]) * sin(data[0])**2 - 1)
    
def salustowicz_2d(data):
    """Salustowicz benchmark function.
    
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\mathbf{x} \in [0, 7]^2`
       * - Function
         - :math:`f(\mathbf{x}) = e^{x_1} x_1^3 \cos(x_1) \sin(x_1) (\cos(x_1) \sin^2(x_1) - 1) (x_2 -5)`
    """
    return exp(-data[0]) * data[0]**3 * cos(data[0]) * sin(data[0]) * (cos(data[0]) * sin(data[0])**2 - 1) * (data[1] - 5)

def unwrapped_ball(data):
    """Unwrapped ball benchmark function.
    
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\mathbf{x} \in [-2, 8]^n`
       * - Function
         - :math:`f(\mathbf{x}) = \\frac{10}{5 + \sum_{i=1}^n (x_i - 3)^2}`
    """
    return 10. / (5. + sum((d - 3)**2 for d in data))

def rational_polynomial(data):
    """Rational polynomial ball benchmark function.
    
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\mathbf{x} \in [0, 2]^3`
       * - Function
         - :math:`f(\mathbf{x}) = \\frac{30 * (x_1 - 1) (x_3 - 1)}{x_2^2 (x_1 - 10)}`
    """
    return 30. * (data[0] - 1) * (data[2] - 1) / (data[1]**2 * (data[0] - 10))

def sin_cos(data):
    """Sine cosine benchmark function.
    
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\mathbf{x} \in [0, 6]^2`
       * - Function
         - :math:`f(\mathbf{x}) = 6\sin(x_1)\cos(x_2)`
    """
    6 * sin(data[0]) * cos(data[1])

def ripple(data):
    """Ripple benchmark function.
    
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\mathbf{x} \in [-5, 5]^2`
       * - Function
         - :math:`f(\mathbf{x}) = (x_1 - 3) (x_2 - 3) + 2 \sin((x_1 - 4) (x_2 -4))`
    """
    return (data[0] - 3) * (data[1] - 3) + 2 * sin((data[0] - 4) * (data[1] - 4))

def rational_polynomial2(data):
    """Rational polynomial benchmark function.
    
    .. list-table:: 
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\mathbf{x} \in [0, 6]^2`
       * - Function
         - :math:`f(\mathbf{x}) = \\frac{(x_1 - 3)^4 + (x_2 - 3)^3 - (x_2 - 3)}{(x_2 - 2)^4 + 10}`
    """
    return ((data[0] - 3)**4 + (data[1] - 3)**3 - (data[1] - 3)) / ((data[1] - 2)**4 + 10)
