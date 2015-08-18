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
"""The :mod:`~deap.tools` module contains the operators for evolutionary
algorithms. They are used to modify, select and move the individuals in their
environment. The set of operators it contains are readily usable in the
:class:`~deap.base.Toolbox`. In addition to the basic operators this module
also contains utility tools to enhance the basic algorithms with
:class:`Statistics`, :class:`HallOfFame`, :class:`Checkpoint`, and
:class:`History`.
"""

from .crossover import *
from .emo import *
from .init import *
from .migration import *
from .mutation import *
from .selection import *
from .support import *