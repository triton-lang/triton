import os
import sys
import pytest

# activate interpreter mode
os.environ['TRITON_INTERPRET'] = '1'

# resolve the relative path to test_core.py 
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CORE = os.path.join(TEST_DIR, '..', 'language')
sys.path.append(TEST_CORE)


from test_core import test_bin_op

# turn off
os.environ['TRITON_INTERPRET'] = '0'