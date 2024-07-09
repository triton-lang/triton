#!/bin/bash

pytest python/test/unit/language/test_core.py::test_dot_max_num_imprecise_acc[0-float8e5-128-256-128-128-256-256] -s &> log1
pytest python/test/unit/language/test_core.py::test_dot_max_num_imprecise_acc[0-float8e5-128-256-128-128-256-256] -s &> log2
diff log1 log2
