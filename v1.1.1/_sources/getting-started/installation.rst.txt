==============
Installation
==============

---------------------
Binary Distributions
---------------------

You can install the latest stable release of Triton from pip:

.. code-block:: bash

      pip install triton

Binary wheels are available for CPython 3.6-3.9 and PyPy 3.6-3.7.

And the latest nightly release:

.. code-block:: bash
  
      pip install -U --pre triton


--------------
From Source
--------------

+++++++++++++++
Python Package
+++++++++++++++

You can install the Python package from source by running the following commands:

.. code-block:: bash

      git clone https://github.com/openai/triton.git;
      cd triton/python;
      pip install cmake; # build time dependency
      pip install -e .

Note that, if llvm-11 is not present on your system, the setup.py script will download the official LLVM11 static libraries link against that.

You can then test your installation by running the unit tests:

.. code-block:: bash

      pip install -r requirements-test.txt
      pytest -vs test/unit/

and the benchmarks

.. code-block:: bash
      
      cd bench/
      python -m run --with-plots --result-dir /tmp/triton-bench
