==============
Installation
==============

---------------------
Binary Distributions
---------------------

You can install the latest stable release of Triton from pip:

      pip install triton


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

      git clone https://github.com/ptillet/triton.git;
      cd triton/python;
      pip install -e .

You can then test your installation by running the unit tests:

.. code-block:: bash

      pytest -vs .

and the benchmarks

.. code-block:: bash
      
      cd bench/
      python -m run --with-plots --result-dir /tmp/triton-bench

+++++++++++++++
C++ Package
+++++++++++++++

Those not interested in Python integration may want to use the internals of Triton (i.e, runtime, parser, codegen, driver, intermediate representation) directly. This can be done by running the following commands:

.. code-block:: bash

      git clone https://github.com/ptillet/triton.git;
      mkdir build;
      cd build;
      cmake ../;
      make -j8;

Note that while direct usage of the C++ API is not officially supported, a usage tutorial can be found  `here <https://github.com/ptillet/triton/blob/master/tutorials/01-matmul.cc>`_
