==============
Installation
==============

--------------
With Pip
--------------

Triton can be installed directly from pip with the following command

.. code-block:: python

   pip install triton


--------------
From Source
--------------

+++++++++++++++
Python Package
+++++++++++++++

You can install the Python package from source by running the following commands:

.. code-block:: bash

      sudo apt-get install llvm-10-dev
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

      sudo apt-get install llvm-10-dev
      git clone https://github.com/ptillet/triton.git;
      mkdir build;
      cd build;
      cmake ../;
      make -j8;

A custom llvm-config binary can also be provided:

.. code-block:: bash
      
      cmake ../ -DLLVM_CONFIG=/path/to/llvm-config

Note that while direct usage of the C++ API is not officially supported, a usage tutorial can be found  `here <https://github.com/ptillet/triton/blob/master/tutorials/01-matmul.cc>`_
