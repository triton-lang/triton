Welcome to Triton's documentation!
==================================

Triton is a language and compiler for parallel programming. It aims to provide a Python-based programming environment for productively writing custom DNN compute kernels capable of running at maximal throughput on modern GPU hardware.


Getting Started
---------------

- Follow the :doc:`installation instructions <getting-started/installation>` for your platform of choice.
- Take a look at the :doc:`tutorials <getting-started/tutorials/index>` to learn how to write your first Triton program.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting-started/installation
   getting-started/tutorials/index


Python API
----------

- :doc:`triton <python-api/triton>`
- :doc:`triton.language <python-api/triton.language>`
- :doc:`triton.testing <python-api/triton.testing>`


.. toctree::
   :maxdepth: 1
   :caption: Python API
   :hidden:

   python-api/triton
   python-api/triton.language
   python-api/triton.testing

   
Going Further
-------------

Check out the following documents to learn more about Triton and how it compares against other DSLs for DNNs:

- Chapter 1: :doc:`Introduction <programming-guide/chapter-1/introduction>`
- Chapter 2: :doc:`Related Work <programming-guide/chapter-2/related-work>`

.. toctree::
   :maxdepth: 1
   :caption: Programming Guide
   :hidden:

   programming-guide/chapter-1/introduction
   programming-guide/chapter-2/related-work
