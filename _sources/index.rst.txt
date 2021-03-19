Welcome to Triton's documentation!
==================================

Triton is an imperative language and compiler for parallel programming. It aims to provide a programming environment for productively writing custom DNN compute kernels capable of running at maximal throughput on modern GPU hardware.

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

Programming Guide
--------------

Check out the following documents to learn more about Triton and how it compares against other DSLs for DNNs:

- Chapter 1: :doc:`Introduction <programming-guide/introduction>`
- Chapter 2: :doc:`Related Work <programming-guide/related-work>`
- Chapter 3: :doc:`The Triton-C Kernel Language <programming-guide/triton-c>`

.. toctree::
   :maxdepth: 1
   :caption: Programming Guide
   :hidden:

   programming-guide/introduction
   programming-guide/related-work
   programming-guide/triton-c