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
------------------

Check out the following documents to learn more about Triton and how it compares against other DSLs for DNNs:

- Chapter 1: :doc:`Introduction <programming-guide/chapter-1/introduction>`
- Chapter 2: :doc:`Related Work <programming-guide/chapter-2/related-work>`
- Chapter 3: :doc:`The Triton-C Language <programming-guide/chapter-3/triton-c>`
- Chapter 4: :doc:`The Triton-IR Intermediate Representation <programming-guide/chapter-4/triton-ir>`

.. toctree::
   :maxdepth: 1
   :caption: Programming Guide
   :hidden:

   programming-guide/chapter-1/introduction
   programming-guide/chapter-2/related-work
   programming-guide/chapter-3/triton-c
   programming-guide/chapter-4/triton-ir