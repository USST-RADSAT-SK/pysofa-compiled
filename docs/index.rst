.. pysofa documentation master file, created by
   sphinx-quickstart on Wed Nov 17 14:57:22 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pysofa's documentation page !
========================================

.. include:: common.inc

:mod:`pysofa` is a `Python <http://www.python.org/>`_ module for accessing
`International Astronomical Union <http://www.iau.org/>`_'s
`SOFA library <http://www.iausofa.org/>`_ from python. |SOFA| (Standards
of Fundamental Astronomy) is a set of algorithms and procedures that
implement standard models used in fundamental astronomy.

:mod:`pysofa` is not a port of |SOFA| routines but a wrapper around the
library. Thus, no calculations are made into the pysofa software, they are
all delegated to the underlying |SOFA| library.

Disclaimer
----------

:mod:`pysofa` is neither distributed, supported nor endorsed by the International
Astronomical Union. In addition to :mod:`pysofa`'s license, any use of this module
should comply with `SOFA's license and terms of use
<http://www.iausofa.org/tandc.html>`_. Especially, but not exclusively, any
published work or commercial products which includes results achieved by using
:mod:`pysofa` shall acknowledge that the |SOFA| software was used in obtaining
those results.


Installation
============

Requirements
------------
Before doing anything useful with :mod:`pysofa`, you'll need:

    * `Python <http://www.python.org/>`_ 2.5 or higher (including 3.x).
    * `numpy <http://numpy.scipy.org>`_
    * and, obviously, the C version of the |SOFA| library.

.. note::
    :mod:`pysofa` use `ctypes <http://docs.python.org/library/ctypes.html>`_ to do
    its job, hence, the |SOFA| C library must be compiled as a shared library
    and findable by the operating system's dynamic loader. Note that the
    default ``makefile`` provided with |SOFA| compile the library as a static
    one on UNIX systems.

Install
-------

Once you have the requirements satisfied, you have a few options for
installlation.

If you have `easy_install/setuptools <http://pypi.python.org/pypi/setuptools>`_
or `pip <http://pypi.python.org/pypi/pip>`_ installed, just do::

    pip install pysofa

or::

    easy_install pysofa

If you are installing from source code, do::

    python ./setup.py install


Bug reports
===========
The best place to report bugs or request features is the `google code bug
tracker <http://code.google.com/p/pysofa/issues/list>`_.


Documentation
=============

.. warning::
    :mod:`pysofa` is still beta software and, although fully functionnal,
    class, method names and calling conventions are subject to change in
    future versions.

Please refer to the :doc:`reference` page for in-depth view of :mod:`pysofa`'s
functions.


.. toctree::
   :hidden:
   :maxdepth: 2

   reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

