Reference
=========

.. include:: common.inc
.. module:: pysofa

:mod:`pysofa` is a *functional* programming tool. This means it is a
collection of independent functions and, apart from input and output
values which are python objects, it does not provide object-oriented
machinery. There are plans to build an OOP version using |SOFA| routines,
but it is yet unclear if these will be included as part of :mod:`pysofa` or of
another module which will make use of :mod:`pysofa`.

Function names in :mod:`pysofa` are almost directly mapped to those of the
|SOFA|_ library, with two slight modifications:

* the ``iau`` prefix is removed to conform with the |SOFA| license
* names are all lowercased, to conform with :pep:`8`

thus, for example, the |SOFA| routine ``iauEra00`` is accessible via
``pysofa.era00``.

All functions in this module are accessible under the ``pysofa`` namespace,
but, for the sake of convenience, the complete list of functions has been
divided into three separate pages, with functions grouped by scope:

.. toctree::
   :maxdepth: 1

   time
   astro
   vector

The documentation of :mod:`pysofa` is quite terse and merely consist of a
brief summary and a list of expected input/output types and values. Users
are urged to refer to the official |SOFA|_ manual for detailed explanations
on these routines, especially to get references for the algorithms used. Each
documented function in :mod:`pysofa` refers to the relevant page of the
official manual.pdf file distributed with |SOFA|. Note that these page numbers
match those of the latest release of |SOFA| and should be adapted
when using older version of the manual.

Apart from the functions directly underlied by |SOFA| listed in the next pages,
:mod:`pysofa` provides the following ones:

.. autofunction:: get_sofa_version
.. autofunction:: has_function


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

