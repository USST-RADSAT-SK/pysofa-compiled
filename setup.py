#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

long_desc = """*pysofa* is a `Python <http://www.python.org/>`_ module for
accessing `International Astronomical Union <http://www.iau.org/>`_'s
`SOFA library <http://www.iausofa.org/>`_ from python. SOFA (Standards
of Fundamental Astronomy) is a set of algorithms and procedures that
implement standard models used in fundamental astronomy.

*pysofa* is not a port of SOFA routines but a wrapper around the SOFA_C
library. Thus, no calculations are made into the pysofa software, they are
all delegated to the underlying SOFA_C library.

*pysofa* is neither distributed, supported nor endorsed by the International
Astronomical Union. In addition to *pysofa*'s license, any use of this module
should comply with `SOFA's license and terms of use
<http://www.iausofa.org/copyr.pdf>`_. Especially, but not exclusively, any
published work or commercial products which includes results achieved by using
*pysofa* shall acknowledge that the SOFA software was used in obtaining those
results."""

setup(name='pysofa',
    version='0.1.1',
    description='Python ctypes wrapper around the SOFA astronomical library',
    long_description = long_desc,
    author='Frédéric Grollier',
    author_email='fred.grollier@gmail.com',
    url='http://code.google.com/p/pysofa/',
    license='MIT License',
    packages=['pysofa',],
    package_data={'pysofa': ['sofa_c.dll']},
    requires=['numpy',],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy'
    ],
)
