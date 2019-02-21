# -*- coding: utf-8 -*-

#
# Copyright 2010 Frédéric Grollier
#
# Distributed under the terms of the MIT license
#

import warnings as _warnings
import os as _os

import ctypes as _ct
from ctypes.util import find_library as _find_library

import numpy as _np
from numpy.ctypeslib import ndpointer as _ndpointer

import pkg_resources

_sofalib_filename = pkg_resources.resource_filename('pysofa', 'sofa_c.dll')
if not _os.path.isfile(_sofalib_filename):
    _sofalib_filename = None
    if _os.environ.get('SOFA_LIBRARY') is not None:
        _sofalib_filename = _os.environ['SOFA_LIBRARY']
    else:
        _sofalib_filename = _find_library('sofa_c')
    if _sofalib_filename is None:
        raise ImportError('Unable to find the shared C library "sofa_c".')
_sofa = _ct.CDLL(_sofalib_filename)


# Try to guess what SOFA version we're dealing with,
# by testing the presence of newly created functions
# between each version.
__sofa_version = None
try:
    _sofa.iauTaitt
    __sofa_version = (2010, 12, 1)
except AttributeError:
    __sofa_version = (2009, 12, 31)

def get_sofa_version():
    """ Return a tuple containing the three components of the
    |SOFA| library release wich has been loaded, in the form
    (year, month, day).

    In case the release number hasn't been resolved (*None*, *None*, *None*)
    is returned. This should never occur and shall be signaled as a bug.

    .. note:: this only deals with *major* release numbers and does not
        account for *revised* versions of |SOFA|.
    """

    if __sofa_version is None:
        return (None, None, None)
    else:
        return __sofa_version


def has_function(funcname):
    """ Helper function that returns True if this particular release of |SOFA|
    provides the function *funcname*, and False otherwise. This is only the
    case with function names that are legal |SOFA| function names, wich means
    that calling ``has_function`` with a *funcname* that isn't known by any
    version of |SOFA| will raise :exc:`AttributeError`.

    >>> pysofa.has_function('era00')
    True
    >>> pysofa.has_function('taitt') # if SOFA release < (2010, 12, 1)
    False
    >>> pysofa.has_function('foo')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "pysofa_ctypes.py", line 62, in has_function
        (__name__, funcname))
    AttributeError: pysofa does not know any function "named" foo

    """

    if not funcname in globals():
        raise AttributeError('%s does not know any function "named" %s' % \
                                (__name__, funcname))
    # convert 'funcname' to its equivalent SOFA name
    funcname = 'iau' + funcname[0].upper() + funcname[1:]
    return hasattr(_sofa, funcname)


def _req_shape_c(a, dtype=None, shape=None, req=None):
    return _np.require(a, dtype=dtype, requirements=req).reshape(shape,
                                                                    order='C')


# iauA2af
_sofa.iauA2af.argtypes = [_ct.c_int, #ndp
                            _ct.c_double, #angle
                            _ct.POINTER(_ct.c_char), #sign
                            _ct.c_int * 4] #idmsf
def a2af(ndp, angle):
    """ Decompose radians into degrees, arcminutes, arcseconds, fraction.

    :param ndp: the requested resolution.
    :type ndp: int

    :param angle: the value to decompose.
    :type angle: float

    :returns: a tuple whose first member is a string containing the sign, and
        the second member is itself a tuple (degrees, arcminutes, arcseconds,
        fraction).

    .. seealso:: |MANUAL| page 19
    """
    sign = _ct.c_char()
    idmsf = (_ct.c_int * 4)()
    _sofa.iauA2af(ndp, float(angle), _ct.byref(sign), idmsf)
    return sign.value, tuple([v for v in idmsf])


# iauA2tf
_sofa.iauA2tf.argtypes = [_ct.c_int, #ndp
                            _ct.c_double, #angle
                            _ct.POINTER(_ct.c_char), #sign
                            _ct.c_int * 4] #ihmsf
def a2tf(ndp, angle):
    """ Decompose radians into hours, arcminutes, arcseconds, fraction.

    :param ndp: the requested resolution.
    :type ndp: int

    :param angle: the value to decompose.
    :type angle: float

    :returns: a tuple whose first member is a string containing the sign, and
        the second member is itself a tuple (hours, arcminutes, arcseconds,
        fraction).

    .. seealso:: |MANUAL| page 20
    """
    sign = _ct.c_char()
    ihmsf = (_ct.c_int * 4)()
    _sofa.iauA2tf(ndp, float(angle), _ct.byref(sign), ihmsf)
    return sign.value, tuple([v for v in ihmsf])


# iauAf2a
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauAf2a.argtypes = [_ct.c_char, #sign
                            _ct.c_int, #ideg
                            _ct.c_int, #iamin
                            _ct.c_double, #asec
                            _ct.POINTER(_ct.c_double)] #rad
    _sofa.iauAf2a.restype = _ct.c_int
except AttributeError:
    pass

_af2a_msg = {0: 'OK', # Unused
                1:'Af2a: degrees outside the range 0-359',
                2:'Af2a: arcminutes outside the range 0-59',
                3:'Af2a: arcseconds outside the range 0-59.999...'}
def af2a(s, ideg, iamin, asec):
    """ Convert degrees, arcminutes, arcseconds to radians.

    :param s: sign, '-' for negative, otherwise positive.

    :param ideg: degrees.
    :type ideg: int

    :param iamin: arcminutes.
    :type iamin: int

    :param asec: arcseconds.
    :type asec: float

    :returns: the converted value in radians as a float.

    :raises: :exc:`UserWarning` in case *ideg*, *iamin* or *asec*
        values are outside the range 0-359, 0-59 or 0-59.999...

        :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 21
    """

    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    rad = _ct.c_double()
    s = _sofa.iauAf2a(s, ideg, iamin, asec, _ct.byref(rad))
    if s != 0:
        _warnings.warn(_af2a_msg[s], UserWarning, 2)
    return rad.value



# iauAnp
_sofa.iauAnp.argtypes = [_ct.c_double]
_sofa.iauAnp.restype = _ct.c_double
def anp(a):
    """ Normalize *a* into the range 0 <= result < 2pi.

    :param a: the value to normalize.
    :type a: float

    :returns: the normalized value as a float.

    .. seealso:: |MANUAL| page 22
    """
    return _sofa.iauAnp(float(a))


# iauAnpm
_sofa.iauAnpm.argtypes = [_ct.c_double]
_sofa.iauAnpm.restype = _ct.c_double
def anpm(a):
    """ Normalize *a* into the range -pi <= result < +pi.

    :param a: the value to normalize.
    :type a: float

    :returns: the normalized value as a float.

    .. seealso:: |MANUAL| page 23
    """
    return _sofa.iauAnpm(float(a))


# iauBi00
_sofa.iauBi00.argtypes = [_ct.POINTER(_ct.c_double), #dpsibi
                            _ct.POINTER(_ct.c_double), #depsbi
                            _ct.POINTER(_ct.c_double)] #dra
def bi00():
    """ Frame bias components of IAU 2000 precession-nutation models.

    :returns: a tuple of three items:

        * longitude correction (float)
        * obliquity correction (float)
        * the ICRS RA of the J2000.0 mean equinox (float).

    .. seealso:: |MANUAL| page 24
    """
    dpsibi = _ct.c_double()
    depsbi = _ct.c_double()
    dra = _ct.c_double()
    _sofa.iauBi00(_ct.byref(dpsibi), _ct.byref(depsbi), _ct.byref(dra))
    return dpsibi.value, depsbi.value, dra.value

# iauBp00
_sofa.iauBp00.argtypes = [_ct.c_double, #date1
                        _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rb
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rp
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rbp
def bp00(date1, date2):
    """ Frame bias and precession, IAU 2000.

    :param date1, date2: TT as a two-part Julian date.

    :returns: a tuple of three items:

        * frame bias matrix (numpy.matrix of shape 3x3)
        * precession matrix (numpy.matrix of shape 3x3)
        * bias-precession matrix (numpy.matrix of shape 3x3)

    .. seealso:: |MANUAL| page 25
    """
    rb = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rbp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauBp00(date1, date2, rb, rp, rbp)
    return rb, rp, rbp


# iauBp06
_sofa.iauBp06.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rb
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rp
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rbp
def bp06(date1, date2):
    """ Frame bias and precession, IAU 2006.

    :param date1, date2: TT as a two-part Julian date.

    :returns: a tuple of three items:

        * frame bias matrix (numpy.matrix of shape 3x3)
        * precession matrix (numpy.matrix of shape 3x3)
        * bias-precession matrix (numpy.matrix of shape 3x3)

    .. seealso:: |MANUAL| page 27
    """
    rb = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rbp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauBp06(date1, date2, rb, rp, rbp)
    return rb, rp, rbp


# iauBpn2xy
_sofa.iauBpn2xy.argtypes = [
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rbpn
                        _ct.POINTER(_ct.c_double), #x
                        _ct.POINTER(_ct.c_double)] #y
def bpn2xy(rbpn):
    """ Extract from the bias-precession-nutation matrix the X,Y coordinates
    of the Celestial Intermediate Pole.

    :param rbpn: celestial-to-true matrix
    :type rbpn: numpy.ndarray, matrix or nested sequences of shape 3x3

    :returns: a tuple of two items containing *x* and *y*, as floats.

    .. seealso:: |MANUAL| page 28
    """
    x = _ct.c_double()
    y = _ct.c_double()
    _sofa.iauBpn2xy(_req_shape_c(rbpn, float, (3,3)), x, y)
    return x.value, y.value


# iauC2i00a
_sofa.iauC2i00a.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rc2i
def c2i00a(date1, date2):
    """ Form the celestial-to-intermediate matrix for a given date using the
    IAU 2000A precession-nutation model.

    :param date1, date2: TT as a two-part Julian date.

    :returns: the celestial-to-intermediate matrix, as a numpy.matrix of
        shape 3x3.

    .. seealso:: |MANUAL| page 29
    """
    rc2i = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauC2i00a(date1, date2, rc2i)
    return rc2i


# iauC2i00b
_sofa.iauC2i00b.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rc2i
def c2i00b(date1, date2):
    """ Form the celestial-to-intermediate matrix for a given date using the
    IAU 2000B precession-nutation model.

    :param date1, date2: TT as a two-part Julian date.

    :returns: the celestial-to-intermediate matrix, as a numpy.matrix of
        shape 3x3.

    .. seealso:: |MANUAL| page 31
    """
    rc2i = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauC2i00b(date1, date2, rc2i)
    return rc2i


# iauC2i06a
_sofa.iauC2i06a.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rc2i
def c2i06a(date1, date2):
    """ Form the celestial-to-intermediate matrix for a given date using the
    IAU 2006 precession-nutation model.

    :param date1, date2: TT as a two-part Julian date.

    :returns: the celestial-to-intermediate matrix, as a numpy.matrix of
        shape 3x3.

    .. seealso:: |MANUAL| page 33
    """
    rc2i = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauC2i06a(date1, date2, rc2i)
    return rc2i


# iauC2ibpn
_sofa.iauC2ibpn.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rbpn
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rc2i
def c2ibpn(date1, date2, rbpn):
    """ Form the celestial-to-intermediate matrix for a given date given the
    bias-precession-nutation matrix. IAU 2000.

    :param date1, date2: TT as a two-part Julian date.

    :param rbpn: celestial-to-true matrix.
    :type rbpn: numpy.ndarray, numpy.matrix or nested sequences of shape 3x3

    :returns: the celestial-to-intermediate matrix, as a numpy.matrix of
        shape 3x3.

    .. seealso:: |MANUAL| page 34
    """
    rc2i = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauC2ibpn(date1, date2, _req_shape_c(rbpn, float, (3,3)), rc2i)
    return rc2i


# iauC2ixy
_sofa.iauC2ixy.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.c_double, #x
                            _ct.c_double, #y
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rc2i
def c2ixy(date1, date2, x, y):
    """ Form the celestial to intermediate-frame-of-date matrix for a given
    date when CIP X,Y coordinates are known. IAU 2000.

    :param date1, date2: TT as a two-part Julian date.

    :param x, y: celestial intermediate pole coordinates.
    :type x, y: float

    :returns: the celestial-to-intermediate matrix as a numpy.matrix of shape
        3x3.

    .. seealso:: |MANUAL| page 36
    """
    rc2i = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauC2ixy(date1, date2, float(x), float(y), rc2i)
    return rc2i


# iauC2ixys
_sofa.iauC2ixys.argtypes = [_ct.c_double, #x
                            _ct.c_double, #y
                            _ct.c_double, #s
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rc2i
def c2ixys(x, y, s):
    """ Form the celestial to intermediate-frame-of-date matrix given the CIP
    X,Y coordinates and the CIO locator s.

    :param x, y: celestial intermediate pole coordinates.
    :type x, y: float

    :param s: the CIO locator.
    :type s: float

    :returns: the celestial-to-intermediate matrix as a numpy.matrix of shape
        3x3.

    .. seealso:: |MANUAL| page 38
    """
    rc2i = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauC2ixys(float(x), float(y), float(s), rc2i)
    return rc2i


# iauC2s
_sofa.iauC2s.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C'), #p
                        _ct.POINTER(_ct.c_double), #theta
                        _ct.POINTER(_ct.c_double)] #phi
def c2s(p):
    """ P-vector to spherical coordinates.

    :param p: p-vector
    :type p: numpy.ndarray, matrix or nested sequences of shape (1,3)

    :returns: a tuple of two items:

        * the longitude angle in radians (float)
        * the latitude angle in radians (float)

    .. seealso:: |MANUAL| page 39
    """
    theta = _ct.c_double()
    phi = _ct.c_double()
    _sofa.iauC2s(_req_shape_c(p, float, (1,3)),
                                            _ct.byref(theta), _ct.byref(phi))
    return theta.value, phi.value


# iauC2t00a
_sofa.iauC2t00a.argtypes = [_ct.c_double, #tta
                            _ct.c_double, #ttb
                            _ct.c_double, #uta
                            _ct.c_double, #utb
                            _ct.c_double, #xp
                            _ct.c_double, #yp
                            _ndpointer(shape=(3,3), dtype=float, flags='C')]
def c2t00a(tta, ttb, uta, utb, xp, yp):
    """ Form the celestial-to-terrestrial matrix given the date, the UT1 and
    the polar motion, using IAU 2000A nutation model.

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param xp, yp: coordinates of the pole in radians.
    :type xp, yp: float

    :returns: the celestial-to-terrestrial matrix, as a numpy.matrix of shape
        3x3.

    .. seealso:: |MANUAL| page 40
    """
    rc2t = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauC2t00a(tta, ttb, uta, utb, float(xp), float(yp), rc2t)
    return rc2t


# iauC2t00b
_sofa.iauC2t00b.argtypes = [_ct.c_double, #tta
                            _ct.c_double, #ttb
                            _ct.c_double, #uta
                            _ct.c_double, #utb
                            _ct.c_double, #xp
                            _ct.c_double, #yp
                            _ndpointer(shape=(3,3), dtype=float, flags='C')]
def c2t00b(tta, ttb, uta, utb, xp, yp):
    """ Form the celestial-to-terrestrial matrix given the date, the UT1 and
    the polar motion, using IAU 2000B nutation model.

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param xp, yp: coordinates of the pole in radians.
    :type xp, yp: float

    :returns: the celestial-to-terrestrial matrix, as a numpy.matrix of shape
        3x3.

    .. seealso:: |MANUAL| page 42
    """
    rc2t = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauC2t00b(tta, ttb, uta, utb, float(xp), float(yp), rc2t)
    return rc2t


# iauC2t06a
_sofa.iauC2t06a.argtypes = [_ct.c_double, #tta
                            _ct.c_double, #ttb
                            _ct.c_double, #uta
                            _ct.c_double, #utb
                            _ct.c_double, #xp
                            _ct.c_double, #yp
                            _ndpointer(shape=(3,3), dtype=float, flags='C')]
def c2t06a(tta, ttb, uta, utb, xp, yp):
    """ Form the celestial-to-terrestrial matrix given the date, the UT1 and
    the polar motion, using the IAU 2006 precession and IAU 2000A nutation
    models.

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param xp, yp: coordinates of the pole in radians.
    :type xp, yp: float

    :returns: the celestial-to-terrestrial matrix, as a nunmp.matrix of shape
        3x3.

    .. seealso:: |MANUAL| page 44
    """
    rc2t = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauC2t06a(tta, ttb, uta, utb, float(xp), float(yp), rc2t)
    return rc2t


# iauC2tcio
_sofa.iauC2tcio.argtypes = [
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rc2i
                        _ct.c_double, #era
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rpom
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rc2t
def c2tcio(rc2i, era, rpom):
    """ Assemble the celestial-to-terrestrial matrix from CIO-based
    components (the celestial-to-intermediate matrix, the Earth Rotation Angle
    and the polar motion matrix).

    :param rc2i: celestial-to-intermediate matrix.
    :type rc2i: array-like object of shape (3,3)

    :param era: Earth rotation angle
    :type era: float

    :param rpom: polar-motion matrix.
    :type rpom: array-like of shape (3,3)

    :returns: celestial-to-terrestrial matrix as a numpy.matrix of shape
        3x3.

    .. seealso:: |MANUAL| page 46
    """
    rc2t = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauC2tcio(_req_shape_c(rc2i, float, (3,3)), float(era),
                                        _req_shape_c(rpom, float, (3,3)), rc2t)
    return rc2t


# iauC2teqx
_sofa.iauC2teqx.argtypes = [
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rbpn
                        _ct.c_double, #gst
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rpom
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rc2t
def c2teqx(rbpn, gst, rpom):
    """ Assemble the celestial-to-terrestrial matrix from equinox-based
    components (the celestial-to-true matrix, the Greenwich Apparent Sidereal
    Time and the polar motion matrix).

    :param rbpn: celestial-to-true matrix.
    :type rbpn: array-like of shape (3,3)

    :param gst: Greenwich apparent sidereal time.
    :type gst: float

    :param rpom: polar-motion matrix.
    :type rpom: array-like of shape (3,3)

    :returns: celestial-to-terrestrial matrix as a numpy.matrix of shape
        3x3.

    *sofa manual.pdp page 47*
    """
    rc2t = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauC2teqx(_req_shape_c(rbpn, float, (3,3)),
                    float(gst), _req_shape_c(rpom, float, (3,3)), rc2t)
    return rc2t


# iauC2tpe
_sofa.iauC2tpe.argtypes = [_ct.c_double, #tta,
                            _ct.c_double, #ttb
                            _ct.c_double, #uta
                            _ct.c_double, #utb
                            _ct.c_double, #dpsi
                            _ct.c_double, #deps
                            _ct.c_double, #xp
                            _ct.c_double, #yp
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rc2t
def c2tpe(tta, ttb, uta, utb, dpsi, deps, xp, yp):
    """ Form the celestial-to-terrestrial matrix given the date, the UT1,
    the nutation and the polar motion. IAU 2000.

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param dpsi, deps: nutation
    :type dpsi, deps: float

    :param xp, yp: coordinates of the pole in radians.
    :type xp, yp: float

    :returns: the celestial-to-terrestrial matrix as a nump.matrix of shape
        3x3.

    .. seealso:: |MANUAL| page 48
    """
    rc2t = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauC2tpe(tta, ttb, uta, utb, float(dpsi), float(deps), float(xp),
                                                            float(yp), rc2t)
    return rc2t



# iauC2txy
_sofa.iauC2txy.argtypes = [_ct.c_double, #tta
                            _ct.c_double, #ttb
                            _ct.c_double, #uta
                            _ct.c_double, #utb
                            _ct.c_double, #x
                            _ct.c_double, #y,
                            _ct.c_double, #xp,
                            _ct.c_double, #yp
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rc2t
def c2txy(tta, ttb, uta, utb, x, y, xp, yp):
    """ Form the celestial-to-terrestrial matrix given the date, the UT1,
    the CIP coordinates and the polar motion. IAU 2000.

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param x, y: Celestial Intermediate Pole.
    :type x, y: float

    :param xp, yp: coordinates of the pole in radians.
    :type xp, yp: float

    :returns: celestial-to-terrestrial matrix as a numpy.matrix of shape
        3x3.

    .. seealso:: |MANUAL| page 50
    """
    rc2t = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauC2txy(tta, ttb, uta, utb, float(x), float(y),
                                                float(xp), float(yp), rc2t)
    return rc2t


# iauCal2jd
_sofa.iauCal2jd.argtypes = [_ct.c_int, #iy
                            _ct.c_int, #im
                            _ct.c_int, #id
                            _ct.POINTER(_ct.c_double), #djm0
                            _ct.POINTER(_ct.c_double)] #djm
_sofa.iauCal2jd.restype = _ct.c_int
_cal2jd_msg = {
            -1: 'minimum year allowed is -4799',
            -2: 'month must be in 1..12',
            -3: 'day is out of range for this month'}

def cal2jd(iy, im, id):
    """ Gregorian calendar to Julian date.

    :param iy: year.
    :type iy: int

    :param im: month.
    :type im: int

    :param id: day.
    :type id: int

    :returns: a tuple of two items:

        * MJD zero-point : always 2400000.5 (float)
        * Modified Julian date for 0 hours (float)

    :raises: :exc:`ValueError` if one of the supplied values is out of its
        allowed range.

    .. seealso:: |MANUAL| page 52
    """
    djm0 = _ct.c_double()
    djm = _ct.c_double()
    s = _sofa.iauCal2jd(iy, im, id, _ct.byref(djm0), _ct.byref(djm))
    if s != 0:
        raise ValueError(_cal2jd_msg[s])
    return djm0.value, djm.value


# iauCp
_sofa.iauCp.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C'), #p
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #c
def cp(p):
    """ Copy a p-vector.

    :param p: p-vector to copy.
    :type p: array-like of shape (1,3)

    :returns: a copy of *p* as a numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 53
    """

    c = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauCp(_req_shape_c(p, float, (1,3)), c)
    return c


# iauCpv
_sofa.iauCpv.argtypes = [_ndpointer(shape=(2,3), dtype=float, flags='C'), #pv
                        _ndpointer(shape=(2,3), dtype=float, flags='C')] #c
def cpv(pv):
    """ Copy a pv-vector.

    :param pv: pv-vector to copy.
    :type pv: array-like of shape (2,3)

    :returns: a copy of *pv* as a numpy.matrix of shape 2x3.

    .. seealso:: |MANUAL| page 54
    """

    c = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    _sofa.iauCpv(_req_shape_c(pv, float, (2,3)), c)
    return c


# iauCr
_sofa.iauCr.argtypes = [_ndpointer(shape=(3,3), dtype=float, flags='C'), #r
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #c
def cr(r):
    """ Copy a rotation matrix.

    :param r: rotation matrix to copy.
    :type r: array-like of shape (3,3)

    :returns: a copy of *r* as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 55
    """

    c = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauCr(_req_shape_c(r, float, (3,3)), c)
    return c


# iauD2dtf
# the routine was added in release 2010-12-01
try:
    _sofa.iauD2dtf.argtypes = [_ct.c_char_p, #scale
                            _ct.c_int, #ndp
                            _ct.c_double, #d1
                            _ct.c_double, #d2
                            _ct.POINTER(_ct.c_int), #iy
                            _ct.POINTER(_ct.c_int), #im
                            _ct.POINTER(_ct.c_int), #id
                            _ct.c_int * 4] #ihmsf
    _sofa.iauD2dtf.restype = _ct.c_int
except AttributeError:
    pass
_d2dtf_msg = {
            1: 'D2dtf: dubious year',
            -1: 'unacceptable date',
            }
def d2dtf(scale, ndp, d1, d2):
    """ Format for output a 2-part Julian Date.

    :param scale: timescale ID.
    :type scale: str

    :param ndp: resolution.
    :type ndp: int

    :param d1, d2: time as a two-part Julian Date.
    :type d1, d2: float

    :returns: a tuple of 7 items:

        * year (int)
        * month (int)
        * day (int)
        * hours (int)
        * minutes (int)
        * seconds (int)
        * fraction of second (int)

    :raises: :exc:`ValueError` if the date is outside the range of valid values
        handled by this function.

        :exc:`UserWarning` if *scale* is "UTC" and the value predates the
        introduction of the timescale or is too far in the future to be
        trusted.

        :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 56
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    iy = _ct.c_int()
    im = _ct.c_int()
    id = _ct.c_int()
    ihmsf = (_ct.c_int * 4)()
    s = _sofa.iauD2dtf(scale, ndp, d1, d2, _ct.byref(iy), _ct.byref(im),
                                                        _ct.byref(id), ihmsf)
    if s < 0:
        raise ValueError(_d2dtf_msg[s])
    elif s > 0:
        _warnings.warn(_d2dtf_msg[s], UserWarning, 2)
    return (iy.value, im.value, id.value) + tuple([v for v in ihmsf])




# iauD2tf
_sofa.iauD2tf.argtypes = [_ct.c_int, #ndp
                            _ct.c_double, #days
                            _ct.POINTER(_ct.c_char), #sign
                            _ct.c_int * 4] #ihmsf
def d2tf(ndp, days):
    """ Decompose days into hours, minutes, seconds, fraction.

    :param ndp: the requested resolution.
    :type ndp: int

    :param days: the value to decompose.
    :type days: float

    :returns: a tuple of two items:

        * the sign as a string ('+' or '-')
        * a tuple (hours, minutes, seconds, fraction).

    .. seealso:: |MANUAL| page 58
    """
    sign = _ct.c_char()
    ihmsf = (_ct.c_int * 4)()
    _sofa.iauD2tf(ndp, days, _ct.byref(sign), ihmsf)
    return sign.value, tuple([v for v in ihmsf])


# iauDat
_sofa.iauDat.argtypes = [_ct.c_int, #iy
                        _ct.c_int, #im
                        _ct.c_int, #id
                        _ct.c_double, #fd
                        _ct.POINTER(_ct.c_double)] #deltat
_sofa.iauDat.restype = _ct.c_int
_dat_msg = {
        1: 'Dat: dubious year',
        -1: 'minimum year allowed is -4799',
        -2: 'month must be in 1..12',
        -3: 'day is out of range for this month',
        -4: 'bad fraction of day',}
def dat(iy, im, id, fd):
    """ Calculate delta(AT) = TAI - UTC for a given UTC date.

    :param iy: UTC year.
    :type iy: int

    :param im: month.
    :type im: int

    :param id: day.
    :type id: int

    :param fd: fraction of day.
    :type fd: float

    :returns: deltat (TAI-UTC) in seconds as a float.

    :raises: :exc:`ValueError` if *iy*, *im*, *id* or *fd* are not in valid
        ranges.

        :exc:`UserWarning` if the value predates the
        introduction of UTC or is too far in the future to be
        trusted.

    .. seealso:: |MANUAL| page 59
    """
    deltat = _ct.c_double()
    s = _sofa.iauDat(iy, im, id, fd, _ct.byref(deltat))
    if s < 0:
        raise ValueError(_dat_msg[s])
    elif s > 0:
        _warnings.warn(_dat_msg[s], UserWarning, 2)
    return deltat.value


# iauDtdb
_sofa.iauDtdb.argtypes = [_ct.c_double, #date1,
                            _ct.c_double, #date2
                            _ct.c_double, #ut
                            _ct.c_double, #elong
                            _ct.c_double, #u
                            _ct.c_double] #v
_sofa.iauDtdb.restype = _ct.c_double
def dtdb(date1, date2, ut, elong, u, v):
    """ Approximation of TDB - TT, the difference between barycentric dynamical
    time and terrestrial time, for an observer on Earth.

    :param date1, date2: TDB as a two-part date.
    :type date1, date2: float

    :param ut: universal time (UT1, fraction of one day).
    :type ut: float

    :param elong: longitude in radians (east positive)
    :type elong: float

    :param u: distance from Earth's spin axis in kilometers.
    :type u: float

    :param v: distance north of equatorial plane in kilometers
    :type v: float

    :returns: TDB - TT in seconds (float)

    .. seealso:: |MANUAL| page 61
    """
    return _sofa.iauDtdb(date1, date2, ut, float(elong), u, v)


# iauDtf2d
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauDtf2d.argtypes = [_ct.c_char_p, #scale
                            _ct.c_int, #iy
                            _ct.c_int, #im
                            _ct.c_int, #id
                            _ct.c_int, #ihr
                            _ct.c_int, #imn
                            _ct.c_double, #sec
                            _ct.POINTER(_ct.c_double), #d1
                            _ct.POINTER(_ct.c_double)] #d2
    _sofa.iauDtf2d.restype = _ct.c_int
except AttributeError:
    pass
_dtf2d_msg = {
        3: 'Dtf2d: dubious year and time is after end of day',
        2: 'Dtf2d: time is after end of day',
        1: 'Dtf2d: dubious year',
        -1: 'minimum year allowed is -4799',
        -2: 'month must be in 1..12',
        -3: 'day is out of range for this month',
        -4: 'hour must be in 0..23',
        -5: 'minute must be in 0..59',
        -6: 'second < 0',}
def dtf2d(scale, iy, im, id, ihr, imn, sec):
    """ Encode date and time fields into a two-part Julian Date.

    :param scale: Timescale id.
    :type scale: str

    :param iy: year.
    :type iy: int

    :param im: month.
    :type im: int

    :param id: day.
    :type id: int

    :param ihr: hour.
    :type ihr: int

    :param imn: minute.
    :type imn: int

    :param sec: seconds.
    :type sec: float

    :returns: the two-part Julian Date as a tuple of floats.

    :raises: :exc:`ValueError` if supplied values for *iy*, *im*, etc. are
        outside their valid range.

        :exc:`UserWarning` if the value predates the
        introduction of UTC or is too far in the future to be
        trusted.

        :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 64
    """

    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    d1 = _ct.c_double()
    d2 = _ct.c_double()
    s = _sofa.iauDtf2d(scale, iy, im, id, ihr, imn, sec, _ct.byref(d1),
                                                                _ct.byref(d2))
    if s < 0:
        raise ValueError(_dtf2d_msg[s])
    elif s > 0:
        _warnings.warn(_dtf2d_msg[s], UserWarning, 2)
    return d1.value, d2.value



# iauEe00
_sofa.iauEe00.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.c_double, # epsa
                            _ct.c_double] #dpsi
_sofa.iauEe00.restype = _ct.c_double
def ee00(date1, date2, epsa, dpsi):
    """ The equation of the equinoxes, compatible with IAU 2000 resolutions,
    given the nutation in longitude and the mean obliquity.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :param epsa: mean obliquity.
    :type epsa: float

    :param dpsi: nutation in longitude.
    :type dpsi: float

    :returns: equation of the equinoxes (float).

    .. seealso:: |MANUAL| page 66
    """
    return _sofa.iauEe00(date1, date2, float(epsa), float(dpsi))

# iauEe00a
_sofa.iauEe00a.argtypes = [_ct.c_double, #date1
                            _ct.c_double] #date2
_sofa.iauEe00a.restype = _ct.c_double
def ee00a(date1, date2):
    """ Equation of the equinoxes, compatible with IAU 2000 resolutions.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: equation of the equinoxes (float)

    .. seealso:: |MANUAL| page 67
    """
    return _sofa.iauEe00a(date1, date2)


# iauEe00b
_sofa.iauEe00b.argtypes = [_ct.c_double, #date1
                            _ct.c_double] #date2
_sofa.iauEe00b.restype = _ct.c_double
def ee00b(date1, date2):
    """ Equation of the equinoxes, compatible with IAU 2000 resolutions, using
    truncated nutation model IAU 2000B.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: equation of the equinoxes (float)

    .. seealso:: |MANUAL| page 68
    """
    return _sofa.iauEe00b(date1, date2)


# iauEe06a
_sofa.iauEe06a.argtypes = [_ct.c_double, #date1
                            _ct.c_double] #date2
_sofa.iauEe06a.restype = _ct.c_double
def ee06a(date1, date2):
    """ Equation of the equinoxes, compatible with IAU 2000 resolutions and
    IAU 2006/2000A precession-nutation.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: equation of the equinoxes (float)

    .. seealso:: |MANUAL| page 69
    """
    return _sofa.iauEe06a(date1, date2)


# iauEect00
_sofa.iauEect00.argtypes = [_ct.c_double, #date1
                            _ct.c_double] #date2
_sofa.iauEect00.restype = _ct.c_double
def eect00(date1, date2):
    """ Equation of the equinoxes complementary terms, consistent with IAU
    2000 resolutions.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: complementary terms (float).

    .. seealso:: |MANUAL| page 70
    """
    return _sofa.iauEect00(date1, date2)


# iauEform
_sofa.iauEform.argtypes = [_ct.c_int, #n
                            _ct.POINTER(_ct.c_double), #a
                            _ct.POINTER(_ct.c_double)] #f
_sofa.iauEform.restype = _ct.c_int
def eform(n):
    """ Earth's reference ellipsoids.

    :param n: ellipsoid identifier, should be one of:

        #. WGS84
        #. GRS80
        #. WGS72
    :type n: int

    :returns: a tuple of two items:

        * equatorial radius in meters (float)
        * flattening (float)

    .. seealso:: |MANUAL| page 72
    """
    a = _ct.c_double()
    f = _ct.c_double()
    s = _sofa.iauEform(n, _ct.byref(a), _ct.byref(f))
    if s != 0:
        raise ValueError('illegal ellipsoid identifier')
    return a.value, f.value


# iauEo06a
_sofa.iauEo06a.argtypes = [_ct.c_double, #date1
                            _ct.c_double] #date2
_sofa.iauEo06a.restype = _ct.c_double
def eo06a(date1, date2):
    """ Equation of the origins, IAU 2006 precession and IAU 2000A nutation.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: equation of the origins in radians (float).

    .. seealso:: |MANUAL| page 73
    """
    return _sofa.iauEo06a(date1, date2)


# iauEors
_sofa.iauEors.argtypes = [
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rnpb
                        _ct.c_double] #s
_sofa.iauEors.restype = _ct.c_double
def eors(rnpb, s):
    """ Equation of the origins, given the classical NPB matrix and the
    quantity s.

    :param rnpb: classical nutation x precession x bias matrix.
    :type rnpb: array-like of shape (3,3)

    :param s: the CIO locator.
    :type s: float

    :returns: the equation of the origins in radians (float).

    .. seealso:: |MANUAL| page 74
    """
    return _sofa.iauEors(_req_shape_c(rnpb, float, (3,3)), float(s))


# iauEpb
_sofa.iauEpb.argtypes = [_ct.c_double, #dj1
                        _ct.c_double] #dj2
_sofa.iauEpb.restype = _ct.c_double
def epb(dj1, dj2):
    """ Julian date to Besselian epoch.

    :param dj1, dj2: two-part Julian date.
    :type date1, date2: float

    :returns: Besselian epoch (float).

    .. seealso:: |MANUAL| page 75
    """
    return _sofa.iauEpb(dj1, dj2)


# iauEpb2jd
_sofa.iauEpb2jd.argtypes = [_ct.c_double, #epb
                            _ct.POINTER(_ct.c_double), #djm0
                            _ct.POINTER(_ct.c_double)] #djm
def epb2jd(epb):
    """ Besselian epoch to Julian date.

    :param epb: Besselian epoch.
    :type epb: float

    :returns: a tuple of two items:

        * MJD zero-point, always 2400000.5 (float)
        * modified Julian date (float).

    .. seealso:: |MANUAL| page 76
    """
    djm0 = _ct.c_double()
    djm = _ct.c_double()
    _sofa.iauEpb2jd(epb, _ct.byref(djm0), _ct.byref(djm))
    return djm0.value, djm.value


# iauEpj
_sofa.iauEpj.argtypes = [_ct.c_double, #dj1
                        _ct.c_double] #dj2
_sofa.iauEpj.restype = _ct.c_double
def epj(dj1, dj2):
    """ Julian date to Julian epoch.

    :param dj1, dj2: two-part Julian date.
    :type dj1, dj2: float

    :returns: Julian epoch (float)

    .. seealso:: |MANUAL| page 77
    """
    return _sofa.iauEpj(dj1, dj2)


# iauEpj2jd
_sofa.iauEpj2jd.argtypes = [_ct.c_double, #epj
                _ct.POINTER(_ct.c_double), #djm0
                _ct.POINTER(_ct.c_double)] #djm
def epj2jd(epj):
    """ Julian epoch to Julian date.

    :param epj: Julian epoch.
    :type epj: float

    :returns: a tuple of two items:

        * MJD zero-point, always 2400000.5 (float)
        * modified Julian date (float).

    .. seealso:: |MANUAL| page 78
    """
    djm0 = _ct.c_double()
    djm = _ct.c_double()
    _sofa.iauEpj2jd(epj, _ct.byref(djm0), _ct.byref(djm))
    return djm0.value, djm.value


# iauEpv00
_sofa.iauEpv00.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(2,3), dtype=float, flags='C'), #pvh
                        _ndpointer(shape=(2,3), dtype=float, flags='C')] # pvb
_sofa.iauEpv00.restype = _ct.c_int
_epv00_msg = {
            1: 'Epv00: date outside the range 1900-2100 AD',
            }
def epv00(date1, date2):
    """ Earth position and velocity, heliocentric and barycentric, with
    respect to the Barycentric Celestial Reference System.

    :param date1, date2: TDB as a two-part Julian date.
    :type date1, date2: float

    :returns: a tuple of two items:

        * heliocentric Earth position velocity as a numpy.matrix of shape \
           2x3.
        * barycentric Earth position/velocity as a numpy.matrix of shape \
           2x3.

    :raises: :exc:`UserWarning` if the date falls outside the range 1900-2100.

    .. seealso:: |MANUAL| page 79
    """
    pvh = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    pvb = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    s = _sofa.iauEpv00(date1, date2, pvh, pvb)
    if s != 0:
        _warnings.warn(_epv00_msg[s], UserWarning, 2)
    return pvh, pvb


# iauEqeq94
_sofa.iauEqeq94.argtypes = [_ct.c_double, #date1
                            _ct.c_double] #date2
_sofa.iauEqeq94.restype = _ct.c_double
def eqeq94(date1, date2):
    """ Equation of the equinoxes, IAU 1994 model.

    :param date1, date2: TDB as a two-part Julian date.
    :type date1, date2: float

    :returns: equation of the equinoxes (float).

    .. seealso:: |MANUAL| page 81
    """
    return _sofa.iauEqeq94(date1, date2)


# iauEra00
_sofa.iauEra00.argtypes = [_ct.c_double, #dj1
                            _ct.c_double] #dj2
_sofa.iauEra00.restype = _ct.c_double
def era00(dj1, dj2):
    """ Earth rotation angle IAU 2000 model.

    :param dj1, dj2: UT1 as a two-part Julian date.
    :type dj1, dj2: float

    :returns: Earth rotation angle in radians, in the range 0-2pi (float).

    .. seealso:: |MANUAL| page 82
    """
    return _sofa.iauEra00(dj1, dj2)


# iauFad03
_sofa.iauFad03.argtypes = [_ct.c_double] #t
_sofa.iauFad03.restype = _ct.c_double
def fad03(t):
    """ Mean elongation of the Moon from the Sun (fundamental argument, IERS
    conventions 2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean elongation of the Moon from the Sun in radians (float).

    .. seealso:: |MANUAL| page 83
    """
    return _sofa.iauFad03(t)


# iauFae03
_sofa.iauFae03.argtypes = [_ct.c_double] #t
_sofa.iauFae03.restype = _ct.c_double
def fae03(t):
    """ Mean longitude of Earth (fundamental argument, IERS conventions 2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Earth in radians (float).

    .. seealso:: |MANUAL| page 84
    """
    return _sofa.iauFae03(t)


# iauFaf03
_sofa.iauFaf03.argtypes = [_ct.c_double] #t
_sofa.iauFaf03.restype = _ct.c_double
def faf03(t):
    """ Mean longitude of the Moon minus mean longitude of the ascending node
    (fundamental argument, IERS conventions 2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: result in radians (float).

    .. seealso:: |MANUAL| page 85
    """
    return _sofa.iauFaf03(t)


# iauFaju03
_sofa.iauFaju03.argtypes = [_ct.c_double] #t
_sofa.iauFaju03.restype = _ct.c_double
def faju03(t):
    """ Mean longitude of Jupiter (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Jupiter in radians (float).

    .. seealso:: |MANUAL| page 86
    """
    return _sofa.iauFaju03(t)


# iauFal03
_sofa.iauFal03.argtypes = [_ct.c_double] #t
_sofa.iauFal03.restype = _ct.c_double
def fal03(t):
    """ Mean anomaly of the Moon (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean anomaly of the Moon in radians (float).

    .. seealso:: |MANUAL| page 87
    """
    return _sofa.iauFal03(t)


# iauFalp03
_sofa.iauFalp03.argtypes = [_ct.c_double] #t
_sofa.iauFalp03.restype = _ct.c_double
def falp03(t):
    """ Mean anomaly of the Sun (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean anomaly of the Sun in radians (float).

    .. seealso:: |MANUAL| page 88
    """
    return _sofa.iauFalp03(t)


# iauFama03
_sofa.iauFama03.argtypes = [_ct.c_double] #t
_sofa.iauFama03.restype = _ct.c_double
def fama03(t):
    """ Mean longitude of Mars (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Mars in radians (float).

    .. seealso:: |MANUAL| page 89
    """
    return _sofa.iauFama03(t)


# iauFame03
_sofa.iauFame03.argtypes = [_ct.c_double] #t
_sofa.iauFame03.restype= _ct.c_double
def fame03(t):
    """ Mean longitude of Mercury (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Mercury in radians (float).

    .. seealso:: |MANUAL| page 90
    """
    return _sofa.iauFame03(t)


# iauFane03
_sofa.iauFane03.argtypes = [_ct.c_double] #t
_sofa.iauFane03.restype = _ct.c_double
def fane03(t):
    """ Mean longitude of Neptune (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Neptune in radians (float).

    .. seealso:: |MANUAL| page 91
    """
    return _sofa.iauFane03(t)


# iauFaom03
_sofa.iauFaom03.argtypes = [_ct.c_double] #t
_sofa.iauFaom03.restype = _ct.c_double
def faom03(t):
    """ Mean longitude of the Moon's ascending node (fundamental argument,
    IERS conventions 2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of of the Moon's ascending node in radians
        (float).

    .. seealso:: |MANUAL| page 92
    """
    return _sofa.iauFaom03(t)


# iauFapa03
_sofa.iauFapa03.argtypes = [_ct.c_double] #t
_sofa.iauFapa03.restype = _ct.c_double
def fapa03(t):
    """ General accumulated precession in longitude (fundamental argument,
    IERS conventions 2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: general accumulated precession in longitude in radians
        (float).

    .. seealso:: |MANUAL| page 93
    """
    return _sofa.iauFapa03(t)


# iauFasa03
_sofa.iauFasa03.argtypes = [_ct.c_double] #t
_sofa.iauFasa03.restype = _ct.c_double
def fasa03(t):
    """ Mean longitude of Saturn (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Saturn in radians (float).

    .. seealso:: |MANUAL| page 94
    """
    return _sofa.iauFasa03(t)


# iauFaur03
_sofa.iauFaur03.argtypes = [_ct.c_double] #t
_sofa.iauFaur03.restype = _ct.c_double
def faur03(t):
    """ Mean longitude of Uranus (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Uranus in radians (float).

    .. seealso:: |MANUAL| page 95
    """
    return _sofa.iauFaur03(t)


# iauFave03
_sofa.iauFave03.argtypes = [_ct.c_double] #t
_sofa.iauFave03.restype = _ct.c_double
def fave03(t):
    """ Mean longitude of Venus (fundamental argument, IERS conventions
    2003).

    :param t: TDB in Julian centuries since J2000.0
    :type t: float

    :returns: mean longitude of Venus in radians (float).

    .. seealso:: |MANUAL| page 96
    """
    return _sofa.iauFave03(t)


# iauFk52h
_sofa.iauFk52h.argtypes = [_ct.c_double, #r5
                            _ct.c_double, #d5
                            _ct.c_double, #dr5
                            _ct.c_double, #dd5
                            _ct.c_double, #px5
                            _ct.c_double, #rv5
                            _ct.POINTER(_ct.c_double), #rh
                            _ct.POINTER(_ct.c_double), #dh
                            _ct.POINTER(_ct.c_double), #drh
                            _ct.POINTER(_ct.c_double), #ddh
                            _ct.POINTER(_ct.c_double), #pxh
                            _ct.POINTER(_ct.c_double)] #rvh
def fk52h(r5, d5, dr5, dd5, px5, rv5):
    """ Transform FK5 (J2000.0) star data into the Hipparcos system.

    :param r5: right ascension in radians.
    :type r5: float

    :param d5: declination in radians.
    :type d5: float

    :param dr5: proper motion in RA (dRA/dt, rad/Jyear).
    :type dr5: float

    :param dd5: proper motion in Dec (dDec/dt, rad/Jyear).
    :type dd5: float

    :param px5: parallax (arcseconds)
    :type px5: float

    :param rv5: radial velocity (km/s, positive = receding)
    :type rv5: float

    :returns: a tuple of six items corresponding to Hipparcos epoch J2000.0:

        * right ascension
        * declination
        * proper motion in RA (dRa/dt, rad/Jyear)
        * proper motion in Dec (dDec/dt, rad/Jyear)
        * parallax in arcseconds
        * radial velocity (km/s, positive = receding).

    .. seealso:: |MANUAL| page 97
    """
    rh = _ct.c_double()
    dh = _ct.c_double()
    drh = _ct.c_double()
    ddh = _ct.c_double()
    pxh = _ct.c_double()
    rvh = _ct.c_double()
    _sofa.iauFk52h(float(r5), float(d5), float(dr5), float(dd5), float(px5),
                    float(rv5),
                    _ct.byref(rh), _ct.byref(dh), _ct.byref(drh),
                    _ct.byref(ddh), _ct.byref(pxh), _ct.byref(rvh))
    return rh.value, dh.value, drh.value, ddh.value, pxh.value, rvh.value


# iauFk5hip
_sofa.iauFk5hip.argtypes = [
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #r5h
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #s5h
def fk5hip():
    """ FK5 to Hipparcos rotation and spin.

    :returns: a tuple of two items:

        * FK5 rotation wrt Hipparcos as a numpy.matrix of shape 3x3
        * FK5 spin wrt Hipparcos as a numpy.matrix of shape 1x3

    .. seealso:: |MANUAL| page 98
    """
    r5h = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    s5h = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauFk5hip(r5h, s5h)
    return r5h, s5h


# iauFk5hz
_sofa.iauFk5hz.argtypes = [_ct.c_double, #r5
                            _ct.c_double, #d5
                            _ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #rh
                            _ct.POINTER(_ct.c_double)] #dh
def fk5hz(r5, d5, date1, date2):
    """ Transform an FK5 (J2000.0) star position into the system of the
    Hipparcos catalogue, assuming zero Hipparcos proper motion.

    :param r5: right ascension in radians, equinox J2000.0, at date.
    :type r5: float

    :param d5: declination in radians, equinox J2000.0, at date.
    :type d5: float

    :param date1, date2: TDB date as a two-part Julian date.
    :type date1, date2: float

    :returns: a tuple of two items:

        * Hipparcos right ascension in radians (float)
        * Hipparcos declination in radians (float).

    .. seealso:: |MANUAL| page 99
    """
    rh = _ct.c_double()
    dh = _ct.c_double()
    _sofa.iauFk5hz(float(r5), float(d5), date1, date2, _ct.byref(rh),
                                                                _ct.byref(dh))
    return rh.value, dh.value


# iauFw2m
_sofa.iauFw2m.argtypes = [_ct.c_double, #gamb
                            _ct.c_double, #phib
                            _ct.c_double, #psi
                            _ct.c_double, #eps
                            _ndpointer(shape=(3,3), dtype=float, flags='C')] #r
def fw2m(gamb, phib, psi, eps):
    """ Form rotation matrix given the Fukushima-Williams angles.

    :param gamb: F-W angle gamma_bar in radians.
    :type gamb: float

    :param phib: F-W angle phi_bar in radians.
    :type phib: float

    :param psi: F-W angle psi in radians.
    :type psi: float

    :param eps: F-W angle epsilon in radians.
    :type epsilon: float

    :returns: rotation matrix, as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 101
    """
    r = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauFw2m(float(gamb), float(phib), float(psi), float(eps), r)
    return r


# iauFw2xy
_sofa.iauFw2xy.argtypes = [_ct.c_double, #gamb
                            _ct.c_double, #phib
                            _ct.c_double, #psi
                            _ct.c_double, #eps
                            _ct.POINTER(_ct.c_double), #x
                            _ct.POINTER(_ct.c_double)] #y
def fw2xy(gamb, phib, psi, eps):
    """ CIP X and Y given Fukushima-Williams bias-precession-nutation angles.

    :param gamb: F-W angle gamma_bar in radians.
    :type gamb: float

    :param phib: F-W angle phi_bar in radians.
    :type phib: float

    :param psi: F-W angle psi in radians.
    :type psi: float

    :param eps: F-W angle epsilon in radians.
    :type epsilon: float

    :returns: a tuple containing CIP X and X in radians (float).

    .. seealso:: |MANUAL| page 103
    """
    x = _ct.c_double()
    y = _ct.c_double()
    _sofa.iauFw2xy(float(gamb), float(phib), float(psi), float(eps),
                                                    _ct.byref(x), _ct.byref(y))
    return x.value, y.value


# iauGc2gd
_sofa.iauGc2gd.argtypes = [_ct.c_int, #n
                        _ndpointer(shape=(1,3), dtype=float, flags='C'), #xyz
                        _ct.POINTER(_ct.c_double), #elong
                        _ct.POINTER(_ct.c_double), #phi
                        _ct.POINTER(_ct.c_double)] #height
_sofa.iauGc2gd.restype = _ct.c_int
_gc2gd_msg = {
        -1: 'Gc2gd: illegal ellipsoid identifier',
        }
def gc2gd(n, xyz):
    """ Transform geocentric coordinates to geodetic using the specified
    reference ellipsoid.

    :param n: ellipsoid identifier, should be one of:

        #. WGS84
        #. GRS80
    :type n: int

    :param xyz: geocentric vector.
    :type xyz: array-like of shape (1,3)

    :returns: a tuple of three items:

        * longitude in radians (float)
        * geodetic latitude in radians (float)
        * geodetic height above ellipsoid (float).

    :raises: :exc:`ValueError` for invalid ellipsoid identifier.

    .. seealso:: |MANUAL| page 104
    """
    elong = _ct.c_double()
    phi = _ct.c_double()
    height = _ct.c_double()
    s = _sofa.iauGc2gd(n, _req_shape_c(xyz, float, (1,3)),
                        _ct.byref(elong), _ct.byref(phi), _ct.byref(height))
    if s != 0:
        raise ValueError(_gc2gd_msg[s])
    return elong.value, phi.value, height.value


# iauGc2gde
_sofa.iauGc2gde.argtypes = [_ct.c_double, #a
                            _ct.c_double, #f
                        _ndpointer(shape=(1,3), dtype=float, flags='C'), #xyz
                        _ct.POINTER(_ct.c_double), #elong
                        _ct.POINTER(_ct.c_double), #phi
                        _ct.POINTER(_ct.c_double)] #height
_sofa.iauGc2gde.restype = _ct.c_int
_gc2gde_msg = {
        -1: 'Gc2gde: illegal value for flattening',
        -2: 'Gc2gde: illegal value for equatorial radius',
        }
def gc2gde(a, f, xyz):
    """ Transform geocentric coordinates to geodetic for a reference
    ellipsoid of specified form.

    :param a: equatorial radius.
    :type a: float

    :param f: flattening.
    :type f: float

    :param xyz: geocentric vector.
    :type xyz: array-like of shape (1,3)

    :returns: a tuple of three items:

        * longitude in radians
        * geodetic latitude in radians
        * geodetic height above ellipsoid

    :raises: :exc:`ValueError` if supplied values for *a* or *f* are nor valid.

    .. seealso:: |MANUAL| page 105
    """
    elong = _ct.c_double()
    phi = _ct.c_double()
    height = _ct.c_double()
    s = _sofa.iauGc2gde(a, f, _req_shape_c(xyz, float, (1,3)),
                        _ct.byref(elong), _ct.byref(phi), _ct.byref(height))
    if s != 0:
        raise ValueError(_gc2gde_msg[s])
    return elong.value, phi.value, height.value


# iauGd2gc
_sofa.iauGd2gc.argtypes = [_ct.c_int, #n
                            _ct.c_double, #elong
                            _ct.c_double, #phi,
                            _ct.c_double, #height
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #xyz
_sofa.iauGd2gc.restype = _ct.c_int
_gd2gc_msg = {
        -1: 'invalid ellipsoid identifier',
        -2: 'illegal case',
        }
def gd2gc(n, elong, phi, height):
    """ Transform geodetic coordinates to geocentric using specified reference
    ellipsoid.

    :param n: ellipsoid identifier, should be one of:

        #. WGS84
        #. GRS80
        #. WGS72
    :type n: int

    :param elong: longitude in radians.
    :type elong: float

    :param phi: geodetic latitude in radians.
    :type phi: float

    :param height: geodetic height above ellipsoid in meters.
    :type height: float

    :returns: geocentric vector as a numpy.matrix of shape 1x3.

    :raises: :exc:`ValueError` in case of invalid ellipsoid identifier or
        invalid coordinate values.

    .. seealso:: |MANUAL| page 106
    """
    xyz = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    s = _sofa.iauGd2gc(n, float(elong), float(phi), height, xyz)
    if s != 0:
        raise ValueError(_gd2gc_msg[s])
    return xyz


# iauGd2gce
_sofa.iauGd2gce.argtypes = [_ct.c_double, #a
                            _ct.c_double, #f
                            _ct.c_double, #elong
                            _ct.c_double, #phi
                            _ct.c_double, #height
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #xyz
_sofa.iauGd2gce.restype = _ct.c_int
_gd2gce_msg = {
        -1: 'illegal case'
        }
def gd2gce(a, f, elong, phi, height):
    """ Transform geodetic coordinates to geocentric for a reference
    ellipsoid of specified form.

    :param a: equatorial radius.
    :type a: float

    :param f: flattening.
    :type f: float

    :param elong: longitude in radians.
    :type elong: float

    :param phi: geodetic latitude in radians.
    :type phi: float

    :param height: geodetic height above ellipsoid in meters.
    :type height: float

    :returns: geocentric vector as a numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 107
    """
    xyz = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    s = _sofa.iauGd2gce(a, f, float(elong), float(phi), height, xyz)
    if s != 0:
        raise ValueError(_gd2gce_msg[s])
    return xyz


# iauGmst00
_sofa.iauGmst00.argtypes = [_ct.c_double, #uta
                            _ct.c_double, #utb
                            _ct.c_double, #tta
                            _ct.c_double] #ttb
_sofa.iauGmst00.restype = _ct.c_double
def gmst00(uta, utb, tta, ttb):
    """ Greenwich mean sidereal time, consistent with IAU 2000 resolutions.

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :returns: Greenwich mean sidereal time in radians (float).

    .. seealso:: |MANUAL| page 108
    """
    return _sofa.iauGmst00(uta, utb, tta, ttb)


# iauGmst06
_sofa.iauGmst06.argtypes = [_ct.c_double, #uta
                            _ct.c_double, #utb
                            _ct.c_double, #tta
                            _ct.c_double] #ttb
_sofa.iauGmst06.restype = _ct.c_double
def gmst06(uta, utb, tta, ttb):
    """ Greenwich mean sidereal time, consistent with IAU 2006 precession.

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :returns: Greenwich mean sidereal time in radians (float).

    .. seealso:: |MANUAL| page 110
    """
    return _sofa.iauGmst06(uta, utb, tta, ttb)


# iauGmst82
_sofa.iauGmst82.argtypes = [_ct.c_double, #dj1
                            _ct.c_double] #dj2
_sofa.iauGmst82.restype = _ct.c_double
def gmst82(dj1, dj2):
    """ Greenwich mean sidereal time, IAU 1982 model.

    :param dj1, dj2: UT1 as a two-part Julian date.
    :type uta, utb: float

    :returns: Greenwich mean sidereal time in radians (float).

    .. seealso:: |MANUAL| page 111
    """
    return _sofa.iauGmst82(dj1, dj2)


# iauGst00a
_sofa.iauGst00a.argtypes = [_ct.c_double, #uta
                            _ct.c_double, #utb
                            _ct.c_double, #tta
                            _ct.c_double] #ttb
_sofa.iauGst00a.restype = _ct.c_double
def gst00a(uta, utb, tta, ttb):
    """ Greenwich apparent sidereal time, consistent with IAU 2000 resolutions.

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :returns: Greenwich apparent sidereal time in radians (float).

    .. seealso:: |MANUAL| page 112
    """
    return _sofa.iauGst00a(uta, utb, tta, ttb)


# iauGst00b
_sofa.iauGst00b.argtypes = [_ct.c_double, #uta
                            _ct.c_double] #utb
_sofa.iauGst00b.restype = _ct.c_double
def gst00b(uta, utb):
    """ Greenwich apparent sidereal time, consistent with IAU 2000 resolutions,
    using truncated nutation model IAU 2000B.

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :returns: Greenwich apparent sidereal time in radians (float).

    .. seealso:: |MANUAL| page 114
    """
    return _sofa.iauGst00b(uta, utb)


# iauGst06
_sofa.iauGst06.argtypes = [_ct.c_double, #uta
                            _ct.c_double, #utb
                            _ct.c_double, #tta
                            _ct.c_double, #ttb
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rnpb
_sofa.iauGst06.restype = _ct.c_double
def gst06(uta, utb, tta, ttb, rnpb):
    """ Greenwich apparent sidereal time, IAU 2006, given the *npb* matrix.

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :param rnpb: nutation x precession x bias matrix.
    :type rnpb: array-like of shape (3,3)

    :returns: Greenwich apparent sidereal time in radians (float).

    .. seealso:: |MANUAL| page 116
    """
    return _sofa.iauGst06(uta, utb, tta, ttb,
                                _req_shape_c(rnpb, float, (3,3)))


# iauGst06a
_sofa.iauGst06a.argtypes = [_ct.c_double, #uta
                            _ct.c_double, #utb
                            _ct.c_double, #tta
                            _ct.c_double] #ttb
_sofa.iauGst06a.restype = _ct.c_double
def gst06a(uta, utb, tta, ttb):
    """ Greenwich apparent sidereal time, consistent with IAU 2000 and 2006
    resolutions.

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :param tta, ttb: TT as a two-part Julian date.
    :type tta, ttb: float

    :returns: Greenwich apparent sidereal time in radians (float).

    .. seealso:: |MANUAL| page 117
    """
    return _sofa.iauGst06a(uta, utb, tta, ttb)


# iauGst94
_sofa.iauGst94.argtypes = [_ct.c_double, #uta
                            _ct.c_double] #utb
_sofa.iauGst94.restype = _ct.c_double
def gst94(uta, utb):
    """ Greenwich apparent sidereal time, consistent with IAU 1982/94
    resolutions.

    :param uta, utb: UT1 as a two-part Julian date.
    :type uta, utb: float

    :returns: Greenwich apparent sidereal time in radians (float).

    .. seealso:: |MANUAL| page 118
    """
    return _sofa.iauGst94(uta, utb)


# iauH2fk5
_sofa.iauH2fk5.argtypes = [_ct.c_double, #rh
                            _ct.c_double, #dh
                            _ct.c_double, #drh
                            _ct.c_double, #ddh
                            _ct.c_double, #pxh
                            _ct.c_double, #rvh
                            _ct.POINTER(_ct.c_double), #r5
                            _ct.POINTER(_ct.c_double), #d5
                            _ct.POINTER(_ct.c_double), #dr5
                            _ct.POINTER(_ct.c_double), #dd5
                            _ct.POINTER(_ct.c_double), #px5
                            _ct.POINTER(_ct.c_double)] #rv5
def h2fk5(rh, dh, drh, ddh, pxh, rvh):
    """ Transform Hipparcos star data into FK5 (J2000.0) system.

    :param rh: right ascension in radians.
    :type rh: float

    :param dh: declination in radians.
    :type dh: float

    :param drh: proper motion in RA (dRA/dt, rad/Jyear).
    :type drh: float

    :param ddh: proper motion in Dec (dDec/dt, rad/Jyear).
    :type ddh: float

    :param pxh: parallax in arcseconds.
    :type pxh: float

    :param rvh: radial velocity (km/s, positive = receding).
    :type rvh: float

    :returns: a tuple of six items:

        * right ascension in radians
        * declination in radians
        * proper motion in RA (dRA/dt, rad/Jyear)
        * proper motion in Dec (dDec/dt, rad/Jyear)
        * parallax in arcseconds
        * radial velocity (km/s, positive = receding).

    .. seealso:: |MANUAL| page 119
    """
    r5 = _ct.c_double()
    d5 = _ct.c_double()
    dr5 = _ct.c_double()
    dd5 = _ct.c_double()
    px5 = _ct.c_double()
    rv5 = _ct.c_double()
    _sofa.iauH2fk5(float(rh), float(dh), float(drh), float(ddh),
                    float(pxh), float(rvh), _ct.byref(r5), _ct.byref(d5),
                    _ct.byref(dr5), _ct.byref(dd5), _ct.byref(px5),
                    _ct.byref(rv5))
    return r5.value, d5.value, dr5.value, dd5.value, px5.value, rv5.value


# iauHfk5z
_sofa.iauHfk5z.argtypes = [_ct.c_double, #rh
                            _ct.c_double, #dh
                            _ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #r5
                            _ct.POINTER(_ct.c_double), #d5
                            _ct.POINTER(_ct.c_double), #dr5
                            _ct.POINTER(_ct.c_double)] #dd5
def hfk5z(rh, dh, date1, date2):
    """ Transform Hipparcos star position into FK5 (J2000.0), assuming
    zero Hipparcos proper motion.

    :param rh: right ascension in radians.
    :type rh: float

    :param dh: declination in radians.
    :type dh: float

    :param date1, date2: TDB as a two-part Julian date.
    :type date1, date2: float

    :returns: a tuple of four items:

        * right ascension in radians
        * declination in radians
        * proper motion in RA (rad/year)
        * proper motion in Dec (rad/year)

    .. seealso:: |MANUAL| page 120
    """
    r5 = _ct.c_double()
    d5 = _ct.c_double()
    dr5 = _ct.c_double()
    dd5 = _ct.c_double()
    _sofa.iauHfk5z(float(rh), float(dh), date1, date2, _ct.byref(r5),
                                _ct.byref(d5) ,_ct.byref(dr5), _ct.byref(dd5))
    return r5.value, d5.value, dr5.value, dd5.value


# iauIr
_sofa.iauIr.argtypes = [_ndpointer(shape=(3,3), dtype=float, flags='C')] #r
def ir():
    """ Create a new rotation matrix initialized to the identity matrix.

    :returns: an identity matrix as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 122
    """

    r = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauIr(r)
    return r


# iauJd2cal
_sofa.iauJd2cal.argtypes = [_ct.c_double, #dj&
                            _ct.c_double, #dj2
                            _ct.POINTER(_ct.c_int), #iy
                            _ct.POINTER(_ct.c_int), #im
                            _ct.POINTER(_ct.c_int), #id
                            _ct.POINTER(_ct.c_double)] #fd
_sofa.iauJd2cal.restype = _ct.c_int
_jd2cal_msg = {
            -1: 'date outside valid range -68569.5 to 1e9',
            }
def jd2cal(dj1, dj2):
    """ Julian date to Gregorian year, month, day and fraction of day.

    :param dj1, dj2: two-part Julian date.
    :type dj1, dj2: float

    :returns: a tuple of five values:

        * year (int)
        * month (int)
        * day (int)
        * fraction of day (float)

    :raises: :exc:`ValueError` if input date is outside valid range.

    .. seealso:: |MANUAL| page 123
    """
    iy = _ct.c_int()
    im = _ct.c_int()
    id = _ct.c_int()
    fd = _ct.c_double()
    s = _sofa.iauJd2cal(dj1, dj2, _ct.byref(iy), _ct.byref(im), _ct.byref(id),
                            _ct.byref(fd))
    if s != 0:
        raise ValueError(_jd2cal_msg[s])
    return iy.value, im.value, id.value, fd.value


# iauJdcalf
_sofa.iauJdcalf.argtypes = [_ct.c_int, #ndp
                            _ct.c_double, #dj1
                            _ct.c_double, #dj2
                            _ct.c_int * 4] #iymdf
_sofa.iauJdcalf.restype = _ct.c_int
_jdcalf_msg = {
            -1: 'date out of range',
            1: 'Jdcalf: invalid value for "ndp", modified to be zero',
            }
def jdcalf(ndp, dj1, dj2):
    """ Julian date to Gregorian calendar, expressed in a form convenient
    for formatting messages: rounded to a specified precision.

    :param ndp: number of decimal places of days fraction.
    :type ndp: int

    :param dj1, dj2: two-part Julian date.
    :type dj1, dj2: float

    :returns: a 4-tuple containing year, month, day, fraction of day.

    :raises: :exc:`ValueError` if date is outside the valid range.

    .. seealso:: |MANUAL| page 124
    """
    iymdf = (_ct.c_int * 4)()
    s = _sofa.iauJdcalf(ndp, dj1, dj2, iymdf)
    if s < 0:
        raise ValueError(_jdcalf_msg[s])
    elif s > 0:
        _warnings.warn(_jdcalf_msg[s], UserWarning, 2)
    return tuple([v for v in iymdf])


# iauNum00a
_sofa.iauNum00a.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rmatn
def num00a(date1, date2):
    """ Form the matrix of nutation for a given date, IAU 2000A model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: nutation matrix, as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 125
    """
    rmatn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauNum00a(date1, date2, rmatn)
    return rmatn


# iauNum00b
_sofa.iauNum00b.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rmatn
def num00b(date1, date2):
    """ Form the matrix of nutation for a given date, IAU 2000B model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: nutation matrix, as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 126
    """
    rmatn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauNum00b(date1, date2, rmatn)
    return rmatn


# iauNum06a
_sofa.iauNum06a.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rmatn
def num06a(date1, date2):
    """ Form the matrix of nutation for a given date, IAU 2006/2000A model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: nutation matrix, as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 127
    """
    rmatn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauNum06a(date1, date2, rmatn)
    return rmatn


# iauNumat
_sofa.iauNumat.argtypes = [_ct.c_double, #epsa
                            _ct.c_double, #dpsi
                            _ct.c_double, #deps
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rmatn
def numat(epsa, dpsi, deps):
    """ Form the matrix of nutation.

    :param epsa: mean obliquity of date.
    :type epsa: float

    :param dpsi, deps: nutation.
    :type dpsi, deps: float

    :returns: nutation matrix as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 128
    """
    rmatn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauNumat(float(epsa), float(dpsi), float(deps), rmatn)
    return rmatn


# iauNut00a
_sofa.iauNut00a.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #dpsi
                            _ct.POINTER(_ct.c_double)] #deps
def nut00a(date1, date2):
    """ Nutation, IAU 2000A model (MHB2000 luni-solar and planetary nutation
    with free core nutation omitted).

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 2-tuple:

        * nutation in longitude in radians (float)
        * nutation in obliquity in radians (float).

    .. seealso:: |MANUAL| page 129
    """
    dpsi = _ct.c_double()
    deps = _ct.c_double()
    _sofa.iauNut00a(date1, date2, _ct.byref(dpsi), _ct.byref(deps))
    return dpsi.value, deps.value


# iauNut00b
_sofa.iauNut00b.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #dpsi
                            _ct.POINTER(_ct.c_double)] #deps
def nut00b(date1, date2):
    """ Nutation, IAU 2000B model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 2-tuple:

        * nutation in longitude in radians (float)
        * nutation in obliquity in radians (float).

    .. seealso:: |MANUAL| page 132
    """
    dpsi = _ct.c_double()
    deps = _ct.c_double()
    _sofa.iauNut00b(date1, date2, _ct.byref(dpsi), _ct.byref(deps))
    return dpsi.value, deps.value


# iauNut06a
_sofa.iauNut06a.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #dpsi
                            _ct.POINTER(_ct.c_double)] #deps
def nut06a(date1, date2):
    """ IAU 2000A nutation with adjustments to match the IAU 2006 precession.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 2-tuple:

        * nutation in longitude in radians (float)
        * nutation in obliquity in radians (float).

    .. seealso:: |MANUAL| page 134
    """
    dpsi = _ct.c_double()
    deps = _ct.c_double()
    _sofa.iauNut06a(date1, date2, _ct.byref(dpsi), _ct.byref(deps))
    return dpsi.value, deps.value


# iauNut80
_sofa.iauNut80.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #dpsi
                            _ct.POINTER(_ct.c_double)] #deps
def nut80(date1, date2):
    """ Nutation, IAU 1980 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 2-tuple:

        * nutation in longitude in radians (float)
        * nutation in obliquity in radians (float).

    .. seealso:: |MANUAL| page 136
    """
    dpsi = _ct.c_double()
    deps = _ct.c_double()
    _sofa.iauNut80(date1, date2, _ct.byref(dpsi), _ct.byref(deps))
    return dpsi.value, deps.value


# iauNutm80
_sofa.iauNutm80.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rmatn
def nutm80(date1, date2):
    """ Form the nutation matrix for a given date, IAU 1980 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: the nutation matrix, as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 137
    """
    rmatn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauNutm80(date1, date2, rmatn)
    return rmatn


# iauObl06
_sofa.iauObl06.argtypes = [_ct.c_double, #date1
                            _ct.c_double] #date2
_sofa.iauObl06.restype = _ct.c_double
def obl06(date1, date2):
    """ Mean obliquity of the ecliptic, IAU 2006 precession model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: obliquity of the ecliptic in radians (float).

    .. seealso:: |MANUAL| page 138
    """
    return _sofa.iauObl06(date1, date2)


# iauObl80
_sofa.iauObl80.argtypes = [_ct.c_double, #date1
                            _ct.c_double] #date2
_sofa.iauObl80.restype = _ct.c_double
def obl80(date1, date2):
    """ Mean obliquity of the ecliptic, IAU 1980 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: obliquity of the ecliptic in radians (float).

    .. seealso:: |MANUAL| page 139
    """
    return _sofa.iauObl80(date1, date2)


# iauP06e
_sofa.iauP06e.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #eps0
                            _ct.POINTER(_ct.c_double), #psia
                            _ct.POINTER(_ct.c_double), #oma
                            _ct.POINTER(_ct.c_double), #bpa
                            _ct.POINTER(_ct.c_double), #bqa
                            _ct.POINTER(_ct.c_double), #pia
                            _ct.POINTER(_ct.c_double), #bpia
                            _ct.POINTER(_ct.c_double), #epsa
                            _ct.POINTER(_ct.c_double), #chia
                            _ct.POINTER(_ct.c_double), #za
                            _ct.POINTER(_ct.c_double), #zetaa
                            _ct.POINTER(_ct.c_double), #thetaa
                            _ct.POINTER(_ct.c_double), #pa
                            _ct.POINTER(_ct.c_double), #gam
                            _ct.POINTER(_ct.c_double), #phi
                            _ct.POINTER(_ct.c_double)] #psi
def p06e(date1, date2):
    """ Precession angles, IAU 2006, equinox based.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 16-tuple:

        * epsilon_0
        * psi_A
        * omega_A
        * P_A
        * Q_A
        * pi_A
        * Pi_A
        * obliquity epsilon_A
        * chi_A
        * z_A
        * zeta_A
        * theta_A
        * p_A
        * F-W angle gamma_J2000
        * F-W angle phi_J2000
        * F-W angle psi_J2000

    .. seealso:: |MANUAL| page 140
    """
    eps0 = _ct.c_double()
    psia = _ct.c_double()
    oma = _ct.c_double()
    bpa = _ct.c_double()
    bqa = _ct.c_double()
    pia = _ct.c_double()
    bpia = _ct.c_double()
    epsa = _ct.c_double()
    chia = _ct.c_double()
    za = _ct.c_double()
    zetaa = _ct.c_double()
    thetaa = _ct.c_double()
    pa = _ct.c_double()
    gam = _ct.c_double()
    phi = _ct.c_double()
    psi = _ct.c_double()
    _sofa.iauP06e(date1, date2, _ct.byref(eps0), _ct.byref(psia),
                    _ct.byref(oma), _ct.byref(bpa), _ct.byref(bqa),
                    _ct.byref(pia), _ct.byref(bpia), _ct.byref(epsa),
                    _ct.byref(chia), _ct.byref(za), _ct.byref(zetaa),
                    _ct.byref(thetaa), _ct.byref(pa), _ct.byref(gam),
                    _ct.byref(phi), _ct.byref(psi))
    return eps0.value, psia.value, oma.value, bpa.value, bqa.value, pia.value,\
            bpia.value, epsa.value, chia.value, za.value, zetaa.value, \
            thetaa.value, pa.value, gam.value, phi.value, psi.value


# iauP2pv
_sofa.iauP2pv.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C'), #p
                        _ndpointer(shape=(2,3), dtype=float, flags='C')] #pv
def p2pv(p):
    """ Extend a p-vector to a pv-vector by appending a zero velocity.

    :param p: p-vector to extend.
    :type p: array-like of shape (1,3)

    :returns: pv-vector as a numpy.matrix of shape 2x3.

    .. seealso:: |MANUAL| page 142
    """
    pv = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    _sofa.iauP2pv(_req_shape_c(p, float, (1,3)), pv)
    return pv


# iauP2s
_sofa.iauP2s.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C'), #p
                        _ct.POINTER(_ct.c_double), #theta
                        _ct.POINTER(_ct.c_double), #phi
                        _ct.POINTER(_ct.c_double)] #r
def p2s(p):
    """ P-vector to spherical polar coordinates.

    :param p: the p-vector
    :type p: array-like of shape (1,3)

    :returns: a 3-tuple:

        * longitude angle in radians (float)
        * latitude angle in radians (float)
        * radial distance (float).

    .. seealso:: |MANUAL| page 143
    """
    theta = _ct.c_double()
    phi = _ct.c_double()
    r = _ct.c_double()
    _sofa.iauP2s(_req_shape_c(p, float, (1,3)), theta, phi, r)
    return theta.value, phi.value, r.value


# iauPap
_sofa.iauPap.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C'), #a
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #b
_sofa.iauPap.restype = _ct.c_double
def pap(a, b):
    """ Position-angle from two p-vectors.

    :param a: direction of the reference point.
    :type a: array-like of shape (1,3)

    :param b: direction of point whose position angle is required.
    :type b: array-like of shape (1,3)

    :returns: position angle of *b* with respect to *a* in radians (float).

    .. seealso:: |MANUAL| page 144
    """
    return _sofa.iauPap(_req_shape_c(a, float, (1,3)),
                                                _req_shape_c(b, float, (1,3)))


# iauPas
_sofa.iauPas.argtypes = [_ct.c_double, #al
                        _ct.c_double, #ap
                        _ct.c_double, #bl
                        _ct.c_double] #bp
_sofa.iauPas.restype = _ct.c_double
def pas(al, ap, bl,bp):
    """ Postion-angle from spherical coordinates.

    :param al: longitude of point A in radians.
    :type al: float

    :param ap: latitude of point A in radians.
    :type ap: float

    :param bl: longitude of point B in radians.
    :type bl: float

    :param bp: latitude of point B in radians.
    :type bp: float

    :returns: position angle of B with respect to A in radians (float).

    .. seealso:: |MANUAL| page 145
    """
    return _sofa.iauPas(float(al), float(ap), float(bl), float(bp))


# iauPb06
_sofa.iauPb06.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #bzeta
                            _ct.POINTER(_ct.c_double), #bz
                            _ct.POINTER(_ct.c_double)] #btheta
def pb06(date1, date2):
    """ Form the three Euler angles which implement general precession from
    epoch J2000.0, using IAU 2006 model. Frame bias is included.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 3-tuple:

        * 1st rotation: radians cw around z (float)
        * 3rd rotation: radians cw around z (float)
        * 2nd rotation: radians ccw around y.

    .. seealso:: |MANUAL| page 146
    """
    bzeta = _ct.c_double()
    bz = _ct.c_double()
    btheta = _ct.c_double()
    _sofa.iauPb06(date1, date2, bzeta, bz, btheta)
    return bzeta.value, bz.value, btheta.value


# iauPdp
_sofa.iauPdp.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C'), #a
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #b
_sofa.iauPdp.restype = _ct.c_double
def pdp(a, b):
    """ P-vector inner product.

    :param a: first p-vector.
    :type a: array-like of shape (1,3)

    :param b: second p-vector.
    :type b: array-like of shape (1,3)

    :returns: a dot b as a numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 147
    """
    return _sofa.iauPdp(_req_shape_c(a, float, (1,3)),
                                                _req_shape_c(b, float, (1,3)))


# iauPfw06
_sofa.iauPfw06.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #gamb
                            _ct.POINTER(_ct.c_double), #phib
                            _ct.POINTER(_ct.c_double), #psib
                            _ct.POINTER(_ct.c_double)] #epsa
def pfw06(date1, date2):
    """ Precession angles, IAU 2006 (Fukushima-Williams 4-angle formulation).

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 4-tuple:

        * F-W angle gamma_bar in radians (float)
        * F-W angle phi_bar in radians (float)
        * F-W angle psi_bar in radians (float)
        * F-W angle epsilon_A in radians (float).

    .. seealso:: |MANUAL| page 148
    """
    gamb = _ct.c_double()
    phib = _ct.c_double()
    psib = _ct.c_double()
    epsa = _ct.c_double()
    _sofa.iauPfw06(date1, date2, gamb, phib, psib, epsa)
    return gamb.value, phib.value, psib.value, epsa.value


# iauPlan94
_sofa.iauPlan94.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.c_int, #np
                        _ndpointer(shape=(2,3), dtype=float, flags='C')] #pv
_sofa.iauPlan94.restype = _ct.c_int
_plan94_msg = {
            -1: 'illegal planet identifier',
            1: 'Plan94: year outside the range 1000-3000',
            2: 'Plan94: failed to converge'
            }
def plan94(date1, date2, np):
    """ Approximate heliocentric position and velocity of a nominated major
    planet : Mercury, Venus, EMB, Mars, Jupiter, Saturn, Uranus or Neptune.

    :param date1, date2: TDB as a two-part Julian date.
    :type date1, date2: float

    :param np: planet identifier (1=Mercury, 2=Venus, 3=EMB, 4=Mars, 5=Jupiter,
                                6=Saturn, 7=Uranus, 8=Neptune).
    :type np: int

    :returns: planet's position and velocity (heliocentric, J2000.0, AU, AU/d)\
            as a numpy.matrix of shape 2x3

    :raises: :exc:`ValueError` if the planet identifier is invalid
        (outside 1..8).

        :exc:`UserWarning` if the year is outside the range 1000-3000.

    .. seealso:: |MANUAL| page 150
    """
    pv = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    s = _sofa.iauPlan94(date1, date2, np, pv)
    if s < 0:
        raise ValueError(_plan94_msg[s])
    elif s > 0:
        _warnings.warn(_plan94_msg[s], UserWarning, 2)
    return pv


# iauPm
_sofa.iauPm.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C')] #p
_sofa.iauPm.restype = _ct.c_double
def pm(p):
    """ Modulus of p-vector.

    :param p: p-vector.
    :type p: array-like of shape (1,3)

    :returns: modulus (float).

    .. seealso:: |MANUAL| page 153
    """
    return _sofa.iauPm(_req_shape_c(p, float, (1,3)))


# iauPmat00
_sofa.iauPmat00.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rbp
def pmat00(date1, date2):
    """ Precession matrix (including frame bias) from GCRS to a specified
    date, IAU 2000 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: bias-precession matrix, as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 154
    """
    rbp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauPmat00(date1, date2, rbp)
    return rbp


# iauPmat06
_sofa.iauPmat06.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rbp
def pmat06(date1, date2):
    """ Precession matrix (including frame bias) from GCRS to a specified 
    date, IAU 2006 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: bias-precession matrix, as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 155
    """
    rbp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauPmat06(date1, date2, rbp)
    return rbp


# iauPmat76
_sofa.iauPmat76.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rmatp
def pmat76(date1, date2):
    """ Precession matrix from J2000.0 to a specified date, IAU 1976 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: bias-precession matrix, as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 156
    """
    rmatp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauPmat76(date1, date2, rmatp)
    return rmatp


# iauPmp
_sofa.iauPmp.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C'), #a
                        _ndpointer(shape=(1,3), dtype=float, flags='C'), #b
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #amb
def pmp(a, b):
    """ P-vector subtraction.

    :param a: first p-vector.
    :type a: array-like of shape (1,3)

    :param b: second p-vector.
    :type b: array-like of shape (1,3)

    :returns: a - b as a numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 158
    """
    amb = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauPmp(_req_shape_c(a, float, (1,3)),
                                            _req_shape_c(b, float, (1,3)), amb)
    return amb


# iauPn
_sofa.iauPn.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C'), #p
                        _ct.POINTER(_ct.c_double), #r
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #u
def pn(p):
    """ Convert a p-vector into modulus and unit vector.

    :param p: p-vector.
    :type p: array-like of shape (1,3)

    :returns: 2-tuple:

            * the modulus (float)
            * unit vector (numpy.matrix of shape 1x3)

    .. seealso:: |MANUAL| page 159
    """
    r = _ct.c_double()
    u = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauPn(_req_shape_c(p, float, (1,3)), _ct.byref(r), u)
    return r.value, u


# iauPn00
_sofa.iauPn00.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.c_double, #dpsi
                            _ct.c_double, #deps
                            _ct.POINTER(_ct.c_double), #epsa
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rb
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rp
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rbp
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rn
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rbpn
def pn00(date1, date2, dpsi, deps):
    """ Precession-nutation, IAU 2000 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :param dpsi, deps: nutation.
    :type dpsi, deps: float

    :returns: a 6-tuple:

            * mean obliquity (float)
            * frame bias matrix (numpy.matrix of shape 3x3)
            * precession matrix (numpy.matrix of shape 3x3)
            * bias-precession matrix (numpy.matrix of shape 3x3)
            * nutation matrix (numpy.matrix of shape 3x3)
            * GCRS-to-true matrix (numpy.matrix of shape 3x3).

    .. seealso:: |MANUAL| page 160
    """
    epsa = _ct.c_double()
    rb = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rbp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rbpn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauPn00(date1, date2, float(dpsi), float(deps), _ct.byref(epsa),
                                                    rb, rp, rbp, rn, rbpn)
    return epsa.value, rb, rp, rbp, rn, rbpn


# iauPn00a
_sofa.iauPn00a.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #dpsi
                            _ct.POINTER(_ct.c_double), #deps
                            _ct.POINTER(_ct.c_double), #epsa
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rb
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rp
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rbp
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rn
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rbpn
def pn00a(date1, date2):
    """ Precession-nutation, IAU 2000A model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 8-tuple:

            * nutation in longitude (float)
            * nutation in obliquity (float)
            * mean obliquity (float)
            * frame bias matrix (numpy.matrix of shape 3x3)
            * precession matrix (numpy.matrix of shape 3x3)
            * bias-precession matrix (numpy.matrix of shape 3x3)
            * nutation matrix (numpy.matrix of shape 3x3)
            * GCRS-to-true matrix (numpy.matrix of shape 3x3).

    .. seealso:: |MANUAL| page 162
    """
    dpsi = _ct.c_double()
    deps = _ct.c_double()
    epsa = _ct.c_double()
    rb = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rbp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rbpn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauPn00a(date1, date2, _ct.byref(dpsi), _ct.byref(deps),
                                        _ct.byref(epsa), rb, rp, rbp, rn, rbpn)
    return dpsi.value, deps.value, epsa.value, rb, rp, rbp, rn, rbpn


# iauPn00b
_sofa.iauPn00b.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #dpsi
                            _ct.POINTER(_ct.c_double), #deps
                            _ct.POINTER(_ct.c_double), #epsa
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rb
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rp
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rbp
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rn
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rbpn
def pn00b(date1, date2):
    """ Precession-nutation, IAU 2000B model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 8-tuple:

            * nutation in longitude (float)
            * nutation in obliquity (float)
            * mean obliquity (float)
            * frame bias matrix (numpy.matrix of shape 3x3)
            * precession matrix (numpy.matrix of shape 3x3)
            * bias-precession matrix (numpy.matrix of shape 3x3)
            * nutation matrix (numpy.matrix of shape 3x3)
            * GCRS-to-true matrix (numpy.matrix of shape 3x3).

    .. seealso:: |MANUAL| page 164
    """
    dpsi = _ct.c_double()
    deps = _ct.c_double()
    epsa = _ct.c_double()
    rb = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rbp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rbpn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauPn00b(date1, date2, _ct.byref(dpsi), _ct.byref(deps),
                                        _ct.byref(epsa), rb, rp, rbp, rn, rbpn)
    return dpsi.value, deps.value, epsa.value, rb, rp, rbp, rn, rbpn


# iauPn06
_sofa.iauPn06.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.c_double, #dpsi
                            _ct.c_double, #deps
                            _ct.POINTER(_ct.c_double), #epsa
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rb
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rp
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rbp
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rn
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rbpn
def pn06(date1, date2, dpsi, deps):
    """ Precession-nutation, IAU 2006 model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :param dpsi, deps: nutation.
    :type dpsi, deps: float

    :returns: a 6-tuple:

            * mean obliquity (float)
            * frame bias matrix (numpy.matrix of shape 3x3)
            * precession matrix (numpy.matrix of shape 3x3)
            * bias-precession matrix (numpy.matrix of shape 3x3)
            * nutation matrix (numpy.matrix of shape 3x3)
            * GCRS-to-true matrix (numpy.matrix of shape 3x3).

    .. seealso:: |MANUAL| page 166
    """
    epsa = _ct.c_double()
    rb = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rbp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rbpn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauPn06(date1, date2, float(dpsi), float(deps), _ct.byref(epsa),
                                                        rb, rp, rbp, rn, rbpn)
    return epsa.value, rb, rp, rbp, rn, rbpn


# iauPn06a
_sofa.iauPn06a.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #dpsi
                            _ct.POINTER(_ct.c_double), #deps
                            _ct.POINTER(_ct.c_double), #epsa
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rb
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rp
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rbp
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #rn
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rbpn
def pn06a(date1, date2):
    """ Precession-nutation, IAU 2006/2000A models.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 8-tuple:

            * nutation in longitude (float)
            * nutation in obliquity (float)
            * mean obliquity (float)
            * frame bias matrix (numpy.matrix of shape 3x3)
            * precession matrix (numpy.matrix of shape 3x3)
            * bias-precession matrix (numpy.matrix of shape 3x3)
            * nutation matrix (numpy.matrix of shape 3x3)
            * GCRS-to-true matrix (numpy.matrix of shape 3x3).

    .. seealso:: |MANUAL| page 168
    """
    dpsi = _ct.c_double()
    deps = _ct.c_double()
    epsa = _ct.c_double()
    rb = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rbp = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    rbpn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauPn06a(date1, date2, _ct.byref(dpsi), _ct.byref(deps),
                                        _ct.byref(epsa), rb, rp, rbp, rn, rbpn)
    return dpsi.value, deps.value, epsa.value, rb, rp, rbp, rn, rbpn


# iauPnm00a
_sofa.iauPnm00a.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rbpn
def pnm00a(date1, date2):
    """ Form the matrix of precession-nutation for a given date (including 
    frame bias), equinox-based, IAU 2000A model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: classical *NPB* matrix, as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 170
    """
    rbpn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauPnm00a(date1, date2, rbpn)
    return rbpn


# iauPnm00b
_sofa.iauPnm00b.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rbpn
def pnm00b(date1, date2):
    """ Form the matrix of precession-nutation for a given date (including 
    frame bias), equinox-based, IAU 2000B model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: bias-precession-nutation matrix, as a numpy.matrix of shape \
        3x3.

    .. seealso:: |MANUAL| page 171
    """
    rbpn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauPnm00b(date1, date2, rbpn)
    return rbpn


# iauPnm06a
_sofa.iauPnm06a.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rbpn
def pnm06a(date1, date2):
    """ Form the matrix of precession-nutation for a given date (including 
    frame bias), IAU 2006 precession and IAU 2000A nutation models.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: bias-precession-nutation matrix, as a numpy.matrix of shape \
        3x3.

    .. seealso:: |MANUAL| page 172
    """
    rbpn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauPnm06a(date1, date2, rbpn)
    return rbpn


# iauPnm80
_sofa.iauPnm80.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rmatpn
def pnm80(date1, date2):
    """ Form the matrix of precession/nutation for a given date, IAU 1976 
    precession model, IAU 1980 nutation model).

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: combined precessoin/nutation matrix, as a numpy.matrix of shape \
        3x3.

    .. seealso:: |MANUAL| page 173
    """
    rmatpn = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauPnm80(date1, date2, rmatpn)
    return rmatpn


# iauPom00
_sofa.iauPom00.argtypes = [_ct.c_double, #xp
                            _ct.c_double, #yp
                            _ct.c_double, #sp
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rpom
def pom00(xp, yp, sp):
    """ Form the matrix of polar motion for a given date, IAU 2000.

    :param xp, yp: coordinates of the pole in radians.
    :type xp, yp: float

    :param sp: the TIO locator in radians.
    :type sp: float

    :returns: the polar motion matrix, as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 174
    """
    rpom = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauPom00(float(xp), float(yp), float(sp), rpom)
    return rpom


# iauPpp
_sofa.iauPpp.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C'), #a
                        _ndpointer(shape=(1,3), dtype=float, flags='C'), #b
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #apb
def ppp(a, b):
    """ P-vector addition.

    :param a: first p-vector.
    :type a: array-like of shape (1,3)

    :param b: second p-vector.
    :type b: array-like of shape (1,3)

    :returns: a + b as a numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 175
    """
    apb = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauPpp(_req_shape_c(a, float, (1,3)),
                                            _req_shape_c(b, float, (1,3)), apb)
    return apb


# iauPpsp
_sofa.iauPpsp.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C'), #a
                        _ct.c_double, #s
                        _ndpointer(shape=(1,3), dtype=float, flags='C'), #b
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #apsb
def ppsp(a, s, b):
    """ P-vector plus scaled p-vector.

    :param a: first p-vector.
    :type a: array-like of shape (1,3)

    :param s: scalar (multiplier for *b*).
    :type s: float

    :param b: second p-vector.
    :type b: array-like of shape (1,3)

    :returns: a + s*b as a numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 176
    """
    apsb = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauPpsp(_req_shape_c(a, float, (1,3)), s,
                                        _req_shape_c(b, float, (1,3)), apsb)
    return apsb


# iauPr00
_sofa.iauPr00.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #dpsipr
                            _ct.POINTER(_ct.c_double)] #depspr
def pr00(date1, date2):
    """ Precession-rate part of the IAU 2000 precession-nutation models.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 2-tuple:

        * precession correction in longitude (float)
        * precession correction in obliquity (float).

    .. seealso:: |MANUAL| page 177
    """
    dpsipr = _ct.c_double()
    depspr = _ct.c_double()
    _sofa.iauPr00(date1, date2, _ct.byref(dpsipr), _ct.byref(depspr))
    return dpsipr.value, depspr.value


# iauPrec76
_sofa.iauPrec76.argtypes = [_ct.c_double, #ep01
                            _ct.c_double, #ep02
                            _ct.c_double, #ep11
                            _ct.c_double, #ep12
                            _ct.POINTER(_ct.c_double), #zeta
                            _ct.POINTER(_ct.c_double), #z
                            _ct.POINTER(_ct.c_double)] #theta
def prec76(ep01, ep02, ep11, ep12):
    """ Form the three Euler angles wich implement general precession between 
    two epochs, using IAU 1976 model (as for FK5 catalog).

    :param ep01, ep02: two-part TDB starting epoch.
    :type ep01, ep02: float

    :param ep11, ep12: two-part TDB ending epoch.
    :type ep11, ep12: float

    :returns: a 3-tuple:

            * 1st rotation: radians cw around z (float)
            * 3rd rotation: radians cw around z (float)
            * 2nd rotation: radians ccw around y (float).

    .. seealso:: |MANUAL| page 179
    """
    zeta = _ct.c_double()
    z = _ct.c_double()
    theta = _ct.c_double()
    _sofa.iauPrec76(ep01, ep02, ep11, ep12, _ct.byref(zeta), _ct.byref(z),
                                                            _ct.byref(theta))
    return zeta.value, z.value, theta.value


# iauPv2p
_sofa.iauPv2p.argtypes = [_ndpointer(shape=(2,3), dtype=float, flags='C'), #pv
                            _ndpointer(shape=(1,3), dtype=float, flags='C')] #p
def pv2p(pv):
    """ Discard velocity component of a pv-vector.

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: p-vector as a numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 181
    """
    p = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauPv2p(_req_shape_c(pv, float, (2,3)), p)
    return p


# iauPv2s
_sofa.iauPv2s.argtypes = [_ndpointer(shape=(2,3), dtype=float, flags='C'), #pv
                            _ct.POINTER(_ct.c_double), #theta
                            _ct.POINTER(_ct.c_double), #phi
                            _ct.POINTER(_ct.c_double), #r
                            _ct.POINTER(_ct.c_double), #td
                            _ct.POINTER(_ct.c_double), #pd
                            _ct.POINTER(_ct.c_double)] #rd
def pv2s(pv):
    """ Convert position/velocity from cartesian to spherical coordinates.

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: a 6-tuple:

        * longitude angle :math:`\\theta`  in radians (float)
        * latitude angle :math:`\phi` in radians (float)
        * radial distance *r* (float)
        * rate of change of :math:`\\theta` (float)
        * rate of change of :math:`\phi` (float)
        * rate of change of *r* (float)

    .. seealso:: |MANUAL| page 182
    """
    theta = _ct.c_double()
    phi = _ct.c_double()
    r = _ct.c_double()
    td = _ct.c_double()
    pd = _ct.c_double()
    rd = _ct.c_double()
    _sofa.iauPv2s(_req_shape_c(pv, float, (2,3)), _ct.byref(theta),
                    _ct.byref(phi), _ct.byref(r), _ct.byref(td), _ct.byref(pd),
                                                                _ct.byref(rd))
    return theta.value, phi.value, r.value, td.value, pd.value, rd.value


# iauPvdpv
_sofa.iauPvdpv.argtypes = [_ndpointer(shape=(2,3), dtype=float, flags='C'), #a
                            _ndpointer(shape=(2,3), dtype=float, flags='C'), #b
                        _ndpointer(shape=(1,2), dtype=float, flags='C')] #adb
def pvdpv(a, b):
    """ Inner product of two pv-vectors.

    :param a: first pv-vector.
    :type a: array-like of shape (2,3)

    :param b: second pv-vector.
    :type b: array-like of shape (2,3)

    :returns: a . b as a numpy.matrix of shape 1x2.

    .. seealso:: |MANUAL| page 183
    """
    adb = _np.asmatrix(_np.zeros(shape=(2), dtype=float, order='C'))
    _sofa.iauPvdpv(_req_shape_c(a, float, (2,3)),
                                            _req_shape_c(b, float, (2,3)), adb)
    return adb


# iauPvm
_sofa.iauPvm.argtypes = [_ndpointer(shape=(2,3), dtype=float, flags='C'), #pv
                        _ct.POINTER(_ct.c_double), #r
                        _ct.POINTER(_ct.c_double)] #s
def pvm(pv):
    """ Modulus of pv-vector.

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: a 2-tuple:

        * modulus of position component (float)
        * modulus of velocity component (float).

    .. seealso:: |MANUAL| page 184
    """
    r = _ct.c_double()
    s = _ct.c_double()
    _sofa.iauPvm(_req_shape_c(pv, float, (2,3)), _ct.byref(r), _ct.byref(s))
    return r.value, s.value


# iauPvmpv
_sofa.iauPvmpv.argtypes = [_ndpointer(shape=(2,3), dtype=float, flags='C'), #a
                            _ndpointer(shape=(2,3), dtype=float, flags='C'), #b
                        _ndpointer(shape=(2,3), dtype=float, flags='C')] #amb
def pvmpv(a, b):
    """ Subtract one pv-vector from another.

    :param a: first pv-vector.
    :type a: array-like of shape (2,3)

    :param b: second pv-vector.
    :type b: array-like of shape (2,3)

    :returns: a - b as a numpy.matrix of shape 2x3.

    .. seealso:: |MANUAL| page 185
    """
    amb = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    _sofa.iauPvmpv(_req_shape_c(a, float, (2,3)),
                                            _req_shape_c(b, float, (2,3)), amb)
    return amb


# iauPvppv
_sofa.iauPvppv.argtypes = [_ndpointer(shape=(2,3), dtype=float, flags='C'), #a
                            _ndpointer(shape=(2,3), dtype=float, flags='C'), #b
                        _ndpointer(shape=(2,3), dtype=float, flags='C')] #apb
def pvppv(a, b):
    """ Add one pv-vector to another.

    :param a: first pv-vector.
    :type a: array-like of shape (2,3)

    :param b: second pv-vector.
    :type b: array-like of shape (2,3)

    :returns: a + b as a numpy.matrix of shape 2x3.

    .. seealso:: |MANUAL| page 186
    """
    apb = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    _sofa.iauPvppv(_req_shape_c(a, float, (2,3)),
                                            _req_shape_c(b, float, (2,3)), apb)
    return apb


# iauPvstar
_sofa.iauPvstar.argtypes = [
                        _ndpointer(shape=(2,3), dtype=float, flags='C'), #pv
                            _ct.POINTER(_ct.c_double), #ra
                            _ct.POINTER(_ct.c_double), #dec
                            _ct.POINTER(_ct.c_double), #pmr
                            _ct.POINTER(_ct.c_double), #pmd
                            _ct.POINTER(_ct.c_double), #px
                            _ct.POINTER(_ct.c_double)] #rv
_sofa.iauPvstar.restype = _ct.c_int
_pvstar_msg = {
            -1: 'superluminal speed',
            -2: 'null position vector'
            }
def pvstar(pv):
    """ Convert star position-velocity vector to catalog coordinates.

    :param pv: pv-vector (AU, AU/day).
    :type pv: array-like of shape (2,3)

    :returns: a 6-tuple:

        * right ascensin in radians (float)
        * declination in radians (float)
        * RA proper motion (radians/year) (float)
        * Dec proper motion (radians/year) (float)
        * parallax in arcseconds (float)
        * radial velocity (km/s, positive = receding)

    :raises: :exc:`ValueError` if the speed is greater than or equal to the
        speed of light.

    .. seealso:: |MANUAL| page 187
    """
    ra = _ct.c_double()
    dec = _ct.c_double()
    pmr = _ct.c_double()
    pmd = _ct.c_double()
    px = _ct.c_double()
    rv = _ct.c_double()
    s = _sofa.iauPvstar(_req_shape_c(pv, float, (2,3)), _ct.byref(ra),
                        _ct.byref(dec), _ct.byref(pmr), _ct.byref(pmd),
                                                _ct.byref(px), _ct.byref(rv))
    if s != 0:
        raise ValueError(_pvstar_msg[s])
    return ra.value, dec.value, pmr.value, pmd.value, px.value, rv.value


# iauPvu
_sofa.iauPvu.argtypes = [_ct.c_double, #dt
                        _ndpointer(shape=(2,3), dtype=float, flags='C'), #pv
                        _ndpointer(shape=(2,3), dtype=float, flags='C')] #upv
def pvu(dt, pv):
    """ Update a pv-vector.

    :param dt: time interval.
    :type dt: float

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: a new pv-vector as a numpy.matrix of shape 2x3, with p \
        updated and v unchanged.

    .. seealso:: |MANUAL| page 189
    """
    upv = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    _sofa.iauPvu(dt, _req_shape_c(pv, float, (2,3)), upv)
    return upv


# iauPvup
_sofa.iauPvup.argtypes = [_ct.c_double, #dt
                        _ndpointer(shape=(2,3), dtype=float, flags='C'), #pv
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #p
def pvup(dt, pv):
    """ Update a pv-vector, discarding the velocity component.

    :param dt: time interval.
    :type dt: float

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: a new p-vector, as a numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 190
    """
    p = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauPvup(dt, _req_shape_c(pv, float, (2,3)), p)
    return p


# iauPvxpv
_sofa.iauPvxpv.argtypes = [_ndpointer(shape=(2,3), dtype=float, flags='C'), #a
                            _ndpointer(shape=(2,3), dtype=float, flags='C'), #b
                        _ndpointer(shape=(2,3), dtype=float, flags='C')] #axb
def pvxpv(a, b):
    """ Outer product of two pv-vectors.

    :param a: first pv-vector.
    :type a: array-like of shape (2,3)

    :param b: second pv-vector.
    :type b: array-like of shape (2,3)

    :returns: a x b as a numpy.matrix of shape 2x3.

    .. seealso:: |MANUAL| page 191
    """
    axb = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    _sofa.iauPvxpv(_req_shape_c(a, float, (2,3)),
                                            _req_shape_c(b, float, (2,3)), axb)
    return axb


# iauPxp
_sofa.iauPxp.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C'), #a
                        _ndpointer(shape=(1,3), dtype=float, flags='C'), #b
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #axb
def pxp(a, b):
    """ P-vector outer product.

    :param a: first p-vector.
    :type a: array-like of shape (1,3)

    :param b: second p-vector.
    :type b: array-like of shape (1,3)

    :returns: a x b as a numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 192
    """
    axb = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauPxp(_req_shape_c(a, float, (1,3)),
                                            _req_shape_c(b, float, (1,3)), axb)
    return axb


# iauRm2v
_sofa.iauRm2v.argtypes = [_ndpointer(shape=(3,3), dtype=float, flags='C'), #r
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #w
def rm2v(r):
    """ Express a r-matrix as a r-vector.

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :returns: rotation vector as a numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 193
    """
    w = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauRm2v(_req_shape_c(r, float, (3,3)), w)
    return w


# iauRv2m
_sofa.iauRv2m.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C'), #w
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #r
def rv2m(w):
    """ Form the rotation matrix corresponding to a given r-vector.

    :param w: rotation vector.
    :type w: array-like of shape (1,3)

    :returns: rotation matrix as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 194
    """
    r = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauRv2m(_req_shape_c(w, float, (1,3)), r)
    return r


# iauRx
_sofa.iauRx.argtypes = [_ct.c_double, #phi
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #r
def rx(phi, r):
    """ Rotate a r-matrix about the x-axis.

    :param phi: angle in radians.
    :type phi: float

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :returns: the new rotation matrix, as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 195
    """
    r2 = _req_shape_c(r, float, (3,3)).copy()
    _sofa.iauRx(float(phi), r2)
    return _np.matrix(r2, dtype=float)


# iauRxp
_sofa.iauRxp.argtypes = [_ndpointer(shape=(3,3), dtype=float, flags='C'), #r
                        _ndpointer(shape=(1,3), dtype=float, flags='C'), #p
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #rp
def rxp(r, p):
    """ Multiply a p-vector by a r-matrix.

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :param p: p-vector.
    :type p: array-like of shape (1,3)

    :returns: r * p as a numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 196
    """
    rp = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauRxp(_req_shape_c(r, float, (3,3)),
                                            _req_shape_c(p, float, (1,3)), rp)
    return rp


# iauRxpv
_sofa.iauRxpv.argtypes = [_ndpointer(shape=(3,3), dtype=float, flags='C'), #r
                        _ndpointer(shape=(2,3), dtype=float, flags='C'), #pv
                        _ndpointer(shape=(2,3), dtype=float, flags='C')] #rpv
def rxpv(r, pv):
    """ Multiply a pv-vector by a r-matrix.

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: r * pv as a numpy.matrix of shape 2x3.

    .. seealso:: |MANUAL| page 197
    """
    rpv = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    _sofa.iauRxpv(_req_shape_c(r, float, (3,3)),
                                        _req_shape_c(pv, float, (2,3)), rpv)
    return rpv


# iauRxr
_sofa.iauRxr.argtypes = [_ndpointer(shape=(3,3), dtype=float, flags='C'), #a
                        _ndpointer(shape=(3,3), dtype=float, flags='C'), #b
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #atb
def rxr(a, b):
    """ Multiply two rotation matrices.

    :param a: first r-matrix.
    :type a: array-like of shape (3,3)

    :param b: second r-matrix.
    :type b: array-like of shape (3,3)

    :returns: a * b as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 198
    """
    atb = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauRxr(_req_shape_c(a, float, (3,3)),
                                            _req_shape_c(b, float, (3,3)), atb)
    return atb


# iauRy
_sofa.iauRy.argtypes = [_ct.c_double, #theta
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #r
def ry(theta, r):
    """ Rotate a r-matrix about the y-axis.

    :param theta: angle in radians.
    :type theta: float

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :returns: the new rotation matrix, as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 199
    """
    r2 = _req_shape_c(r, float, (3,3)).copy()
    _sofa.iauRy(float(theta), r2)
    return _np.matrix(r2, dtype=float)


# iauRz
_sofa.iauRz.argtypes = [_ct.c_double, #psi
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #r
def rz(psi, r):
    """ Rotate a r-matrix about the z-axis.

    :param psi: angle in radians.
    :type psi: float

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :returns: the new rotation matrix, as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 200
    """
    r2 = _req_shape_c(r, float, (3,3)).copy()
    _sofa.iauRz(float(psi), r2)
    return _np.matrix(r2, dtype=float)


# iauS00
_sofa.iauS00.argtypes = [_ct.c_double, #date1
                        _ct.c_double, #date2
                        _ct.c_double, #x
                        _ct.c_double] #y
_sofa.iauS00.restype = _ct.c_double
def s00(date1, date2, x, y):
    """ The CIO locator *s*, positioning the celestial intermediate 
    origin on the equator of the celestial intermediate pole, given the
    CIP's X,Y coordinates. Compatible with IAU 2000A precession-nutation.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :param x, y: CIP coordinates.
    :type x, y: float

    :returns: the CIO locator *s* in radians (float).

    .. seealso:: |MANUAL| page 201
    """
    return _sofa.iauS00(date1, date2, float(x), float(y))


# iauS00a
_sofa.iauS00a.argtypes = [_ct.c_double, #date1
                            _ct.c_double] #date2
_sofa.iauS00a.restype = _ct.c_double
def s00a(date1, date2):
    """ The CIO locator, positioning the celestial intermediate origin 
    on the equator of the celestial intermediate pole, using IAU 2000A
    precession-nutation model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: the CIO locator *s* in radians (float):

    .. seealso:: |MANUAL| page 203
    """
    return _sofa.iauS00a(date1, date2)


# iauS00b
_sofa.iauS00b.argtypes = [_ct.c_double, #date1
                            _ct.c_double] #date2
_sofa.iauS00b.restype = _ct.c_double
def s00b(date1, date2):
    """ The CIO locator, positioning the celestial intermediate origin 
    on the equator of the celestial intermediate pole, using IAU 2000B
    precession-nutation model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: the CIO locator *s* in radians (float):

    .. seealso:: |MANUAL| page 205
    """
    return _sofa.iauS00b(date1, date2)


# iauS06
_sofa.iauS06.argtypes = [_ct.c_double, #date1
                        _ct.c_double, #date2
                        _ct.c_double, #x
                        _ct.c_double] #y
_sofa.iauS06.restype = _ct.c_double
def s06(date1, date2, x, y):
    """ The CIO locator *s*, positioning the celestial intermediate 
    origin on the equator of the celestial intermediate pole, given the
    CIP's X,Y coordinates. Compatible with IAU 2006/2000A precession-nutation.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :param x, y: CIP coordinates.
    :type x, y: float

    :returns: the CIO locator *s* in radians (float).

    .. seealso:: |MANUAL| page 207
    """
    return _sofa.iauS06(date1, date2, float(x), float(y))


# iauS06a
_sofa.iauS06a.argtypes = [_ct.c_double, #date1
                            _ct.c_double] #date2
_sofa.iauS06a.restype = _ct.c_double
def s06a(date1, date2):
    """ The CIO locator, positioning the celestial intermediate origin 
    on the equator of the celestial intermediate pole, using IAU 2006
    precession and IAU 2000A nutation models.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: the CIO locator *s* in radians (float):

    .. seealso:: |MANUAL| page 209
    """
    return _sofa.iauS06a(date1, date2)


# iauS2c
_sofa.iauS2c.argtypes = [_ct.c_double, #theta
                        _ct.c_double, #phi
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #c
def s2c(theta, phi):
    """ Convert spherical coordinates to cartesian.

    :param theta: longitude angle in radians.
    :type theta: float

    :param phi: latitude angle in radians.
    :type phi: float

    :returns: direction cosines as a numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 211
    """
    c = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauS2c(float(theta), float(phi), c)
    return c


# iauS2p
_sofa.iauS2p.argtypes = [_ct.c_double, #theta
                        _ct.c_double, #phi
                        _ct.c_double, #r
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #p
def s2p(theta, phi, r):
    """ Convert spherical polar coordinates to p-vector.

    :param theta: longitude angle in radians.
    :type theta: float

    :param phi: latitude angle in radians.
    :type phi: float

    :param r: radial distance.
    :type r: float

    :returns: cartesian coordinates as a numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 212
    """
    p = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauS2p(float(theta), float(phi), r, p)
    return p


# iauS2pv
_sofa.iauS2pv.argtypes = [_ct.c_double, #theta
                            _ct.c_double, #phi
                            _ct.c_double, #r
                            _ct.c_double, #td
                            _ct.c_double, #pd
                            _ct.c_double, #rd
                        _ndpointer(shape=(2,3), dtype=float, flags='C')] #pv
def s2pv(theta, phi, r, td, pd, rd):
    """ Convert position/velocity from spherical to cartesian coordinates.

    :param theta: longitude angle in radians.
    :type theta: float

    :param phi: latitude angle in radians.
    :type phi: float

    :param r: radial distance.
    :type r: float

    :param td: rate of change of *theta*.
    :type td: float

    :param pd: rate of change of *phi*.
    :type pd: float

    :param rd: rate of change of *r*.
    :type rd: float

    :returns: pv-vector as a numpy.matrix of shape 2x3.

    .. seealso:: |MANUAL| page 213
    """
    pv = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    _sofa.iauS2pv(float(theta), float(phi), r, float(td), float(pd), rd, pv)
    return pv


# iauS2xpv
_sofa.iauS2xpv.argtypes = [_ct.c_double, #s1
                            _ct.c_double, #s2
                        _ndpointer(shape=(2,3), dtype=float, flags='C'), #pv
                        _ndpointer(shape=(2,3), dtype=float, flags='C')] #spv
def s2xpv(s1, s2, pv):
    """ Multiply a pv-vector by two scalars.

    :param s1: scalar to multiply position component by.
    :type s1: float

    :param s2: scalar to multiply velocity component by.
    :type s2: float

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: a new pv-vector (with p scaled by s1 and v scaled by s2) as a \
        numpy.matrix of shape 2x3.

    .. seealso:: |MANUAL| page 214
    """
    spv = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    _sofa.iauS2xpv(s1, s2, _req_shape_c(pv, float, (2,3)), spv)
    return spv


# iauSepp
_sofa.iauSepp.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C'), #a
                            _ndpointer(shape=(1,3), dtype=float, flags='C')] #b
_sofa.iauSepp.restype = _ct.c_double
def sepp(a, b):
    """ Angular separation between two p-vectors.

    :param a: first p-vector.
    :type a: array-like of shape (1,3)

    :param b: second p-vector.
    :type b: array-like of shape (1,3)

    :returns: angular separation in radians, always positive (float).

    .. seealso:: |MANUAL| page 215
    """
    return _sofa.iauSepp(_req_shape_c(a, float, (1,3)),
                                                _req_shape_c(b, float, (1,3)))


# iauSeps
_sofa.iauSeps.argtypes = [_ct.c_double, #al
                            _ct.c_double, #ap
                            _ct.c_double, #bl
                            _ct.c_double] #bp
_sofa.iauSeps.restype = _ct.c_double
def seps(al, ap, bl, bp):
    """ Angular separation between two sets of spherical coordinates.

    :param al: first longitude in radians.
    :type al: float

    :param ap: first latitude in radians.
    :type ap: float

    :param bl: second longitude in radians.
    :type bl: float

    :param bl: second latitude in radians.
    :type bp: float

    :returns: angular separation in radians (float).

    .. seealso:: |MANUAL| page 216
    """
    return _sofa.iauSeps(float(al), float(ap), float(bl), float(bp))


# iauSp00
_sofa.iauSp00.argtypes = [_ct.c_double, #date1
                            _ct.c_double] #date2
_sofa.iauSp00.restype = _ct.c_double
def sp00(date1, date2):
    """ The TIO locator, positioning the terrestrial intermediate origin on 
    the equator of the celestial intermediate pole.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: the TIO locator in radians (float).

    .. seealso:: |MANUAL| page 217
    """
    return _sofa.iauSp00(date1, date2)


# iauStarpm
_sofa.iauStarpm.argtypes = [_ct.c_double, #ra1
                            _ct.c_double, #dec1
                            _ct.c_double, #pmr1
                            _ct.c_double, #pmd1
                            _ct.c_double, #px1
                            _ct.c_double, #rv1
                            _ct.c_double, #ep1a
                            _ct.c_double, #ep1b
                            _ct.c_double, #ep2a
                            _ct.c_double, #ep2b
                            _ct.POINTER(_ct.c_double), #ra2
                            _ct.POINTER(_ct.c_double), #dec2
                            _ct.POINTER(_ct.c_double), #pmr2
                            _ct.POINTER(_ct.c_double), #pmd2
                            _ct.POINTER(_ct.c_double), #px2
                            _ct.POINTER(_ct.c_double)] #rv2
_sofa.iauStarpm.restype = _ct.c_int
_starpm_msg = {0: 'OK', # Unused
                1:'Starpm: distance overridden',
                2:'Starpm: excessive velocity',
                4:'Starpm: solution didn\'t converge'}
def starpm(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b):
    """ Update star catalog data for space motion.

    :param ra1: right ascension in radians.
    :type ra1: float

    :param dec1: declination in radians.
    :type dec1: float

    :param pmr1: proper motion in RA (radians/year).
    :type pmr1: float

    :param pmd1: proper motion in Dec (radians/year).
    :type pmd1: float

    :param px1: parallax in arcseconds.
    :type px1: float

    :param rv1: radial velocity (km/s, positive = receding).
    :type rv1: float

    :param ep1a, ep1b: two-part starting epoch.
    :type ep1a, ep1b: float

    :param ep2a, ep2b: two-part ending epoch.
    :type ep2a, ep2b: float

    :returns: a 6-tuple:

        * the new right ascension in radians (float)
        * the new declination in radians (float)
        * the new RA proper motion in radians/year (float)
        * the new Dec proper motion in radians/year (float)
        * the new parallax in arcseconds (float)
        * the new radial velocity (km/s)

    .. seealso:: |MANUAL| page 218
    """
    ra2 = _ct.c_double()
    dec2 = _ct.c_double()
    pmr2 = _ct.c_double()
    pmd2 = _ct.c_double()
    px2 = _ct.c_double()
    rv2 = _ct.c_double()
    status = _sofa.iauStarpm(float(ra1), float(dec1), float(pmr1), float(pmd1),
                                float(px1), float(rv1), ep1a, ep1b, ep2a,
                    ep2b, _ct.byref(ra2), _ct.byref(dec2), _ct.byref(pmr2),
                            _ct.byref(pmd2), _ct.byref(px2), _ct.byref(rv2))
    if status >= 4:
        _warnings.warn(_starpm_msg[status], UserWarning, 2)
        status -= 4
    if status >= 2:
        _warnings.warn(_starpm_msg[status], UserWarning, 2)
        status -= 2
    if status >= 1:
        _warnings.warn(_starpm_msg[status], UserWarning, 2)
    return ra2.value, dec2.value, pmr2.value, pmd2.value, px2.value, rv2.value


# iauStarpv
_sofa.iauStarpv.argtypes = [_ct.c_double, #ra
                            _ct.c_double, #dec
                            _ct.c_double, #pmr
                            _ct.c_double, #pmd
                            _ct.c_double, #px
                            _ct.c_double, #rv
                        _ndpointer(shape=(2,3), dtype=float, flags='C')] #pv
_sofa.iauStarpv.restype = _ct.c_int
_starpv_msg = {0: 'OK', # Unused
                1:'Starpv: distance overridden',
                2:'Starpv: excessive speed',
                4:'Starpv: solution didn\'t converge'}
def starpv(ra, dec, pmr, pmd, px, rv):
    """ Convert star catalog coordinates to position+velocity vector.

    :param ra: right ascension in radians.
    :type ra: float

    :param dec: declination in radians.
    :type dec: float

    :param pmr: proper motion in RA (radians/year).
    :type pmr: float

    :param pmd: proper motion in Dec (radians/year).
    :type pmd: float

    :param px: parallax in arcseconds.
    :type px: float

    :param rv: radial velocity (km/s, positive = receding).
    :type rv: float

    :returns: the pv-vector (AU, AU/day) as a numpy.matrix of shape 2x3

    .. seealso:: |MANUAL| page 220
    """
    pv = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    status = _sofa.iauStarpv(float(ra), float(dec), float(pmr), float(pmd),
                                                    float(px), float(rv), pv)
    if status >= 4:
        _warnings.warn(_starpv_msg[status], UserWarning, 2)
        status -= 4
    if status >= 2:
        _warnings.warn(_starpv_msg[status], UserWarning, 2)
        status -= 2
    if status >= 1:
        _warnings.warn(_starpv_msg[status], UserWarning, 2)
    return pv


# iauSxp
_sofa.iauSxp.argtypes = [_ct.c_double, #s
                        _ndpointer(shape=(1,3), dtype=float, flags='C'), #p
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #sp
def sxp(s, p):
    """ Multiply a p-vector by a scalar.

    :param s: scalar.
    :type s: float

    :param p: p-vector
    :type p: array-like of shape (1,3)

    :returns: s * p as a numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 222
    """
    sp = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauSxp(s, _req_shape_c(p, float, (1,3)), sp)
    return sp


# iauSxpv
_sofa.iauSxpv.argtypes = [_ct.c_double, #s
                        _ndpointer(shape=(2,3), dtype=float, flags='C'), #pv
                        _ndpointer(shape=(2,3), dtype=float, flags='C')] #spv
def sxpv(s, pv):
    """ Multiply a pv-vector by a scalar.

    :param s: scalar.
    :type s: float

    :param pv: pv-vector
    :type pv: array-like of shape (2,3)

    :returns: s * pv as a numpy.matrix of shape 2x3.

    .. seealso:: |MANUAL| page 223
    """
    spv = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    _sofa.iauSxpv(s, _req_shape_c(pv, float, (2,3)), spv)
    return spv


# iauTaitt
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauTaitt.argtypes = [_ct.c_double, #tai1
                            _ct.c_double, #tai2
                            _ct.POINTER(_ct.c_double), #tt1
                            _ct.POINTER(_ct.c_double)] #tt2
    _sofa.iauTaitt.restype = _ct.c_int
except AttributeError:
    pass
def taitt(tai1, tai2):
    """ Timescale transformation: International Atomic Time (TAI) to
    Terrestrial Time (TT).

    :param tai1, tai2: TAI as a two-part Julian Date.
    :type tai1, tai2: float

    :returns: TT as a two-part Julian Date.

    :raises: :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 224
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    tt1 = _ct.c_double()
    tt2 = _ct.c_double()
    s = _sofa.iauTaitt(tai1, tai2, _ct.byref(tt1), _ct.byref(tt2))
    return tt1.value, tt2.value


# iauTaiut1
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauTaiut1.argtypes = [_ct.c_double, #tai1
                            _ct.c_double, #tai2
                            _ct.c_double, #dta
                            _ct.POINTER(_ct.c_double), #ut11
                            _ct.POINTER(_ct.c_double)] #ut12
    _sofa.iauTaiut1.restype = _ct.c_int
except AttributeError:
    pass
def taiut1(tai1, tai2, dta):
    """ Timescale transformation: International Atomic Time (TAI) to
    Universal Time (UT1).

    :param tai1, tai2: TAI as a two-part Julian Date.
    :type tai1, tai2: float

    :param dta: UT1-TAI in seconds.
    :type dta: float

    :returns: UT1 as a two-part Julian Date.

    :raises: :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 225
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    ut11 = _ct.c_double()
    ut12 = _ct.c_double()
    s = _sofa.iauTaiut1(tai1, tai2, dta, _ct.byref(ut11), _ct.byref(ut12))
    return ut11.value, ut12.value


# iauTaiutc
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauTaiutc.argtypes = [_ct.c_double, #tai1
                            _ct.c_double, #tai2
                            _ct.POINTER(_ct.c_double), #utc1
                            _ct.POINTER(_ct.c_double)] #utc2
    _sofa.iauTaiutc.restype = _ct.c_int
except AttributeError:
    pass
_taiutc_msg = {
            1: 'Taiutc: dubious year',
            -1: 'unacceptable date'
            }
def taiutc(tai1, tai2):
    """ Timescale transformation: International Atomic Time (TAI) to
    Coordinated Universal Time (UTC).

    :param tai1, tai2: TAI as a two-part Julian Date.
    :type tai1, tai2: float

    :returns: UTC as a two-part Julian Date.

    :raises: :exc:`ValueError` if the date is outside the range of valid values
        handled by this function.

        :exc:`UserWarning` if the value predates the
        introduction of UTC or is too far in the future to be
        trusted.

        :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 226
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    utc1 = _ct.c_double()
    utc2 = _ct.c_double()
    s = _sofa.iauTaiutc(tai1, tai2, _ct.byref(utc1), _ct.byref(utc2))
    if s < 0:
        raise ValueError(_taiutc_msg[s])
    elif s > 0:
        _warnings.warn(_taiutc_msg[s], UserWarning, 2)
    return utc1.value, utc2.value


# iauTcbtdb
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauTcbtdb.argtypes = [_ct.c_double, #tcb1
                            _ct.c_double, #tcb2
                            _ct.POINTER(_ct.c_double), #tdb1
                            _ct.POINTER(_ct.c_double)] #tdb2
    _sofa.iauTcbtdb.restype = _ct.c_int
except AttributeError:
    pass
def tcbtdb(tcb1, tcb2):
    """ Timescale transformation: Barycentric Coordinate Time (TCB) to
    Barycentric Dynamical Time (TDB).

    :param tcb1, tcb2: TCB as a two-part Julian Date.
    :type tcb1, tcb2: float

    :returns: TDB as a two-part Julian Date.

    :raises: :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 227
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    tdb1 = _ct.c_double()
    tdb2 = _ct.c_double()
    s = _sofa.iauTcbtdb(tcb1, tcb2, _ct.byref(tdb1), _ct.byref(tdb2))
    return tdb1.value, tdb2.value


# iauTcgtt
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauTcgtt.argtypes = [_ct.c_double, #tcg1
                            _ct.c_double, #tcg2
                            _ct.POINTER(_ct.c_double), #tt1
                            _ct.POINTER(_ct.c_double)] #tt2
    _sofa.iauTcgtt.restype = _ct.c_int
except AttributeError:
    pass
def tcgtt(tcg1, tcg2):
    """ Timescale transformation: Geocentric Coordinate Time (TCG) to
    Terrestrial Time (TT).

    :param tcg1, tcg2: TCG as a two-part Julian Date.
    :type tcg1, tcg2: float

    :returns: TT as a two-part Julian Date.

    :raises: :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 228
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    tt1 = _ct.c_double()
    tt2 = _ct.c_double()
    s = _sofa.iauTcgtt(tcg1, tcg2, _ct.byref(tt1), _ct.byref(tt2))
    return tt1.value, tt2.value


# iauTdbtcb
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauTdbtcb.argtypes = [_ct.c_double, #tdb1
                            _ct.c_double, #tdb2
                            _ct.POINTER(_ct.c_double), #tcb1
                            _ct.POINTER(_ct.c_double)] #tcb2
    _sofa.iauTdbtcb.restype = _ct.c_int
except AttributeError:
    pass
def tdbtcb(tdb1, tdb2):
    """ Timescale transformation: Barycentric Dynamical Time (TDB) to
    Barycentric Coordinate Time (TCB).

    :param tdb1, tdb2: TDB as a two-part Julian Date.
    :type tdb1, tdb2: float

    :returns: TCB as a two-part Julian Date.

    :raises: :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 229
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    tcb1 = _ct.c_double()
    tcb2 = _ct.c_double()
    s = _sofa.iauTdbtcb(tdb1, tdb2, _ct.byref(tcb1), _ct.byref(tcb2))
    return tcb1.value, tcb2.value


# iauTdbtt
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauTdbtt.argtypes = [_ct.c_double, #tdb1
                            _ct.c_double, #tdb2
                            _ct.c_double, #dtr
                            _ct.POINTER(_ct.c_double), #tt1
                            _ct.POINTER(_ct.c_double)] #tt2
    _sofa.iauTdbtt.restype = _ct.c_int
except AttributeError:
    pass
def tdbtt(tdb1, tdb2, dtr):
    """ Timescale transformation: Barycentric Dynamical Time (TDB) to
    Terrestrial Time (TT).

    :param tdb1, tdb2: TDB as a two-part Julian Date.
    :type tdb1, tdb2: float

    :param dtr: TDB-TT in seconds.
    :type dtr: float

    :returns: TT as a two-part Julian Date.

    :raises: :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 230
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    tt1 = _ct.c_double()
    tt2 = _ct.c_double()
    s = _sofa.iauTdbtt(tdb1, tdb2, dtr, _ct.byref(tt1), _ct.byref(tt2))
    return tt1.value, tt2.value


# iauTf2a
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauTf2a.argtypes = [_ct.c_char, #s
                            _ct.c_int, #ihour
                            _ct.c_int, #imin
                            _ct.c_double, #sec
                            _ct.POINTER(_ct.c_double)] #rad
    _sofa.iauTf2a.restype = _ct.c_int
except AttributeError:
    pass
_tf2a_msg = {0: 'OK', # Unused
                1:'Tf2a: hours outside range 0-23',
                2:'Tf2a: minutes outside the range 0-59',
                3:'Tf2a: seconds outside the range 0-59.999...'}
def tf2a(s, ihour, imin, sec):
    """ Convert hours, minutes, seconds to radians.

    :param s: sign, '-' is negative, everything else positive.

    :param ihour: hours.
    :type ihour: int

    :param imin: minutes.
    :type imin: int

    :param sec: seconds.
    :type sec: float

    :returns: the converted value in radians (float).

    :raises: :exc:`ValueError` if at least one input value is outside its
        valid range.

        :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 231
    """

    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    rad = _ct.c_double()
    s = _sofa.iauTf2a(s, ihour, imin, sec, _ct.byref(rad))
    if s > 0:
        _warnings.warn(_tf2a_msg[s], UserWarning, 2)
    return rad.value


# iauTf2d
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauTf2d.argtypes = [_ct.c_char, #s
                            _ct.c_int, #ihour
                            _ct.c_int, #imin
                            _ct.c_double, #sec
                            _ct.POINTER(_ct.c_double)] #days
    _sofa.iauTf2d.restype = _ct.c_int
except AttributeError:
    pass
_tf2d_msg = {0: 'OK', # Unused
                1:'Tf2d: hours outside range 0-23',
                2:'Tf2d: minutes outside the range 0-59',
                3:'Tf2d: seconds outside the range 0-59.999...'}
def tf2d(s, ihour, imin, sec):
    """ Convert hours, minutes, seconds to days.

    :param s: sign, '-' is negative, everything else positive.

    :param ihour: hours.
    :type ihour: int

    :param imin: minutes.
    :type imin: int

    :param sec: seconds.
    :type sec: float

    :returns: the converted value in days (float).

    :raises: :exc:`ValueError` if at least one input value is outside its
        valid range.

        :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 232
    """

    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    days = _ct.c_double()
    s = _sofa.iauTf2d(s, ihour, imin, sec, _ct.byref(days))
    if s > 0:
        _warnings.warn(_tf2d_msg[s], UserWarning, 2)
    return days.value


# iauTr
_sofa.iauTr.argtypes = [_ndpointer(shape=(3,3), dtype=float, flags='C'), #r
                        _ndpointer(shape=(3,3), dtype=float, flags='C')] #rt
def tr(r):
    """ Transpose a rotation matrix.

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :returns: transpose as a numpy.matrix of shape 3x3.

    .. seealso:: |MANUAL| page 233
    """
    rt = _np.asmatrix(_np.zeros(shape=(3,3), dtype=float, order='C'))
    _sofa.iauTr(_req_shape_c(r, float, (3,3)), rt)
    return rt


# iauTrxp
_sofa.iauTrxp.argtypes = [_ndpointer(shape=(3,3), dtype=float, flags='C'), #r
                        _ndpointer(shape=(1,3), dtype=float, flags='C'), #p
                        _ndpointer(shape=(1,3), dtype=float, flags='C')] #trp
def trxp(r, p):
    """ Multiply a p-vector by the transpose of a rotation matrix.

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :param p: p-vector.
    :type p: array-like of shape (1,3)

    :returns: numpy.matrix of shape 1x3.

    .. seealso:: |MANUAL| page 234
    """
    trp = _np.asmatrix(_np.zeros(shape=(1,3), dtype=float, order='C'))
    _sofa.iauTrxp(_req_shape_c(r, float, (3,3)),
                                            _req_shape_c(p, float, (1,3)), trp)
    return trp


# iauTrxpv
_sofa.iauTrxpv.argtypes = [_ndpointer(shape=(3,3), dtype=float, flags='C'), #r
                        _ndpointer(shape=(2,3), dtype=float, flags='C'), #pv
                        _ndpointer(shape=(2,3), dtype=float, flags='C')] #trpv
def trxpv(r, pv):
    """ Multiply a pv-vector by the transpose of a rotation matrix.

    :param r: rotation matrix.
    :type r: array-like of shape (3,3)

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: numpy.matrix of shape 2x3.

    .. seealso:: |MANUAL| page 235
    """
    trpv = _np.asmatrix(_np.zeros(shape=(2,3), dtype=float, order='C'))
    _sofa.iauTrxpv(_req_shape_c(r, float, (3,3)),
                                        _req_shape_c(pv, float, (2,3)), trpv)
    return trpv


# iauTttai
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauTttai.argtypes = [_ct.c_double, #tt1
                            _ct.c_double, #tt2
                            _ct.POINTER(_ct.c_double), #tai1
                            _ct.POINTER(_ct.c_double)] #tai2
    _sofa.iauTttai.restype = _ct.c_int
except AttributeError:
    pass
def tttai(tt1, tt2):
    """ Timescale transformation: Terrestrial Time (TT) to
    International Atomic Time (TAI).

    :param tt1, tt2: TT as a two-part Julian Date.
    :type tt1, tt2: float

    :returns: TAI as a two-part Julian Date.

    :raises: :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 236
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    tai1 = _ct.c_double()
    tai2 = _ct.c_double()
    s = _sofa.iauTttai(tt1, tt2, _ct.byref(tai1), _ct.byref(tai2))
    return tai1.value, tai2.value


# iauTttcg
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauTttcg.argtypes = [_ct.c_double, #tt1
                            _ct.c_double, #tt2
                            _ct.POINTER(_ct.c_double), #tcg1
                            _ct.POINTER(_ct.c_double)] #tcg2
    _sofa.iauTttcg.restype = _ct.c_int
except AttributeError:
    pass
def tttcg(tt1, tt2):
    """ Timescale transformation: Terrestrial Time (TT) to
    Geocentric Coordinate Time (TCG).

    :param tt1, tt2: TT as a two-part Julian Date.
    :type tt1, tt2: float

    :returns: TCG as a two-part Julian Date.

    :raises: :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 237
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    tcg1 = _ct.c_double()
    tcg2 = _ct.c_double()
    s = _sofa.iauTttcg(tt1, tt2, _ct.byref(tcg1), _ct.byref(tcg2))
    return tcg1.value, tcg2.value


# iauTttdb
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauTttdb.argtypes = [_ct.c_double, #tt1
                            _ct.c_double, #tt2
                            _ct.c_double, #dtr
                            _ct.POINTER(_ct.c_double), #tdb1
                            _ct.POINTER(_ct.c_double)] #tdb2
    _sofa.iauTttdb.restype = _ct.c_int
except AttributeError:
    pass
def tttdb(tt1, tt2, dtr):
    """ Timescale transformation: Terrestrial Time (TT) to
    Barycentric Dynamical Time (TDB)

    :param tt1, tt2: TT as a two-part Julian Date.
    :type tt1, tt2: float

    :param dtr: TDB-TT in seconds.
    :type dtr: float

    :returns: TDB as a two-part Julian Date.

    :raises: :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 238
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    tdb1 = _ct.c_double()
    tdb2 = _ct.c_double()
    s = _sofa.iauTttdb(tt1, tt2, dtr, _ct.byref(tdb1), _ct.byref(tdb2))
    return tdb1.value, tdb2.value


# iauTtut1
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauTtut1.argtypes = [_ct.c_double, #tt1
                            _ct.c_double, #tt2
                            _ct.c_double, #dt
                            _ct.POINTER(_ct.c_double), #ut11
                            _ct.POINTER(_ct.c_double)] #ut12
    _sofa.iauTtut1.restype = _ct.c_int
except AttributeError:
    pass
def ttut1(tt1, tt2, dt):
    """ Timescale transformation: Terrestrial Time (TT) to
    Universal Time (UT1).

    :param tt1, tt2: TT as a two-part Julian Date.
    :type tt1, tt2: float

    :param dt: TT-UT1 in seconds.
    :type dt: float

    :returns: UT1 as a two-part Julian Date.

    :raises: :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 239
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    ut11 = _ct.c_double()
    ut12 = _ct.c_double()
    s = _sofa.iauTtut1(tt1, tt2, dt, _ct.byref(ut11), _ct.byref(ut12))
    return ut11.value, ut12.value


# iauUt1tai
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauUt1tai.argtypes = [_ct.c_double, #ut11
                            _ct.c_double, #ut12
                            _ct.c_double, #dta
                            _ct.POINTER(_ct.c_double), #tai1
                            _ct.POINTER(_ct.c_double)] #tai2
    _sofa.iauUt1tai.restype = _ct.c_int
except AttributeError:
    pass
def ut1tai(ut11, ut12, dta):
    """ Timescale transformation: Universal Time (UT1) to
    International Atomic Time (TAI).

    :param ut11, ut12: UT1 as a two-part Julian Date.
    :type ut11, ut12: float

    :param dta: UT1-TAI in seconds.
    :type dta: float

    :returns: TAI as a two-part Julian Date

    :raises: :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 240
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    tai1 = _ct.c_double()
    tai2 = _ct.c_double()
    s = _sofa.iauUt1tai(ut11, ut12, dta, _ct.byref(tai1), _ct.byref(tai2))
    return tai1.value, tai2.value


# iauUt1tt
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauUt1tt.argtypes = [_ct.c_double, #ut11
                            _ct.c_double, #ut12
                            _ct.c_double, #dt
                            _ct.POINTER(_ct.c_double), #tt1
                            _ct.POINTER(_ct.c_double)] #tt2
    _sofa.iauUt1tt.restype = _ct.c_int
except AttributeError:
    pass
def ut1tt(ut11, ut12, dt):
    """ Timescale transformation: Universal Time (UT1) to
    Terrestrial Time (TT).

    :param ut11, ut12: UT1 as a two-part Julian Date.
    :type ut11, ut12: float

    :param dt: TT-UT1 in seconds.
    :type dt: float

    :returns: TT as a two-part Julian Date.

    :raises: :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 241
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    tt1 = _ct.c_double()
    tt2 = _ct.c_double()
    s = _sofa.iauUt1tt(ut11, ut12, dt, _ct.byref(tt1), _ct.byref(tt2))
    return tt1.value, tt2.value


# iauUt1utc
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauUt1utc.argtypes = [_ct.c_double, #ut11
                            _ct.c_double, #ut12
                            _ct.c_double, #dut1
                            _ct.POINTER(_ct.c_double), #utc1
                            _ct.POINTER(_ct.c_double)] #utc2
    _sofa.iauUt1utc.restype = _ct.c_int
except AttributeError:
    pass
_ut1utc_msg = {
            1: 'Ut1utc: dubious year',
            -1: 'unacceptable date'
            }
def ut1utc(ut11, ut12, dut1):
    """ Timescale transformation: Universal Time (UT1) to
    Coordinated Universal Time (UTC)

    :param ut11, ut12: UT1 as a two-part Julian Date.
    :type ut11, ut12: float

    :param dut1: UT1-UTC in seconds.
    :type dut1: float

    :returns: UTC as a two-part Julian Date.

    :raises: :exc:`ValueError` if the date is outside the range of valid values
        handled by this function.

        :exc:`UserWarning` if the value predates the
        introduction of UTC or is too far in the future to be
        trusted.

        :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 242
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    utc1 = _ct.c_double()
    utc2 = _ct.c_double()
    s = _sofa.iauUt1utc(ut11, ut12, dut1, _ct.byref(utc1), _ct.byref(utc2))
    if s < 0:
        raise ValueError(_ut1utc_msg[s])
    elif s > 1:
        _warnings.warn(_ut1utc_msg[s], UserWarning, 2)
    return utc1.value, utc2.value


# iauUtctai
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauUtctai.argtypes = [_ct.c_double, #utc1
                            _ct.c_double, #utc2
                            _ct.POINTER(_ct.c_double), #tai1
                            _ct.POINTER(_ct.c_double)] #tai2
    _sofa.iauUtctai.restype = _ct.c_int
except AttributeError:
    pass
_utctai_msg = {
            1: 'Utctai: dubious year',
            -1: 'unacceptable date'
            }
def utctai(utc1, utc2):
    """ Timescale transformation: Coordinated Universal Time (UTC) to
    International Atomic Time (TAI).

    :param utc1, utc2: UTC as a two-part Julian Date.
    :type utc1, utc2: float

    :returns: TAI as a two-part Julian Date.

    :raises: :exc:`ValueError` if the date is outside the range of valid values
        handled by this function.

        :exc:`UserWarning` if the value predates the
        introduction of UTC or is too far in the future to be
        trusted.

        :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 243
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    tai1 = _ct.c_double()
    tai2 = _ct.c_double()
    s = _sofa.iauUtctai(utc1, utc2, _ct.byref(tai1), _ct.byref(tai2))
    if s < 0:
        raise ValueError(_utctai_msg[s])
    elif s > 1:
        _warnings.warn(_utctai_msg[s], UserWarning, 2)
    return tai1.value, tai2.value


# iauUtcut1
# this routine was added in release 2010-12-01 of SOFA
try:
    _sofa.iauUtcut1.argtypes = [_ct.c_double, #utc1
                            _ct.c_double, #utc2
                            _ct.c_double, #dut1
                            _ct.POINTER(_ct.c_double), #ut11
                            _ct.POINTER(_ct.c_double)] #ut12
    _sofa.iauUtcut1.restype = _ct.c_int
except AttributeError:
    pass
_utcut1_msg = {
            1: 'Utcut1: dubious year',
            -1: 'unacceptable date'
            }
def utcut1(utc1, utc2, dut1):
    """ Timescale transformation: Coordinated Universal Time (UTC) to
    Universal Time (UT1)

    :param utc1, utc2: UTC as a two-part Julian Date.
    :type utc1, utc2: float

    :param dut1: UT1-UTC in seconds.
    :type dut1: float

    :returns: UT1 as a two-part Julian Date.

    :raises: :exc:`ValueError` if the date is outside the range of valid values
        handled by this function.

        :exc:`UserWarning` if the value predates the
        introduction of UTC or is too far in the future to be
        trusted.

        :exc:`NotImplementedError` if called with a |SOFA| release prior
        to 2010/12/01.

    .. seealso:: |MANUAL| page 244
    """
    if __sofa_version < (2010, 12, 1):
        raise NotImplementedError
    ut11 = _ct.c_double()
    ut12 = _ct.c_double()
    s = _sofa.iauUtcut1(utc1, utc2, dut1, _ct.byref(ut11), _ct.byref(ut12))
    if s < 0:
        raise ValueError(_utcut1_msg[s])
    elif s > 1:
        _warnings.warn(_utcut1_msg[s], UserWarning, 2)
    return ut11.value, ut12.value


# iauXy06
_sofa.iauXy06.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #x
                            _ct.POINTER(_ct.c_double)] #y
def xy06(date1, date2):
    """ X,Y coordinates of the celestial intermediate pole from series 
    based on IAU 2006 precession and IAU 2000A nutation.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 2-tuple containing X and Y CIP coordinates.

    .. seealso:: |MANUAL| page 246
    """
    x = _ct.c_double()
    y = _ct.c_double()
    _sofa.iauXy06(date1, date2, _ct.byref(x), _ct.byref(y))
    return x.value, y.value


# iauXys00a
_sofa.iauXys00a.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #x
                            _ct.POINTER(_ct.c_double), #y
                            _ct.POINTER(_ct.c_double)] #s
def xys00a(date1, date2):
    """ For a given TT date, compute X, Y coordinates of the celestial 
    intermediate pole and the CIO locator *s*, using IAU 2000A 
    precession-nutation model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 3-tuple:

        * X CIP coordinate
        * Y CIP coordinate
        * the CIO locator *s*.

    .. seealso:: |MANUAL| page 248
    """
    x = _ct.c_double()
    y = _ct.c_double()
    s = _ct.c_double()
    _sofa.iauXys00a(date1, date2, _ct.byref(x), _ct.byref(y), _ct.byref(s))
    return x.value, y.value, s.value


# iauXys00b
_sofa.iauXys00b.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #x
                            _ct.POINTER(_ct.c_double), #y
                            _ct.POINTER(_ct.c_double)] #s
def xys00b(date1, date2):
    """ For a given TT date, compute X, Y coordinates of the celestial 
    intermediate pole and the CIO locator *s*, using IAU 2000B 
    precession-nutation model.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 3-tuple:

        * X CIP coordinate
        * Y CIP coordinate
        * the CIO locator *s*.

    .. seealso:: |MANUAL| page 249
    """
    x = _ct.c_double()
    y = _ct.c_double()
    s = _ct.c_double()
    _sofa.iauXys00b(date1, date2, _ct.byref(x), _ct.byref(y), _ct.byref(s))
    return x.value, y.value, s.value


# iauXys06a
_sofa.iauXys06a.argtypes = [_ct.c_double, #date1
                            _ct.c_double, #date2
                            _ct.POINTER(_ct.c_double), #x
                            _ct.POINTER(_ct.c_double), #y
                            _ct.POINTER(_ct.c_double)] #s
def xys06a(date1, date2):
    """ For a given TT date, compute X, Y coordinates of the celestial 
    intermediate pole and the CIO locator *s*, using IAU 2006 precession 
    and IAU 2000A nutation models.

    :param date1, date2: TT as a two-part Julian date.
    :type date1, date2: float

    :returns: a 3-tuple:

        * X CIP coordinate
        * Y CIP coordinate
        * the CIO locator *s*.

    .. seealso:: |MANUAL| page 250
    """
    x = _ct.c_double()
    y = _ct.c_double()
    s = _ct.c_double()
    _sofa.iauXys06a(date1, date2, _ct.byref(x), _ct.byref(y), _ct.byref(s))
    return x.value, y.value, s.value


# iauZp
_sofa.iauZp.argtypes = [_ndpointer(shape=(1,3), dtype=float, flags='C')] #p
def zp(p):
    """ Zero a p-vector.

    :param p: p-vector.
    :type p: array-like of shape (1,3)

    :returns: a new p-vector filled with zeros. *p* isn't modified.

    .. seealso:: |MANUAL| page 251
    """
    p2 = _np.asmatrix(_np.require(p, float, 'C').copy())
    _sofa.iauZp(p2)
    return p2


# iauZpv
_sofa.iauZpv.argtypes = [_ndpointer(shape=(2,3), dtype=float, flags='C')] #pv
def zpv(pv):
    """ Zero a pv-vector.

    :param pv: pv-vector.
    :type pv: array-like of shape (2,3)

    :returns: a new pv-vector filled with zeros. *pv* isn't modified.

    .. seealso:: |MANUAL| page 252
    """
    pv2 = _np.asmatrix(_req_shape_c(pv, float, (2,3)).copy())
    _sofa.iauZpv(pv2)
    return pv2


# iauZr
_sofa.iauZr.argtypes = [_ndpointer(shape=(3,3), dtype=float, flags='C')] #r
def zr(r):
    """ Initialize a rotation matrix to the null matrix.

    :param r: rotation matrix.
    :type r: array-like shape (3,3)

    :returns: a new rotation matrix filled with zeros. *r* isn't modified.

    .. seealso:: |MANUAL| page 253
    """
    r2 = _np.asmatrix(_req_shape_c(r, float, (3,3)).copy())
    _sofa.iauZr(r2)
    return r2

