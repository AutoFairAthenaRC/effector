import scipy.integrate as integrate
import numpy as np


def normalization_constant_1D(func, start, stop):
    """Computes normalization constant of PDP of feature

    :param s: index of feature
    :returns:

    """

    def wrapper(x):
        x = np.array([x])
        return func(x)

    y = integrate.quad(wrapper, start, stop)[0]
    return y


def expectation_1D(xs, func, p_xc, s, start=-np.inf, stop=np.inf):
    """

    :param func:  (N, D) -> (N)
    :param xs: np.float
    :param p_xc: p_xc float -> [0,1]
    :param s: index of feature of interest
    :param start: left limit of integral
    :param stop: right limit of integral
    :returns:

    """
    def func_wrap(xc):
        x = np.array([xs, xc]) if s == 0 else np.array([xc, xs])
        x = np.expand_dims(x, axis=0)
        return func(x)[0] * p_xc(xc)
    return integrate.quad(func_wrap, start, stop)


def expecation_2D(xs, func, p_xc, s, axis_limits):
    """

    :param func:  (N, D) -> (N)
    :param xs: np.float
    :param p_xc: p_xc (float, float)-> [0,1]
    :param s: index of feature of interest
    :param start: left limit of integral
    :param stop: right limit of integral
    :returns:

    """
    def func_wrap(xc1, xc2):
        if s == 0:
            x = np.array([xs, xc1, xc2])
        elif s == 1:
            x = np.array([xc1, xs, xc2])
        elif s == 2:
            x = np.array([xc1, xc2, xs])
        x = np.expand_dims(x, axis=0)
        return func(x)[0] * p_xc(xc1, xc2)

    def gfun(xc2):
        return axis_limits[0, 1]

    def hfun(xxc2):
        return axis_limits[1, 1]

    return integrate.dblquad(func_wrap, axis_limits[0, 2], axis_limits[1, 2], gfun, hfun)