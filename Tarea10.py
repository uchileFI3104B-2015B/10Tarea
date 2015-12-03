#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats



def modelo_Gaussiano(x, m, n, A, mu, sigma):
    '''
    Retorna la diferencia entre una recta y un ajuste Gaussiano
    '''
    R = m * x + n
    L = A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return R - L


def modelo_Lorentz(x, m, n, A, mu, sigma):
    '''
    Retorna la diferencia entre una recta y un ajuste Lorentziano
    '''
    R = m * x + n
    L = A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)
    return R - L


def chi2(x , y, parametros, f):
    m, n, A, mu, sigma = parametros_modelo
    y_m = f(x, m, n, A, mu, sigma)
    sigma = (y - y_m)**2
    return np.sum(sigma)


def cdf(data, model):
    return np.array([np.sum(model <= yy) for yy in data]) / len(model)
