#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import cauchy

def modelo1(x, params):
    A, mu, sigma, a, b = params
    return A * norm(loc=mu, scale=sigma).pdf(x) + a*x + b

def modelo2(x, params):
    A, mu, sigma, a, b = params
    return A * cauchy(loc=mu, scale=sigma).pdf(x) + a*x + b

def deriv_modelo(x, params, indice_der, modelo):
    epsilon = 10**(-7)
    dpar = np.zeros(len(params))
    dpar[indice_der] = epsilon
    return (modelo(x, params + dpar) - modelo(x, params - dpar))/(2*epsilon)

def chi2(data, modelo, params):
    return np.sum(np.power((data[1] - modelo(data[0], params)), 2))

def beta(data, modelo, params):
    betak = np.zeros((5, len(data[0])))
    f1 = data[1] - modelo(data[0], params)
    for k in range(5):
        f2 = deriv_modelo(data[0], params, k, modelo)
        betak[i] = np.sum(f1*f2)
    return betak

def alpha(data, modelo, params, lamb):
    derivadas = np.zeros((5, len(data[0])))
    for k in range(5):
        derivadas[k] = deriv_modelo(data[0], params, k, modelo)
    alphakl = np.zeros((5,5))
    for k in range(5):
        for l in range(5):
            alphakl[k][l] = np.sum(derivadas[k]*derivadas[l])
    for j in range(5):
        alphakl[j][j] = alphakl[j][j] * (1 + lamb)
    return alphakl

wavelen = np.loadtxt("espectro.dat", usecols=(0,))
frec = np.loadtxt("espectro.dat", usecols=(1,))
data = np.zeros((2, len(wavelen)))
data[0] = wavelen
data[1] = (10**16)*frec
