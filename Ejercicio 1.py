from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit

np.random.seed(1911)

def datos():
    arch = np.loadtxt("espectro.dat")
    wave = arch[:, 0]
    flux = arch[:, 1]
    return wave, flux

def rectagauss(x, a, b, A, mu, sigma):
    recta = a + b * x
    gauss = A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return  recta - gauss

def rectalorentz(x, a, b, A, mu, sigma):
    recta = a + b * x
    lorentz = A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)
    return recta - lorentz

def fit(funcion, x, y, seed):
    '''
    Calcula coeficientes a y b de la ecuacion que mejor fitea los vectores x,y.
    '''
    popt, pcov = curve_fit(funcion, x, y, seed)
    return popt