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
    gauss = A * scipy.stats.norm(loc = mu, scale = sigma).pdf(x)
    return  recta - gauss

def rectalorentz(x, a, b, A, mu, sigma):
    recta = a + b * x
    lorentz = A * scipy.stats.cauchy(loc = mu, scale = sigma).pdf(x)
    return recta - lorentz

def fit(funcion, x, y, seed):
    '''
    Calcula coeficientes a y b de la ecuacion que
    mejor fitea los vectores x,y.
    '''
    popt, pcov = curve_fit(funcion, x, y, seed)
    return popt

# Setup
wave, flux = datos()
n = len(wave)
seeds = [9.e-17, 7.e-21, 1.3e-16, 6560., 7.]  # Semillas
poptg = fit(rectagauss, wave, flux, seeds)  # Gaussiana
poptl = fit(rectalorentz, wave, flux, seeds)  # Lorentziana

print 'Modelo Gauss:'
print 'a = ', poptg[0], 'b = ', poptg[1], 'A = ', poptg[2], 'mu = ', poptg[3], 'sigma = ', poptg[4]
print 'Modelo Lorentz:'
print 'a = ', poptl[0], 'b = ', poptl[1], 'A = ', poptl[2], 'mu = ', poptl[3], 'sigma = ', poptl[4]


# Graficos
plt.plot(wave, rectagauss(wave, poptg[0], poptg[1], poptg[2], poptg[3], poptg[4]), 'm--', linewidth = '2')
plt.plot(wave, flux, 'g')
plt.title("Espectro, ajuste gauss")
plt.xlabel('Longitus de onda $[Angstrom]$')
plt.ylabel('Flujo $[erg s^{-1} Hz^{-1} cm^{-2}]$')
plt.show()


plt.plot(wave, rectalorentz(wave, poptl[0], poptl[1], poptl[2], poptl[3], poptl[4]), 'm--', linewidth = '2')
plt.plot(wave, flux, 'g')
plt.title("Espectro, ajuste lorentz")
plt.xlabel('Longitus de onda $[Angstrom]$')
plt.ylabel('Flujo $[erg s^{-1} Hz^{-1} cm^{-2}]$')
plt.show()