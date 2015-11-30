from __future__ import division
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
from scipy.stats import kstest
import matplotlib.pyplot as plt
from scipy.stats import kstest



def recta_gaussiana(x, a, b, A, mu, sigma):
    return a * x + b - A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)


def recta_lorentz(x, a, b, A, mu, sigma):
    return a * x + b - A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)


def cdf(data, model):
    return np.array([np.sum(model <= yy) for yy in data]) / len(model)


wavelength = np.loadtxt('espectro.dat', usecols=(0,))
fnu = np.loadtxt('espectro.dat', usecols=(1,))

n = len(wavelength)
mean = sum(wavelength)/n
sigma = np.sqrt(sum((wavelength-mean)**2)/n)


popt1, pcov1 = curve_fit(recta_gaussiana, wavelength, fnu,
                         [1e-16, 1e-20, 1e-17, mean, sigma])
popt2, pcov2 = curve_fit(recta_lorentz, wavelength, fnu,
                         [1e-16, 1e-20, 1e-17, mean, sigma])


xmin = np.min(wavelength)
xmax = np.max(wavelength)

y1_func_sorted = np.sort(recta_gaussiana(np.linspace(xmin, xmax, 1000), *popt1))
y2_func_sorted = np.sort(recta_lorentz(np.linspace(xmin, xmax, 1000), *popt2))
y_data_sorted = np.sort(fnu)

Dn_scipy_1, prob_scipy_1 = kstest(y_data_sorted, cdf, args=(y1_func_sorted,))
Dn_scipy_2, prob_scipy_2 = kstest(y_data_sorted, cdf, args=(y2_func_sorted,))

print "Dn_scipy 1   : ", Dn_scipy_1
print "Nivel de confianza 1 : ", prob_scipy_1
print "Dn_scipy 2   : ", Dn_scipy_2
print "Nivel de confianza 2 : ", prob_scipy_2
