from __future__ import division
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def recta_gaussian(x, a, b, A, mu, sigma):
    return a * x +b - A * norm(loc=mu, scale=sigma).pdf(x)


wavelength = np.loadtxt('espectro.dat', usecols=(0,))
fnu = np.loadtxt('espectro.dat', usecols=(1,))

n = len(wavelength)
mean = sum(wavelength)/n
sigma = np.sqrt(sum((wavelength-mean)**2)/n)

popt, pcov = curve_fit(recta_gaussian, wavelength, fnu,
                       [1e-16, 1e-20, 1e-17, mean, sigma])
fig = plt.figure(1)
plt.clf()
plt.axvline(mean, color='g')
plt.plot(wavelength, fnu, 'o', label='Datos')
plt.plot(wavelength, recta_gaussian(wavelength,*popt),'ro:',label='fit')
plt.legend()
plt.draw()
plt.show()
