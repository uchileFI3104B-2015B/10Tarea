from __future__ import division
import numpy as np
import scipy.stats
from scipy.stats import cauchy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def recta_gaussiana(x, a, b, A, mu, sigma):
    return a * x + b - A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)


def recta_lorentz(x, a, b, A, mu, sigma):
    return a * x + b - A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)


wavelength = np.loadtxt('espectro.dat', usecols=(0,))
fnu = np.loadtxt('espectro.dat', usecols=(1,))

n = len(wavelength)
mean = sum(wavelength)/n
sigma = np.sqrt(sum((wavelength-mean)**2)/n)

# primer caso
popt1, pcov1 = curve_fit(recta_gaussiana, wavelength, fnu,
                       [1e-16, 1e-20, 1e-17, mean, sigma])
fig = plt.figure(1)
plt.clf()
plt.axvline(mean, color='g')
plt.plot(wavelength, fnu, 'o', label='Datos')
plt.plot(wavelength, recta_gaussiana(wavelength,*popt1),'ro:',
         label='Ajuste recta-gaussiana')
plt.xlabel('Longitud de onda [$\AA$]')
plt.ylabel('Frecuencia [$erg$ $s^{-1} Hz^{-1} cm^{-2}$]')
plt.legend(loc=4)
plt.draw()
plt.show()
plt.savefig('figura1.png')

#segundo caso
popt2, pcov2 = curve_fit(recta_lorentz, wavelength, fnu,
                       [1e-16, 1e-20, 1e-17, mean, sigma])
fig = plt.figure(2)
plt.clf()
plt.axvline(mean, color='g')
plt.plot(wavelength, fnu, 'o', label='Datos')
plt.plot(wavelength, recta_lorentz(wavelength,*popt2),'ro:',
         label='Ajuste recta-lorentz')
plt.xlabel('Longitud de onda [$\AA$]')
plt.ylabel('Frecuencia [$erg$ $s^{-1} Hz^{-1} cm^{-2}$]')
plt.legend(loc=4)
plt.draw()
plt.show()
plt.savefig('figura2.png')
