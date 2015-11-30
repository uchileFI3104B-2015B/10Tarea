'''
Este script ajusta los datos del archivo espectro.dat
(frecuencias [[erg s-1 Hz-1 cm-2]] vs longitudes de onda en [Angstrom])
de dos formas (modelos):
1) recta - gaussiana
2) recta - lorentz
Utilizando curve_fit.
Devuelve un grafico con el ploteo de los datos y de ambos ajustes, mas los
arreglos de los mejores parametros para cada caso.
'''
from __future__ import division
import numpy as np
import scipy.stats
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


popt1, pcov1 = curve_fit(recta_gaussiana, wavelength, fnu,
                         [1e-16, 1e-20, 1e-17, mean, sigma])
popt2, pcov2 = curve_fit(recta_lorentz, wavelength, fnu,
                         [1e-16, 1e-20, 1e-17, mean, sigma])
fig = plt.figure(1)
plt.clf()
plt.plot(wavelength, fnu, 'o', label='Datos')
plt.plot(wavelength, recta_gaussiana(wavelength, *popt1), 'ro:',
         label='Ajuste recta-gaussiana', alpha=0.4)
plt.plot(wavelength, recta_lorentz(wavelength, *popt2), 'go:',
         label='Ajuste recta-lorentz', alpha=0.4)
plt.xlabel('Longitud de onda [$\AA$]')
plt.ylabel('Frecuencia [$erg$ $s^{-1} Hz^{-1} cm^{-2}$]')
plt.legend(loc=4)
plt.draw()
plt.show()
plt.savefig('figura1.png')

print 'mejores parametros: a, b, A, mu, sigma'
print 'mejores parametros caso1'
print popt1
print 'mejores parametros caso 2'
print popt2
