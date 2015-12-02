from __future__ import division
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

'''
Este script toma los datos de espectro.dat y los ajusta de forma "gaussiana"
y "lorentziana".
'''


def gauss(x, m, n, A, mu, sigma):
    g = m * x + n - A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return g


def lorentz(x, m, n, A, mu, sigma):
    l = m * x + n - A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)
    return l


def chi2(d, m, params):
    x = d[0]
    y = m(x, params[0], params[1], params[2], params[3], params[4])
    chi_2 = (d[1] - y)**2
    return np.sum(chi_2)


# DATOS
datos = np.loadtxt('espectro.dat')
l = datos[:, 0]  # longitud de onda (A)
f = datos[:, 1]  # flujo (erg / s / Hz / cm2)
n = len(l)  # número de datos
mu = sum(l) / n
sigma = np.sqrt(sum((l - mu)**2) / n)

# POPT = VALOR OPTIMO DE LOS PARÁMETROS, PCOV = COVARIANZA DE POPT
popt1, pcov1 = curve_fit(gauss, l, f, [1e-16, 1e-18, 1e-16, mu, sigma])
popt2, pcov2 = curve_fit(lorentz, l, f, [1e-16, 1e-18, 1e-16, mu, sigma])

# CHI CUADRADO
chi2_1 = chi2([l, f], gauss, popt1)
chi2_2 = chi2([l, f], lorentz, popt2)

# GRÁFICOS
fig = plt.figure(1)
plt.clf()
plt.plot(l, f, '*', label='Datos Experimentales', color='m')
plt.plot(l, gauss(l, *popt1), color='c', label='Ajuste Gaussiano')
plt.xlabel('Longitud de Onda [$\AA$]')
plt.ylabel('Flujo [erg $s^{-1}$ $Hz^{-1}$ $cm^{-2}$]')
plt.legend(loc=0)
plt.draw()
plt.show()
plt.savefig('p1gauss.png')

fig2 = plt.figure(2)
plt.plot(l, f, '*', label='Datos Experimentales', color='m')
plt.plot(l, lorentz(l, *popt2), color='g', label='Ajuste Lorentziano')
plt.xlabel('Longitud de Onda [$\AA$]')
plt.ylabel('Flujo [erg $s^{-1}$ $Hz^{-1}$ $cm^{-2}$]')
plt.legend(loc=0)
plt.draw()
plt.show()
plt.savefig('p1lorentz.png')

# RESULTADOS
print popt1  # m, n, A, mu, sigma
print popt2
print chi2_1
print chi2_2
