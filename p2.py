from __future__ import division
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import kstest


def gauss(x, m, n, A, mu, sigma):
    g = m * x + n - A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return g


def lorentz(x, m, n, A, mu, sigma):
    l = m * x + n - A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)
    return l


def cdf(data, model):
    x = np.array([np.sum(model <= yy) for yy in data]) / len(model)
    return x

# DATOS
datos = np.loadtxt('espectro.dat')
l = datos[:, 0]  # longitud de onda (A)
f = datos[:, 1]  # flujo (erg / s / Hz / cm2)
n = len(l)  # número de datos
mu = sum(l) / n
sigma = np.sqrt(sum((l - mu)**2) / n)
xmin = np.min(l)
xmax = np.max(l)
y_data_sorted = np.sort(f)
num = len(y_data_sorted)

popt1, pcov1 = curve_fit(gauss, l, f, [1e-16, 1e-18, 1e-16, mu, sigma])
popt2, pcov2 = curve_fit(lorentz, l, f, [1e-16, 1e-18, 1e-16, mu, sigma])

y1_func_sorted = np.sort(gauss(np.linspace(xmin, xmax, 1000), *popt1))
y2_func_sorted = np.sort(lorentz(np.linspace(xmin, xmax, 1000), *popt2))

# TEST DE KS
Dn_scipy_1, prob_scipy_1 = kstest(y_data_sorted, cdf, args=(y1_func_sorted,))
Dn_scipy_2, prob_scipy_2 = kstest(y_data_sorted, cdf, args=(y2_func_sorted,))

CDFg = np.array([np.sum(y1_func_sorted <= yy)
                for yy in y_data_sorted]) / len(y1_func_sorted)
CDFl = np.array([np.sum(y2_func_sorted <= yy)
                for yy in y_data_sorted]) / len(y2_func_sorted)

# GRAFICO
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111)
ax.plot(y_data_sorted, np.arange(num) / num, '-*', drawstyle='steps-post',
        color='c', label='Modelo Gauss')
ax.plot(y_data_sorted, np.arange(1, num+1) / num, '-o', drawstyle='steps-post',
        color='m', label='Modelo Lorentz')
ax.plot(y_data_sorted, CDFg, '-x', drawstyle='steps-post', color='g',
        label='Ajuste Gaussiano')
ax.plot(y_data_sorted, CDFl, '-x', drawstyle='steps-post',
        label='Ajuste Lorentziano', color='r')
ax.set_xlabel('Flujo [erg $s^{-1}$ $Hz^{-1}$ $cm^{-2}$]')
ax.set_ylabel('Probabilidad')
plt.legend(loc=0)
plt.draw()
plt.show()
plt.savefig('p2.png')

# RESULTADOS
print "Dn_scipy Gauss:", Dn_scipy_1
print "Nivel de confianza Gauss:", prob_scipy_1
print "Dn_scipy Lorentz: ", Dn_scipy_2
print "Nivel de confianza Lorentz : ", prob_scipy_2
