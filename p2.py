'''
Este script ajusta los datos del archivo espectro.dat
(frecuencias [[erg s-1 Hz-1 cm-2]] vs longitudes de onda en [Angstrom])
de dos formas (modelos):
1) recta - gaussiana
2) recta - lorentz
Utilizando curve_fit. Realiza test KS con scipy (kstest y kstwobign
[aproximacion] para N grande) para cada modelo entregando
Dn y Nivel de confianza en cada caso.
'''
from __future__ import division
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
from scipy.stats import kstest
import matplotlib.pyplot as plt
from scipy.stats import kstest
from scipy.stats import kstwobign
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


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
y_data_sorted = np.sort(fnu)
N = len(y_data_sorted)

y1_func_sorted = np.sort(recta_gaussiana(np.linspace(xmin, xmax, 1000),
                         *popt1))
y2_func_sorted = np.sort(recta_lorentz(np.linspace(xmin, xmax, 1000), *popt2))

# Test KS con scipy kstest
Dn_scipy_1, prob_scipy_1 = kstest(y_data_sorted, cdf, args=(y1_func_sorted,))
Dn_scipy_2, prob_scipy_2 = kstest(y_data_sorted, cdf, args=(y2_func_sorted,))

print "Dn_scipy 1   : ", Dn_scipy_1
print "Nivel de confianza 1 : ", prob_scipy_1
print "Dn_scipy 2   : ", Dn_scipy_2
print "Nivel de confianza 2 : ", prob_scipy_2


CDF_gauss = np.array([np.sum(y1_func_sorted <= yy)
                     for yy in y_data_sorted]) / len(y1_func_sorted)
CDF_lorentz = np.array([np.sum(y2_func_sorted <= yy)
                       for yy in y_data_sorted]) / len(y2_func_sorted)

# plot
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111)
ax.plot(y_data_sorted, np.arange(N) / N, '-^', drawstyle='steps-post',
         color='b')
ax.plot(y_data_sorted, np.arange(1, N+1) / N, '-.', drawstyle='steps-post',
         color='b')
ax.plot(y_data_sorted, CDF_gauss, '-x', drawstyle='steps-post',
         label='Ajuste recta-gaussiana', color='g')
ax.plot(y_data_sorted, CDF_lorentz, '-x', drawstyle='steps-post',
         label='Ajuste recta-lorentz', color='r')
plt.legend(loc=2)
plt.title('Funciones de probabilidad acumulada')
ax.set_xlabel('Frecuencia [$erg$ $s^{-1} Hz^{-1} cm^{-2}$]')

# zoom en el plot
axins = zoomed_inset_axes(ax, 6, loc=6)
axins.plot(y_data_sorted, np.arange(N) / N, '-^', drawstyle='steps-post',
         color='b')
axins.plot(y_data_sorted, np.arange(1, N+1) / N, '-.', drawstyle='steps-post',
         color='b')
axins.plot(y_data_sorted, CDF_gauss, '-x', drawstyle='steps-post',
         label='Ajuste recta-gaussiana', color='g')
axins.plot(y_data_sorted, CDF_lorentz, '-x', drawstyle='steps-post',
         label='Ajuste recta-lorentz', color='r')

plt.yticks(visible=False)
plt.xticks(visible=False)
x1, x2, y1, y2 = 1.375*10**-16, 1.39*10**-16, 0.05, 0.09 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")

plt.draw()
plt.show()
plt.savefig('figura2.png')


# Dn critico
ks_dist = kstwobign() # aproximacion para N grande
alpha = 0.05
Dn_critico = ks_dist.ppf(1 - alpha) / np.sqrt(N)
print "Dn_critico = ", Dn_critico
