from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import leastsq, curve_fit
from scipy.stats import kstest


def datos():
    '''
    Lee el archivo y retorna sus columnas como 2 listas
    '''
    arch = np.loadtxt("espectro.dat")
    wave = arch[:, 0]
    flux = arch[:, 1]
    return wave, flux


def rectagauss(x, a, b, A, mu, sigma):
    '''
    Retorna el ajuste para un modelo gaussiano
    '''
    recta = a + b * x
    gauss = A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return recta - gauss


def rectalorentz(x, a, b, A, mu, sigma):
    '''
    Retorna el ajuste para un modelo lorentziano
    '''
    recta = a + b * x
    lorentz = A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)
    return recta - lorentz


def fit(funcion, x, y, seeds):
    '''
    Calcula coeficientes a y b de la ecuacion que
    mejor fitea los vectores x,y.
    '''
    popt, pcov = curve_fit(funcion, x, y, seeds)
    return popt


def chi(funcion, x, y, seeds_op):
    '''
    Retorna el chi^2 asociado a la ecuacion entrante
    '''
    X = 0
    n = len(x)
    for i in range(n):
        X += (y[i] - funcion(x[i], *seeds_op))**2
    return X


def errorgauss(seeds, wave, flux):
    '''
    Residuo del modelo gaussiano
    '''
    error = flux - rectagauss(wave, *seeds)
    return error


def errorlorentz(seeds, wave, flux):
    '''
    Residuo del modelo lorentziano
    '''
    error = flux - rectalorentz(wave, *seeds)
    return error


def cdf(data, funcion):
    return np.array([np.sum(funcion <= yy) for yy in data]) / len(funcion)


def test_ks(wave, flux, p_optg, p_optl):
    xmin = np.min(wave)
    xmax = np.max(wave)
    x = np.linspace(xmin, xmax, 1000)
    # Gauss
    y_gauss_ord = np.sort(rectagauss(x, *p_optg))
    y_expg_ord = np.sort(flux)
    dng, probg = kstest(y_expg_ord, cdf, args=(y_gauss_ord,))

    # Lorentz
    y_lorentz_ord = np.sort(rectalorentz(x, *p_optl))
    y_expl_ord = np.sort(flux)
    dnl, probl = kstest(y_expl_ord, cdf, args=(y_lorentz_ord,))

    return dng, probg, dnl, probl


# Setup
wave, flux = datos()
n = len(wave)

# Semillas para el ajuste, sacadas del grafico espectro (a, b, A, mu, sigma)
seeds = 9.e-17, 8.e-21, 1.e-17, 6550., 10.

# Main
poptg = fit(rectagauss, wave, flux, seeds)  # Gaussiana
poptl = fit(rectalorentz, wave, flux, seeds)  # Lorentziana

print 'Modelo Gauss:'
print 'a =', poptg[0], 'b =', poptg[1], 'A =', poptg[2], 'mu =', poptg[3],\
    'sigma =', poptg[4], 'chi**2 = ', chi(rectagauss, wave, flux, poptg)
print 'Modelo Lorentz:'
print 'a =', poptl[0], 'b =', poptl[1], 'A =', poptl[2], 'mu =', poptl[3],\
    'sigma =', poptl[4], 'chi**2 = ', chi(rectalorentz, wave, flux, poptl)

optimog = leastsq(errorgauss, seeds, args=(wave, flux))[0]
optimol = leastsq(errorlorentz, seeds, args=(wave, flux))[0]
ks = test_ks(wave, flux, optimog, optimol)

print 'Test KS Gauss:', ks[:2]
print 'Test KS Lorentz:', ks[2:]

# Graficos
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(wave, flux, 'y*-', label="Datos Experimentales")
ax1.plot(wave, rectalorentz(wave, *poptl), 'r-', linewidth='1.5',
         label="Ajuste Lorentz")
ax1.plot(wave, rectagauss(wave, *poptg), 'g-', linewidth='1.5',
         label="Ajuste Gauss")
ax1.set_xlabel("Longitud de onda $[Angstrom]$")
ax1.set_ylabel("Flujo por unidad de frecuencia $[erg s^{-1} Hz^{-1} cm^{-2}]$")
ax1.set_title("Espectro y sus ajustes")

plt.legend(loc='lower right')
plt.show()
