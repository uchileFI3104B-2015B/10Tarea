##############################################################################
'''
Metodos Numericos para la Ciencia e Ingenieria
FI3104-1
Tarea 10
Maximiliano Dirk Vega Aguilera
18.451.231-9
'''
##############################################################################

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scist
from scipy.optimize import curve_fit
from scipy.stats import kstwobign
from scipy.stats import kstest

##############################################################################
##############################################################################
# funciones


def ajuste_gauss(x, a, b, A, mu, sigma):
    '''
    ajuste de una recta menos una gaussiana, los parametros (a,b) pertenecen
    a la recta y los paramatros (A, mu, sigma) a la gaussiana
    '''
    y = (a + b * x) - A * scist.norm(loc=mu, scale=sigma).pdf(x)
    return y


def ajuste_lorentz(x, a, b, A, mu, sigma):
    '''
    ajuste de una recta menos una perfil de lorentz, los parametros (a,b)
    pertenecen a la recta y los paramatros (A, mu, sigma) al perfil de lorentz
    '''
    y = (a + b * x) - A * scist.cauchy(loc=mu, scale=sigma).pdf(x)
    return y


def obtener_parametros(f, x, y, parametros=False):
    '''
    entrega los parametros para el ajuste de las funciones anteriores a partir
    de los datos. Si se le entrega 'parametros' los usa como punto de partida
    '''
    if parametros is False:
        popt, pcov = curve_fit(f, x, y)
        return popt
    else:
        popt, pcov = curve_fit(f, x, y, p0=parametros)
        return popt


def chi(f, x, y, parametros):
    chi2 = 0
    for i in range(len(x)):
        chi2 += (y[i] - f(x[i], *parametros))**2
    return chi2


def cdf(data, model):
    '''
    funcion de distribucion acumulada para los datos segun un modelo.
    '''
    return np.array([np.sum(model <= yy) for yy in data]) / len(model)


##############################################################################
##############################################################################
# P1
File_espectro = np.loadtxt('espectro.dat')  # se lee archivo
wavelength = File_espectro[:, 0]  # se asigna variable lambda
fnu = File_espectro[:, 1]  # se asigna variable flujo por frecuencia
# parametros iniciales para el ajuste, obtenidos del grafico de datos
parametros = [9.1e-17, 7.4e-21, 1e-17, 6563., 7.]
#############
# gauss
#############
# obtener los parametros para la funcion ajuste_gauss
poptg = obtener_parametros(ajuste_gauss, wavelength,
                           fnu, parametros=parametros)
# resultados
print '----------------------------------------------'
print 'f(x) = (a + b * x) - A * 1 / sqrt(2pi sigma^2) * exp((x - mu) / sigma)'
print 'a = ', poptg[0]
print 'b = ', poptg[1]
print 'A = ', poptg[2]
print 'mu = ', poptg[3]
print 'sigma = ', poptg[4]
print 'chi**2 = ', chi(ajuste_gauss, wavelength, fnu, poptg)
print '----------------------------------------------'
# Graficar
plt.plot(wavelength, ajuste_gauss(wavelength, *poptg), 'r--')
plt.plot(wavelength, fnu, 'g^')
plt.axis([6450, 6675, 1.28e-16, 1.42e-16])
plt.title('Perfil Gaussiano')
plt.xlabel('$\AA$')
plt.ylabel('$erg\ \  s^{-1}\ Hz^{-1}\ cm^{-2}$')
plt.show()


###############
# lorentz
###############

# obtener los parametros para la funcion ajuste_lorentz
poptl = obtener_parametros(ajuste_lorentz, wavelength,
                           fnu, parametros=parametros)
# resultados
print '----------------------------------------------'
print 'f(x) = (a + b * x) - A / (pi * (1 + (x - mu)**2 / sigma**2))'
print 'a = ', poptl[0]
print 'b = ', poptl[1]
print 'A = ', poptl[2]
print 'mu = ', poptl[3]
print 'sigma = ', poptl[4]
print 'chi**2 = ', chi(ajuste_lorentz, wavelength, fnu, poptl)
print '----------------------------------------------'
# Graficar
plt.plot(wavelength, ajuste_lorentz(wavelength, *poptl), 'r--')
plt.plot(wavelength, fnu, 'g^')
plt.axis([6450, 6675, 1.28e-16, 1.42e-16])
plt.title('Perfil Lorentziano')
plt.xlabel('$\AA$')
plt.ylabel('$erg\ \  s^{-1}\ Hz^{-1}\ cm^{-2}$')
plt.show()

###################
# gauss vs lorentz
###################

plt.plot(wavelength, ajuste_gauss(wavelength, *poptg), 'b-', label='Gauss')
plt.plot(wavelength, ajuste_lorentz(wavelength, *poptl),
         'g-', label='Lorentz')
plt.plot(wavelength, fnu, 'r^', label='Datos')
plt.axis([6450, 6675, 1.28e-16, 1.42e-16])
plt.title('Perfil Gaussiano vs Perfil Lorentziano')
plt.xlabel('longitud de onda [$\AA$]')
plt.ylabel('$F_{nu} $[$erg\ \  s^{-1}\ Hz^{-1}\ cm^{-2}$]')
plt.legend(loc='lower right')
plt.show()

##############################################################################
##############################################################################
# P2

N = 1e+5  # numero de datos a modelar
n = len(fnu)  # largo de los datos
# determina valores extremos de la longitud de onda
xmin = np.min(wavelength)  # minimo wavelength
xmax = np.max(wavelength)  # maximo wavelength

##########
# gauss
##########

parametros_gauss = obtener_parametros(ajuste_gauss, wavelength, fnu,
                                      parametros=parametros)

# crea y ordena datos fnu a partir del modelo
fnu_modelo_gauss = np.sort(ajuste_gauss(np.linspace(xmin, xmax, N),
                           *parametros_gauss))
fnu_datos_gauss = np.sort(fnu)  # ordena los valores fnu
cdf_gauss = np.array([np.sum(fnu_modelo_gauss <= yy)
                     for yy in fnu_datos_gauss]) / len(fnu_modelo_gauss)

plt.plot(fnu_datos_gauss, np.arange(n) / n, 'b', drawstyle='steps-post')
plt.plot(fnu_datos_gauss, np.arange(1, n + 1) / n, 'g',
         drawstyle='steps-post')
plt.plot(fnu_datos_gauss, cdf_gauss, 'r-^', drawstyle='steps-post')
plt.title('CDF Gauss')
plt.xlabel('$F_{nu} $[$erg\ \  s^{-1}\ Hz^{-1}\ cm^{-2}$]')
plt.ylabel('Probabilidad')
plt.show()

# obtiene valores maximos de la distancia entre cdf de los datos y el modelo
max1g = np.max(cdf_gauss - np.arange(n) / n)
max2g = np.max(np.arange(1, n + 1) / n - cdf_gauss)
# guarda el maximo de los maximos
Dng = max(max1g, max2g)
print '----------------------------------------------'
print 'Gauss'
print 'Distacia Maxima entre CDF\'s =', Dng

ks_dist_g = kstwobign()  # crea el objeto de Kolmogorov-Smirnov two-sided test
alpha = 0.05  # tolerancia 5%
# obtiene la distancia critica como el punto en 95% de la distribucion
Dng_crit = ks_dist_g.ppf(1-alpha) / np.sqrt(n)

print 'Distacia critica al 5% =', Dng_crit

print 'Nivel de confianza =', 1 - ks_dist_g.cdf(Dng*np.sqrt(n))

# test ya implementado
Dn_testg, prob_scipyg = kstest(fnu_datos_gauss, cdf, args=(fnu_modelo_gauss,))

print 'Distacia Maxima entre CDF\'s implementado =', Dn_testg
print 'Nivel de confianza implementado =', prob_scipyg
print '----------------------------------------------'

###############
# lorentz
###############

parametros_lorentz = obtener_parametros(ajuste_lorentz, wavelength, fnu,
                                        parametros=parametros)

# crea y ordena datos fnu a partir del modelo
fnu_modelo_lorentz = np.sort(ajuste_lorentz(np.linspace(xmin, xmax, N),
                             *parametros_lorentz))
fnu_datos_lorentz = np.sort(fnu)  # ordena los valores fnu
cdf_lorentz = np.array([np.sum(fnu_modelo_lorentz <= yy)
                       for yy in fnu_datos_lorentz]) / len(fnu_modelo_lorentz)

plt.plot(fnu_datos_lorentz, np.arange(n) / n, 'b', drawstyle='steps-post')
plt.plot(fnu_datos_lorentz, np.arange(1, n + 1) / n, 'g',
         drawstyle='steps-post')
plt.plot(fnu_datos_lorentz, cdf_lorentz, 'r-^', drawstyle='steps-post')
plt.title('CDF Lorentz')
plt.xlabel('$F_{nu} $[$erg\ \  s^{-1}\ Hz^{-1}\ cm^{-2}$]')
plt.ylabel('Probabilidad')
plt.show()

# obtiene valores maximos de la distancia entre cdf de los datos y el modelo
max1l = np.max(cdf_lorentz - np.arange(n) / n)
max2l = np.max(np.arange(1, n + 1) / n - cdf_lorentz)
# guarda el maximo de los maximos
Dnl = max(max1l, max2l)
print '----------------------------------------------'
print 'Lorentz'
print 'Distacia Maxima entre CDF\'s =', Dnl

ks_dist_l = kstwobign()  # crea el objeto de Kolmogorov-Smirnov two-sided test
alpha = 0.05  # tolerancia 5%
# obtiene la distancia critica como el punto en 95% de la distribucion
Dnl_crit = ks_dist_l.ppf(1-alpha) / np.sqrt(n)

print 'Distacia critica al 5% =', Dng_crit

print 'Nivel de confianza =', 1 - ks_dist_l.cdf(Dnl*np.sqrt(n))

# test ya implementado
Dn_testl, prob_scipyl = kstest(fnu_datos_lorentz, cdf,
                               args=(fnu_modelo_lorentz,))

print 'Distacia Maxima entre CDF\'s implementado =', Dn_testl
print 'Nivel de confianza implementado =', prob_scipyl
print '----------------------------------------------'

##############################################################################
##############################################################################
