##############################################################################
'''
Metodos Numericos para la Ciencia e Ingenieria
FI3104-1
Tarea 10
Maximiliano Dirk Vega Aguilera
18.451.231-9
'''
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scist
from scipy.optimize import curve_fit
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


def cdf(data, model):
	'''
	funcion de distribucion acumulada para los datos segun un modelo.
	'''
	return np.array([np.sum(model <= yy) for yy in data]) / len(model)


##############################################################################
##############################################################################
# P1
File_espectro = np.loadtxt('espectro.dat')  # se lee archivo
wavelength = File_espectro[:,0]  # se asigna variable lambda
fnu = File_espectro[:,1]  # se asigna variable flujo por frecuencia
parametros = [9.1e-17, 7.4e-21, 1e-17, 6563., 7.]  # parametros iniciales para el ajuste
#plt.plot(wavelength, fnu, '-o')
#plt.show()
#############
# gauss
#############
# obtienes los parametros para la funcion ajuste_gauss
poptg = obtener_parametros(ajuste_gauss, wavelength, fnu, parametros=parametros)
# Escribir resultados
print '----------------------------------------------'
print 'f(x) = (a + b * x) - A * exp((x - mu) / sigma)'
print 'a = ', poptg[0]
print 'b = ', poptg[1]
print 'A = ', poptg[2]
print 'mu = ', poptg[3]
print 'sigma = ', poptg[4]
print '----------------------------------------------'
# Graficar
plt.plot(wavelength, ajuste_gauss(wavelength, *poptg), 'r--')
plt.plot(wavelength, fnu, 'g^')
plt.axis([6450, 6675, 1.28e-16, 1.42e-16])
plt.xlabel('$\AA$')
plt.ylabel('$erg\ \  s^{-1}\ Hz^{-1}\ cm^{-2}$')
plt.show()


###############
# lorentz
###############

# Ajustar recta-gaussiana
poptl = obtener_parametros(ajuste_lorentz, wavelength, fnu, parametros=parametros)
# Escribir resultados
print '----------------------------------------------'
print 'f(x) = (a + b * x) - A / (pi * (1 + (x - mu)**2 / sigma**2))'
print 'a = ', poptl[0]
print 'b = ', poptl[1]
print 'A = ', poptl[2]
print 'mu = ', poptl[3]
print 'sigma = ', poptl[4]
print '----------------------------------------------'
# Graficar
plt.plot(wavelength, ajuste_lorentz(wavelength, *poptl), 'r--')
plt.plot(wavelength, fnu, 'g^')
plt.axis([6450, 6675, 1.28e-16, 1.42e-16])
plt.xlabel('$\AA$')
plt.ylabel('$erg\ \  s^{-1}\ Hz^{-1}\ cm^{-2}$')
plt.show()

###################
# gauss vs lorentz
###################

plt.plot(wavelength, ajuste_gauss(wavelength, *poptg), 'r--')
plt.plot(wavelength, ajuste_lorentz(wavelength, *poptl), 'g--')
plt.plot(wavelength, fnu, 'y^')
plt.axis([6450, 6675, 1.28e-16, 1.42e-16])
plt.xlabel('$\AA$')
plt.ylabel('$erg\ \  s^{-1}\ Hz^{-1}\ cm^{-2}$')
plt.show()

##############################################################################
##############################################################################
# P2

N = 1e+5  # numero de datos a modelar
parametros = [9.1e-17, 7.4e-21, 1e-17, 6563., 7.]  # parametros iniciales para el ajuste

# determina valores extremos de la longitud de onda
xmin = np.min(wavelength)  # minimo wavelength
xmax = np.max(wavelength)  # maximo wavelength

##########
# gauss
##########

parametros_gauss = obtener_parametros(ajuste_gauss, wavelength, fnu, parametros=parametros)

# crea y ordena datos fnu a partir del modelo
fnu_modelo_gauss = np.sort(ajuste_gauss(np.linspace(xmin, xmax, N), *parametros_gauss))
fnu_datos_gauss = np.sort(fnu)  # ordena los valores fnu
cdf_gauss = np.array([np.sum(fnu_modelo_gauss <= yy) \
					 for yy in fnu_datos_gauss]) / len(fnu_modelo_gauss)
n = len(fnu)
plt.plot(fnu_datos_gauss, np.arange(n) / n, '-^', drawstyle = 'steps-post')
plt.plot(fnu_datos_gauss, np.arange(1, n + 1) / n, '-.', drawstyle = 'steps-post')
plt.plot(fnu_datos_gauss, cdf_gauss, '-x', drawstyle = 'steps-post')
plt.show()

max1g = np.max(cdf_gauss - np.arange(n) / n)
max2g = np.max(np.arange(1,n + 1) / n - cdf_gauss)
Dng = max(max1g, max2g)
print Dng


Dng_testg, prob_scipyg = kstest(fnu_datos_gauss, cdf, args=(fnu_modelo_gauss,))

print Dng_testg
print prob_scipyg


##############################################################################
##############################################################################
