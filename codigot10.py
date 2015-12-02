from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
from scipy.stats import kstwobign
from scipy.stats import kstest

'''Este script ocupa el archivo espectro.dat para modelar los datos
de espectroscopia de una fuente radiante mediante 2 modelos '''

# Funciones principales


def chi(func, x, y, parametros):
    chi2 = 0
    for i in range(len(x)):
        chi2 += (y[i] - func(x[i], *parametros))**2
    return chi2


def modelo_gauss(x, a, b, A, mu, sigma):
    return a + b*x - A*scipy.stats.norm(loc=mu, scale=sigma).pdf(x)


def modelo_Lorentz(x, a, b, A, mu, sigma):
    return a + b*x - A*scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)

# Archivo
datos = np.loadtxt("espectro.dat")
flujo = datos[:, 1]
londa = datos[:, 0]

# Vector estimacion de parametros segun grafico de datos
# Otros valores que se ajustaban: mu=6563,a=0 o 1, sigma=7,etc.
a0 = [3, (0.04e-16)/200, 0.1e-16, 6550, 10]

# Se ajustan los datos con curve-fit para cada modelo
popt, pcov = curve_fit(modelo_gauss, londa, flujo, a0)
popt2, pcov2 = curve_fit(modelo_Lorentz, londa, flujo, a0)

# test de chicuadrado
print 'chi gauss', chi(modelo_gauss, londa, flujo, popt)
print '[erg2/s2/Hz2/cm^4]'

print 'chi lorentz', chi(modelo_Lorentz, londa, flujo, popt2)
print '[erg2/s2/Hz2/cm^4]'

# Grafico de modelo gauss mas datos
plt.figure(1)
plt.plot(londa, modelo_gauss(londa, *popt), 'r--', label='Curva Modelo gauss')
plt.plot(londa, flujo, 'b^', label='Datos')
plt.axis([6450, 6675, 1.25e-16, 1.44e-16])
plt.xlabel('Longitud de Onda [Angstrom = 10^-10 m]')
plt.ylabel("$F_v$ [erg/s/Hz/cm^2] ")
plt.title('Grafico Modelo gaussiano + Datos')
plt.legend(loc='best', fontsize=8.3)
plt.savefig('fig1.png')
print '                           '
print 'Parametros para Modelo gaussiano:'
print 'a = ', popt[0], '[erg/s/Hz/cm^2]'
print 'b = ', popt[1], '[erg/s/Hz/cm^2/Angstrom]'
print 'A = ', popt[2], '[erg/s/Hz/cm^2]*[Angstrom]'
print 'Mu = ', popt[3], '[Angstrom]'
print 'Sigma = ', popt[4], '[Angstrom]'

# Grafico de modelo de Lorentz mas datos
plt.figure(2)
plt.plot(londa, modelo_Lorentz(londa, *popt2), 'g--', label='Modelo lorentz')
plt.plot(londa, flujo, 'b^', label='Datos')
plt.axis([6450, 6675, 1.25e-16, 1.44e-16])
plt.xlabel('Longitud de Onda [Angstrom = 10^-10 m]')
plt.ylabel("$F_v$ [erg/s/Hz/cm^2] ")
plt.title('Grafico Modelo de lorentz + Datos')
plt.legend(loc='best', fontsize=8.3)
plt.savefig('fig2.png')
print '                         '
print 'Parametros para Modelo lorentz:'
print 'a = ', popt2[0], '[erg/s/Hz/cm^2]'
print 'b = ', popt2[1], '[erg/s/Hz/cm^2/Angstrom]'
print 'A = ', popt2[2], '[erg/s/Hz/cm^2]*[Angstrom]'
print 'Mu = ', popt2[3], '[Angstrom]'
print 'Sigma = ', popt2[4], '[Angstrom]'
print '                         '

# Grafico de ambos modelos mas los datos de espectro.dat
plt.figure(3)
plt.plot(londa, modelo_gauss(londa, *popt), 'r--', label='Modelo gauss')
plt.plot(londa, modelo_Lorentz(londa, *popt2), 'g--', label='Modelo lorentz')
plt.plot(londa, flujo, 'b^', label='Datos')
plt.axis([6450, 6675, 1.25e-16, 1.44e-16])
plt.xlabel('Longitud de Onda [Angstrom = 10^-10 m]')
plt.ylabel("$F_v$ [erg/s/Hz/cm^2] ")
plt.title('Grafico comparativo de los Modelos')
plt.legend(loc='best', fontsize=8.3)
plt.savefig('fig3.png')

''' Ocupamos K-S para saber si los fits son buenos'''

# Crea datos
x = np.linspace(6.4600e+03, 6.6596e+03, 122)

# El maximo y el minimo se obtiene de los datos
xmin = np.min(londa)
xmax = np.max(londa)

arreglo = np.linspace(xmin, xmax, 122)

# Define arreglos para los modelos segun datos
y_m_sort_gauss = np.sort(modelo_gauss(arreglo, *popt))

y_m_sort_Lorentz = np.sort(modelo_Lorentz(arreglo, *popt2))

# Arreglo de los datos
y_d_sort = np.sort(flujo)
N = len(y_d_sort)

# Funcion distribucion acumulada de cada modelo
CDF_model_gauss = np.array([np.sum(y_m_sort_gauss <= yy) for yy in y_d_sort])
gaussnormalizada = CDF_model_gauss / len(y_m_sort_gauss)
CDF_model_Lor = np.array([np.sum(y_m_sort_Lorentz <= yy) for yy in y_d_sort])
Lorentznormalizada = CDF_model_Lor / len(y_m_sort_Lorentz)

# Se definen los maximos de distancia entre las CDF
max1_gauss = np.max(gaussnormalizada - np.arange(N) / N)
max2_gauss = np.max(np.arange(1, N+1)/N - gaussnormalizada)

max1_Lorentz = np.max(Lorentznormalizada - np.arange(N) / N)
max2_Lorentz = np.max(np.arange(1, N+1)/N - Lorentznormalizada)

Dn_gauss = max(max1_gauss, max2_gauss)
Dn_Lorentz = max(max1_Lorentz, max2_Lorentz)

# Muestra en pantalla Dn obtenido para cada modelo
print "Dn para nuestro fit con gauss = ", Dn_gauss
print '                                  '
print "Dn para nuestro fit con Lorentz = ", Dn_Lorentz


# Calcula el Dn critico dado alfa 0.05 con scipy
# Aproximacion para N grande
ks_dist = kstwobign()
alpha = 0.05
Dn_critico = ks_dist.ppf(1 - alpha) / np.sqrt(N)
print '                                '
print "Dn_critico(kstwobign) = ", Dn_critico
print '                                '
print "Nivel confianza Ho(kstwobign):", 1 - ks_dist.cdf(Dn_gauss * np.sqrt(N))
print '                                '
print "Nivelconfianza Ho(kstwobign):", 1 - ks_dist.cdf(Dn_Lorentz * np.sqrt(N))

# Forma 2
# Version scipy Dn y nivel de confianza


def cdf(data, model):
    return np.array([np.sum(model <= yy) for yy in data]) / len(model)

Dn_scipy_gauss, pro_scipy_gauss = kstest(y_d_sort, cdf, args=(y_m_sort_gauss,))
Dn_scipy_Lor, pro_scipy_Lor = kstest(y_d_sort, cdf, args=(y_m_sort_Lorentz,))
print '                                '
print " Dn segun scipy(kstest) para gauss es: ", Dn_scipy_gauss
print '                                '
print " Dn segun scipy(kstest) para Lorentz es: ", Dn_scipy_Lor
print '                                '
print 'Nivel de confianza para gauss segun scipy(kstest): ', pro_scipy_gauss
print '                                '
print 'Nivel de confianza para Lorentz segun scipy(kstest): ', pro_scipy_Lor


# Graficos distribucion de ambos modelos
a = np.arange(1, N+1) / N

plt.figure(4)
plt.title(' Distribucion acumulada Modelo gauss v/s Datos ')
plt.plot(y_d_sort, np.arange(N) / N, '-^', drawstyle='steps-post', label='F1')
plt.plot(y_d_sort, a, '-.', drawstyle='steps-post', label='F2')
plt.plot(y_d_sort, gaussnormalizada, '-x', drawstyle='steps-post', label='Fg')
plt.xlabel('Datos de flujo (ordenados) [erg/s/Hz/cm^2]')
plt.ylabel('Distribucion acumulada CDF')
plt.legend(loc='best', fontsize=8.3)
plt.savefig('fig4.png')

plt.figure(5)
plt.title(' Distribucion acumulada Modelo de Lorentz v/s Datos ')
plt.plot(y_d_sort, np.arange(N) / N, '-^', drawstyle='steps-post', label='F1')
plt.plot(y_d_sort, a, '-.', drawstyle='steps-post', label='F2')
plt.plot(y_d_sort, Lorentznormalizada, '-x', drawstyle='steps-post', label='L')
plt.xlabel('Datos de flujo (ordenados) [erg/s/Hz/cm^2]')
plt.ylabel('Distribucion acumulada CDF')
plt.legend(loc='best', fontsize=8.3)
plt.savefig('fig5.png')


plt.show()
