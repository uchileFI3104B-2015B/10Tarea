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

def modelo_gauss(x,a,b,A,mu,sigma):
    return a + b*x - A*scipy.stats.norm(loc=mu, scale=sigma).pdf(x)

def modelo_Lorentz(x,a,b,A,mu,sigma):
    return a + b*x - A*scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)

# Archivo
datos = np.loadtxt("espectro.dat")
flujo = datos[:,1]
londa = datos[:,0]

# Vector estimacion de parametros segun grafico de datos
a0 = [3, (0.04e-16)/200, 0.1e-16, 6550, 10]

# Se ajustan los datos con curve-fit para cada modelo
popt, pcov = curve_fit(modelo_gauss,londa,flujo,a0)
popt2, pcov2 = curve_fit(modelo_Lorentz,londa,flujo,a0)

# test de chicuadrado
print 'chi gauss', chi(modelo_gauss,londa,flujo,popt),'[erg2/s2/Hz2/cm^4]'
print 'chi lorentz', chi(modelo_Lorentz,londa,flujo,popt2),'[erg2/s2/Hz2/cm^4]'

# Grafico de modelo gauss mas datos
plt.figure(1)
plt.plot(londa, modelo_gauss(londa, *popt), 'r--',label='Curva Modelo gaussiano')
plt.plot(londa, flujo, 'b^',label='Datos')
plt.axis([6450, 6675, 1.25e-16, 1.44e-16])
plt.xlabel('Longitud de Onda [Angstrom = 10^-10 m]')
plt.ylabel("$F_v$ [erg/s/Hz/cm^2] ")
plt.title('Grafico Modelo gaussiano + Datos')
plt.legend(loc='best',fontsize=8.3)
plt.savefig('fig1.png')
print '                           '
print 'Parametros para Modelo gaussiano:'
print 'a = ',popt[0] ,'[erg/s/Hz/cm^2]'
print 'b = ',popt[1] ,'[erg/s/Hz/cm^2/Angstrom]'
print 'A = ',popt[2] ,'[erg/s/Hz/cm^2]*[Angstrom]'
print 'Mu = ',popt[3] ,'[Angstrom]'
print 'Sigma = ',popt[4] ,'[Angstrom]'

# Grafico de modelo de Lorentz mas datos
plt.figure(2)
plt.plot(londa, modelo_Lorentz(londa, *popt2), 'g--',label='Curva Modelo lorentz')
plt.plot(londa, flujo, 'b^',label='Datos')
plt.axis([6450, 6675, 1.25e-16, 1.44e-16])
plt.xlabel('Longitud de Onda [Angstrom = 10^-10 m]')
plt.ylabel("$F_v$ [erg/s/Hz/cm^2] ")
plt.title('Grafico Modelo de lorentz + Datos')
plt.legend(loc='best',fontsize=8.3)
plt.savefig('fig2.png')
print '                         '
print 'Parametros para Modelo lorentz:'
print 'a = ',popt2[0] ,'[erg/s/Hz/cm^2]'
print 'b = ',popt2[1] ,'[erg/s/Hz/cm^2/Angstrom]'
print 'A = ',popt2[2] ,'[erg/s/Hz/cm^2]*[Angstrom]'
print 'Mu = ',popt2[3] , '[Angstrom]'
print 'Sigma = ',popt2[4] ,'[Angstrom]'
print '                         '

# Grafico de ambos modelos mas los datos de espectro.dat
plt.figure(3)
plt.plot(londa, modelo_gauss(londa, *popt), 'r--',label='Curva Modelo gaussiano')
plt.plot(londa, modelo_Lorentz(londa, *popt2), 'g--',label='Curva Modelo lorentz')
plt.plot(londa, flujo, 'b^',label='Datos')
plt.axis([6450, 6675, 1.25e-16, 1.44e-16])
plt.xlabel('Longitud de Onda [Angstrom = 10^-10 m]')
plt.ylabel("$F_v$ [erg/s/Hz/cm^2] ")
plt.title('Grafico comparativo de los Modelos')
plt.legend(loc='best',fontsize=8.3)
plt.savefig('fig3.png')

''' Ocupamos K-S para saber si los fits son buenos'''

# Crea datos
x = np.linspace(6.4600e+03, 6.6596e+03, 122)

# El maximo y el minimo se obtiene de los datos
xmin = np.min(londa)
xmax = np.max(londa)

# Define arreglos para los modelos segun datos
y_model_sorted_gauss = np.sort(modelo_gauss(np.linspace(xmin, xmax, 122), *popt))
y_model_sorted_Lorentz = np.sort(modelo_Lorentz(np.linspace(xmin, xmax, 122), *popt2))

# Arreglo de los datos
y_data_sorted = np.sort(flujo)
N = len(y_data_sorted)

# Funcion distribucion acumulada de cada modelo
CDF_model_gauss = np.array([np.sum(y_model_sorted_gauss <= yy) for yy in y_data_sorted]) / len(y_model_sorted_gauss)
CDF_model_Lorentz = np.array([np.sum(y_model_sorted_Lorentz <= yy) for yy in y_data_sorted]) / len(y_model_sorted_Lorentz)

# Se definen los maximos de distancia entre las CDF
max1_gauss = np.max(CDF_model_gauss - np.arange(N) / N)
max2_gauss = np.max( np.arange(1,N+1)/N - CDF_model_gauss)

max1_Lorentz = np.max(CDF_model_Lorentz - np.arange(N) / N)
max2_Lorentz = np.max(np.arange(1,N+1)/N - CDF_model_Lorentz)

Dn_gauss = max(max1_gauss, max2_gauss)
Dn_Lorentz = max(max1_Lorentz, max2_Lorentz)

# Muestra en pantalla Dn obtenido para cada modelo
print "Dn para nuestro fit con gauss = ", Dn_gauss
print '                                  '
print "Dn para nuestro fit con Lorentz = ",Dn_Lorentz


# Calcula el Dn critico dado alfa 0.05
# Aproximacion para N grande
ks_dist = kstwobign()
alpha = 0.05
Dn_critico = ks_dist.ppf(1 - alpha) / np.sqrt(N)
print '                                '
print "Dn_critico = ", Dn_critico
print '                                '
print "Nivel de confianza que Ho sea correcta(datos consistentes con modelo gauss):", 1-ks_dist.cdf(Dn_gauss * np.sqrt(N))
print '                                '
print "Nivel de confianza que Ho sea correcta(datos consistentes con modelo Lorentz):", 1- ks_dist.cdf(Dn_Lorentz * np.sqrt(N))

# Forma 2
# Version scipy Dn y nivel de confianza
def cdf(data, model):
    return np.array([np.sum(model <= yy) for yy in data]) / len(model)

Dn_scipy_gauss, prob_scipy_gauss = kstest(y_data_sorted, cdf, args=(y_model_sorted_gauss,))
Dn_scipy_Lorentz, prob_scipy_Lorentz = kstest(y_data_sorted, cdf, args=(y_model_sorted_Lorentz,))
print '                                '
print " Dn segun scipy para gauss es: ",Dn_scipy_gauss
print '                                '
print " Dn segun scipy para Lorentz es: ",Dn_scipy_Lorentz
print '                                '
print 'Nivel de confianza para gauss segun scipy: ',prob_scipy_gauss
print '                                '
print 'Nivel de confianza para Lorentz segun scipy: ',prob_scipy_Lorentz


# Graficos distribucion de ambos modelos
plt.figure(4)
plt.title(' Distribucion acumulada Modelo gauss v/s Datos ')
plt.plot(y_data_sorted, np.arange(N) / N, '-^', drawstyle='steps-post',label='CDF Datos extremo 1 ')
plt.plot(y_data_sorted, np.arange(1, N+1) / N, '-.', drawstyle='steps-post',label='CDF Datos extremo 2')
plt.plot(y_data_sorted, CDF_model_gauss, '-x', drawstyle='steps-post',label='CDF Modelo gauss')
plt.xlabel('Datos de flujo (ordenados) [erg/s/Hz/cm^2]')
plt.ylabel('Distribucion acumulada CDF')
plt.legend(loc='best',fontsize=8.3)
plt.savefig('fig4.png')

plt.figure(5)
plt.title(' Distribucion acumulada Modelo de Lorentz v/s Datos ')
plt.plot(y_data_sorted, np.arange(N) / N, '-^', drawstyle='steps-post',label='CDF Datos extremo 1')
plt.plot(y_data_sorted, np.arange(1, N+1) / N, '-.', drawstyle='steps-post',label='CDF datos extremo 2')
plt.plot(y_data_sorted, CDF_model_Lorentz, '-x', drawstyle='steps-post',label='Distribucion acumulada Modelo de Lorentz')
plt.xlabel('Datos de flujo (ordenados) [erg/s/Hz/cm^2]')
plt.ylabel('Distribucion acumulada CDF')
plt.legend(loc='best',fontsize=8.3)
plt.savefig('fig5.png')


plt.show()
