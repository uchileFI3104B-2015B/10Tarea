
# coding: utf-8

# In[ ]:

from __future__ import division
import scipy as sp
from scipy.stats import norm
from scipy.stats import cauchy
from scipy.stats import kstest
from scipy.optimize import curve_fit as cv
import matplotlib.pyplot as plt


# P1 Tarea 10


def gaussiana(x, C, mu, sigma):
    gauss = C * norm(loc=mu, scale=sigma).pdf(x)
    return gauss


def lorentz(x, C, mu, sigma):
    lorentz = C * cauchy(loc=mu, scale=sigma).pdf(x)
    return lorentz


def recta(x, A, B):
    recta = A * x + B
    return recta


def modelo1(x, A, B, C, mu, sigma):
    modelo1 = recta(x, A, B) - gaussiana(x, C, mu, sigma)
    return modelo1


def modelo2(x, A, B, C, mu, sigma):
    modelo2 = recta(x, A, B) - lorentz(x, C, mu, sigma)
    return modelo2

# Main Setup

x = sp.loadtxt('espectro.dat')[:, 0]  # long de onda
y = sp.loadtxt('espectro.dat')[:, 1]  # Flujo
sigma = sp.std(x)
mu = sp.mean(x)
A, B = sp.polyfit(x, y, 1)
C = 1 * 10**(-16)  # valor mas pequeno de y
a1, b1 = cv(modelo1, x, y, [A, B, C, mu, sigma])
a2, b2 = cv(modelo2, x, y, [A, B, C, mu, sigma])
modelo1v = modelo1(x, *a1)
modelo2v = modelo2(x, *a2)

# Grafico modelo 1

fig1 = plt.figure(1)
fig1.clf()
plt.plot(x, y, 'b-')
plt.plot(x, modelo1v, 'r-')
plt.xlabel(r'Longitud de onda [$\AA$]')
plt.ylabel(r'Flujo [$erg s^{-1} Hz^{-1}cm^{-2}$]')
plt.title('Curva y modelamiento de Espectro ' +
          'por Gaussiana')
plt.grid(True)
fig1.savefig('gauss')
plt.show()

# Grafico modelo 2

fig2 = plt.figure(2)
fig2.clf()
plt.plot(x, y, 'b-')
plt.plot(x, modelo2v, 'g-')
plt.xlabel(r'Longitud de onda [$\AA$]')
plt.ylabel(r'Flujo [$erg s^{-1} Hz^{-1}cm^{-2}$]')
plt.title('Curva y modelamiento de Espectro ' +
          'por perfil de Lorentz')
plt.grid(True)
fig2.savefig('lorentz')
plt.show()

# Grafico modelo 1 vs modelo 2

fig3 = plt.figure(3)
fig3.clf()
plt.plot(x, y, 'b-')
plt.plot(x, modelo1v, 'r-')
plt.plot(x, modelo2v, 'g-')
plt.xlim(6530, 6600)
plt.xlabel(r'Longitud de onda [$\AA$]')
plt.ylabel(r'Flujo [$erg s^{-1} Hz^{-1}cm^{-2}$]')
plt.title('Curva y modelamiento de Espectro')
plt.grid(True)
fig3.savefig('gl')
plt.show()

# Valores de constantes

print 'Parametros modelo gaussiana :'
print 'A= ', a1[0]
print 'B= ', a1[1]
print 'C= ', a1[2]
print 'mu= ', a1[3]
print 'sigma= ', a1[4]
print 'Chi cuadrado modelo gaussiana', sum((y - modelo1v)**2)
print 'Parametros modelo lorentz :'
print 'A= ', a2[0]
print 'B= ', a2[1]
print 'C= ', a2[2]
print 'mu= ', a2[3]
print 'sigma= ', a2[4]
print 'Chi cuadrado modelo lorentz', sum((y - modelo2v)**2)

# P2 Tarea 10


def CDF(datos, datosmodelo):
    modelosort = sp.sort(datosmodelo)
    datosort = sp.sort(datos)
    cdf = (sp.array([sp.sum(modelosort <= yy) for yy in datosort]) /
           (len(datosmodelo)))
    return cdf

numinter = 1000
xsort = sp.sort(x)
xmin = xsort[0]
xmax = xsort[-1]
x = sp.linspace(xmin, xmax, numinter)
modelo1k = modelo1(x, *a1)
modelo2k = modelo2(x, *a2)
CDF1 = CDF(y, modelo1k)
CDF2 = CDF(y, modelo2k)
Dn1, p1 = kstest(y, CDF, args=(sp.sort(modelo1k),))
Dn2, p2 = kstest(y, CDF, args=(sp.sort(modelo2k),))
print 'Dn modelo 1 = ', Dn1
print 'Probabilidad modelo 1 = ', p1
print 'Dn modelo 2 = ', Dn2
print 'Probabilidad modelo 2 = ', p2

# Graficos cdf

fig4 = plt.figure(4)
fig4.clf()
plt.plot(modelo1k, sp.arange(len(modelo1k)) /
         len(modelo1k), 'g*')
plt.plot(modelo1k, sp.arange(1, len(modelo1k)+1) /
         len(modelo1k), 'g*')
plt.plot(y, CDF1, 'r*')
plt.xlabel(r'Longitud de onda [$\AA$]')
plt.ylabel('Probabilidad')
plt.title('Funcion distribucion acumulada')
plt.grid(True)
fig4.savefig('cdf1')
plt.show()

# No funciono


# In[ ]:



