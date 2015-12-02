#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import cauchy
from scipy.stats import kstest


def modelo1(x, params):
    A, mu, sigma, a, b = params
    return -A * norm(loc=mu, scale=sigma).pdf(x) + a*x + b


def modelo2(x, params):
    A, mu, sigma, a, b = params
    return -A * cauchy(loc=mu, scale=sigma).pdf(x) + a*x + b


def deriv_modelo(x, params, indice_der, modelo):
    epsilon = 10**(-7)
    dpar = np.zeros(len(params))
    dpar[indice_der] = epsilon
    return (modelo(x, params + dpar) - modelo(x, params - dpar))/(2*epsilon)


def chi2(data, modelo, params):
    return np.sum(np.power((data[1] - modelo(data[0], params)), 2))


def beta(data, modelo, params):
    betak = np.zeros(5)
    f1 = data[1] - modelo(data[0], params)
    for k in range(5):
        f2 = deriv_modelo(data[0], params, k, modelo)
        betak[k] = np.sum(f1*f2)
    return betak


def alpha(data, modelo, params, lamb):
    derivadas = np.zeros((5, len(data[0])))
    for k in range(5):
        derivadas[k] = deriv_modelo(data[0], params, k, modelo)
    alphakl = np.zeros((5, 5))
    for k in range(5):
        for l in range(5):
            alphakl[k][l] = np.sum(derivadas[k]*derivadas[l])
    for j in range(5):
        alphakl[j][j] = alphakl[j][j] * (1 + lamb)
    return alphakl


def cdf(datos, modelo):
    return np.array([np.sum(modelo <= yy) for yy in datos]) / (1.0*len(modelo))


wavelen = np.loadtxt("espectro.dat", usecols=(0,))
frec = np.loadtxt("espectro.dat", usecols=(1,))
data = np.zeros((2, len(wavelen)))
data[0] = wavelen
data[1] = (10**16)*frec
coefin = np.polyfit(data[0], data[1], 1)
act_params1 = (1, 6565, 5, coefin[0], coefin[1])
act_params2 = (1, 6565, 5, coefin[0], coefin[1])
lamb1 = 0.001
lamb2 = 0.001
chi21 = chi2(data, modelo1, act_params1)
chi22 = chi2(data, modelo2, act_params2)
deltachi1 = 10
deltachi2 = 10

# Modelo 1: recta más gaussiana
while deltachi1 > 0.00001:
    alp = alpha(data, modelo1, act_params1, lamb1)
    bet = beta(data, modelo1, act_params1)
    delparams = np.dot(np.linalg.inv(alp), bet)
    chi2t = chi2(data, modelo1, act_params1 + delparams)
    if chi2t >= chi21:
        lamb1 = lamb1 * 10
    else:
        lamb1 = lamb1 * 0.1
        act_params1 = act_params1 + delparams
        deltachi1 = chi21 - chi2t
        chi21 = chi2t

# Modelo 2: recta más perfil de Lorentz
while deltachi2 > 0.00001:
    alp = alpha(data, modelo2, act_params2, lamb2)
    bet = beta(data, modelo2, act_params2)
    delparams = np.dot(np.linalg.inv(alp), bet)
    chi2t = chi2(data, modelo2, act_params2 + delparams)
    if chi2t >= chi22:
        lamb2 = lamb2 * 10
    else:
        lamb2 = lamb2 * 0.1
        act_params2 = act_params2 + delparams
        deltachi2 = chi22 - chi2t
        chi22 = chi2t

# Test de Kolmogorov-Smirnov
xmin = np.min(data[0])
xmax = np.max(data[0])
xrang = np.linspace(xmin, xmax, 10000)
ymod1_orden = np.sort(modelo1(xrang, act_params1)) / 10.0**16
ymod2_orden = np.sort(modelo2(xrang, act_params2)) / 10.0**16
y_ordenado = np.sort(data[1]) / 10.0**16
CDF_mod1 = cdf(y_ordenado, ymod1_orden)
CDF_mod2 = cdf(y_ordenado, ymod2_orden)
N = len(y_ordenado)
Dn_mod1, prob_mod1 = kstest(y_ordenado, cdf, args=(ymod1_orden,))
Dn_mod2, prob_mod2 = kstest(y_ordenado, cdf, args=(ymod2_orden,))

# Resultados
print ""
print "Modelo Gaussiano:"
print "chi2 =", chi21, "10^-32 [erg^2 s^-2 Hz^-2 cm^-4]"
print "A =", act_params1[0], "10^-16 [erg s^-1 Hz^-1 cm^-2 Angstrom]"
print "mu =", act_params1[1], "Angstrom"
print "sigma =", act_params1[2], "Angstrom"
print "a =", act_params1[3], "10^-16 [erg s^-1 Hz^-1 cm^-2 Angstrom^-1]"
print "b =", act_params1[4], "10^-16 [erg s^-1 Hz^-1 cm^-2]"
print "Dn =", Dn_mod1
print "Nivel de confianza:", prob_mod1

print ""
print "Modelo Lorentziano:"
print "chi2 =", chi22, "10^-32 [erg^2 s^-1 Hz^-2 cm^-4]"
print "A =", act_params2[0], "10^-16 [erg s^-1 Hz^-1 cm^-2 Angstrom]"
print "mu =", act_params2[1], "Angstrom"
print "sigma =", act_params2[2], "Angstrom"
print "a =", act_params2[3], "10^-16 [erg s^-1 Hz^-1 cm^-2 Angstrom^-1]"
print "b =", act_params2[4], "10^-16 [erg s^-1 Hz^-1 cm^-2]"
print "Dn =", Dn_mod2
print "Nivel de confianza:", prob_mod2


# Plots
plt.clf()
plt.figure(1)
plt.plot(data[0], data[1]/10.0**16, 'k--', label='Espectro observado')
plt.plot(xrang, modelo1(xrang, act_params1)/10.0**16, 'r',
         label='Perfil gaussiano sobre recta')
plt.plot(xrang, modelo2(xrang, act_params2)/10.0**16, 'b',
         label='Perfil lorentziano sobre recta')
plt.xlabel('Longitud de onda [$\AA$]')
plt.ylabel('$F_{\\nu} \, [erg \, s^{-1}Hz^{-1}cm^{-2}]$')
plt.title('Mejores fits a espectro observado')
plt.legend(loc=4)
plt.xlim([6450, 6750])
plt.savefig('fits.eps')
plt.figure(2)
plt.plot(y_ordenado, np.arange(N) / (1.0*N), '-^', drawstyle='steps-post',
         label='CDF observada inf')
plt.plot(y_ordenado, np.arange(1, N+1) / (1.0*N), '-.', drawstyle='steps-post',
         label='CDF observada sup')
plt.plot(y_ordenado, CDF_mod1, '-x', drawstyle='steps-post',
         label='CDF modelo gaussiano')
plt.xlabel('$F_{\\nu} \, [erg \, s^{-1}Hz^{-1}cm^{-2}]$')
plt.title('Funci'u'ó''n de distribuci'u'ó''n acumulada')
plt.legend(loc=2)
plt.savefig('ksgauss.eps')
plt.figure(3)
plt.plot(y_ordenado, np.arange(N) / (1.0*N), '-^', drawstyle='steps-post',
         label='CDF observada inf')
plt.plot(y_ordenado, np.arange(1, N+1) / (1.0*N), '-.', drawstyle='steps-post',
         label='CDF observada sup')
plt.plot(y_ordenado, CDF_mod2, '-x', drawstyle='steps-post',
         label='CDF modelo lorentziano')
plt.xlabel('$F_{\\nu} \, [erg \, s^{-1}Hz^{-1}cm^{-2}]$')
plt.title('Funci'u'ó''n de distribuci'u'ó''n acumulada')
plt.legend(loc=2)
plt.savefig('kslorentz.eps')
plt.show()
