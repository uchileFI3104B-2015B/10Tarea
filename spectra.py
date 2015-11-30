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
ymod1_orden = np.sort(modelo1(np.linspace(xmin, xmax, 10000), act_params1))
ymod2_orden = np.sort(modelo2(np.linspace(xmin, xmax, 10000), act_params2))
y_ordenado = np.sort(data[1])
CDF_mod1 = cdf(y_ordenado, ymod1_orden)
CDF_mod2 = cdf(y_ordenado, ymod2_orden)
N = len(y_ordenado)
Dn_mod1, prob_mod1 = kstest(y_ordenado, cdf, args=(ymod1_orden,))
Dn_mod2, prob_mod2 = kstest(y_ordenado, cdf, args=(ymod2_orden,))
print "Dn modelo 1 : ", Dn_mod1
print "Nivel de confianza modelo 1 :", prob_mod1
print "Dn modelo 2 : ", Dn_mod2
print "Nivel de confianza modelo 2 :", prob_mod2


# Plots
plt.figure(1)
plt.plot(data[0], data[1]/10.0**16, 'k')
plt.plot(data[0], modelo1(data[0], act_params1)/10.0**16, 'r')
plt.figure(2)
plt.plot(data[0], data[1]/10.0**16, 'k')
plt.plot(data[0], modelo2(data[0], act_params2)/10.0**16, 'b')
plt.figure(3)
plt.plot(y_ordenado, np.arange(N) / (1.0*N), '-^', drawstyle='steps-post')
plt.plot(y_ordenado, np.arange(1, N+1) / (1.0*N), '-.', drawstyle='steps-post')
plt.plot(y_ordenado, CDF_mod1, '-x', drawstyle='steps-post')
plt.figure(4)
plt.plot(y_ordenado, np.arange(N) / (1.0*N), '-^', drawstyle='steps-post')
plt.plot(y_ordenado, np.arange(1, N+1) / (1.0*N), '-.', drawstyle='steps-post')
plt.plot(y_ordenado, CDF_mod2, '-x', drawstyle='steps-post')
plt.show()
