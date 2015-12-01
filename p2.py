from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import scipy.stats
from scipy.stats import kstest


def leer(algo):
    '''lee el archivo'''
    data= np.loadtxt(algo)
    w=data[:,0] #wavelength Angstrom x
    f=data[:,1] #frecuencia erg/(s Hz cm^2) y
    return w,f

#ingredientes
def recta(a,b,x):
    yy= a*x +b
    return yy

def gauss(A,mu,sigma,x):
    A=A
    mu=mu
    sigma=sigma
    x=x
    g= A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    return g

def lorentz(A,mu,sigma,x):
    A=A
    mu=mu
    sigma=sigma
    x=x
    l= A * scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)
    return l

def resta_gauss(c,x):
    '''modelo de recta menos gaussiana'''
    A,mu,sigma,a,b= c
    x=x
    g=gauss(A,mu,sigma,x)
    yy=recta(a,b,x)
    return yy-g

def resta_lorentz(c,x):
    '''modelo de recta menos lorentz'''
    A,mu,sigma,a,b= c
    x=x
    l=lorentz(A,mu,sigma,x)
    yy=recta(a,b,x)
    return yy-l

def cdf(data, modelo):
    return np.array([np.sum(modelo <= yy) for yy in data]) / len(modelo)

def test (modelo,c,x,y):
    xmin = np.min(x)
    xmax = np.max(x)
    y_modelo_sorted = np.sort(modelo(c, np.linspace(xmin, xmax, 1000)))
    y_data_sorted = np.sort(y)
    Dn, proba = kstest(y_data_sorted, cdf, args=(y_modelo_sorted,))

#definimos las funciones a minimizar

def res_gauss(c,x,y):
    rg= y - resta_gauss(c,x)
    return rg

def res_lorentz(c,x,y):
    rl= y - resta_lorentz(c,x)
    return rl

#para correr
aa=leer("espectro.dat")
w,f= aa
c=1*(10**(-16)),6550,1,1*(10**(-16)),0  #segun grafico A mu sigma a b
parametros=c[0],c[1],c[2],c[3],c[4]
aprox_gauss=leastsq(res_gauss, parametros, args=(w,f))
aprox_lorentz=leastsq(res_lorentz, parametros, args=(w,f))

ga= aprox_gauss[0]
lo= aprox_lorentz[0]

print test(resta_gauss,c,w,f)
print test(resta_lorentz,c,w,f)
