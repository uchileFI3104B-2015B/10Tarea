from __future__ import division
from math import *
#from scipy import integrate as int
import scipy.stats
from scipy.optimize import (leastsq, curve_fit)
from scipy.integrate import ode
import pyfits #modulo para leer archivos fits
import matplotlib.pyplot as p #modulo para graficar
import numpy as np #este modulo es para trabajar con matrices como en matlab
import matplotlib as mp
from mpl_toolkits.mplot3d import Axes3D
import random

def f(x,n,m,B,mu,sigma):
    return n+m*x-B*scipy.stats.norm(loc=mu, scale=sigma).pdf(x)
    
def g(x,n,m,B,mu,sigma):
    return n+m*x-B*scipy.stats.cauchy(loc=mu, scale=sigma).pdf(x)


datos=np.loadtxt('espectro.dat')

Lambda=datos[:,0]
flujo=datos[:,1]

N=np.sum(flujo)

a0=10**(-18),10**(-20),0.001,6562.3,3.21
par,cov=curve_fit(f,Lambda,flujo,a0)
par2,cov2=curve_fit(g,Lambda,flujo,a0)

n=par2[0]
m=par2[1]
B=par2[2]
mu=par2[3]
sigma=par2[4]


chi=scipy.stats.chisquare(flujo/N,g(Lambda,n,m,B,mu,sigma)/N)
#chi=scipy.stats.chisquare(flujo,Lambda)
print chi

x=np.linspace(6450.0,6670.0,1001)


   


p.plot(Lambda,flujo)#, label='datos tomados')
p.plot(x,g(x,n,m,B,mu,sigma), label='curva de ajuste')
p.xlabel('Longitud de onda [A]')
p.ylabel('Irradiancia espectral [erg/s/Hz/cm^2')
p.title('Espectro')
p.legend(loc='upper right')
p.show()
