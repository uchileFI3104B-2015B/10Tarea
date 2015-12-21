


'''modelando con dos mecanismos distintos, gauss y lorentz'''

from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import scipy.stats


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

def chi(c,x,y,rest):
    X= np.sum((y-rest(c,x))**2)
    return X

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
c=1*(10**(-16)),6550,1,1*(10**(-16)),0  #segun grafico
parametros=c[0],c[1],c[2],c[3],c[4]
aprox_gauss=leastsq(res_gauss, parametros, args=(w,f))
aprox_lorentz=leastsq(res_lorentz, parametros, args=(w,f))

ga= aprox_gauss[0]
lo= aprox_lorentz[0]

gy= resta_gauss(ga,w)
ly= resta_lorentz(lo,w)

fig= plt.figure()
plt.plot(w,f,'+',label="datos")
plt.plot(w,gy,'-', label= "ajuste gauss")
plt.plot(w,ly,'-',label="ajuste lorentz")
plt.xlabel("w [A]")
plt.ylabel("F_nu [erg/(s Hz cm^2)]")
plt.title('Ajustes para el espectro')
plt.legend()
plt.savefig('1')
plt.show()

GA,Gmu,Gsigma,Ga,Gb= gauss_valores
LA,Lmu,Lsigma,La,Lb= lorentz_valores

chig=chi(ga,w,f,resta_gauss)
chil=chi(lo,w,f,resta_lorentz)

print ("Gauss")
print (gauss_valores)


print ("Lorentz")
print(lorentz_valores)

print ("Chi^2 Gauss")
print(chig)
print ("Chi^2 Lorentz")
print(chil)
