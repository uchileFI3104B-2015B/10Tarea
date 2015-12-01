#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' este codigo modelará la funcion del espectro
a traves de una gaussiana y de lorentz'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import kstest
from scipy.stats import kstwobign


datos = np.loadtxt('espectro.dat')
print datos
x=datos[:,0]
y=datos[:,1]
plt.plot(x,y)
plt.show()


def linea(x,a,b):
    '''funcion que modela la linea recta'''

    return a+ b*x

def gaussiana(A,mu,sigma,x):
    '''funcion que modela la parte gaussiana'''

    return  A * scipy.stats.norm(loc=mu, scale=sigma).pdf(x)

def espectro():
    '''funcion que se le aplicará el fiting'''
    
    return linea(x,a,b) - gaussiana(A,mu,sigma,x)
