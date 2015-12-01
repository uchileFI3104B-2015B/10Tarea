# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (leastsq, curve_fit)
from scipy import optimize as opt
import os

'''
Este script modela 5 parametros a la vez a partir de los datos de
espectro.dat, al pltear se observa un segmento del espectro de una fuente
que muesta continuo (con una leve pendiente) y una linea de absorcion, se modela
el primero con una linea recta (2parametros) y el segundo con una gaussiana
o con un perfil de lorentz
'''
