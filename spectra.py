#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import cauchy

def modelo1(x, A, mu, sigma, a, b):
    return A * norm(loc=mu, scale=sigma).pdf(x) + a*x + b

def modelo2(x, A, mu, sigma, a, b):
    return A * cauchy(loc=mu, scale=sigma).pdf(x) + a*x + b
