"""
The module contains definitions for 

__author__ = 'Isabel Corona Guevara'

"""

import numpy as np
import math


def likelihood_fun(a, k, x, y, Psi, active_cols):
    '''
    Psi: Polynomial basis function that can be evaluated at values of x
    '''
    
    Psi_x = Psi(x)
    
    # In the case that a comes from the sparse model, we fill a_full with zeros so that it matches
    # the dimension of Psi
    a_full = np.zeros(Psi_x.shape[1])
    a_full[active_cols] = a.flatten()
    val = (k/(2*np.pi))**(1/2)*np.exp(-k/2*(y-Psi_x@a_full)**2)
    
    return float(val)


def BME(x, y, a, k, Psi, active_cols, N_mc, rank, nprocs):

    Psi_x = Psi(x) 
    
    val = 0
    for j in range(rank, N_mc, nprocs):
        a_full = np.zeros(Psi_x.shape[1])
        a_full[active_cols] = a[j].flatten()
        
        val += (k[j]/(2*np.pi))**(1/2)*np.exp(-k[j]/2*(y-Psi_x@a_full)**2)
    val /= N_mc
    
    return float(val)