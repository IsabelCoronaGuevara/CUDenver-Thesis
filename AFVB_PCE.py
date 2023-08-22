#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Author: 
Date: 

Contains classes for Chaos expansions, Exponential families (containing Beta, Bernoulli, Gauss and Gamma),
Optimizer for the standard RVM and sparse RVM.
"""

__all__ = ['ChaosModel', 'AFVB_PCE']

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
from scipy.special import legendre
import math
from itertools import product
from scipy.special import gamma
import sys


class ChaosModel(object):
    """
    dim: number of variables
    order: max degree
    """

    _dim = None

    _order = None
    
    # _basis_MI needs to be a function that takes the multi-indexes 
    # and can be evaluated at the xi's
    _basis = None
    
    _basis_shape = None

    _coeffs = None
    
    # Only works for total degree (TD)
    #def __init__(self, dim, order, basis_MI, coeffs = None, trunc = 'TD', q = None):
    def __init__(self, dim, order, basis, basis_shape, coeffs = None):
        """
        Initializes the object
        """
        assert isinstance(dim, int)
        assert isinstance(order, int)
        self._dim = dim
        self._order = order
        self._basis = basis
        self._basis_shape = basis_shape

        if coeffs is None:
            self._coeffs = np.zeros(self._basis_shape[0])
        else:
            assert self._basis_shape[0] == coeffs.shape[0]
            self._coeffs = coeffs
            
    ### Evaluates the spectral representation of f (Eq. 5)
    ### Need to save coefficients first
    def eval(self, xi, active_indices = None):
        if active_indices is None:
            return np.dot(self._basis(xi), self._coeffs)
        else:
            return np.dot(self._basis(xi)[:, active_indices], self._coeffs[active_indices])


# In[3]:


class AFVB_PCE(object):
    
    _N = None
    
    _d = None
    
    _p = None
    
    _params = None
    
    _n = None
    
    
    def __init__(self, params, p, data, basis):
        """
        Initializes the object
        """
        self._params = params
        
        self._p = p
        
        self._data = data
        
        self._basis = basis
        
        self._N = self._data['Xi'].shape[0]
        
        self._d = self._data['Xi'].shape[1]
        
        self._n = int(math.factorial(self._d + self._p)/(math.factorial(self._d)*math.factorial(self._p)))
        
        self._Phi = self._basis(self._data['Xi'])
        
    def algorithm(self, T_ell, e):
        
        N = self._N
        
        for k in range(2): # Running loop twice, second time will store optimal values
            if k == 1:
                max_beta = np.argmax(ell_beta) + 1
            else:
                max_beta = 0
            
            ########################################
            #### Compute initial E_theta[Theta] ####
            ########################################
            E_Theta = np.zeros((self._n, self._n))
            
            for i in range(self._n):
                E_Theta[i,i] = self._params['C_0']/self._params['D_0']

            #### Compute A_r and C_r
            A_r = self._params['A_0'] + self._N/2
            C_r = self._params['C_0'] + 1/2

            #### List that keeps track of L_beta values and corresponding Beta value
            ell_beta = []

            ### An array made in order to keep track of which column numbers
            ### Phi_hat kept in the end
            col = np.array(range(0,self._Phi.shape[1]))

            Beta = 1
            ell_old = None
            
            Phi = self._Phi
            while Phi.shape[1] > 1: ## Beta 
                n = Phi.shape[1]

                e_ell = sys.float_info.max
                r = 1

                while e_ell > T_ell: ## r
                    ###############################
                    #### Compute chi_r and a_r ####
                    ###############################

                    #### chi_r
                    chi_r_inv = np.zeros((n, n))

                    for i in range(N):
                        chi_r_inv += np.c_[Phi[i,:]]@np.c_[Phi[i,:]].transpose()

                    chi_r_inv += E_Theta
                    chi_r = np.linalg.inv(chi_r_inv)


                    #### a_r
                    chi_r_inv_times_a_r = np.zeros((n, 1))
                    for i in range(N):
                        chi_r_inv_times_a_r += np.c_[Phi[i,:]]*self._data['Y'][i]

                    a_r = chi_r@chi_r_inv_times_a_r

                    #########################################
                    #### Update B_r, D_r, E_Theta, ell_r ####
                    #########################################

                    #### B_r
                    B_r = self._params['B_0'] + 1/2*(np.sum(self._data['Y']**2) - a_r.transpose()@chi_r_inv@a_r)

                    #### D_r
                    D_r = np.zeros((n,1))
                    for i in range(n):
                        D_r[i] = self._params['D_0'] + 1/2*(a_r[i]**2*A_r/B_r + chi_r[i,i])         

                    #### E_Theta
                    for i in range(n):
                        E_Theta[i,i] = float(C_r/D_r[i])

                    #### ell_r
                    s_2 = 0
                    for i in range(N):
                        s_2 += A_r/B_r*(self._data['Y'][i] - np.c_[Phi[i,:]].transpose()@a_r)**2 \
                        + np.c_[Phi[i,:]].transpose()@chi_r@np.c_[Phi[i,:]]

                    det_chi_r = np.linalg.det(chi_r)  # determinant of Xr
                    if det_chi_r == 0.0:
                        det_chi_r = np.finfo(np.float64).tiny  
                           
                    ell_r = -N/2*math.log(2*math.pi) \
                        - 1/2*s_2 \
                        + math.log(gamma(A_r)) \
                        + A_r*(1 - math.log(B_r) - self._params['B_0']/B_r) \
                        - math.log(gamma(self._params['A_0'])) \
                        + self._params['A_0']*math.log(self._params['B_0']) \
                        - np.sum(C_r*np.log(D_r)) \
                        + n*(1/2 - math.log(gamma(self._params['C_0'])) + self._params['C_0']*math.log(self._params['D_0']) + math.log(gamma(C_r))) \
                        + 1/2*math.log(det_chi_r)

                    
                    ##############################
                    ####    Compute e_ell     ####
                    ##############################
                    if ell_old is None:
                        ell_old = ell_r
                    else:
                        e_ell = np.abs(100*(float(ell_r) - ell_old)/ell_old)
                        ell_old = float(ell_r)
                        
                    r += 1   

                #### Save ell_r value in an array
                ell_beta.append(float(ell_r))

                ########################################################################################
                ############################### Save the optimal vectors ###############################
                ########################################################################################
                if Beta == 1:
                    Phi_full = self._Phi
                    a_full = a_r
                if Beta == max_beta:
                    Phi_hat = Phi
                    a_hat = a_r
                if k == 1:    
                    print(Beta, Phi.shape, float(ell_r))
                ########################################################################################
                ########################################################################################

                #### Compute lambda_beta ####
                lambda_beta = np.diag(np.linalg.inv(E_Theta))

                #### Compute ln_T_beta ####
                ln_T_beta = min(np.log(lambda_beta)) + (max(np.log(lambda_beta)) - min(np.log(lambda_beta)))/e

                ########################################################################################
                ############################# Update columns we are keeping ############################
                ########################################################################################
                if Beta < max_beta:
                    col = col[np.log(lambda_beta) >= ln_T_beta]
                ########################################################################################
                ########################################################################################

                #### Prune Phi and E_Theta
                Phi = Phi[:,np.log(lambda_beta) >= ln_T_beta]
                E_Theta = E_Theta[:, np.log(lambda_beta) >= ln_T_beta]
                E_Theta = E_Theta[np.log(lambda_beta) >= ln_T_beta,:]

                Beta += 1

        return Phi_hat, a_hat, Phi_full, a_full, col
    


# In[28]:


#### Need class that computes S_1, ... 


# In[ ]:




