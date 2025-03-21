"""
The module contains definitions for test functions

__author__ = 'Isabel Corona Guevara'

"""

import numpy as np
import math
from itertools import product
from scipy.special import gamma
import sys
from sklearn.base import BaseEstimator
from scipy.special import legendre
from scipy.special import hermitenorm
from scipy.special import norm
from aPCE import *
from basis_functions import *

class basis(object):
    
    
    def __init__(self, d, p, domain = None, aPCE_model = None, P = None, sigma_vals = None, mu_vals = None, l_bound = None, u_bound = None):
        """
        Initializes the object
        """
        
        self.d = d
        self.p = p
        self.domain = domain
        self.aPCE_model = aPCE_model
        self.P = P
        self.sigma_vals = sigma_vals
        self.mu_vals = mu_vals
        self.l_bound = l_bound
        self.u_bound = u_bound
        
        
    def multivariate_pce_index(d, max_deg):
        """
        Generate all the d-dimensional polynomial indices with the 
        constraint that the sum of the indexes is <= max_deg

        input:
        d: int, number of random variables
        max_deg: int, the max degree allowed

        return: 
        2d array with shape[1] equal to d, the multivariate indices
        """
        indices = []
    
        # Iterate over all possible total degrees (0 to max_deg)
        for total_degree in range(max_deg + 1):
        
            # Generate all weak compositions of total_degree into d parts
            for comb in combinations_with_replacement(range(d), total_degree):
            
                index = np.zeros(d, dtype=int)
            
                for i in comb:
                
                    index[i] += 1
                
                indices.append(index)

        # Convert to numpy array and sort in lexicographic order
        return np.array(indices)[np.lexsort(np.array(indices).T[::-1])]

    def basis_aPCE(self, Z):
        """
        PCE_method: aPCE or PCE_Legendre
        aPCE_model: mod or None
        P: P or P_Steiltjs or None
        domain: Looks like np.array([[a,b], [a,b], [a,b], ...])
        """


        N = Z.shape[0]
        n = int(math.factorial(self.d + self.p)/(math.factorial(self.d)*math.factorial(self.p)))

        Phi = np.ones((N, n))
        idx = self.multivariate_pce_index(self.d, self.p)

        for i in range(n):
            for j in range(self.d):

                Phi[:,i] *=  self.aPCE_model.Pol_eval(self.P[j][idx[i][j]], Z[:,j])

        return Phi
 

    def basis_PCE_Legendre(self, Z):
        """
        PCE_method: aPCE or PCE_Legendre
        aPCE_model: mod or None
        P: P or P_Steiltjs or None
        domain: Looks like np.array([[a,b], [a,b], [a,b], ...])
        """


        N = Z.shape[0]
        n = int(math.factorial(self.d + self.p)/(math.factorial(self.d)*math.factorial(self.p)))

        Phi = np.ones((N, n))
        idx = self.multivariate_pce_index(self.d, self.p)

        a = np.array(self.domain)[:,0]
        b = np.array(self.domain)[:,1]
        for i in range(n):
            for j in range(self.d):
                Phi[:,i] *=  math.sqrt((2*idx[i][j]+1)/1)*legendre(idx[i][j])((a[j]+b[j]-2*Z[:,j])/(a[j]-b[j]))

        return Phi


    def basis_PCE_Hermite(self, Z):
        """
        PCE_method: aPCE or PCE_Legendre
        aPCE_model: mod or None
        P: P or P_Steiltjs or None
        domain: Looks like np.array([[a,b], [a,b], [a,b], ...])
        """


        N = Z.shape[0]
        n = int(math.factorial(self.d + self.p)/(math.factorial(self.d)*math.factorial(self.p)))

        Phi = np.ones((N, n))
        idx = self.multivariate_pce_index(self.d, self.p)

        mu_vals = self.mu_vals
        sigma_vals = self.sigma_vals

        for i in range(n):
            for j in range(self.d):
                Phi[:,i] *=  hermitenorm(idx[i][j])((Z[:,j]-mu_vals[j])/sigma_vals[j])

        return Phi
    
        def basis_PCE_Truncated_Hermite(self, Z):
        """
        PCE_method: aPCE or PCE_Legendre
        aPCE_model: mod or None
        P: P or P_Steiltjs or None
        domain: Looks like np.array([[a,b], [a,b], [a,b], ...])
        """


        N = Z.shape[0]
        n = int(math.factorial(self.d + self.p)/(math.factorial(self.d)*math.factorial(self.p)))

        Phi = np.ones((N, n))
        idx = self.multivariate_pce_index(self.d, self.p)

        mu_vals = self.mu_vals
        sigma_vals = self.sigma_vals
        a = self.l_bound
        b = self.u_bound

        
        for i in range(n):
            for j in range(self.d):
                Phi[:,i] *=  norm.pdf(Z[:,j])/(norm.cdf(b)-norm.cdf(a))*hermitenorm(idx[i][j])((Z[:,j]-mu_vals[j])/sigma_vals[j])
                        

        return Phi

    
    
