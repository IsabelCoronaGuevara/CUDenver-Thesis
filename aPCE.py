"""
The module contains ...
Add description

__author__ = 'Isabel Corona Guevara'
__version__ = '0.1'
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import math
from itertools import product

class aPCE(object):
    """
    Description Here
    
    """
    
    def __init__(self, X, p, idx):
        """
        Implement the initializer of aPCE class.

        data: array, is of size (n_sample, n_param)
        p: integer, the degree of the PCE
        model: function, the model, the model takes a numpy array
               of size (n_sample, n_param) as input and the output
               is of size (n_sample, n_outputs)
        method: string, method used to calculate the coefficients of the PCE
                choices from 'LR' (Linear Regression) and 'Quadrature'
        q: Number of Quadrature points used for method = "Quadrature"

        """
        
        self.X = X
        self.p = p
        self.idx = idx
        self.N = self.X.shape[0] # sample size
        self.d = self.X.shape[1] # number of parameters
        self.n = int(math.factorial(self.d + self.p)/(math.factorial(self.d)*math.factorial(self.p)))
    
    def Pol_eval(self, coeff, x):
        """
        Evaluates a polynomial at the value x.
        Polynomial is assumed to be in the form: c_0 + c_1*x + ... + c_n*x^n
        
        """
        
        val = 0
        for i in range(self.p + 1):
            val += coeff[i]*x**i

        return val

    def Inner_prod(self, coeff, dat):
        """
        Approximates the integral of the product of 2 polynomials.
        """
        
        s = np.mean(self.Pol_eval(coeff, dat)*self.Pol_eval(coeff, dat))
        
        return s
    
    def Create_Orthonormal_Polynomials(self, p):
        """
        
        """
        P = []
                    
        for j in range(self.d):
            # Creating Orthogonal Polynomials of size n_quad
            P_temp = np.zeros((p+1, p+1))

            P_temp[0,0] = 1

            mu = np.zeros(2*p) 
            for i in range(2*p): 
                mu[i] = np.mean(self.X[:,j]**i)

            mu_mat = np.zeros((p, p+1))
            for i in range(p):
                mu_mat[i,:] = mu[i:(p+1+i)]

            for i in range(1, p+1):
                A = np.zeros((i+1, i+1))
                A[-1,-1] = 1
                A[0:i,:] = mu_mat[0:i, 0:i+1]
                b = A[-1]
                a = np.zeros(p+1)
                a[0:i+1] = np.linalg.inv(A)@b
                P_temp[i, :] = a.transpose()

            # Normalizing Polynomials
            P_temp_norm = np.zeros((p+1, p+1))
            for i in range(p+1):
                P_temp_norm[i,:] = P_temp[i,:]/np.sqrt(self.Inner_prod(P_temp[i,:], self.X[:,j]))

            # Adding Matrix with Polynomial Coefficients to P
            P.append(P_temp_norm)

        return P
    