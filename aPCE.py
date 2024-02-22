"""
The module contains ...
Add description

__author__ = 'Isabel Corona Guevara'
__version__ = '0.1'
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import math
import scipy.integrate as integrate
from itertools import product

class aPCE(object):
    """
    Description Here
    
    """
    
    def __init__(self, X, p):
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
        self.N = self.X.shape[0] # sample size
        self.d = self.X.shape[1] # number of parameters
        self.n = int(math.factorial(self.d + self.p)/(math.factorial(self.d)*math.factorial(self.p)))
    
    
    def multivariate_pce_index(self, d, max_deg):
        """
        Generate all the d-dimensional polynomial indices with the 
        constraint that the sum of the indexes is <= max_deg

        input:
        d: int, number of random variables
        max_deg: int, the max degree allowed

        return: 
        2d array with shape[1] equal to d, the multivariate indices
        """
        maxRange = max_deg*np.ones(d, dtype = 'int')
        index = np.array([i for i in product(*(range(i + 1) for i in maxRange)) if sum(i) <= max_deg])

        return index    
    
    
    def Pol_eval(self, coeff, x):
        """
        Evaluates a polynomial at the value x.
        Polynomial is assumed to be in the form: c_0 + c_1*x + ... + c_n*x^n
        
        """
        
        val = 0
        for i in range(self.p + 1):
            val += coeff[i]*x**i

        return val
    
    def Inner_prod_3D(self, coeff1, coeff2, coeff3, dat):
        """
        Approximates the integral of the product of 2 polynomials.
        """
        
        s = np.mean(self.Pol_eval(coeff1, dat[:,0])*self.Pol_eval(coeff2, dat[:,1])*self.Pol_eval(coeff3, dat[:,2])\
                   *self.Pol_eval(coeff1, dat[:,0])*self.Pol_eval(coeff2, dat[:,1])*self.Pol_eval(coeff3, dat[:,2]))
        
        return s

    def Inner_prod(self, coeff1, coeff2, dat):
        """
        Approximates the integral of the product of 2 polynomials.
        """
        
        s = np.mean(self.Pol_eval(coeff1, dat)*self.Pol_eval(coeff2, dat))
        
        return s
    
    def coeff_index(self, d, max_deg):
        """
        Generate all the d-dimensional polynomial indices with the 
        constraint that the sum of the indexes is <= max_deg

        input:
        d: int, number of random variables
        max_deg: int, the max degree allowed

        return: 
        2d array with shape[1] equal to d, the multivariate indices
        """
        maxRange = max_deg*np.ones(d, dtype = 'int')
        index = np.array([i for i in product(*(range(i + 1) for i in maxRange)) if sum(i) <= 2*max_deg])

        return index    

    def Norm(self, P, dat, pol_d):
        """
        Approximates the norm of a polynomial.
        """
        
        
        
        c_idx = self.coeff_index(2, pol_d)
        val = 0
        for i in range(c_idx.shape[0]):
            val += P[pol_d][c_idx[i][0]]*P[pol_d][c_idx[i][1]]*np.mean(dat**(np.sum(c_idx[i])))
        return np.sqrt(val)
    
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
                #P_temp_norm[i,:] = P_temp[i,:]/self.Norm(P_temp, self.X[:,j], i)
                P_temp_norm[i,:] = P_temp[i,:]/(np.sqrt(self.Inner_prod(P_temp[i,:],P_temp[i,:],self.X[:,j])))

            # Adding Matrix with Polynomial Coefficients to P
            P.append(P_temp_norm)
            
        return P
    
    def P_ip1(self, P_i, P_im1, alpha, beta):
        return np.roll(P_i,1) - alpha*P_i - beta*P_im1

    def alpha(self, P_i, X):
        return np.mean(X*self.Pol_eval(P_i, X)*self.Pol_eval(P_i, X))/np.mean(self.Pol_eval(P_i, X)*self.Pol_eval(P_i, X))

    def beta(self, P_i, P_im1, X):
        return np.mean(self.Pol_eval(P_i, X)*self.Pol_eval(P_i, X))/np.mean(self.Pol_eval(P_im1, X)*self.Pol_eval(P_im1, X))

    
    def Create_Orthonormal_Polynomials_Stieltjes(self, p):

        P = []
        for j in range(self.d):
            P_temp = np.zeros((self.p+1, self.p+1)) 
            P_temp[0,0] = 1
            P_m1 = np.zeros(self.p+1)
            P_temp[1,:] = self.P_ip1(P_temp[0,:], P_m1, self.alpha(P_temp[0,:], self.X[:,j]), 1)

            for i in range(2,self.p+1):
                P_temp[i,:] = self.P_ip1(P_temp[i-1,:], P_temp[i-2,:], self.alpha(P_temp[i-1,:], self.X[:,j]), self.beta(P_temp[i-1,:], P_temp[i-2,:], self.X[:,j]))

            P_temp_norm = np.zeros((self.p+1, self.p+1))
            for i in range(self.p+1):
                #P_temp_norm[i,:] = P_temp[i,:]/self.Norm(P_temp, self.X[:,j], i)
                P_temp_norm[i,:] = P_temp[i,:]/(np.sqrt(self.Inner_prod(P_temp[i,:],P_temp[i,:],self.X[:,j])))

            # Adding Matrix with Polynomial Coefficients to P
            P.append(P_temp_norm)
            
        return P
        
        
        
        
        
    