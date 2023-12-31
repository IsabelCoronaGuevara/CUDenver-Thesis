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

class aPCE_withQuad(object):
    """
    Description Here
    
    """
    
    def __init__(self, X, model, p, idx):
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
        self.model = model
        self.idx = idx
        self.N = X.shape[0] # sample size
        self.d = X.shape[1] # number of parameters
        self.n = int(math.factorial(self.d + self.p)/(math.factorial(self.d)*math.factorial(self.p))) # number of terms in PCE
    
    def Pol_eval(self, coeff, x):
        """
        Evaluates a polynomial at the value x.
        Polynomial is assumed to be in the form: c_0 + c_1*x + ... + c_n*x^n
        
        """
        
        deg = coeff.shape[0]
        val = 0
        for i in range(deg):
            val += coeff[i]*x**i

        return val

    def Int_eval(self, coeff1, coeff2, dat):
        """
        Approximates the integral of the product of 2 polynomials.
        """
        
        s = np.mean(self.Pol_eval(coeff1, dat)*self.Pol_eval(coeff2, dat))
        
        return s
    
    def Create_Orthonormal_Polynomials(self, n_quad):
        """
        
        """
        P = []
                    
        for j in range(self.d):
            # Creating Orthogonal Polynomials of size n_quad
            P_temp = np.zeros((n_quad+1, n_quad+1))

            P_temp[0,0] = 1

            mu = np.zeros(2*n_quad) 
            for i in range(2*n_quad): 
                mu[i] = np.mean(self.X[:,j]**i)

            mu_mat = np.zeros((n_quad, n_quad+1))
            for i in range(n_quad):
                mu_mat[i,:] = mu[i:(n_quad+1+i)]

            for i in range(1, n_quad+1):
                A = np.zeros((i+1, i+1))
                A[-1,-1] = 1
                A[0:i,:] = mu_mat[0:i, 0:i+1]
                b = A[-1]
                a = np.zeros(n_quad+1)
                a[0:i+1] = np.linalg.inv(A)@b
                P_temp[i, :] = a.transpose()

            # Normalizing Polynomials
            P_temp_norm = np.zeros((n_quad+1, n_quad+1))
            for i in range(n_quad+1):
                P_temp_norm[i,:] = P_temp[i,:]/np.sqrt(self.Int_eval(P_temp[i,:], P_temp[i,:], self.X[:,j]))

            # Adding Matrix with Polynomial Coefficients to P
            P.append(P_temp_norm)

        return P
    
    def multivariate_pce_index_quad(self, d, max_deg):
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
        index = np.array([i for i in product(*(range(i + 1) for i in maxRange)) if max(i) <= max_deg])
        
        return index
    
    
    def Quadrature(self, n_quad_in):
        """
        Need to fix restriction that n_quad needs to be larger than p

        """
        
        if n_quad_in < self.p:
            n_quad = self.p
        else:
            n_quad = n_quad_in
            
        pce_index = self.idx
        pce_coef = np.zeros(pce_index.shape[0])

        P_quad = self.Create_Orthonormal_Polynomials(n_quad)
        P = self.Create_Orthonormal_Polynomials(self.p)

        w = []
        nodes = []

        for j in range(self.d):

            # Finding the weights for each parameter
            nodes_temp = np.roots(P_quad[j][-1][::-1])
            V = np.zeros((n_quad, n_quad))
            for i in range(n_quad):
                V[i,:] = nodes_temp**i
            b_cond = np.zeros(n_quad)
            for i in range(n_quad):
                b_cond[i] = np.mean(self.X[:,j]**i)
            w_temp = np.linalg.inv(V)@b_cond

            w.append(w_temp)
            nodes.append(nodes_temp)

        quad_index = self.multivariate_pce_index_quad(self.d, n_quad_in-1)
        
        for i in range(pce_coef.size):
            for j in range(quad_index.shape[0]):
                w_P = 1
                Z = np.zeros((1, self.d))
                for k in range(self.d):
                    w_P *= w[k][quad_index[j,k]]*self.Pol_eval(P_quad[k][pce_index[i,k],:], nodes[k][quad_index[j,k]])
                    Z[:,k] = nodes[k][quad_index[j,k]]
                    
                pce_coef[i] += w_P*self.model(Z)

        return pce_coef, P
    
    def LR(self):
        """
        Need to not require n_quad input
        
        """
        
        P = self.Create_Orthonormal_Polynomials(self.p)
        
        index = self.idx
        Phi = np.ones((self.N, self.n))

        for i in range(self.n):
            for j in range(self.d):
                Phi[:,i] *=  self.Pol_eval(P[j][index[i][j],:], self.X[:,j])
        
        Y = self.model(self.X)
        mod = LinearRegression(fit_intercept = False).fit(Phi, Y)
        
        return mod.coef_.reshape(self.n), P