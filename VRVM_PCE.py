#!/usr/bin/env python
# coding: utf-8

# In[12]:


"""
Author: Panagiotis Tsilifis
Date: 11.10.2019

Contains classes for Chaos expansions, Exponential families (containing Beta, Bernoulli, Gauss and Gamma),
Optimizer for the standard RVM and sparse RVM.
"""

__all__ = ['ChaosModel', 'ExponFam', 'VRVM_PCE']

import numpy as np
import math 
import scipy.stats as st 
import scipy.special as sp
from sklearn.base import BaseEstimator
from itertools import product
import sys
from scipy.special import legendre
from basis_functions import *


# In[13]:


class ChaosModel(object):

    _dim = None

    _order = None
    
    # _basis_MI needs to be a function that takes the multi-indexes 
    # and can be evaluated at the xi's
    _basis = None
    
    _basis_shape = None

    _coeffs = None
    
    # Only works for total degree (TD)
    #def __init__(self, dim, order, basis_MI, coeffs = None, trunc = 'TD', q = None):
    def __init__(self, dim, order, basis, coeffs = None):
        """
        Initializes the object
        """
        
        assert isinstance(dim, int)
        assert isinstance(order, int)
        self._dim = dim
        self._order = order
        self._basis = basis
        self._n = int(math.factorial(self._dim + self._order)/(math.factorial(self._dim)*math.factorial(self._order)))
        self._basis_shape = (self._n, self._dim)
        
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
    


# In[ ]:


### Distributions which are part of the exponential family can have their probability densities
### written in their canonical form: h(theta)exp(eta^T*R(theta) - A(eta))
### The sufficient statistic R(.), log-normalization constant A(.) and function h(.) is the same
### for all distributions. 
### Only the natural parameter eta differs between distributions

class ExponFam(object):
	"""
	"""
	_pdf = None

	def __init__(self, pdf):
		"""
		Initializes the object
		"""
		self._pdf = pdf

#### Formula for A(eta) found in Apendix A
	def A(self, eta):
		if self._pdf == 'Gauss':
			return - eta[0]**2/(4*eta[1]) - np.log(-2*eta[1]) / 2.
		elif self._pdf == 'Gamma':
			return np.log( sp.gamma(eta[0]+1) ) - (eta[0]+1.) * np.log(-eta[1])
		elif self._pdf == 'Bernoulli':
			return - np.log(sp.expit(eta))
			#if A == np.inf:
			#	return eta
			#else:
			#	return A
		elif self._pdf == 'Beta':
			return np.log(sp.gamma(eta[0])) + np.log(sp.gamma(eta[1])) - np.log(sp.gamma(eta[0]+eta[1]))


	def A_grad(self, eta):
		if self._pdf == 'Gauss':
			return np.array([-eta[0]/(2*eta[1]), eta[0]**2 / (4*eta[1]**2) - 1./(2*eta[1])])
		elif self._pdf == 'Gamma':
			return np.array([sp.digamma(eta[0]+1) - np.log(-eta[1]) , -(eta[0]+1)/eta[1]])
		elif self._pdf == 'Bernoulli':
			return sp.expit(eta)
		elif self._pdf == 'Beta':
			return np.array([sp.digamma(eta[0]) - sp.digamma(eta[0]+eta[1]), sp.digamma(eta[1]) - sp.digamma(eta[0]+eta[1])])

	def map_from_eta(self, eta):
		if self._pdf == 'Gauss':
			return np.array([-eta[0]/(2*eta[1]), -2*eta[1]])
		elif self._pdf == 'Gamma':
			return np.array([eta[0]+1., -eta[1]])
		elif self._pdf == 'Bernoulli':
			return sp.expit(eta)
		elif self._pdf == 'Beta':
			return np.array([eta[0], eta[1]])

#### Defining the parameter eta, based on equations A1, A2, A3 and A4 on Appendix
	def map_to_eta(self, params):
		if self._pdf == 'Gauss':
			return np.array([params[0]*params[1], -params[1] / 2.])
		elif self._pdf == 'Gamma':
			return np.array([params[0]-1., -params[1]])
		elif self._pdf == 'Bernoulli':
			return np.log(params/(1. - params))
		elif self._pdf == 'Beta':
			return np.array([params[0], params[1]])


class VRVM_PCE(BaseEstimator):

	_chaos_model = None

	_data = None

	_W = None # Won't be used. Delete it later. 

	_Psi = None

	_yPsi = None

	_PsiPsi = None

	_prior_params = None
	# Number of observations 
	_K = None
	# L component
	_expL = None

	def __init__(self, PCE_method, d, p = 8, domain = None, aPCE_model = None, P = None, omega_a = 10**(-6), omega_b = 10**(-6), tau_a = 10**(-6), tau_b = 10**(-6), pi_a = 0.2, pi_b = 1.0, sigma_vals = None, mu_vals = None):
		"""
		Initializes the object
		"""
		self.p = p   
		self.d = d
		self.domain = domain
		self.PCE_method = PCE_method
		self.aPCE_model = aPCE_model
		self.P = P
		self.omega_a = omega_a
		self.omega_b = omega_b
		self.tau_a = tau_a
		self.tau_b = tau_b
		self.pi_a = pi_a
		self.pi_b = pi_b
		self._prior_params = {'omega' : [self.omega_a, self.omega_b], 'tau': [self.tau_a, self.tau_b], 'pi': [self.pi_a, self.pi_b]}
        
		if (PCE_method == 'aPCE'):
			self.basis = basis(self.d, self.p, self.domain, self.aPCE_model, self.P).basis_aPCE
            
		elif (PCE_method == 'aPCE_Stieltjes'):
			self.basis = basis(self.d, self.p, None, self.aPCE_model, self.P).basis_aPCE
            
		elif (PCE_method == 'PCE_Legendre'):
			self.basis = basis(self.d, self.p, self.domain).basis_PCE_Legendre
        
		elif (PCE_method == 'PCE_Hermite'):
			self.basis = basis(self.d, self.p, self.domain, self.aPCE_model, self.P, self.sigma_vals, self.mu_vals).basis_PCE_Hermite
        
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

#### For the following z_c_... definitions we can look at Eq. 28 - 32        

	def z_c_mean(self, val, arg = 'omega'):
		if arg == 'omega':
			exp_F = ExponFam('Gamma')
			eta = exp_F.map_to_eta(val)
		elif arg == 'eta_om':
			eta = val
		return np.array([0, (eta[0]+1) / (2*eta[1])])

#### If the argument is pi then set the distribution to be a Beta distribution
#### map_to_eta takes 2 parameter values in (called val here) and computes eta from Eq. A.4
	def z_z_mean(self, val, arg = 'pi'):
		exp_F = ExponFam('Beta')
		if arg == 'pi':
			eta = exp_F.map_to_eta(val)
		elif arg == 'eta_pi':
			eta = val
		return sp.digamma(eta[0]) - sp.digamma(eta[1])

#### Creates eta based on the prior parameters entered by the user
	def z_tau(self):
		exp_F = ExponFam('Gamma')
		return exp_F.map_to_eta(self._prior_params['tau'])

#### Creates eta based on the prior parameters entered by the user
#### They are using z_omega to be z_Zeta (from paper)
	def z_omega(self):
		exp_F = ExponFam('Gamma')
		return exp_F.map_to_eta(self._prior_params['omega'])

#### Creates eta based on the prior parameters entered by the user
	def z_pi(self):
		exp_F = ExponFam('Beta')
		return exp_F.map_to_eta(self._prior_params['pi'])

	def update_Psi(self): # Won't be used (most likely). Delete it later. 
		self._Psi = self._chaos_model._basis(self.X)

	def update_L(self):
		xi = self.X
		self._L = np.array([self.K / 2., - np.linalg.norm(self.Y - self._chaos_model.eval(xi) ) ** 2 / 2.])

	def expL(self, m, rho, pi):
		eta = self.X
		Psi = self._chaos_model._basis(eta)
		self._expL = np.array([self.K / 2., - np.linalg.norm(self.Y - np.dot(Psi, pi*m) ) ** 2 / 2. - 0.5 * np.trace( np.dot(self._PsiPsi, np.diag( pi*(1./rho) + (pi-pi**2)*m**2) )  )  ])

	def compELBO(self, eta_c, eta_om, eta_tau, eta_z, eta_pi):
		expF_gauss = ExponFam('Gauss')
		expF_gamma = ExponFam('Gamma')
		expF_beta = ExponFam('Beta')
		expF_bernoulli = ExponFam('Bernoulli')

		ELBO = - 0.5 * self.K * np.log(2*np.pi) + np.sum(expF_gamma.A_grad(eta_tau) * self._expL)
		# Expected log-likelihood
		for i in range(eta_c.shape[0]):
			# Adding entropy and expected log-prior of omega
			ELBO += expF_gamma.A(eta_om[i,:]) - np.sum( (eta_om[i,:] - self.z_omega())* expF_gamma.A_grad(eta_om[i,:]) ) - expF_gamma.A(self.z_omega())
			omega = expF_gamma.map_from_eta(eta_om[i,:])
			# Adding entropy and expected log-prior of c
			ELBO += expF_gauss.A(eta_c[i,:]) - np.sum( eta_c[i,:] * expF_gauss.A_grad(eta_c[i,:]) ) + np.sum( self.z_c_mean(omega) * expF_gauss.A_grad(eta_c[i,:]) ) - (-sp.digamma(eta_om[i,0]+1) + np.log(-eta_om[i,0]) ) / 2.
			# Adding entropy and expected log-prior of pi 
			ELBO += expF_beta.A(eta_pi[i,:]) - np.sum( (eta_pi[i,:] - self.z_pi()) * expF_beta.A_grad(eta_pi[i,:]) ) - expF_beta.A(self.z_pi())
			pi = expF_beta.map_from_eta(eta_pi[i,:])
			# Adding entropy and expected log-prior of z
			ELBO += expF_bernoulli.A(eta_z[i])  - eta_z[i] * expF_bernoulli.A_grad(eta_z[i])  + self.z_z_mean(pi) * expF_bernoulli.A_grad(eta_z[i])  - (sp.digamma(eta_pi[i,0]+eta_pi[i,1]) - sp.digamma(eta_pi[i,1]))
		ELBO +=  - (eta_tau[0]+1.) * np.log(-eta_tau[1]) + (self.z_tau()[0]+1.) * np.log(-self.z_tau()[1]) - np.sum( eta_tau - self.z_tau() * np.array([sp.digamma(eta_tau[0]+1) - np.log(-eta_tau[1]) , -(eta_tau[0]+1)/eta_tau[1]]) )
		# Adding entropy and expected log-prior of tau
		return ELBO

	#def elbo_grad(self, var):
		"""
		Computes the updating term of ELBO's natural gradient 
		corresponding to the variable var.
		"""


	def fit(self, X, Y):
		tol = 1e-4
		method = 'ascent'
		self.d = X.shape[1]
		self.K = X.shape[0]
		sys.path.append('..')

		data={'xi': X}
		data['y'] = Y.reshape(self.K)

		self.X = data['xi']
		self.Y = data['y']

		self._chaos_model = ChaosModel(self.d, self.p, self.basis)
        
		expF_gauss = ExponFam('Gauss')
		expF_gamma = ExponFam('Gamma')
		expF_beta = ExponFam('Beta')
		expF_bernoulli = ExponFam('Bernoulli')
        
		self._Psi = self._chaos_model._basis(X)
		self.n = self._Psi.shape[1]
		self._PsiPsi = np.dot(self._Psi.T, self._Psi)
		self._yPsi = (self.Y * self._Psi.T).T

		if method == 'ascent':

			eta_c = np.zeros((self._chaos_model._coeffs.shape[0],2))
			eta_om = np.zeros((self._chaos_model._coeffs.shape[0],2))
			eta_z = np.zeros(self._chaos_model._coeffs.shape[0]) 
			eta_pi = np.zeros((self._chaos_model._coeffs.shape[0], 2)) 
			for i in range(self._chaos_model._coeffs.shape[0]):
				eta_om[i,:] = self.z_omega()
				omega = expF_gamma.map_from_eta(eta_om[i,:])
				eta_c[i,:] = self.z_c_mean(omega)
				eta_pi[i,:] = self.z_pi()
				pi = expF_beta.map_from_eta(eta_pi[i,:])
				eta_z[i] = self.z_z_mean(pi)

			eta_tau = self.z_tau().reshape((1,2)) 
			elbo = np.array([-np.inf])

			eta_c_new = 1e+5 * np.ones(eta_c.shape)
			eta_om_new = 1e+5 * np.ones(eta_om.shape)
			eta_tau_new = 1e+5 * np.ones(eta_tau.shape)
			eta_z_new = 1e+5 * np.ones(eta_z.shape)
			eta_pi_new = 1e+5 * np.ones(eta_pi.shape)

			c_sol = np.zeros(eta_c.shape)
			omega_sol = np.zeros(eta_om.shape)
			tau_sol = np.zeros(eta_tau.shape)
			z_sol = np.zeros(eta_z.shape)
			pi_sol = np.zeros(eta_pi.shape)
			#print 'Eta_c :' + str(eta_c)
			err = 1e+6
			iters = 0
			pi_sum_prev = 0
			pi_err = 1e+5

			active_indices = range(eta_c.shape[0])

			while err > tol:
				params_c = np.zeros(eta_c.shape)
				params_z = np.zeros(eta_z.shape)
				
				for i in range(params_c.shape[0]):
					params_c[i,:] = expF_gauss.map_from_eta(eta_c[i,:])
					params_z[i] = expF_bernoulli.map_from_eta(eta_z[i])
				self.expL(params_c[:,0], params_c[:,1], params_z)
				# ---- Updating tau
                #### Eq. 43a
				eta_tau_new = self.z_tau() + self._expL # - np.array([20., 0]) 

				for i in active_indices:#range(eta_om.shape[0]):
					# ---- Updating omega
                    #### Eq. 43b
					eta_om_new[i,:] = self.z_omega() + 0.5 * np.array([1., - expF_gauss.A_grad(eta_c[i,:])[1]])

				#for i in range(eta_om.shape[0]):
					# ---- Updating pi 
                    #### Eq. 43c
					eta_pi_new[i,:] = self.z_pi() + np.array([0, 1]) + np.array([ 1., -1.]) * expF_bernoulli.A_grad(eta_z[i])

				#for i in range(eta_c.shape[0]):
					# ---- Updating zeta
                    #### Eq. 43e
					v_i = np.sum( self._yPsi[:,i] * params_c[i,0]) - 0.5 * self._PsiPsi[i,i] * ( (1 - 2*params_z[i]) * params_c[i,0]**2)# + params_c[i,1])
					m_i = params_z * params_c[:,0].copy()#).reshape(eta_c.shape[0], 1)
					u_i = - np.sum( self._PsiPsi[i,:] * m_i) * params_c[i,0]
					eta_z_new[i] = (v_i + u_i) * expF_gamma.A_grad(eta_tau_new)[1] + self.z_z_mean(eta_pi_new[i,:], arg = 'eta_pi')
					params_z[i] = expF_bernoulli.map_from_eta(eta_z_new[i])


				#for i in range(eta_c.shape[0]):
					# ---- Updating c
                    #### Eq. 43d
					v_i = np.array([np.sum( self._yPsi[:,i]) * params_z[i], - 0.5 * self._PsiPsi[i,i] * params_z[i]]) ## testing 
					m_i = params_z * params_c[:,0].copy()#.reshape(eta_c.shape[0], 1)
					m_i[i] = 0.
					u_i = np.array([- np.sum(self._PsiPsi[i,:] * m_i) * params_z[i], 0 ])
					eta_c_new[i,:] = (v_i + u_i) * expF_gamma.A_grad(eta_tau_new)[1] + self.z_c_mean(eta_om_new[i,:], arg = 'eta_om')
					params_c[i,:] = expF_gauss.map_from_eta(eta_c_new[i,:])

				ETA = np.hstack([eta_tau.flatten(), eta_c.flatten(), eta_z ])
				ETA_new = np.hstack([eta_tau_new.flatten(), eta_c_new.flatten(), eta_z_new ])
				ZETA_err = np.sum( expF_bernoulli.map_from_eta(eta_z) - expF_bernoulli.map_from_eta(eta_z_new) )
				pi_sum_new = expF_bernoulli.map_from_eta(eta_z_new).sum()
				pi_err =  np.abs(pi_sum_prev - pi_sum_new)
				#print(pi_err)
				pi_sum_prev = pi_sum_new.copy()

				if pi_err < 1e-4 and iters > 100:
					active_indices = list(np.where(expF_bernoulli.map_from_eta(eta_z_new) > .01)[0])
					#print('----- Iterating only on sparse terms -----')
					#print('active_indices = ', active_indices)
                
				#print ZETA_err
				err = np.linalg.norm(ETA-ETA_new)
				#if iters%10==0:
					#print('Iters = '+str(iters) + ' '*10+ 'Relative error : ' + str(err) + ' '*10)# + 'ELBO value: ' + str(elbo[iters])
				elbo = np.append(elbo, self.compELBO(eta_c_new, eta_om_new, eta_tau_new, eta_z_new, eta_pi_new) )
				eta_c = eta_c_new.copy()
				eta_tau = eta_tau_new.copy()
				eta_om = eta_om_new.copy()
				eta_z = eta_z_new.copy()
				eta_pi = eta_pi_new.copy()
				iters += 1

            #### Saving the active_indices
			active_indices_final = active_indices
            
			tau_sol[:] = expF_gamma.map_from_eta(eta_tau)
			for i in range(c_sol.shape[0]):
				c_sol[i,:] = expF_gauss.map_from_eta(eta_c[i,:])
				omega_sol[i,:] = expF_gamma.map_from_eta(eta_om[i,:])
				z_sol[i] = expF_bernoulli.map_from_eta(eta_z[i])
				pi_sol[i,:] = expF_beta.map_from_eta(eta_pi[i,:])

			ZETA_err = np.sum(z_sol)

			#print omega_sol
			#print('Total number of iterations : ' + str(iters))            
			self.c_sol = c_sol
			self.omega_sol = omega_sol
			self.tau_sol = tau_sol
			self.z_sol = z_sol
			self.pi_sol = pi_sol
			self.iters = iters
			self.elbo = elbo
            
			self.active_cols = np.array(range(0,self.n))[self.z_sol>0.01]
			self.a_hat = self.c_sol[self.active_cols,0]
			self.n_star = self.active_cols.shape[0]
			self.a_full = self.c_sol[:,0]
			#print('n_star = ', self.n_star)
            
			return self

		#elif method == 'stoch_ascent':

	def predict(self, X, sparse = True):
		if sparse is True:
			if self.n_star != 0:
				return self.basis(X)[:,self.active_cols]@self.c_sol[self.active_cols,0]
			else:
				return self.basis(X)@self.c_sol[:,0]
		else:
			return self.basis(X)@self.c_sol[:,0]
# In[ ]:




