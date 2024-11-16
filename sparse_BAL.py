"""
The module contains definitions for test functions

__author__ = 'Isabel Corona Guevara'

"""

import numpy as np
import time
from aPCE import *
from AFVB_PCE import *
from VRVM_PCE import *
from PCE_Full_Model import *
import test_functions as tf
import random
from mpi4py import MPI
import sys
from sklearn.metrics import mean_squared_error

def BME(x_pool, y_pool, a, k, Psi, active_cols, N_mc):

    Psi_x = Psi(x_pool) # each row is a sample
    BME_vals = []

    for i in range(x_pool.shape[0]):
        val = 0
        for j in range(N_mc):
            a_full = np.zeros(Psi_x.shape[1])
            a_full[active_cols] = a[j].flatten()

            val += (k[j]/(2*np.pi))**(1/2)*np.exp(-k[j]/2*(y_pool[i]-Psi_x[i,:]@a_full)**2)
        val /= N_mc
        BME_vals.append(val.item())

    return np.array(BME_vals)

def KL_Divergence(x_pool, y_pool, a, k, Psi, active_cols, N_mc):
    
    Psi_x = Psi(x_pool) # each row is a sample
    KL_val1 = []
    KL_val2 = []

    for i in range(x_pool.shape[0]):
        val1 = 0
        val2 = 0
        for j in range(N_mc):
            a_full = np.zeros(Psi_x.shape[1])
            a_full[active_cols] = a[j].flatten()

            val1 += np.log((k[j]/(2*np.pi))**(1/2)*np.exp(-k[j]/2*(y_pool[i]-Psi_x[i,:]@a_full)**2))
            val2 += (k[j]/(2*np.pi))**(1/2)*np.exp(-k[j]/2*(y_pool[i]-Psi_x[i,:]@a_full)**2)
        val1 /= N_mc
        val2 /= N_mc
        KL_val1.append(val1.item())
        KL_val2.append(val2.item())
    
    return np.array(KL_val1), np.array(KL_val2)

def BAL(d, p, domain, x_pool, y_pool, X_test, Y_test, N_BAL, N_mc, model, method):
    
    '''
    model is PCE_Full_Model or AFVB_PCE for example
    '''

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if N_mc % size != 0:
        sys.exit("N_mc must be divisible by n_tasks")

    A_0 = 1e-02
    B_0 = 1e-04
    C_0 = 1e-02
    D_0 = 1e-04

    mod = model('PCE_Legendre', d, p, domain, A_0 = A_0, B_0 = B_0, C_0 = C_0, D_0 = D_0)

    # Training model on the first data point
    x_train = x_pool[0:1,:]
    y_train = y_pool[0]
    arg_train = np.array([0])
    mod.fit(x_train, y_train)

    x_pool = np.delete(x_pool, 0, axis = 0)
    y_pool = np.delete(y_pool, 0, axis = 0)
    N_pool = x_pool.shape[0]
    arg_pool = np.array(list(range(1,N_pool+1)))


    index_opt = [np.nan]
    VLB = []
    VLB.append(mod.L_beta[mod.max_beta])
    n_star = []
    n_star.append(mod.n_star)

    Errors = [np.sqrt(mean_squared_error(mod.predict(X_test), Y_test))/np.mean(Y_test)]

    for t in range(N_BAL-1):

        t0 = time.time()

        a_mean = mod.a_hat
        chi = mod.chi
        A = mod.A
        B = mod.B
        C = mod.C
        D = mod.D
        Psi = mod.basis
        active_cols = mod.active_cols

        if rank == 0:
            a = []
            k = list(np.random.gamma(A, 1/B, N_mc//size))

            for i in range(N_mc//size):
                a.append(np.random.multivariate_normal(a_mean.reshape(a_mean.shape[0]), 1/k[i]*chi))

            if method == 'BME_min':
                method_vals = BME(x_pool, y_pool, a, k, Psi, active_cols, N_mc//size)
            elif method == 'KL_Divergence_max':
                val1, val2 = KL_Divergence(x_pool, y_pool, a, k, Psi, active_cols, N_mc//size)
                method_vals = val1 - np.log(val2)
            else:
                print('No method given')

            for i in range(1, size):
                method_vals += comm.recv(source = i, tag = 1)
            method_vals = method_vals/size

        else:
            a = []
            k = list(np.random.gamma(A, 1/B, N_mc//size))

            for i in range(N_mc//size):
                a.append(np.random.multivariate_normal(a_mean.reshape(a_mean.shape[0]), 1/k[i]*chi))

            if method == 'BME_min':
                method_vals = BME(x_pool, y_pool, a, k, Psi, active_cols, N_mc//size)
            elif method == 'KL_Divergence_max':
                val1, val2 = KL_Divergence(x_pool, y_pool, a, k, Psi, active_cols, N_mc//size)
                method_vals = val1 - np.log(val2)
            else:
                print('No method given')

            comm.send(method_vals, dest = 0, tag = 1)

        method_vals = comm.bcast(method_vals, root = 0)

        if method == 'BME_min':
            index_BAL = np.argmin(method_vals)
        elif method == 'KL_Divergence_max':
            index_BAL = np.argmax(method_vals)


        index_opt.append(method_vals[index_BAL])

        # Add data point to training set for BME
        x_train = np.concatenate((x_train, x_pool[index_BAL:index_BAL+1,:]), axis = 0)
        y_train = np.concatenate((y_train, y_pool[index_BAL]), axis = 0)
        arg_train = np.append(arg_train, arg_pool[index_BAL])

        # Delete data point from pool for BME
        x_pool = np.delete(x_pool, index_BAL, axis = 0)
        y_pool = np.delete(y_pool, index_BAL, axis = 0)
        N_pool = x_pool.shape[0]
        arg_pool = np.delete(arg_pool, index_BAL, axis = 0)

        # Re-train the model
        mod.fit(x_train, y_train)
        Errors.append(np.sqrt(mean_squared_error(mod.predict(X_test), Y_test))/np.mean(Y_test))
        if rank == 0:
            print(t, 'One point selected:', time.time()-t0, 'sec')
        #VLB.append(mod.L_beta[mod.max_beta])
        #n_star.append(mod.n_star)

    return Errors





