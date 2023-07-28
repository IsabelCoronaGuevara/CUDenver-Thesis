{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9d9be5de",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "from matplotlib.pylab import rcParams\n",
    "from scipy.integrate import solve_ivp\n",
    "import math\n",
    "from scipy.special import hermitenorm as hermite\n",
    "import scipy.integrate as integrate\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from NormalScoreTransform import *\n",
    "from numpy.polynomial.hermite_e import hermeval, hermegauss\n",
    "import seaborn as sns\n",
    "from itertools import product"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "be068508",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Mixture of Gaussian and log-Gaussian\n",
    "z = np.random.uniform(0, 1, N) < 0.6 # assume mixture coefficients are (0.8,0.2)\n",
    "dat = np.zeros(N)\n",
    "for i in range(N):\n",
    "    if z[i] == True:\n",
    "        dat[i] = np.random.normal(1.5, 0.5, 1)\n",
    "    else:\n",
    "        dat[i] = np.random.lognormal(1, 0.25, 1)\n",
    "        \n",
    "noise = np.random.normal(0, 0.01, N) \n",
    "dat += noise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "211e709e",
   "metadata": {},
   "outputs": [],
   "source": [
    "N = 500 # Sample size\n",
    "N_mc = 10000\n",
    "d = 1  # d --> number of random variables\n",
    "p = 6 # p --> degree of PCE polynomial\n",
    "n = int(math.factorial(d + p)/(math.factorial(d)*math.factorial(p))) # number of terms in PCE"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "634e3893",
   "metadata": {},
   "source": [
    "## aPCE"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d5b8ee5b",
   "metadata": {},
   "source": [
    "## Making one Function to Calculate any Degree n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4fc8a3de",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Checking Polynomials we created are orthogonal\n",
    "def Pol_eval(coeff, x):\n",
    "    deg = coeff.shape[0]\n",
    "    val = 0\n",
    "    for i in range(deg):\n",
    "        val += coeff[i]*x**i\n",
    "        \n",
    "    return val\n",
    "\n",
    "def Int_eval(coeff1, coeff2, dat):\n",
    "    s = np.mean(Pol_eval(coeff1, dat)*Pol_eval(coeff2, dat))\n",
    "    return s"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "168960df",
   "metadata": {},
   "outputs": [],
   "source": [
    "def aPCE(d, p, N, dat):\n",
    "    n = int(math.factorial(d + p)/(math.factorial(d)*math.factorial(p)))\n",
    "\n",
    "    #### Calculating the Orthogonal Polynomials\n",
    "    P = np.zeros((p+1, p+1))\n",
    "\n",
    "    P[0,0] = 1\n",
    "\n",
    "    mu = np.zeros(2*p) \n",
    "    for i in range(2*p): \n",
    "        mu[i] = np.mean(dat**i)\n",
    "\n",
    "    mu_mat = np.zeros((p, p+1))\n",
    "    for i in range(p):\n",
    "        mu_mat[i,:] = mu[i:(p+1+i)]\n",
    "\n",
    "    for i in range(1,p+1):\n",
    "        A = np.zeros((i+1, i+1))\n",
    "        A[-1,-1] = 1\n",
    "        A[0:i,:] = mu_mat[0:i, 0:i+1]\n",
    "        b = A[-1]\n",
    "        a = np.zeros(p+1)\n",
    "        a[0:i+1] = np.linalg.inv(A)@b\n",
    "        P[i, :] = a.transpose()\n",
    "    \n",
    "    P_norm = np.zeros((p+1, p+1))\n",
    "    for i in range(p+1):\n",
    "        P_norm[i,:] = P[i,:]/np.sqrt(Int_eval(P[i,:], P[i,:], dat))\n",
    "    \n",
    "    # Finding the weights for c_i\n",
    "    nodes = np.roots(P_norm[-1][::-1])\n",
    "    V = np.zeros((p,p))\n",
    "    for i in range(p):\n",
    "        V[i,:] = nodes**i\n",
    "    b_cond = np.zeros(p)\n",
    "    for i in range(p):\n",
    "        b_cond[i] = np.mean(dat**i)\n",
    "    w_s = np.linalg.inv(V)@b_cond\n",
    "\n",
    "    c = np.zeros(p+1)\n",
    "    for i in range(p+1):\n",
    "        c[i] = np.dot(w_s, np.exp(-nodes)*Pol_eval(P_norm[i,:], nodes))\n",
    "        \n",
    "    r = 0\n",
    "    for i in range(n):\n",
    "        r += c[i]*Pol_eval(P_norm[i,:], dat)\n",
    "    return r"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a36b3de3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "@webio": {
   "lastCommId": null,
   "lastKernelId": null
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
