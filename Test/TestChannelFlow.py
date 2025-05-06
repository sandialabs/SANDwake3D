#!/usr/bin/env python
# coding: utf-8

# # Test laminar 3D equations: channel flow

# In[1]:


# Add any possible locations of amr-wind-frontend here
import sys, os
curdir = os.getcwd()
extradirs = ['../', './',
             os.path.dirname(curdir),
            ]
for x in extradirs: sys.path.insert(1, x)

import numpy as np
import matplotlib.pyplot as plt
import SANDwake3D_laminar as SANDwake3D


# In[2]:


# Set boundary conditions
ubc = {}
ubc['ylo'] = {'type':'dirichlet', 'value':0.0}
ubc['yhi'] = {'type':'dirichlet', 'value':0.0}
ubc['zlo'] = {'type':'neumann', 'value':0.0}
ubc['zhi'] = {'type':'neumann', 'value':0.0}

vbc = {}
vbc['ylo'] = {'type':'dirichlet', 'value':0.0}
vbc['yhi'] = {'type':'dirichlet', 'value':0.0}
vbc['zlo'] = {'type':'dirichlet', 'value':0.0}
vbc['zhi'] = {'type':'dirichlet', 'value':0.0}

wbc = {}
wbc['ylo'] = {'type':'dirichlet', 'value':0.0}
wbc['yhi'] = {'type':'dirichlet', 'value':0.0}
wbc['zlo'] = {'type':'dirichlet', 'value':0.0}
wbc['zhi'] = {'type':'dirichlet', 'value':0.0}

allbc = {'u':ubc, 'v':vbc, 'w':wbc}


# In[3]:


# Initial condition
Ny=21
Nz=5
yvec = np.linspace(-1,1,Ny)
zvec = np.linspace(-1,1,Nz)

dx   = 1
dy   = np.mean(np.diff(yvec))
dz   = np.mean(np.diff(zvec))
print(dy)
print(dz)


# In[4]:


ym, zm = np.meshgrid(yvec, zvec)

Uinit  = (1.0 - ym.transpose()**2) #np.ones((Ny, Nz))*uval
Vinit  = np.zeros((Ny, Nz))
Winit  = np.zeros((Ny, Nz))

phiinit = {}
phiinit['u'] = Uinit
phiinit['v'] = Vinit
phiinit['w'] = Winit


# In[5]:


Uinit


# In[6]:


# Set parameters
params = {}
params['nu']=1
params['fx_const']=2.0*params['nu']


# In[7]:


phi1 = SANDwake3D.advanceSystem(phiinit, dx, dy, dz, params, allbc, SANDwake3D.laminar_eqns, verbose=True, maxiter=2)


# In[ ]:


phi = SANDwake3D.marchSystem(phiinit, xvec, dy, dz, params, allbc, SANDwake3D.laminar_eqns, verbose=True, maxiter=2)


# In[8]:


err = np.sum(np.abs(phi1['u'] - Uinit))
print('Error = %e'%err)


# In[9]:


