#!/usr/bin/env python
# coding: utf-8

# # Test laminar 3D equations: const flow

# In[1]:


# Add any possible locations of amr-wind-frontend here
import sys
extradirs = ['../',
            ]
for x in extradirs: sys.path.insert(1, x)

import numpy as np
import matplotlib.pyplot as plt
import SANDwake3D_laminar as SANDwake3D


# In[2]:


# Set boundary conditions
uval=1.23
ubc = {}
ubc['ylo'] = {'type':'dirichlet', 'value':uval}
ubc['yhi'] = {'type':'dirichlet', 'value':uval}
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
Ny=11
Nz=11
yvec = np.linspace(-1,1,Ny)
zvec = np.linspace(-3,3,Nz)

dx   = 1
dy   = np.mean(np.diff(yvec))
dz   = np.mean(np.diff(zvec))
print(dy)
print(dz)


# In[4]:


ym, zm = np.meshgrid(yvec, zvec)

Uinit  = np.ones((Ny, Nz))*uval
Vinit  = np.zeros((Ny, Nz))
Winit  = np.zeros((Ny, Nz))

phiinit = {}
phiinit['u'] = Uinit
phiinit['v'] = Vinit
phiinit['w'] = Winit


# In[5]:


params = {'nu':1}


# In[6]:


phi1 = SANDwake3D.advanceSystem(phiinit, dx, dy, dz, params, allbc, SANDwake3D.laminar_eqns, verbose=True, maxiter=2)


# In[7]:


phi = SANDwake3D.marchSystem(phiinit, [0.0, 1.0, 2.0], dy, dz, params, allbc, SANDwake3D.laminar_eqns, verbose=True, maxiter=2)


# In[8]:


print(phi['u'][-1,:,:])


# In[9]:


err = np.sum(np.abs(phi['u'][-1,:,:]-phi['u'][0,:,:]))
print('Error = %e'%err)


# In[10]:


