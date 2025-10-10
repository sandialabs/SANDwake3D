#!/usr/bin/env python

# Add any possible locations of amr-wind-frontend here
import sys, os
curdir = os.getcwd()
extradirs = ['../', '../../',
             '../../SANDwake3D',
             os.path.dirname(curdir),
            ]
for x in extradirs: sys.path.insert(1, x)

import numpy as np
import matplotlib.pyplot as plt
import SANDwake3D_keps as SANDwake3D
import SANDwake3D_base as sdb

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle

## Set Mesh and BC's

# y-z mesh
dz   = 10
zmin = 2 #2.00
zmax = zmin + 400
Nz   = int((zmax-zmin)/dz+1)
zvec = np.linspace(zmin,zmax,Nz)

dy   = 10
ymax = 400
Ny   = int((2*ymax)/dy+1)
yvec = np.linspace(-ymax,ymax,Ny)

ym, zm = np.meshgrid(yvec, zvec, indexing='ij')

dy   = np.mean(np.diff(yvec))
dz   = np.mean(np.diff(zvec))

turbhh = 150.0
rotorD = 240
turbR  = rotorD*0.5

# [0.08998533, 1.62033396, 2.43429404, 1]
# [0.080678   1.41483508 1.94730793 0.77932559]
Cmu, C1eps, C2eps, kfactor = 0.07586245, 1.45631433, 1.92560166, 0.71556826

# Set parameters
params = {
    'rho' : 1.225,
    'cp'  : 1005,
    'g'   : 9.81,
    'nu'  : 1.5E-5,
    'Cmu' :  Cmu, # [0.025 - 0.06],  #5.48**(-2),
    'C1eps' : C1eps, #[1-1.76], #1.3, #1.76,
    'C2eps' : C2eps, #[1.92 - 3.50], #1.92,
    'C3eps' : None, #2.0, #-1, #0.033,
    'sigmak' : 1.0, 
    'sigmaeps' : 1.3,
    'sigmaT':1.0,
    'dtau':  0.5,
    
    'ustar' : 0.175, 
    'qw': -0.4, #-0.40, #-0.050,
    'L' : 500,
    'kappa' : 0.42,
    'z0' : 0.0001, #0.0005,

    'zlo':zmin,
    'Tw':300.0,
    'beta':1.0/300.0,
    'Tref':None, #302.0,
    
    # Extra mesh variables needed for turbine forcing
    'ym':ym,
    'zm':zm,
    
    'turbforcing':{'turbx':-120.0E6,     # Turbine x location
                   'turby':0.0,     # Turbine y location
                   'zhh':turbhh,      # Turbine hub-height
                   'turbD':rotorD,   # Turbine diameter
                   'Ct':0.80,
                   'Rdelta':24, #40,
                   'turbfunc':sdb.tanhADM,
                  },
    'turbinelist': [
            {'name':'T0',
             'turbx':120.0,     # Turbine x location
             'turby':0.0,       # Turbine y location
             'zhh':turbhh,      # Turbine hub-height
             'turbD':rotorD,    # Turbine diameter
             'turbnormal':[-1, 0, 0],
             'Ct':0.80,
             'power':1000.123,
             'Rdelta':24,
             'turbfunc':'SANDwake3D_base.UnifCtADM',
            },
        ]
}

# This parameter defines V(z) -- deg per meter
veerpermeter = 0.05 #0.05 #0.02
Tpermeter    = 0.002 #0.002 #0.0025

# Set up ABL profile
UABL = np.zeros(Nz)
for iz, z in enumerate(zvec):
    UABL[iz] = SANDwake3D.init_U_ABL(z,params)
    
# Set up veer profile
veerProf = np.zeros(Nz)
for i, z in enumerate(zvec):
    veerProf[i] = veerpermeter*(z-turbhh)
    
Uprof, Vprof = SANDwake3D.getUVfromUhVeer(UABL, veerProf)
    
tempT = np.zeros(Nz)
for i, z in enumerate(zvec):
    tempT[i] = params['Tw'] + Tpermeter*z
    
# Compute the wind direction
ztop = turbhh + turbR
zbot = turbhh - turbR
Utop, Ubot = np.interp(ztop, zvec, Uprof), np.interp(zbot, zvec, Uprof)
Vtop, Vbot = np.interp(ztop, zvec, Vprof), np.interp(zbot, zvec, Vprof)
Wdirtop = np.arctan2(Vtop, Utop)*180/np.pi
Wdirbot = np.arctan2(Vbot, Ubot)*180/np.pi

Uhh, Vhh = np.interp(turbhh, zvec, UABL), np.interp(turbhh, zvec, UABL)
alpha = np.log(Utop/Ubot)/np.log(ztop/zbot)
print('Uhh = ',Uhh)
print('alpha = ',alpha)
print('rotor veer = ',Wdirtop-Wdirbot)

Uinit  = np.zeros((Ny, Nz))
Vinit  = np.zeros((Ny, Nz))
Winit  = np.zeros((Ny, Nz))
Kinit  = np.zeros((Ny, Nz))
Tinit  = np.zeros((Ny, Nz)) #np.ones((Ny, Nz))*params['Tref']
EPSinit= np.zeros((Ny, Nz))
pinit  = np.zeros((Ny, Nz))

for iz, z in enumerate(zvec):
    Uinit[:, iz] = Uprof[iz] #SANDwake3D.init_U_ABL(z,ABLparam)
    Vinit[:, iz] = Vprof[iz]
    Winit[:, iz] = 0.0
    
    Kinit[:, iz]   = SANDwake3D.init_k_ABL(z,params)*np.exp(-0.0*z)*kfactor 
    EPSinit[:, iz] = SANDwake3D.init_e_ABL(z,params)
    Tinit[:, iz] = tempT[iz]
    
phiinit = {}
phiinit['u'] = Uinit
phiinit['v'] = Vinit
phiinit['w'] = Winit
phiinit['k'] = Kinit
phiinit['T'] = Tinit
phiinit['eps'] = EPSinit
phiinit['p'] = pinit

dx  = 120
xvecplot=[0, 120+240*8]

xvec = np.arange(xvecplot[0], xvecplot[-1]+1.0E-6, dx)

# March the RANS system
start_time = time.time()
phi = SANDwake3D.marchSystem(phiinit, xvec, dy, dz, params, 
                             SANDwake3D.getTypicalWMBC(Uinit[0,-1], tempT, #[-1], 
                                                       veerBC=Vprof, dTdz=Tpermeter, uBC_y=Uprof),
                             SANDwake3D.kepsT_eqns, 
                             advanceSys=SANDwake3D.advanceSystemKEPS,
                             verbose=2, maxiter=100, tol=1.0E-4)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

savepklfile = 'Phi_LowWS_LowTI.pkl'

if len(savepklfile)>0:
    db={'params':params, 'xvec':xvec, 'phi':phi}
    dbfile = open(savepklfile, 'wb')
    pickle.dump(db, dbfile, protocol=2)
    dbfile.close()
