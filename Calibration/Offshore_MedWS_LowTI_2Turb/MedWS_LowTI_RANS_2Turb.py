#!/usr/bin/env python
# Add any possible locations of amr-wind-frontend here
import sys, os
curdir = os.getcwd()
extradirs = ['../', '../../',
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

## Set the Mesh and BC's

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

Cmu, C1eps, C2eps, kfactor = 0.07586245, 1.45631433, 1.92560166, 0.71556826
#Cmu, C1eps, C2eps, kfactor = 0.080678,   1.41483508, 1.94730793, 0.77932559

# These values from iea_15MW.yaml
IEA15MW_Ct=[0.000000    , 0.000000    , 0.80742173    , 0.784655297    , 0.781771245    , 0.785377072    , 0.788045584    , 0.789922119    , 0.790464625    , 0.789868339    , 0.788727582    , 0.787359348    , 0.785895402    , 0.778275899    , 0.778275899    , 0.778275899    , 0.778275899    , 0.778275899    , 0.778275899    , 0.778275899    , 0.778275899    , 0.778275899    , 0.778275899    , 0.778275899    , 0.778275899    , 0.778275899    , 0.778275899    , 0.778275899    , 0.778275899    , 0.778275899    , 0.77176172    , 0.747149663    , 0.562338457    , 0.463477777    , 0.389083718    , 0.329822385    , 0.281465071    , 0.241494345    , 0.208180574    , 0.180257568    , 0.156747535    , 0.136877529    , 0.120026379    , 0.105689427    , 0.093453742    , 0.082979637    , 0.073986457    , 0.066241166    , 0.059552107    , 0.053756866    , 0.048721662    , 0.044334197    , 0.0    , 0.0]
IEA15MW_WS=[0.000    , 2.9    , 3.0    , 3.54953237    , 4.067900771    , 4.553906848    , 5.006427063    , 5.424415288    , 5.806905228    , 6.153012649    , 6.461937428    , 6.732965398    , 6.965470002    , 7.158913742    , 7.312849418    , 7.426921164    , 7.500865272    , 7.534510799    , 7.541241633    , 7.58833327    , 7.675676842    , 7.803070431    , 7.970219531    , 8.176737731    , 8.422147605    , 8.70588182    , 9.027284445    , 9.385612468    , 9.780037514    , 10.20964776    , 10.67345004    , 10.86770694    , 11.17037214    , 11.6992653    , 12.25890683    , 12.84800295    , 13.46519181    , 14.10904661    , 14.77807889    , 15.470742    , 16.18543466    , 16.92050464    , 17.67425264    , 18.44493615    , 19.23077353    , 20.02994808    , 20.8406123    , 21.66089211    , 22.4888912    , 23.32269542    , 24.1603772    , 25    , 25.020    , 50.0]
IEA15MW_PC=[0.000000    , 0.000000    , 42.733312    , 292.585981    , 607.966543    , 981.097693    , 1401.98084    , 1858.67086    , 2337.575997    , 2824.097302    , 3303.06456    , 3759.432328    , 4178.637714    , 4547.19121    , 4855.342682    , 5091.537139    , 5248.453137    , 5320.793207    , 5335.345498    , 5437.90563    , 5631.253025    , 5920.980626    , 6315.115602    , 6824.470067    , 7462.846389    , 8238.359448    , 9167.96703    , 10285.211    , 11617.23699    , 13194.41511    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 15000.00    , 0.0    , 0.0]


# Set parameters
params = {
    'rho' : 1.225,
    'cp'  : 1005,
    'g'   : 9.81,
    'nu'  : 1.5E-5,
    'Cmu' : Cmu, # [0.025 - 0.06] #5.48**(-2),
    'C1eps' : C1eps, #[1 - 1.76] 1.76,
    'C2eps' : C2eps, #[1.92 - 3.50] #1.92,
    'C3eps' : None, #-1, #0.033,
    'sigmak' : 1.0, 
    'sigmaeps' : 1.3,
    'sigmaT':1.0,
    'dtau':  0.5,
    
    'ustar' : 0.2125, 
    'qw': -0.40, #-0.050,
    'L' : 275.0,
    'kappa' : 0.42,
    'z0' : 0.00004, #0.0005,

    'zlo':zmin,
    'Tw':300.0,
    'beta':1.0/300.0,
    'Tref':None,
    
    # Extra mesh variables needed for turbine forcing
    'ym':ym,
    'zm':zm,
    
    'turbinelist': [
            {'name':'T0',
             'turbx':120.0,     # Turbine x location
             'turby':0.0,       # Turbine y location
             'zhh':turbhh,      # Turbine hub-height
             'turbD':rotorD,    # Turbine diameter
             'turbnormal':[-1, 0, 0],
             'Ct':IEA15MW_Ct,
             'WS':IEA15MW_WS,
             'power':IEA15MW_PC,
             #'Ct':0.80,
             #'power':1000.123,
             'Rdelta':24,
             'turbfunc':'SANDwake3D_base.UnifCtADM',
            },
            {'name':'T1',
             'turbx':120.0+5*240,     # Turbine x location
             'turby':0.0,       # Turbine y location
             'zhh':turbhh,      # Turbine hub-height
             'turbD':rotorD,    # Turbine diameter
             'turbnormal':[-1, 0, 0],
             'Ct':IEA15MW_Ct,
             'WS':IEA15MW_WS,
             'power':IEA15MW_PC,
             'Rdelta':24,
             'turbfunc':'SANDwake3D_base.UnifCtADM',
            },

        ]

}

# This parameter defines V(z) -- deg per meter
veerpermeter = 0.05 #0.05 #0.02
Tpermeter    = 0.0025 #0.002 #0.0025

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
   
    Kinit[:, iz]   = SANDwake3D.init_k_ABL(z,params)*np.exp(-0.0*z)*kfactor + 0.0 #0.02 #0.15 #0.0075 #0.01 #0.02
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
xvecplot=[0, 120+240*10]

xvec = np.arange(xvecplot[0], xvecplot[-1]+1.0E-6, dx)

# Start marching the solution
start_time = time.time()
phi = SANDwake3D.marchSystem(phiinit, xvec, dy, dz, params, 
                             SANDwake3D.getTypicalWMBC(Uinit[0,-1], tempT, 
                                                       veerBC=Vprof, dTdz=Tpermeter, uBC_y=Uprof),
                             SANDwake3D.kepsT_eqns, 
                             advanceSys=SANDwake3D.advanceSystemKEPS,
                             verbose=1, maxiter=250, tol=1.0E-4)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

savepklfile = 'Phi_MedWS_LowTI.pkl'

if len(savepklfile)>0:
    db={'params':params, 'xvec':xvec, 'phi':phi}
    dbfile = open(savepklfile, 'wb')
    pickle.dump(db, dbfile, protocol=2)
    dbfile.close()
