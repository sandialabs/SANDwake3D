import sys, os
curdir = os.getcwd()
extradirs = ['../', './',
             os.path.dirname(curdir),
            ]
for x in extradirs: sys.path.insert(1, x)

import numpy as np
import matplotlib.pyplot as plt
import SANDwake3D_base as sdb
import SANDwake3D_keps as SANDwake3D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from datetime import timedelta
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

def main():

    # Set up mesh and BL parameters
    nu   = 1.5E-5 # m^2/s

    # y-z mesh 
    zmin = 5
    zmax = 250
    Nz   = int((zmax-zmin)/5+1)
    zvec = np.linspace(zmin,zmax,Nz)

    Ny=51
    ymax = 125
    yvec = np.linspace(-ymax,ymax,Ny)

    # x grid
    xvecplot = [0, 10]
    ym, zm = np.meshgrid(yvec, zvec, indexing='ij')
    dy   = np.mean(np.diff(yvec))
    dz   = np.mean(np.diff(zvec))


    # Set parameters
    params = {
        'rho' : 1.225,
        'cp'  : 1005,
        'g'   : 9.81,
        'nu'  : nu,
        'Cmu' : 5.48**(-2),
        'C1eps' : 1.76,
        'C2eps' : 1.92,
        'C3eps' : 0.033,
        'sigmak' : 1.0,
        'sigmaeps' : 1.3,
        'sigmaT':1.0,
        'dtau':1.0,

        'ustar' : 0.4, #0.4, #0.575652,
        'qw': 0.0,
        'L' : float('inf'),
        'kappa' : 0.42,
        'z0' : 0.05,
        'beta':1.0/300.0,

        'zlo':zmin,
        'Tw':300,

        # Extra mesh variables needed for turbine forcing
        'ym':ym,
        'zm':zm,
        'turbforcing':{'turbx':0.0,     # Turbine x location
                       'turby':0.0,     # Turbine y location
                       'zhh':80.0,      # Turbine hub-height
                       'turbD':100.0,   # Turbine diameter
                       'Ct':0.8,
                       'Rdelta':0.5,
                       'turbfunc':sdb.tanhADM,
                      }
    }

    # This parameter defines V(z) -- m/s per meter
    veerpermeter = 0.0 #0.02

    # Set up ABL profile
    UABL = np.zeros(Nz)
    for iz, z in enumerate(zvec):
        UABL[iz] = SANDwake3D.init_U_ABL(z,params)

    # Set up veer profile
    turbhh = params['turbforcing']['zhh']
    turbR  = params['turbforcing']['turbD']*0.5
    veerV = np.zeros(Nz)
    for i, z in enumerate(zvec):
        veerV[i] = veerpermeter*(z-turbhh)

    # Compute the wind direction
    ztop = turbhh + turbR
    zbot = turbhh - turbR
    Utop, Ubot = np.interp(ztop, zvec, UABL), np.interp(zbot, zvec, UABL)
    Vtop, Vbot = np.interp(ztop, zvec, veerV), np.interp(zbot, zvec, veerV)
    Wdirtop = np.arctan2(Vtop, Utop)*180/np.pi
    Wdirbot = np.arctan2(Vbot, Ubot)*180/np.pi

    Uhh, Vhh = np.interp(turbhh, zvec, UABL), np.interp(turbhh, zvec, UABL)
    alpha = np.log(Utop/Ubot)/np.log(ztop/zbot)

    Uinit  = np.zeros((Ny, Nz))
    Vinit  = np.zeros((Ny, Nz))
    Winit  = np.zeros((Ny, Nz))
    Kinit  = np.zeros((Ny, Nz))
    Tinit  = np.ones((Ny, Nz))*300
    EPSinit= np.zeros((Ny, Nz))
    pinit  = np.zeros((Ny, Nz))

    for iz, z in enumerate(zvec):
        Uinit[:, iz] = UABL[iz] #SANDwake3D.init_U_ABL(z,ABLparam)
        Vinit[:, iz] = veerV[iz]
        Winit[:, iz] = 0.0
        #Kinit[:, iz]   = SANDwake3D.init_k_ABL(z,ABLparam) #*0.5 #0.2
        Kinit[:, iz]   = SANDwake3D.init_k_ABL(z,params)*1.0 #*0.75
        #EPSinit[:, iz] = 5.0E-3 #9.75E-4 #5.0E-4 #SANDwake3D.init_e_ABL(z,ABLparam)*0.1  #5.0E-4 is good
        EPSinit[:, iz] = SANDwake3D.init_e_ABL(z,params)
        #EPSinit[:, iz] = SANDwake3D.init_e_ABL(z,ABLparam)*0.50  #5.0E-4 is good

    # Set boundary conditions
    ubc = {}
    ubc['ylo'] = {'type':'neumann', 'value':0.0}
    ubc['yhi'] = {'type':'neumann', 'value':0.0}
    ubc['zlo'] = {'type':'bcfunc', 'value':None, 'tag':'ZLO_WALLBC',  'func':SANDwake3D.MO_wallmodel}
                  #'func':(lambda phi, aux, param: {'u':{'zlo':{'type':'dirichlet','value':Uinit[0,0]}}})}
    #ubc['zlo'] = {'type':'dirichlet', 'value':Uinit[0,0]}
    ubc['zhi'] = {'type':'dirichlet', 'value':Uinit[0,-1]}
    #ubc['zhi'] = {'type':'neumann', 'value':0.0}

    vbc = {}
    #vbc['ylo'] = {'type':'dirichlet', 'value':veerV}
    #vbc['ylo'] = {'type':'bcfunc', 'value':None, 'tag':'MyYLO_VBC', 'func':(lambda phi, aux, param, debugout=False: {'v':{'ylo':{'type':'dirichlet','value':veerV}}})}
    vbc['ylo'] = {'type':'neumann', 'value':0.0}
    vbc['yhi'] = {'type':'neumann', 'value':0.0}
    vbc['zlo'] = {'type':'neumann', 'value':0.0}
    vbc['zhi'] = {'type':'neumann', 'value':0.0}

    wbc = {}
    wbc['ylo'] = {'type':'neumann', 'value':0.0}
    wbc['yhi'] = {'type':'neumann', 'value':0.0}
    wbc['zlo'] = {'type':'dirichlet', 'value':0.0}
    wbc['zhi'] = {'type':'neumann', 'value':0.0}

    kbc = {}
    kbc['ylo'] = {'type':'neumann', 'value':0.0}
    kbc['yhi'] = {'type':'neumann', 'value':0.0}
    kbc['zlo'] = {'type':'bcfunc', 'value':None, 'tag':'ZLO_WALLBC',  'func':SANDwake3D.MO_wallmodel}
    #kbc['zlo'] = {'type':'dirichlet', 'value':Kinit[0,0]}
    kbc['zhi'] = {'type':'neumann', 'value':0.0}

    epsbc = {}
    epsbc['ylo'] = {'type':'neumann', 'value':0.0}
    epsbc['yhi'] = {'type':'neumann', 'value':0.0}
    epsbc['zlo'] = {'type':'bcfunc', 'value':None, 'tag':'ZLO_WALLBC',  'func':SANDwake3D.MO_wallmodel}
    #epsbc['zlo'] = {'type':'dirichlet', 'value':EPSinit[0,0]}
    epsbc['zhi'] = {'type':'neumann', 'value':0.0}

    Tbc = {}
    Tbc['ylo'] = {'type':'dirichlet', 'value':params['Tw']}
    Tbc['yhi'] = {'type':'dirichlet', 'value':params['Tw']}
    Tbc['zlo'] = {'type':'bcfunc', 'value':None, 'tag':'ZLO_WALLBC',  'func':SANDwake3D.MO_wallmodel}
    #Tbc['zlo'] = {'type':'dirichlet', 'value':params['Tw']}
    Tbc['zhi'] = {'type':'dirichlet', 'value':params['Tw']}

    pbc = {}
    pbc['ylo'] = {'type':'neumann', 'value':0.0}
    pbc['yhi'] = {'type':'neumann', 'value':0.0}
    pbc['zlo'] = {'type':'dirichlet', 'value':0.0}
    pbc['zhi'] = {'type':'neumann', 'value':0.0}

    allbc = {'u':ubc, 'v':vbc, 'w':wbc, 'k':kbc, 'eps':epsbc, 'T':Tbc, 'p':pbc}
        
    phiinit = {}
    phiinit['u'] = Uinit
    phiinit['v'] = Vinit
    phiinit['w'] = Winit
    phiinit['k'] = Kinit
    phiinit['T'] = Tinit
    phiinit['eps'] = EPSinit
    phiinit['p'] = pinit

    xvecplot=[-100, 0, 100, 300, 400, 500] 
    dx  = 50
    xvec = np.arange(xvecplot[0], xvecplot[-1]+1.0E-6, dx)

    start = time.time()
    phi = SANDwake3D.marchSystem(phiinit, xvec, dy, dz, params, allbc, SANDwake3D.kepsT_eqns, 
                             advanceSys=SANDwake3D.advanceSystemKEPS,
                             verbose=True, maxiter=100, tol=1.0E-4)
    end = time.time() - start
    print(f"Elapsed time {timedelta(seconds=end)} (or {end:f} seconds)")

    np.savez("phi.npz", **phi)
    phi_old = np.load("phi-bkp.npz")
    for k, v in phi_old.items():
        try:
            np.testing.assert_allclose(phi[k], v, rtol=1e-14, atol=1e-14)
        except AssertionError as e:
            raise ValueError(f"Arrays are not close:\n{e}")

    
    
if __name__ == "__main__":
    main()
    
