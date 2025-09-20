#!/usr/bin/env python

import numpy as np
from scipy.linalg.lapack import dgbsv
import copy
from collections import OrderedDict
import yaml
import inspect
import sys

def solvetridiag(matrow, b, verbose=False):
    """
    Solve tridiagonal system
    """
    N  = len(matrow)

    kl = 1
    ku = 1
    ldab = 2 * kl + ku + 1
    ab = np.zeros((ldab, N))
    ab[1, 1:] = matrow[:-1, 2]
    ab[2, :] = matrow[:, 1]
    ab[3, :-1] = matrow[1:, 0]
    if verbose:
        print(ab.shape)
        print(b.shape)
    lu, piv, x, info = dgbsv(kl, ku, ab, b, overwrite_ab=0, overwrite_b=0)
    if info != 0:
        raise RuntimeError(f"dgbsv failed with info = {info}")

    # We could use this instead (many more checks) but it is slower
    # ab = np.zeros((3, N))
    # ab[0,1:]  = matrow[:-1,2]
    # ab[1,:]   = matrow[:,1]
    # ab[2,:-1] = matrow[1:,0]
    # x  = solve_banded((kl, ku), ab, b)

    return x

# Set up differentiation operators on the RHS
# ==================================
# first derivative
def D1yfor(u, i, j):
    return (u[i+1, j] - u[i, j])

def D1zfor(u, i, j):
    return (u[i, j+1] - u[i, j])

def D1yback(u, i, j):
    return (u[i, j] - u[i-1, j])

def D1zback(u, i, j):
    return (u[i, j] - u[i, j-1])

def D1y(u, i, j):
    return 0.5*(u[i+1, j] - u[i-1, j])

def D1z(u, i, j):
    return 0.5*(u[i, j+1] - u[i, j-1])

# second derivative
def D2yfor(u, i, j):
    return u[i, j] - 2.0*u[i+1, j] + u[i+2, j]

def D2zfor(u, i, j):
    return u[i, j] - 2.0*u[i, j+1] + u[i, j+2]

def D2yback(u, i, j):
    return u[i, j] - 2.0*u[i-1, j] + u[i-2, j]

def D2zback(u, i, j):
    return u[i, j] - 2.0*u[i, j-1] + u[i, j-2]

def D2y(u, i, j):
    return u[i+1, j] - 2.0*u[i, j] + u[i-1, j]

def D2z(u, i, j):
    return u[i, j+1] - 2.0*u[i, j] + u[i, j-1]
# ==================================

def applyBC(bcdict, location, dz):
    # Stencils
    if bcdict['type'] == 'dirichlet':
        return np.array([0,   1,  0]),  bcdict['value']
    elif bcdict['type'] == 'neumann':
        Dstencil =  np.array([0,  -1,  1])/(dz) if location=='lower' else np.array([-1,  1,  0])/(dz)
        return Dstencil, bcdict['value']
    return None

def getTildeVel(phi_np1, phi_n):
    """
    Get the averaged velocities for the convective term
    """
    phi_tilde = getTildeVars(phi_np1, phi_n)
    return phi_tilde['u'], phi_tilde['v'], phi_tilde['w']

def getTildeVars(phi_np1, phi_n):
    """
    Get the averaged quantities for the convective term
    """
    phiTilde = {}
    for v in phi_n:
        phiTilde[v] = 0.5*(phi_np1[v] + phi_n[v])
    return phiTilde

def convergetest(phi_new, phi_old, tol, testvars=None):
    """
    """
    if testvars is None:
        varlist = [v for v, g in phi_new.items()]
    else:
        varlist = testvars
    convergevar = {}
    for v in varlist:
        convergevar[v] = np.linalg.norm(phi_new[v] - phi_old[v])
    converged = True
    for v in varlist:
        if (convergevar[v] > tol):
            converged = False
            break
    return converged, convergevar

def advanceSystem(phi_n, x, dx, dy, dz, params, allbcs, eqnsys, maxiter=100,
                  tol=1.0E-6, verbose=False):
    """
    Advance equation system 1 step in x
    """
    varlist = [v for v, g in eqnsys.items()]

    phi_n1 = copy.deepcopy(phi_n)

    for k in range(maxiter):
        phi_next = OrderedDict()
        # Loop over all variables
        for v in varlist:
            bcvar = allbcs[v]
            phi_next[v] = eqnsys[v](phi_n1, phi_n, x, dx, dy, dz, params,
                                    bcvar['ylo'], bcvar['yhi'], bcvar['zlo'], bcvar['zhi'])
        # Test for convergence
        converged, convergedat = convergetest(phi_next, phi_n1, tol)
        phi_n1 = copy.deepcopy(phi_next)
        if verbose:
            print(k, convergedat)
        if converged:
            break
    # Check if k hit maxiter:
    # -->TODO!
    return phi_n1

def marchSystemBase(phi_init, xvec, dy, dz, params, allbcs, eqnsys,
                    advanceSys=advanceSystem, maxiter=100, tol=1.0E-6, verbose=False,
                    postadvfunc=None):
    """
    March the system in x according xvec
    """
    # Get the list of variables
    varlist = [v for v, g in eqnsys.items()]
    # Get the shape of the vectors
    N  = phi_init[varlist[0]].shape
    Ny = N[0]
    Nz = N[1]
    Nx = len(xvec)
    
    # Populate the storage dict
    phi = {}
    for v in varlist:
        phi[v] = np.zeros((Nx, Ny, Nz))
    # Initialize phi with phi_init
    for v in varlist:
        phi[v][0,:,:] = phi_init[v]

    xprev = xvec[0]
    phiprev = phi_init
    # Start the march
    for xi, x in enumerate(xvec[1:]):
        dx = x-xprev
        if verbose:
            print(f'x = {x} dx = {dx}')
        phinext = advanceSys(phiprev, x, dx, dy, dz, params, allbcs, eqnsys, verbose=verbose, maxiter=maxiter, tol=tol)
        for v in varlist:
            phi[v][xi+1,:,:] = phinext[v]
        phiprev = phinext.copy()
        xprev = x
        if postadvfunc is not None:
            postadvfunc(xi+1, phi, params)
    return phi

############################################
########## TURBINE MODEL ROUTINES ##########
############################################

def computeTurbineForces(x, dx, ym, zm, phi_n, tparams, verbose=0):
    """
    Compute turbine model forces from a list of turbines
    """
    debugout = {}  # Any turbine debugging output goes here
    eps = 1.0E-5
    fx = np.zeros_like(phi_n['u'])
    fy = np.zeros_like(phi_n['u'])
    fz = np.zeros_like(phi_n['u'])

    for iturb, turbdict in enumerate(tparams):
        xturb    = turbdict['turbx']
        if np.abs(x-dx+eps <= xturb) and (xturb < x+eps):
            name = turbdict['name'] if 'name' in turbdict else 'T'+repr(iturb)
            zhh  = turbdict['zhh']
            yhh  = turbdict['turby']
            R    = turbdict['turbD']*0.5
            Uinf = rotorAvgUh(ym, zm, phi_n['u'], phi_n['v'], yhh, zhh, R)
            # Compute the power
            turbdict['power'] = 0.0    # --> TO-DO!

            # Compute the forces
            if verbose: print(f'Computing forces for turbine {name}')
            turbADfunc = turbdict['turbfunc']
            if inspect.isfunction(turbADfunc):
                func     = turbADfunc
            else:
                modname  = turbADfunc.split('.')[0]
                funcname = turbADfunc.split('.')[1]
                func     = getattr(sys.modules[modname], funcname)
            fdict = func(dx, ym, zm, Uinf, phi_n, turbdict)

            fx += fdict['u']
            fy += fdict['v']
            fz += fdict['w']

    return {'u':fx, 'v':fy, 'w':fz}, debugout

def rotorAvgUh(ym, zm, u, v, yhh, zhh, R):
    """
    Compute the rotor average velocity
    """
    Uh = np.sqrt(u**2 + v**2)
    maskoutside = ((zm-zhh)**2 + (ym-yhh)**2 > R**2)
    masked_vel  = np.ma.array(Uh, mask=maskoutside)
    return masked_vel.mean()

def CtTableLookup(Uinf, params):
    """
    Return the Ct value based on tables or file inputs from the params dictionary
    Options:
    (A) params['CtCpSource']=='FlorisFile'
        ==> pull Ct from FLORIS definition in params['turbinefilename']
    (B) params['CtCpSource']!='FlorisFile'
        if params['Ct'] is scalar, return that
        if params['Ct'] is list, return interpolated value from params['WS'] and params['Ct'] 
    """
    if ('CtCpSource' in params) and (params['CtCpSource']=='FlorisFile'):
        turbinefile=params['turbinefilename']
        # Extract wind speed / Ct data from yaml file
        with open(turbinefile, 'r') as file:
            data = yaml.safe_load(file)
        wind_speeds = data['power_thrust_table']['wind_speed']
        thrust_coefficients = data['power_thrust_table']['thrust_coefficient']
    else:
        thrust_coefficients = params['Ct']
        # If it's a scalar value, just return that
        if isinstance(thrust_coefficients, float) or isinstance(thrust_coefficients, int):
            return thrust_coefficients
        wind_speeds = params['WS']
    return np.interp(Uinf, wind_speeds, thrust_coefficients, left=0.0, right=0.0)
    

def tanhADM(dx, y,z, Uinf, phi, params):
    """
    A uniformly loaded actuator disk
    """
    zhh    = params['zhh']
    yhh    = params['turby']
    Rdelta = params['Rdelta']
    turbR  = params['turbD']*0.5
    Ct     = CtTableLookup(Uinf, params)
    r  = np.sqrt((y-yhh)**2 + (z-zhh)**2)
    Ulocal = np.sqrt(phi['u']**2 + phi['v']**2)
    F1 = 0.0                      # Force at infinity (should be zero)
    F0 = (0.5*Ct*Ulocal**2)/dx      # Force on disk
    Fr = 0.5*(F1-F0)*(1.0 + np.tanh((r-turbR)/Rdelta)) + F0
    return -Fr 

def UnifCtADM(dx, y,z, Uinf, phi, params):
    """
    A uniformly loaded actuator disk
    """
    zhh    = params['zhh']
    yhh    = params['turby']
    Rdelta = params['Rdelta']
    turbR  = params['turbD']*0.5
    turbnormal = params['turbnormal'] if 'turbnormal' in params else [-1.0, 0.0, 0.0]
    Ct     = CtTableLookup(Uinf, params)
    r      = np.sqrt((y-yhh)**2 + (z-zhh)**2)
    Ulocal = np.sqrt(phi['u']**2 + phi['v']**2)
    Fdisk  = (0.5*Ct*Ulocal**2)/dx      # Force on disk
    Faxial = Fdisk - 0.5*Fdisk*(1.0 + np.tanh((r-turbR)/Rdelta))
    Fx     = turbnormal[0]*Faxial
    Fy     = turbnormal[1]*Faxial
    Fz     = turbnormal[2]*Faxial
    return {'u':Fx, 'v':Fy, 'w':Fz}

