#!/usr/bin/env python

import numpy as np
from scipy.linalg import solve_banded
import copy
from collections import OrderedDict

def solvetridiag(matrow, b, verbose=False):
    """
    Solve tridiagonal system
    """
    lu = (1,1)
    N  = len(matrow)
    ab = np.zeros((3, N))
    ab[0,1:]  = matrow[:-1,2]
    ab[1,:]   = matrow[:,1]
    ab[2,:-1] = matrow[1:,0]
    if verbose: 
        print(ab.shape)
        print(b.shape)
    x  = solve_banded(lu, ab, b)
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

def getTilde(phi_np1, phi_n):
    """
    Get the averaged velocities for the convective term
    """
    u_np1, u_n = phi_np1['u'], phi_n['u']
    v_np1, v_n = phi_np1['v'], phi_n['v']
    w_np1, w_n = phi_np1['w'], phi_n['w']
    u_tilde = 0.5*(u_np1 + u_n)
    v_tilde = 0.5*(v_np1 + v_n)
    w_tilde = 0.5*(w_np1 + w_n)
    return u_tilde, v_tilde, w_tilde

def convergetest(phi_new, phi_old, tol):
    """
    """
    varlist = [v for v, g in phi_old.items()]
    convergevar = {}
    for v in varlist:
        convergevar[v] = np.linalg.norm(phi_new[v] - phi_old[v])
    converged = True
    for v in varlist:
        if (convergevar[v] > tol):
            converged = False
            break
    return converged, convergevar

def advanceSystem(phi_n, dx, dy, dz, params, allbcs, eqnsys, maxiter=100,
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
            phi_next[v] = eqnsys[v](phi_n1, phi_n, dx, dy, dz, params,
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
                    advanceSys=advanceSystem, maxiter=100, tol=1.0E-6, verbose=False):
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
        phinext = advanceSys(phiprev, dx, dy, dz, params, allbcs, eqnsys, verbose=verbose, maxiter=maxiter, tol=tol)
        for v in varlist:
            phi[v][xi+1,:,:] = phinext[v]
        phiprev = phinext.copy()
        xprev = x
    return phi
