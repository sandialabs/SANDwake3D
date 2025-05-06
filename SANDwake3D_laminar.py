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

def applyBC(bcdict, location, dz, z=None):
    # Stencils
    Irow   = np.array([0,   1,  0])
    D1for  = np.array([0,  -1,  1])/(dz)
    D1back = np.array([-1,  1,  0])/(dz)
    Dstencil =  D1for if location=='lower' else D1back
    #value = bcdict['func'](z) if bcdict['value'] is None else bcdict['value']
    value  = bcdict['value']
    
    if bcdict['type'] == 'dirichlet':
        return Irow,  value
    elif bcdict['type'] == 'neumann':
        return Dstencil, value
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

def RHS_u_nhalf(phi_np1, phi_n, dx, dy, dz, params):
    """
    Go from n to n+1/2 for the u-momentum equation
    
    """
    u_np1, u_n = phi_np1['u'], phi_n['u']
    v_np1, v_n = phi_np1['v'], phi_n['v']
    w_np1, w_n = phi_np1['w'], phi_n['w']
    u_tilde, v_tilde, w_tilde = getTilde(phi_np1, phi_n)

    nu = params['nu']
    
    N  = u_n.shape
    Ny = N[0]
    Nz = N[1]
    RHS = np.zeros((Ny, Nz))
    # These loops can be optimized
    for i in range(Ny):
        j=0
        RHS[i,j] = u_tilde[i,j]*u_n[i,j]/(0.5*dx) - w_tilde[i,j]/dz*D1zfor(u_n, i, j) + nu*D2zfor(u_n, i, j)/(dz*dz)  
        for j in range(1,Nz-1):
            RHS[i,j] = u_tilde[i,j]*u_n[i,j]/(0.5*dx) - w_tilde[i,j]/dz*D1z(u_n, i, j) + nu*D2z(u_n, i, j)/(dz*dz)
        j=Nz-1
        RHS[i,j] = u_tilde[i,j]*u_n[i,j]/(0.5*dx) - w_tilde[i,j]/dz*D1zback(u_n, i, j) + nu*D2zback(u_n, i, j)/(dz*dz)
    return RHS

def RHS_u_np1(phi_np1, u_nhalf, phi_n, dx, dy, dz, params):
    """
    """
    u_np1, u_n = phi_np1['u'], phi_n['u']
    v_np1, v_n = phi_np1['v'], phi_n['v']
    w_np1, w_n = phi_np1['w'], phi_n['w']
    u_tilde, v_tilde, w_tilde = getTilde(phi_np1, phi_n)

    nu = params['nu']
    
    N  = u_n.shape
    Ny = N[0]
    Nz = N[1]
    RHS = np.zeros((Ny, Nz))
    # These loops can be optimized
    for j in range(Nz):
        i=0
        RHS[i,j] = u_tilde[i,j]*u_nhalf[i,j]/(0.5*dx) - v_tilde[i,j]/dy*D1yfor(u_nhalf, i, j) + nu*D2yfor(u_nhalf, i, j)/(dy*dy)
        for i in range(1,Ny-1):
            RHS[i,j] = u_tilde[i,j]*u_nhalf[i,j]/(0.5*dx) - v_tilde[i,j]/dy*D1y(u_nhalf, i, j) + nu*D2y(u_nhalf, i, j)/(dy*dy)
        i=Ny-1
        RHS[i,j] = u_tilde[i,j]*u_nhalf[i,j]/(0.5*dx) - v_tilde[i,j]/dy*D1yback(u_nhalf, i, j) + nu*D2yback(u_nhalf, i, j)/(dy*dy) 
    return RHS

def advanceU(phi_np1old, phi_n, dx, dy, dz, params, bc_ylo, bc_yhi, bc_zlo, bc_zhi):
    """
    Advance the u-momentum one full step
    """
    u_tilde, v_tilde, w_tilde = getTilde(phi_np1old, phi_n)
    
    N  = u_tilde.shape
    Ny = N[0]
    Nz = N[1]

    # Load parameters
    nu       = params['nu']
    fx_const = params['fx_const'] if 'fx_const' in params else 0.0
    
    # --- differentiation stencils ---
    #                  j-1  j  j+1
    Irow   = np.array([0,   1,  0])
    Dcen   = np.array([-1,  0,  1])*0.5
    D2cen  = np.array([1,  -2,  1])
    # -------------------------------

    # First sweep: n -> n+1/2
    # -----------------------
    u_nhalf   = np.zeros((Ny, Nz))
    RHS_nhalf = RHS_u_nhalf(phi_np1old, phi_n, dx, dy, dz, params) + fx_const
    for j in range(Nz):
        LHS_nhalf = np.zeros((Ny,3))
        # == Set up the LHS matrices ==
        for i in range(1,Ny-1):
            LHS_nhalf[i,:] = u_tilde[i,j]/(0.5*dx)*Irow + v_tilde[i,j]*Dcen/dy - nu*D2cen/(dy*dy)
        # Apply BC's
        row_lo, dentry_lo = applyBC(bc_ylo, 'lower', dy)
        row_hi, dentry_hi = applyBC(bc_yhi, 'upper', dy)
        LHS_nhalf[0,:]  = row_lo
        LHS_nhalf[-1,:] = row_hi
        RHS_nhalf[0,:]  = dentry_lo
        RHS_nhalf[-1,:] = dentry_hi
        #print(f'j = {j}\nRHS_nhalf = ',RHS_nhalf[:,j], '\nLHS = ', LHS_nhalf)        
        # Solve the triadiagonal system
        u_nhalf[:,j] = solvetridiag(LHS_nhalf, RHS_nhalf[:,j])

    #print(f'u_nhalf = ',u_nhalf)
    # Second sweep: n+1/2 -> n+1
    # -----------------------
    u_np1   = np.zeros((Ny, Nz))
    RHS_np1 = RHS_u_np1(phi_np1old, u_nhalf, phi_n, dx, dy, dz, params) + fx_const
    # == Set up the LHS matrices ==
    for i in range(Ny):
        LHS_np1 = np.zeros((Nz,3))
        for j in range(1,Nz-1):
            LHS_np1[j,:] = u_tilde[i,j]/(0.5*dx)*Irow + w_tilde[i,j]*Dcen/dz - nu*D2cen/(dz*dz) 
        # Apply BC's
        row_lo, dentry_lo = applyBC(bc_zlo, 'lower', dz)
        row_hi, dentry_hi = applyBC(bc_zhi, 'upper', dz)
        if i==0:
            row_lo, dentry_lo = applyBC({'type':'dirichlet', 'value':u_tilde[i,0]}, 'lower', dz)
        if i==Ny-1:
            row_hi, dentry_hi = applyBC({'type':'dirichlet', 'value':u_tilde[i,-1]}, 'lower', dz)            
        LHS_np1[0,:]  = row_lo
        LHS_np1[-1,:] = row_hi
        RHS_np1[:,0]  = dentry_lo
        RHS_np1[:,-1] = dentry_hi
        #print(f'i = {i}\nRHS_np1 = ',RHS_np1[i,:], '\nLHS = ', LHS_np1)                
        # Solve the triadiagonal system
        u_np1[i,:] = solvetridiag(LHS_np1, RHS_np1[i,:], verbose=False)
        #print('u_np1 = ',u_np1[i,:])
    return u_np1

def advanceW(phi_np1old, phi_n, dx, dy, dz, params, bc_ylo, bc_yhi, bc_zlo, bc_zhi):
    """
    Advance the W-momentum one full step
    """
    u_tilde, v_tilde, w_tilde = getTilde(phi_np1old, phi_n)

    N  = u_tilde.shape
    Ny = N[0]
    Nz = N[1]
    return np.zeros((Ny, Nz))


def advanceMass(phi_np1old, phi_n, dx, dy, dz, params, bc_ylo, bc_yhi, bc_zlo, bc_zhi):
    """
    Advance the continuity equation one full step
    return v^(n+1)
    """
    u_np1, u_n = phi_np1old['u'], phi_n['u']
    v_np1, v_n = phi_np1old['v'], phi_n['v']
    w_np1, w_n = phi_np1old['w'], phi_n['w']

    N  = u_n.shape
    Ny = N[0]
    Nz = N[1]

    # --- differentiation stencils ---
    #                  j-1  j  j+1
    Irow   = np.array([0,   1,  0])
    Dcen   = np.array([-1,  0,  1])*0.5
    D2cen  = np.array([1,  -2,  1])
    # -------------------------------

    v_np1   = np.zeros((Ny, Nz))

    Dz_w    = np.zeros((Ny, Nz))
    # Note: This loop can definitely be optimized
    for i in range(Ny):
        for j in range(Nz):
            if j==0:
                Dz_w[i,j] = (Dz_w[i,j+1] - Dz_w[i,j])/dz
            elif j==Nz-1:
                Dz_w[i,j] = (Dz_w[i,j] - Dz_w[i,j-1])/dz
            else:
                Dz_w[i,j] = D1z(w_np1, i, j) #0.5*(w[i,j+1] - w[i,j-1])/dz
    RHS     = -dy*(u_np1 - u_n)/dx - dy*Dz_w

    # == Set up the LHS matrices ==
    for j in range(Nz):
        v_np1[:,j] = np.cumsum(RHS[:,j])
    
    # # == Set up the LHS matrices ==
    # for j in range(Nz):
    #     LHS = np.zeros((Ny,3))
    #     # == Set up the LHS matrices ==
    #     for i in range(1,Ny-1):
    #         LHS[i,:] = Dcen
    #     # Apply BC's
    #     row_lo, dentry_lo = applyBC(bc_ylo, 'lower', dy)
    #     row_hi, dentry_hi = applyBC(bc_yhi, 'upper', dy)
    #     LHS[0,:]  = row_lo
    #     LHS[-1,:] = row_hi
    #     RHS[0,:]  = dentry_lo
    #     RHS[-1,:] = dentry_hi
    #     # Solve the triadiagonal system
    #     v_np1[:,j] = solvetridiag(LHS, RHS[:,j])
        
    return v_np1

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
        phi_next = {}
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

def marchSystem(phi_init, xvec, dy, dz, params, allbcs, eqnsys, maxiter=100,
                tol=1.0E-6, verbose=False):
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
        if verbose:
            print('x = %f'%x)
        dx = x-xprev
        phinext = advanceSystem(phiprev, dx, dy, dz, params, allbcs, eqnsys, verbose=verbose, maxiter=maxiter, tol=tol)
        for v in varlist:
            phi[v][xi+1,:,:] = phinext[v]
        phiprev = phinext.copy()
    return phi

########################################################
# Define the laminar equation system
laminar_eqns=OrderedDict()
laminar_eqns['u'] = advanceU
laminar_eqns['w'] = advanceW
laminar_eqns['v'] = advanceMass
