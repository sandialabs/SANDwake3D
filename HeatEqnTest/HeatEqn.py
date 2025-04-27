#!/usr/bin/env python

import numpy as np
from scipy.linalg import solve_banded

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

def RHS_u_nhalf(u, dx, delta, alpha):
    """
    Go from n to n+1/2
    """
    N = u.shape
    Ny = N[0]
    Nz = N[1]
    RHS = np.zeros((Ny, Nz))
    for i in range(1,Ny-1):
        for j in range(1,Nz-1):
            RHS[i,j] = delta*delta/alpha*u[i,j] + 0.5*dx*(u[i,j+1] -2.0*u[i,j] + u[i,j-1])
    return RHS

def RHS_u_np1(u, dx, delta, alpha):
    """
    Go from n+1/2 to n+1
    """
    N = u.shape
    Ny = N[0]
    Nz = N[1]
    RHS = np.zeros((Ny, Nz))
    for i in range(1,Ny-1):
        for j in range(1,Nz-1):
            RHS[i,j] = delta*delta/alpha*u[i,j] + 0.5*dx*(u[i+1,j] -2.0*u[i,j] + u[i-1,j])
    return RHS

def applyBC(bcdict, location, dz):
    # Stencils
    Irow   = np.array([0,   1,  0])
    D1for  = np.array([0,  -1,  1])/(dz)
    D1back = np.array([-1,  1,  0])/(dz)
    Dstencil =  D1for if location=='lower' else D1back
    if bcdict['type'] == 'dirichlet':
        return Irow,  bcdict['value']
    elif bcdict['type'] == 'neumann':
        return Dstencil, bcdict['value']
    return None

def advanceU(u, dx, dy, dz, alpha, bc_ylo, bc_yhi, bc_zlo, bc_zhi):
    """
    Advance the heat equation one step
    """
    N = u.shape
    Ny = N[0]
    Nz = N[1]
    
    # --- differentiation stencils ---
    #                  j-1  j  j+1
    Irow   = np.array([0,   1,  0])
    Dcen   = np.array([-1,  0,  1])
    D2cen  = np.array([1,  -2,  1])
    # -------------------------------

    # First sweep: n -> n+1/2
    # -----------------------
    u_nhalf   = np.zeros((Ny, Nz))
    RHS_nhalf = RHS_u_nhalf(u, dx, dz, alpha)
    LHS_nhalf = np.zeros((Ny,3))
    # == Set up the LHS matrices ==
    for i in range(1,Ny-1):
        LHS_nhalf[i,:] = Irow*(dz*dz/alpha) - 0.5*dx*D2cen

    # Apply BC's
    row_lo, dentry_lo = applyBC(bc_zlo, 'lower', dz)
    row_hi, dentry_hi = applyBC(bc_zhi, 'upper', dz)
    LHS_nhalf[0,:]  = row_lo
    LHS_nhalf[-1,:] = row_hi
    RHS_nhalf[0,:]  = dentry_lo
    RHS_nhalf[-1,:] = dentry_hi
    # Solve the triadiagonal system
    for j in range(Nz):
        u_nhalf[:,j] = solvetridiag(LHS_nhalf, RHS_nhalf[:,j])


    # Second sweep: n+1/2 -> n+1
    # -----------------------
    u_np1   = np.zeros((Ny, Nz))
    RHS_np1 = RHS_u_np1(u_nhalf, dx, dy, alpha)
    LHS_np1 = np.zeros((Nz,3))
    # == Set up the LHS matrices ==
    for i in range(1,Nz-1):
        LHS_np1[i,:] = Irow*(dy*dy/alpha) - 0.5*dx*D2cen
        
    # Apply BC's
    row_lo, dentry_lo = applyBC(bc_ylo, 'lower', dy)
    row_hi, dentry_hi = applyBC(bc_yhi, 'upper', dy)
    LHS_np1[0,:]  = row_lo
    LHS_np1[-1,:] = row_hi
    RHS_np1[:,0]  = dentry_lo
    RHS_np1[:,-1] = dentry_hi
    #print(RHS_np1.shape)
    # Solve the triadiagonal system
    for i in range(Ny):
        u_np1[i,:] = solvetridiag(LHS_np1, RHS_np1[i,:], verbose=False)

    return u_np1

def marchHeatEqn(uinit, xvec, dy, dz, alpha, bc_ylo, bc_yhi, bc_zlo, bc_zhi, verbose=False):
    """
    """
    Nx = len(xvec)
    Usol = []
    Usol.append(uinit)
    xprev = xvec[0]
    uprev = uinit
    Nx = len(xvec[1:])
    for ix, x in enumerate(xvec[1:]):
        dx = x-xprev
        if verbose:
            print(f"[{ix+1}/{Nx}] x = {x}")
        Ustep = advanceU(uprev, dx, dy, dz, alpha, bc_ylo, bc_yhi, bc_zlo, bc_zhi)
        
        # Store the solution
        xprev = x
        uprev = Ustep
        Usol.append(Ustep)
    return Usol
