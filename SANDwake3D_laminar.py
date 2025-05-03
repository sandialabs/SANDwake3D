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

# Set up differentiation operators on the RHS
# ==================================
def D1y(u, i, j):
    return u[i+1, j] - u[i-1, j]

def D1z(u, i, j):
    return u[i, j+1] - u[i, j-1]

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
    value = bcdict['func'](z) if bcdict['value'] is None else bcdict['value']
    
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
    for i in range(1,Ny-1):
        for j in range(1,Nz-1):
            RHS[i,j] = u_tilde[i,j]*u_n[i,j]/(0.5*dx) - w_tilde[i,j]/dz*D1z(u_n, i, j) + nu*D2z(u_n, i, j)/(dz*dz)
    return RHS

def RHS_u_np1(phi_np1, phi_nhalf, phi_n, dx, dy, dz, params):
    """
    """
    u_nhalf = phi_nhalf['u']
    v_nhalf = phi_nhalf['v']
    w_nhalf = phi_nhalf['w']

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
    for i in range(1,Ny-1):
        for j in range(1,Nz-1):
            RHS[i,j] = u_tilde[i,j]*u_nhalf[i,j]/(0.5*dx) - v_tilde[i,j]/dy*D1y(u_nhalf, i, j) + nu*D2y(u_nhalf, i, j)/(dy*dy)
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
    nu = params['nu']
    
    # --- differentiation stencils ---
    #                  j-1  j  j+1
    Irow   = np.array([0,   1,  0])
    Dcen   = np.array([-1,  0,  1])
    D2cen  = np.array([1,  -2,  1])
    # -------------------------------

    # First sweep: n -> n+1/2
    # -----------------------
    u_nhalf   = np.zeros((Ny, Nz))
    RHS_nhalf = RHS_u_nhalf(phi_np1old, phi_n, dx, dy, dz, params)
    for j in range(Nz):
        LHS_nhalf = np.zeros((Ny,3))
        # == Set up the LHS matrices ==
        for i in range(1,Ny-1):
            LHS_nhalf[i,:] = u_tilde[i,j]/(0.5*dx) + v_tilde[i,j]*Dcen/dy + nu*D2cen/(dy*dy)
        # Apply BC's
        row_lo, dentry_lo = applyBC(bc_ylo, 'lower', dy)
        row_hi, dentry_hi = applyBC(bc_yhi, 'upper', dy)
        LHS_nhalf[0,:]  = row_lo
        LHS_nhalf[-1,:] = row_hi
        RHS_nhalf[0,:]  = dentry_lo
        RHS_nhalf[-1,:] = dentry_hi
        # Solve the triadiagonal system
        u_nhalf[:,j] = solvetridiag(LHS_nhalf, RHS_nhalf[:,j])

    # Second sweep: n+1/2 -> n+1
    # -----------------------
    u_np1   = np.zeros((Ny, Nz))
    RHS_np1 = RHS_u_np1(u_nhalf, dx, dy, alpha)
    # == Set up the LHS matrices ==
    for i in range(Ny):
        LHS_np1 = np.zeros((Nz,3))
        for i in range(1,Nz-1):
            LHS_np1[i,:] = u_tilde[i,j]/(0.5*dx) + w_tilde[i,j]*Dcen/dz + nu*D2cen/(dz*dz) 
        # Apply BC's
        row_lo, dentry_lo = applyBC(bc_zlo, 'lower', dz)
        row_hi, dentry_hi = applyBC(bc_zhi, 'upper', dz)
        LHS_np1[0,:]  = row_lo
        LHS_np1[-1,:] = row_hi
        RHS_np1[0,:]  = dentry_lo
        RHS_np1[-1,:] = dentry_hi
        # Solve the triadiagonal system
        u_np1[i,:] = solvetridiag(LHS_np1, RHS_np1[i,:], verbose=False)

    return u_np1

def advanceW(phi_np1old, phi_n, dx, dy, dz, params, bc_ylo, bc_yhi, bc_zlo, bc_zhi):
    """
    Advance the W-momentum one full step
    """
    u_tilde, v_tilde, w_tilde = getTilde(phi_np1old, phi_n)

    N  = u_tilde.shape
    Ny = N[0]
    Nz = N[1]
    return
