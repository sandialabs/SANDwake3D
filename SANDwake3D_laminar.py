#!/usr/bin/env python

import numpy as np
from scipy.linalg import solve_banded
import copy
from collections import OrderedDict
from SANDwake3D_base import *


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
    Go from n+1/2 to n+1 for the u-momentum equation
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

def RHS_w_nhalf(phi_np1, phi_n, dx, dy, dz, params):
    """
    Go from n to n+1/2 for the w-momentum equation
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
        RHS[i,j] = u_tilde[i,j]*w_n[i,j]/(0.5*dx) - w_tilde[i,j]/dz*D1zfor(w_n, i, j) + nu*D2zfor(w_n, i, j)/(dz*dz)  
        for j in range(1,Nz-1):
            RHS[i,j] = u_tilde[i,j]*w_n[i,j]/(0.5*dx) - w_tilde[i,j]/dz*D1z(w_n, i, j) + nu*D2z(w_n, i, j)/(dz*dz)
        j=Nz-1
        RHS[i,j] = u_tilde[i,j]*w_n[i,j]/(0.5*dx) - w_tilde[i,j]/dz*D1zback(w_n, i, j) + nu*D2zback(w_n, i, j)/(dz*dz)
    return RHS

def RHS_w_np1(phi_np1, w_nhalf, phi_n, dx, dy, dz, params):
    """
    Go from n+1/2 to n+1 for the w-momentum equation
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
        RHS[i,j] = u_tilde[i,j]*w_nhalf[i,j]/(0.5*dx) - v_tilde[i,j]/dy*D1yfor(w_nhalf, i, j) + nu*D2yfor(w_nhalf, i, j)/(dy*dy)
        for i in range(1,Ny-1):
            RHS[i,j] = u_tilde[i,j]*w_nhalf[i,j]/(0.5*dx) - v_tilde[i,j]/dy*D1y(w_nhalf, i, j) + nu*D2y(w_nhalf, i, j)/(dy*dy)
        i=Ny-1
        RHS[i,j] = u_tilde[i,j]*w_nhalf[i,j]/(0.5*dx) - v_tilde[i,j]/dy*D1yback(w_nhalf, i, j) + nu*D2yback(w_nhalf, i, j)/(dy*dy) 
    return RHS

def advanceW(phi_np1old, phi_n, dx, dy, dz, params, bc_ylo, bc_yhi, bc_zlo, bc_zhi):
    """
    Advance the W-momentum one full step
    """
    u_tilde, v_tilde, w_tilde = getTilde(phi_np1old, phi_n)

    N  = u_tilde.shape
    Ny = N[0]
    Nz = N[1]

    # Load parameters
    nu       = params['nu']
    fz_const = params['fz_const'] if 'fz_const' in params else 0.0
    
    # --- differentiation stencils ---
    #                  j-1  j  j+1
    Irow   = np.array([0,   1,  0])
    Dcen   = np.array([-1,  0,  1])*0.5
    D2cen  = np.array([1,  -2,  1])
    # -------------------------------

    # First sweep: n -> n+1/2
    # -----------------------
    w_nhalf   = np.zeros((Ny, Nz))
    RHS_nhalf = RHS_w_nhalf(phi_np1old, phi_n, dx, dy, dz, params) + fz_const
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
        # Solve the triadiagonal system
        w_nhalf[:,j] = solvetridiag(LHS_nhalf, RHS_nhalf[:,j])

    # Second sweep: n+1/2 -> n+1
    # -----------------------
    w_np1   = np.zeros((Ny, Nz))
    RHS_np1 = RHS_w_np1(phi_np1old, w_nhalf, phi_n, dx, dy, dz, params) + fz_const
    # == Set up the LHS matrices ==
    for i in range(Ny):
        LHS_np1 = np.zeros((Nz,3))
        for j in range(1,Nz-1):
            LHS_np1[j,:] = u_tilde[i,j]/(0.5*dx)*Irow + w_tilde[i,j]*Dcen/dz - nu*D2cen/(dz*dz) 
        # Apply BC's
        row_lo, dentry_lo = applyBC(bc_zlo, 'lower', dz)
        row_hi, dentry_hi = applyBC(bc_zhi, 'upper', dz)
        if i==0:
            row_lo, dentry_lo = applyBC({'type':'dirichlet', 'value':w_tilde[i,0]}, 'lower', dz)
        if i==Ny-1:
            row_hi, dentry_hi = applyBC({'type':'dirichlet', 'value':w_tilde[i,-1]}, 'lower', dz)            
        LHS_np1[0,:]  = row_lo
        LHS_np1[-1,:] = row_hi
        RHS_np1[:,0]  = dentry_lo
        RHS_np1[:,-1] = dentry_hi
        #print(f'i = {i}\nRHS_np1 = ',RHS_np1[i,:], '\nLHS = ', LHS_np1)                
        # Solve the triadiagonal system
        w_np1[i,:] = solvetridiag(LHS_np1, RHS_np1[i,:], verbose=False)

    return w_np1


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
                Dz_w[i,j] = (w_np1[i,j+1] - w_np1[i,j])/dz
            elif j==Nz-1:
                Dz_w[i,j] = (w_np1[i,j] - w_np1[i,j-1])/dz
            else:
                Dz_w[i,j] = D1z(w_np1, i, j)/dz #0.5*(w[i,j+1] - w[i,j-1])/dz
    RHS     = -dy*(u_np1 - u_n)/(dx) - dy*Dz_w

    # == Set up the LHS matrices ==
    for j in range(Nz):
        row_lo, dentry_lo = applyBC(bc_ylo, 'lower', dy)
        v_np1[:,j] = np.cumsum(RHS[:,j]) + dentry_lo
    
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

########################################################
# Define the laminar equation system
laminar_eqns=OrderedDict()
laminar_eqns['u'] = advanceU
laminar_eqns['w'] = advanceW
laminar_eqns['v'] = advanceMass

# Use the same marchSystemBase in SANDWake3D_base to advance the equations
marchSystem = marchSystemBase
