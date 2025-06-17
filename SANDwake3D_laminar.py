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
    u_tilde, v_tilde, w_tilde = getTildeVel(phi_np1, phi_n)

    nu = params['nu']
    
    RHS = np.empty_like(u_n)
    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz
    inv_dz2 = 1.0 / (dz * dz)
    inv_half_dx = 2.0 * inv_dx

    RHS[:, 0] = (
    u_tilde[:, 0] * u_n[:, 0] * inv_half_dx
    - w_tilde[:, 0] * inv_dz * (u_n[:, 1] - u_n[:, 0])
    + nu * (u_n[:, 0] - 2.0 * u_n[:, 1] + u_n[:, 2]) * inv_dz2
    )

    RHS[:, 1:-1] = (
        u_tilde[:, 1:-1] * u_n[:, 1:-1] * inv_half_dx
        - w_tilde[:, 1:-1] * inv_dz * 0.5 * (u_n[:, 2:] - u_n[:, :-2])
        + nu * (u_n[:, 2:] - 2.0 * u_n[:, 1:-1] + u_n[:, :-2]) * inv_dz2
    )

    RHS[:, -1] = (
        u_tilde[:, -1] * u_n[:, -1] * inv_half_dx
        - w_tilde[:, -1] * inv_dz * (u_n[:, -1] - u_n[:, -2])
        + nu * (u_n[:, -1] - 2.0 * u_n[:, -2] + u_n[:, -3]) * inv_dz2
    )
    return RHS

def RHS_u_np1(phi_np1, u_nhalf, phi_n, dx, dy, dz, params):
    """
    Go from n+1/2 to n+1 for the u-momentum equation
    """
    u_np1, u_n = phi_np1['u'], phi_n['u']
    v_np1, v_n = phi_np1['v'], phi_n['v']
    w_np1, w_n = phi_np1['w'], phi_n['w']
    u_tilde, v_tilde, w_tilde = getTildeVel(phi_np1, phi_n)

    nu = params['nu']

    RHS = np.empty_like(u_n)
    inv_dx = 1.0 / dx
    inv_half_dx = 2.0 * inv_dx
    inv_dy = 1.0 / dy
    inv_dy2 = 1.0 / (dy * dy)

    RHS[1:-1, :] = (
        u_tilde[1:-1, :] * u_nhalf[1:-1, :] * inv_half_dx
        - v_tilde[1:-1, :] * inv_dy * 0.5 * (u_nhalf[2:, :] - u_nhalf[:-2, :])
        + nu * (u_nhalf[2:, :] - 2.0 * u_nhalf[1:-1, :] + u_nhalf[:-2, :]) * inv_dy2
    )

    RHS[0, :] = (
        u_tilde[0, :] * u_nhalf[0, :] * inv_half_dx
        - v_tilde[0, :] * inv_dy * (u_nhalf[1, :] - u_nhalf[0, :])
        + nu * (u_nhalf[0, :] - 2.0 * u_nhalf[1, :] + u_nhalf[2, :]) * inv_dy2
    )

    RHS[-1, :] = (
        u_tilde[-1, :] * u_nhalf[-1, :] * inv_half_dx
        - v_tilde[-1, :] * inv_dy * (u_nhalf[-1, :] - u_nhalf[-2, :])
        + nu * (u_nhalf[-1, :] - 2.0 * u_nhalf[-2, :] + u_nhalf[-3, :]) * inv_dy2
    )
    
    return RHS

def advanceU(phi_np1old, phi_n, dx, dy, dz, params, bc_ylo, bc_yhi, bc_zlo, bc_zhi):
    """
    Advance the u-momentum one full step
    """
    u_tilde, v_tilde, w_tilde = getTildeVel(phi_np1old, phi_n)
    
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
    inv_half_dx = 2.0 / dx
    inv_dy = 1.0 / dy
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz = 1.0 / dz
    inv_dz2 = 1.0 / (dz * dz)
    row_lo, dentry_lo = applyBC(bc_ylo, 'lower', dy)
    row_hi, dentry_hi = applyBC(bc_yhi, 'upper', dy)
    for j in range(Nz):
        LHS_nhalf = np.zeros((Ny,3))
        # == Set up the LHS matrices ==
        LHS_nhalf[1:-1, :] = (
            u_tilde[1:-1, j, np.newaxis] * inv_half_dx * Irow
            + v_tilde[1:-1, j, np.newaxis] * inv_dy * Dcen
            - nu * inv_dy2 * D2cen
        )
        # Apply BC's
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
    row_lo_base, dentry_lo_base = applyBC(bc_zlo, 'lower', dz)
    row_hi_base, dentry_hi_base = applyBC(bc_zhi, 'upper', dz)
    for i in range(Ny):
        LHS_np1 = np.zeros((Nz,3))
        LHS_np1[1:-1, :] = (
            u_tilde[i, 1:-1, np.newaxis] * inv_half_dx * Irow
            + w_tilde[i, 1:-1, np.newaxis] * inv_dz * Dcen
            - nu * inv_dz2 * D2cen
        )
        # Apply BC's
        row_lo, dentry_lo = row_lo_base, dentry_lo_base
        row_hi, dentry_hi = row_hi_base, dentry_hi_base
        if i==0:
            row_lo, dentry_lo = applyBC({'type':'dirichlet', 'value':u_tilde[i,0]}, 'lower', dz)
        elif i==Ny-1:
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
    u_tilde, v_tilde, w_tilde = getTildeVel(phi_np1, phi_n)

    nu = params['nu']

    RHS = np.empty_like(u_n)
    inv_dx = 1.0 / dx
    inv_half_dx = 2.0 * inv_dx
    inv_dz = 1.0 / dz
    inv_dz2 = 1.0 / (dz * dz)

    RHS[:, 1:-1] = (
        u_tilde[:, 1:-1] * w_n[:, 1:-1] * inv_half_dx
        - w_tilde[:, 1:-1] * inv_dz * 0.5 * (w_n[:, 2:] - w_n[:, :-2])
        + nu * (w_n[:, 2:] - 2.0 * w_n[:, 1:-1] + w_n[:, :-2]) * inv_dz2
    )

    RHS[:, 0] = (
    u_tilde[:, 0] * w_n[:, 0] * inv_half_dx
    - w_tilde[:, 0] * inv_dz * (w_n[:, 1] - w_n[:, 0])
    + nu * (w_n[:, 0] - 2.0 * w_n[:, 1] + w_n[:, 2]) * inv_dz2
    )

    RHS[:, -1] = (
        u_tilde[:, -1] * w_n[:, -1] * inv_half_dx
        - w_tilde[:, -1] * inv_dz * (w_n[:, -1] - w_n[:, -2])
        + nu * (w_n[:, -1] - 2.0 * w_n[:, -2] + w_n[:, -3]) * inv_dz2
    )

    return RHS

def RHS_w_np1(phi_np1, w_nhalf, phi_n, dx, dy, dz, params):
    """
    Go from n+1/2 to n+1 for the w-momentum equation
    """
    u_np1, u_n = phi_np1['u'], phi_n['u']
    v_np1, v_n = phi_np1['v'], phi_n['v']
    w_np1, w_n = phi_np1['w'], phi_n['w']
    u_tilde, v_tilde, w_tilde = getTildeVel(phi_np1, phi_n)

    nu = params['nu']

    RHS = np.empty_like(u_n)
    inv_dx = 1.0 / dx
    inv_half_dx = 2.0 * inv_dx
    inv_dy = 1.0 / dy
    inv_dy2 = 1.0 / (dy * dy)

    RHS[1:-1, :] = (
        u_tilde[1:-1, :] * w_nhalf[1:-1, :] * inv_half_dx
        - v_tilde[1:-1, :] * inv_dy * 0.5 * (w_nhalf[2:, :] - w_nhalf[:-2, :])
        + nu * (w_nhalf[2:, :] - 2.0 * w_nhalf[1:-1, :] + w_nhalf[:-2, :]) * inv_dy2
    )

    RHS[0, :] = (
    u_tilde[0, :] * w_nhalf[0, :] * inv_half_dx
    - v_tilde[0, :] * inv_dy * (w_nhalf[1, :] - w_nhalf[0, :])
    + nu * (w_nhalf[0, :] - 2.0 * w_nhalf[1, :] + w_nhalf[2, :]) * inv_dy2
    )

    RHS[-1, :] = (
        u_tilde[-1, :] * w_nhalf[-1, :] * inv_half_dx
        - v_tilde[-1, :] * inv_dy * (w_nhalf[-1, :] - w_nhalf[-2, :])
        + nu * (w_nhalf[-1, :] - 2.0 * w_nhalf[-2, :] + w_nhalf[-3, :]) * inv_dy2
    )
    return RHS

def advanceW(phi_np1old, phi_n, dx, dy, dz, params, bc_ylo, bc_yhi, bc_zlo, bc_zhi):
    """
    Advance the W-momentum one full step
    """
    u_tilde, v_tilde, w_tilde = getTildeVel(phi_np1old, phi_n)

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
    inv_half_dx = 2.0 / dx
    inv_dy = 1.0 / dy
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz = 1.0 / dz
    inv_dz2 = 1.0 / (dz * dz)
    row_lo, dentry_lo = applyBC(bc_ylo, 'lower', dy)
    row_hi, dentry_hi = applyBC(bc_yhi, 'upper', dy)
    for j in range(Nz):
        LHS_nhalf = np.zeros((Ny,3))
        LHS_nhalf[1:-1, :] = (
            u_tilde[1:-1, j, np.newaxis] * inv_half_dx * Irow
            + v_tilde[1:-1, j, np.newaxis] * inv_dy * Dcen
            - nu * inv_dy2 * D2cen
        )
        # Apply BC's
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
    row_lo_base, dentry_lo_base = applyBC(bc_zlo, 'lower', dz)
    row_hi_base, dentry_hi_base = applyBC(bc_zhi, 'upper', dz)
    for i in range(Ny):
        LHS_np1 = np.zeros((Nz,3))
        LHS_np1[1:-1, :] = (
            u_tilde[i, 1:-1, np.newaxis] * inv_half_dx * Irow
            + w_tilde[i, 1:-1, np.newaxis] * inv_dz * Dcen
            - nu * inv_dz2 * D2cen
        )
        # Apply BC's
        row_lo, dentry_lo = row_lo_base, dentry_lo_base
        row_hi, dentry_hi = row_hi_base, dentry_hi_base
        if i==0:
            row_lo, dentry_lo = applyBC({'type':'dirichlet', 'value':w_tilde[i,0]}, 'lower', dz)
        elif i==Ny-1:
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

    # --- differentiation stencils ---
    #                  j-1  j  j+1
    Irow   = np.array([0,   1,  0])
    Dcen   = np.array([-1,  0,  1])*0.5
    D2cen  = np.array([1,  -2,  1])
    # -------------------------------

    Dz_w    = np.empty_like(u_n)
    inv_dz = 1.0 / dz
    Dz_w[:, 0] = (w_np1[:, 1] - w_np1[:, 0]) * inv_dz
    Dz_w[:, 1:-1] = 0.5 * (w_np1[:, 2:] - w_np1[:, :-2]) * inv_dz
    Dz_w[:, -1] = (w_np1[:, -1] - w_np1[:, -2]) * inv_dz
    RHS     = -dy*(u_np1 - u_n)/(dx) - dy*Dz_w

    row_lo, dentry_lo = applyBC(bc_ylo, 'lower', dy)
    v_np1 = np.cumsum(RHS, axis=0) + dentry_lo
    
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
