#!/usr/bin/env python

import numpy as np
from scipy.linalg import solve_banded
import copy
from collections import OrderedDict
from SANDwake3D_base import *
from functools import partial


def RHS_f_nhalf(phi_np1, phi_n, phi_tilde, Dz_nuT_tilde, dx, dy, dz, params, field):
    """
    Go from n to n+1/2
    """
    vel_n = phi_n[field]
    u_tilde, w_tilde = phi_tilde['u'], phi_tilde['w']
    nu_total = params['nu'] + phi_tilde['nuT']
    w_total  = w_tilde - Dz_nuT_tilde

    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz
    inv_dz2 = 1.0 / (dz * dz)
    inv_half_dx = 2.0 * inv_dx

    RHS = np.empty_like(vel_n)
    RHS[:, 0] = (
        u_tilde[:, 0] * vel_n[:, 0] * inv_half_dx
        - w_total[:, 0] * inv_dz * (vel_n[:, 1] - vel_n[:, 0])
        + nu_total[:, 0] * inv_dz2 * (vel_n[:, 0] - 2.0 * vel_n[:, 1] + vel_n[:, 2])
    )

    RHS[:, 1:-1] = (
        u_tilde[:, 1:-1] * vel_n[:, 1:-1] * inv_half_dx
        - w_total[:, 1:-1] * inv_dz * 0.5 * (vel_n[:, 2:] - vel_n[:, :-2])
        + nu_total[:, 1:-1] * inv_dz2 * (vel_n[:, 2:] - 2.0 * vel_n[:, 1:-1] + vel_n[:, :-2])
    )

    RHS[:, -1] = (
        u_tilde[:, -1] * vel_n[:, -1] * inv_half_dx
        - w_total[:, -1] * inv_dz * (vel_n[:, -1] - vel_n[:, -2])
        + nu_total[:, -1] * inv_dz2 * (vel_n[:, -1] - 2.0 * vel_n[:, -2] + vel_n[:, -3])
    )

    return RHS

def RHS_f_np1(f_nhalf, phi_np1, phi_n, phi_tilde, Dy_nuT_tilde, dx, dy, dz, params):
    """
    Go from n+1/2 to n+1 for the u-momentum equation
    """
    u_tilde, v_tilde = phi_tilde['u'], phi_tilde['v']
    nu_total = params['nu'] + phi_tilde['nuT']
    v_total  = v_tilde - Dy_nuT_tilde

    inv_half_dx = 2.0 / dx
    inv_dy = 1.0 / dy
    inv_dy2 = 1.0 / (dy * dy)

    RHS = np.empty_like(f_nhalf)
    RHS[0, :] = (
    u_tilde[0, :] * f_nhalf[0, :] * inv_half_dx
    - v_total[0, :] * inv_dy * (f_nhalf[1, :] - f_nhalf[0, :])
    + nu_total[0, :] * inv_dy2 * (f_nhalf[0, :] - 2.0 * f_nhalf[1, :] + f_nhalf[2, :])
    )

    RHS[1:-1, :] = (
        u_tilde[1:-1, :] * f_nhalf[1:-1, :] * inv_half_dx
        - v_total[1:-1, :] * inv_dy * 0.5 * (f_nhalf[2:, :] - f_nhalf[:-2, :])
        + nu_total[1:-1, :] * inv_dy2 * (f_nhalf[2:, :] - 2.0 * f_nhalf[1:-1, :] + f_nhalf[:-2, :])
    )

    RHS[-1, :] = (
        u_tilde[-1, :] * f_nhalf[-1, :] * inv_half_dx
        - v_total[-1, :] * inv_dy * (f_nhalf[-1, :] - f_nhalf[-2, :])
        + nu_total[-1, :] * inv_dy2 * (f_nhalf[-1, :] - 2.0 * f_nhalf[-2, :] + f_nhalf[-3, :])
    )
    return RHS

def advanceF(phi_np1old, phi_n, dx, dy, dz, params, bc_ylo, bc_yhi, bc_zlo, bc_zhi, field):
    """
    Advance the field one full step
    """
    phi_tilde = getTildeVars(phi_np1old, phi_n)
    u_tilde, v_tilde, w_tilde = phi_tilde['u'], phi_tilde['v'], phi_tilde['w']
    nuT_tilde = phi_tilde['nuT']
    
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

    # Compute some quantities related to nuT
    Dy_nuT, Dz_nuT = np.gradient(nuT_tilde, dy, dz, edge_order=1)

    nu_total = nu + nuT_tilde
    v_total  = v_tilde - Dy_nuT
    w_total  = w_tilde - Dz_nuT

    # First sweep: n -> n+1/2
    # -----------------------
    f_nhalf   = np.zeros((Ny, Nz))
    RHS_nhalf = RHS_f_nhalf(phi_np1old, phi_n, phi_tilde, Dz_nuT, dx, dy, dz, params, field) + fx_const
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
            + v_total[1:-1, j, np.newaxis] * inv_dy * Dcen
            - nu_total[1:-1, j, np.newaxis] * inv_dy2 * D2cen
        )
        # Apply BC's
        LHS_nhalf[0,:]  = row_lo
        LHS_nhalf[-1,:] = row_hi
        RHS_nhalf[0,:]  = dentry_lo
        RHS_nhalf[-1,:] = dentry_hi
        #print(f'j = {j}\nRHS_nhalf = ',RHS_nhalf[:,j], '\nLHS = ', LHS_nhalf)        
        # Solve the triadiagonal system
        f_nhalf[:,j] = solvetridiag(LHS_nhalf, RHS_nhalf[:,j])

    #print(f'f_nhalf = ',f_nhalf)
    # Second sweep: n+1/2 -> n+1
    # -----------------------
    f_np1   = np.zeros((Ny, Nz))
    RHS_np1 = RHS_f_np1(f_nhalf, phi_np1old, phi_n, phi_tilde, Dy_nuT, dx, dy, dz, params) + fx_const
    # == Set up the LHS matrices ==
    row_lo_base, dentry_lo_base = applyBC(bc_zlo, 'lower', dz)
    row_hi_base, dentry_hi_base = applyBC(bc_zhi, 'upper', dz)
    for i in range(Ny):
        LHS_np1 = np.zeros((Nz,3))
        LHS_np1[1:-1, :] = (
            u_tilde[i, 1:-1, np.newaxis] * inv_half_dx * Irow
            + w_total[i, 1:-1, np.newaxis] * inv_dz * Dcen
            - nu_total[i, 1:-1, np.newaxis] * inv_dz2 * D2cen
        )
        # Apply BC's
        row_lo, dentry_lo = row_lo_base, dentry_lo_base
        row_hi, dentry_hi = row_hi_base, dentry_hi_base
        if i==0:
            row_lo, dentry_lo = applyBC({'type':'dirichlet', 'value':phi_tilde[field][i,0]}, 'lower', dz)
        elif i==Ny-1:
            row_hi, dentry_hi = applyBC({'type':'dirichlet', 'value':phi_tilde[field][i,-1]}, 'lower', dz)            
        LHS_np1[0,:]  = row_lo
        LHS_np1[-1,:] = row_hi
        RHS_np1[:,0]  = dentry_lo
        RHS_np1[:,-1] = dentry_hi
        #print(f'i = {i}\nRHS_np1 = ',RHS_np1[i,:], '\nLHS = ', LHS_np1)                
        # Solve the triadiagonal system
        f_np1[i,:] = solvetridiag(LHS_np1, RHS_np1[i,:], verbose=False)
        #print('f_np1 = ',f_np1[i,:])
    return f_np1

def RHS_tke_nhalf(phi_np1, phi_n, phi_tilde, Dz_nuT_tilde, dx, dy, dz, params):
    """
    Go from n to n+1/2 for the TKE equation
    """
    k_n = phi_n['k']
    u_tilde, v_tilde, w_tilde = phi_tilde['u'], phi_tilde['v'], phi_tilde['w']
    nuT_tilde = phi_tilde['nuT']

    nu     = params['nu']
    sigmak = params['sigmak']

    N  = u_tilde.shape
    Ny = N[0]
    Nz = N[1]
    RHS = np.zeros((Ny, Nz))

    nu_total = nu + nuT_tilde/sigmak
    w_total  = w_tilde - Dz_nuT_tilde/sigmak

    # These loops can be optimized
    for i in range(Ny):
        j=0
        RHS[i,j] = u_tilde[i,j]*k_n[i,j]/(0.5*dx) - w_total[i,j]/dz*D1zfor(k_n, i, j) + nu_total[i,j]*D2zfor(k_n, i, j)/(dz*dz)  
        for j in range(1,Nz-1):
            RHS[i,j] = u_tilde[i,j]*k_n[i,j]/(0.5*dx) - w_total[i,j]/dz*D1z(k_n, i, j) + nu_total[i,j]*D2z(k_n, i, j)/(dz*dz)
        j=Nz-1
        RHS[i,j] = u_tilde[i,j]*k_n[i,j]/(0.5*dx) - w_total[i,j]/dz*D1zback(k_n, i, j) + nu_total[i,j]*D2zback(k_n, i, j)/(dz*dz)
    return RHS

def RHS_tke_np1(tke_nhalf, phi_np1, phi_n, phi_tilde, Dy_nuT_tilde, dx, dy, dz, params):
    """
    Go from n+1/2 to n+1 for the TKE equation
    """
    u_np1, u_n = phi_np1['u'], phi_n['u']
    v_np1, v_n = phi_np1['v'], phi_n['v']
    w_np1, w_n = phi_np1['w'], phi_n['w']
    u_tilde, v_tilde, w_tilde = phi_tilde['u'], phi_tilde['v'], phi_tilde['w']
    nuT_tilde = phi_tilde['nuT']

    nu = params['nu']
    sigmak = params['sigmak']

    N  = u_n.shape
    Ny = N[0]
    Nz = N[1]
    RHS = np.zeros((Ny, Nz))
    
    nu_total = nu + nuT_tilde/sigmak
    v_total  = v_tilde - Dy_nuT_tilde/sigmak

    # These loops can be optimized
    for j in range(Nz):
        i=0
        RHS[i,j] = u_tilde[i,j]*tke_nhalf[i,j]/(0.5*dx) - v_total[i,j]/dy*D1yfor(tke_nhalf, i, j) + nu_total[i,j]*D2yfor(tke_nhalf, i, j)/(dy*dy)
        for i in range(1,Ny-1):
            RHS[i,j] = u_tilde[i,j]*tke_nhalf[i,j]/(0.5*dx) - v_total[i,j]/dy*D1y(tke_nhalf, i, j) + nu_total[i,j]*D2y(tke_nhalf, i, j)/(dy*dy)
        i=Ny-1
        RHS[i,j] = u_tilde[i,j]*tke_nhalf[i,j]/(0.5*dx) - v_total[i,j]/dy*D1yback(tke_nhalf, i, j) + nu_total[i,j]*D2yback(tke_nhalf, i, j)/(dy*dy) 
    return RHS

def advanceTKE(phi_np1old, phi_n, dx, dy, dz, params, bc_ylo, bc_yhi, bc_zlo, bc_zhi):
    """
    Advance the TKE equation one full step
    """
    phi_tilde = getTildeVars(phi_np1old, phi_n)
    u_tilde, v_tilde, w_tilde = phi_tilde['u'], phi_tilde['v'], phi_tilde['w']
    nuT_tilde = phi_tilde['nuT']
    k_tilde = phi_tilde['k']
    eps_tilde = phi_tilde['eps']

    N  = u_tilde.shape
    Ny = N[0]
    Nz = N[1]

    # Load parameters
    nu       = params['nu']
    sigmak   = params['sigmak']
    fk_const = params['fk_const'] if 'fk_const' in params else 0.0
    
    # --- differentiation stencils ---
    #                  j-1  j  j+1
    Irow   = np.array([0,   1,  0])
    Dcen   = np.array([-1,  0,  1])*0.5
    D2cen  = np.array([1,  -2,  1])
    # -------------------------------

    # Compute some quantities
    Dy_nuT, Dz_nuT = np.gradient(nuT_tilde, dy, dz, edge_order=1)
    Dy_U, Dz_U = np.gradient(u_tilde, dy, dz, edge_order=1)

    nu_total = nu + nuT_tilde/sigmak
    v_total  = v_tilde - Dy_nuT/sigmak
    w_total  = w_tilde - Dz_nuT/sigmak

    RHS_extra_forcing = nuT_tilde*(Dy_U*Dy_U + Dz_U*Dz_U) - eps_tilde + fk_const
    
    # First sweep: n -> n+1/2
    # -----------------------
    tke_nhalf   = np.zeros((Ny, Nz))
    RHS_nhalf = RHS_tke_nhalf(phi_np1old, phi_n, phi_tilde, Dz_nuT, dx, dy, dz, params) + RHS_extra_forcing
    for j in range(Nz):
        LHS_nhalf = np.zeros((Ny,3))
        # == Set up the LHS matrices ==
        for i in range(1,Ny-1):
            LHS_nhalf[i,:] = u_tilde[i,j]/(0.5*dx)*Irow + v_total[i,j]*Dcen/dy - nu_total[i,j]*D2cen/(dy*dy)
        # Apply BC's
        row_lo, dentry_lo = applyBC(bc_ylo, 'lower', dy)
        row_hi, dentry_hi = applyBC(bc_yhi, 'upper', dy)
        LHS_nhalf[0,:]  = row_lo
        LHS_nhalf[-1,:] = row_hi
        RHS_nhalf[0,:]  = dentry_lo
        RHS_nhalf[-1,:] = dentry_hi
        # Solve the triadiagonal system
        tke_nhalf[:,j] = solvetridiag(LHS_nhalf, RHS_nhalf[:,j])

    # Second sweep: n+1/2 -> n+1
    # -----------------------
    tke_np1   = np.zeros((Ny, Nz))
    RHS_np1 = RHS_tke_np1(tke_nhalf, phi_np1old, phi_n, phi_tilde, Dy_nuT, dx, dy, dz, params) + RHS_extra_forcing
    # == Set up the LHS matrices ==
    for i in range(Ny):
        LHS_np1 = np.zeros((Nz,3))
        for j in range(1,Nz-1):
            LHS_np1[j,:] = u_tilde[i,j]/(0.5*dx)*Irow + w_total[i,j]*Dcen/dz - nu_total[i,j]*D2cen/(dz*dz) 
        # Apply BC's
        row_lo, dentry_lo = applyBC(bc_zlo, 'lower', dz)
        row_hi, dentry_hi = applyBC(bc_zhi, 'upper', dz)
        if i==0:
            row_lo, dentry_lo = applyBC({'type':'dirichlet', 'value':k_tilde[i,0]}, 'lower', dz)
        if i==Ny-1:
            row_hi, dentry_hi = applyBC({'type':'dirichlet', 'value':k_tilde[i,-1]}, 'lower', dz)            
        LHS_np1[0,:]  = row_lo
        LHS_np1[-1,:] = row_hi
        RHS_np1[:,0]  = dentry_lo
        RHS_np1[:,-1] = dentry_hi
        #print(f'i = {i}\nRHS_np1 = ',RHS_np1[i,:], '\nLHS = ', LHS_np1)                
        # Solve the triadiagonal system
        tke_np1[i,:] = solvetridiag(LHS_np1, RHS_np1[i,:], verbose=False)

    return tke_np1

def RHS_eps_nhalf(phi_np1, phi_n, phi_tilde, Dz_nuT_tilde, dx, dy, dz, params):
    """
    Go from n to n+1/2 for the EPS equation
    """
    eps_n     = phi_n['eps']
    u_tilde, v_tilde, w_tilde = phi_tilde['u'], phi_tilde['v'], phi_tilde['w']
    nuT_tilde = phi_tilde['nuT']

    nu     = params['nu']
    sigmaeps = params['sigmaeps']

    N  = u_tilde.shape
    Ny = N[0]
    Nz = N[1]
    RHS = np.zeros((Ny, Nz))

    nu_total = nu + nuT_tilde/sigmaeps
    w_total  = w_tilde - Dz_nuT_tilde/sigmaeps

    # These loops can be optimized
    for i in range(Ny):
        j=0
        RHS[i,j] = u_tilde[i,j]*eps_n[i,j]/(0.5*dx) - w_total[i,j]/dz*D1zfor(eps_n, i, j) + nu_total[i,j]*D2zfor(eps_n, i, j)/(dz*dz)  
        for j in range(1,Nz-1):
            RHS[i,j] = u_tilde[i,j]*eps_n[i,j]/(0.5*dx) - w_total[i,j]/dz*D1z(eps_n, i, j) + nu_total[i,j]*D2z(eps_n, i, j)/(dz*dz)
        j=Nz-1
        RHS[i,j] = u_tilde[i,j]*eps_n[i,j]/(0.5*dx) - w_total[i,j]/dz*D1zback(eps_n, i, j) + nu_total[i,j]*D2zback(eps_n, i, j)/(dz*dz)
    return RHS

def RHS_eps_np1(eps_nhalf, phi_np1, phi_n, phi_tilde, Dy_nuT_tilde, dx, dy, dz, params):
    """
    Go from n+1/2 to n+1 for the EPS equation
    """
    u_np1, u_n = phi_np1['u'], phi_n['u']
    v_np1, v_n = phi_np1['v'], phi_n['v']
    w_np1, w_n = phi_np1['w'], phi_n['w']
    u_tilde, v_tilde, w_tilde = phi_tilde['u'], phi_tilde['v'], phi_tilde['w']
    nuT_tilde = phi_tilde['nuT']

    nu = params['nu']
    sigmaeps = params['sigmaeps']

    N  = u_n.shape
    Ny = N[0]
    Nz = N[1]
    RHS = np.zeros((Ny, Nz))
    
    nu_total = nu + nuT_tilde/sigmaeps
    v_total  = v_tilde - Dy_nuT_tilde/sigmaeps

    # These loops can be optimized
    for j in range(Nz):
        i=0
        RHS[i,j] = u_tilde[i,j]*eps_nhalf[i,j]/(0.5*dx) - v_total[i,j]/dy*D1yfor(eps_nhalf, i, j) + nu_total[i,j]*D2yfor(eps_nhalf, i, j)/(dy*dy)
        for i in range(1,Ny-1):
            RHS[i,j] = u_tilde[i,j]*eps_nhalf[i,j]/(0.5*dx) - v_total[i,j]/dy*D1y(eps_nhalf, i, j) + nu_total[i,j]*D2y(eps_nhalf, i, j)/(dy*dy)
        i=Ny-1
        RHS[i,j] = u_tilde[i,j]*eps_nhalf[i,j]/(0.5*dx) - v_total[i,j]/dy*D1yback(eps_nhalf, i, j) + nu_total[i,j]*D2yback(eps_nhalf, i, j)/(dy*dy) 
    return RHS

def advanceEPS(phi_np1old, phi_n, dx, dy, dz, params, bc_ylo, bc_yhi, bc_zlo, bc_zhi):
    """
    Advance the dissipation equation one full step
    """
    phi_tilde = getTildeVars(phi_np1old, phi_n)
    u_tilde, v_tilde, w_tilde = phi_tilde['u'], phi_tilde['v'], phi_tilde['w']
    nuT_tilde = phi_tilde['nuT']
    eps_tilde = phi_tilde['eps']

    N  = u_tilde.shape
    Ny = N[0]
    Nz = N[1]
    
    # Load parameters
    nu       = params['nu']
    sigmaeps = params['sigmaeps']
    C1eps    = params['C1eps']
    C2eps    = params['C2eps']
    C3eps    = params['C3eps']
    feps_const = params['feps_const'] if 'feps_const' in params else 0.0
    
    # --- differentiation stencils ---
    #                  j-1  j  j+1
    Irow   = np.array([0,   1,  0])
    Dcen   = np.array([-1,  0,  1])*0.5
    D2cen  = np.array([1,  -2,  1])
    # -------------------------------

    # Compute some quantities
    Dy_nuT, Dz_nuT = np.gradient(nuT_tilde, dy, dz, edge_order=1)
    Dy_U, Dz_U = np.gradient(u_tilde, dy, dz, edge_order=1)

    nu_total = nu + nuT_tilde/sigmaeps
    v_total  = v_tilde - Dy_nuT/sigmaeps
    w_total  = w_tilde - Dz_nuT/sigmaeps

    Tscale = TimeScale(phi_tilde['k'], phi_tilde['eps'], nu)
    RHS_extra_forcing = C1eps/Tscale*(nuT_tilde*(Dy_U*Dy_U) + nuT_tilde*(Dz_U*Dz_U)) - C2eps*eps_tilde/Tscale + feps_const
    
    # First sweep: n -> n+1/2
    # -----------------------
    eps_nhalf   = np.zeros((Ny, Nz))
    RHS_nhalf = RHS_eps_nhalf(phi_np1old, phi_n, phi_tilde, Dz_nuT, dx, dy, dz, params) 
    for j in range(Nz):
        LHS_nhalf = np.zeros((Ny,3))
        # == Set up the LHS matrices ==
        for i in range(1,Ny-1):
            LHS_nhalf[i,:] = u_tilde[i,j]/(0.5*dx)*Irow + v_total[i,j]*Dcen/dy - nu_total[i,j]*D2cen/(dy*dy)
        # Apply BC's
        row_lo, dentry_lo = applyBC(bc_ylo, 'lower', dy)
        row_hi, dentry_hi = applyBC(bc_yhi, 'upper', dy)
        LHS_nhalf[0,:]  = row_lo
        LHS_nhalf[-1,:] = row_hi
        RHS_nhalf[0,:]  = dentry_lo
        RHS_nhalf[-1,:] = dentry_hi
        # Solve the triadiagonal system
        eps_nhalf[:,j] = solvetridiag(LHS_nhalf, RHS_nhalf[:,j])

    # Second sweep: n+1/2 -> n+1
    # -----------------------
    eps_np1   = np.zeros((Ny, Nz))
    RHS_np1 = RHS_eps_np1(eps_nhalf, phi_np1old, phi_n, phi_tilde, Dy_nuT, dx, dy, dz, params)
    # == Set up the LHS matrices ==
    for i in range(Ny):
        LHS_np1 = np.zeros((Nz,3))
        for j in range(1,Nz-1):
            LHS_np1[j,:] = u_tilde[i,j]/(0.5*dx)*Irow + w_total[i,j]*Dcen/dz - nu_total[i,j]*D2cen/(dz*dz) 
        # Apply BC's
        row_lo, dentry_lo = applyBC(bc_zlo, 'lower', dz)
        row_hi, dentry_hi = applyBC(bc_zhi, 'upper', dz)
        if i==0:
            row_lo, dentry_lo = applyBC({'type':'dirichlet', 'value':eps_tilde[i,0]}, 'lower', dz)
        if i==Ny-1:
            row_hi, dentry_hi = applyBC({'type':'dirichlet', 'value':eps_tilde[i,-1]}, 'lower', dz)            
        LHS_np1[0,:]  = row_lo
        LHS_np1[-1,:] = row_hi
        RHS_np1[:,0]  = dentry_lo
        RHS_np1[:,-1] = dentry_hi
        #print(f'i = {i}\nRHS_np1 = ',RHS_np1[i,:], '\nLHS = ', LHS_np1)                
        # Solve the triadiagonal system
        eps_np1[i,:] = solvetridiag(LHS_np1, RHS_np1[i,:], verbose=False)

    return eps_np1 #np.zeros((Ny, Nz))

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
        row_lo, dentry_lo_lst = applyBC(bc_ylo, 'lower', dy)
        dentry_lo = dentry_lo_lst[j]
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

def TimeScale(k, eps, nu):
    return np.fmax(k/eps, 6.0*np.sqrt(nu/eps))

def getNuT(phi, Cmu, nu):
    # timescale
    #Tscale = np.fmax(phi['k']/phi['eps'], 6.0*np.sqrt(nu/phi['eps'][:,:]))
    Tscale = TimeScale(phi['k'], phi['eps'], nu)
    return Cmu*phi['k']*Tscale

def advanceSystemKEPS(phi_n, dx, dy, dz, params, allbcs, eqnsys, maxiter=100,
                      tol=1.0E-6, verbose=False):
    """
    Advance equation system 1 step in x
    """
    varlist = [v for v, g in eqnsys.items()]

    if 'nuT' not in phi_n:
        phi_n['nuT'] = getNuT(phi_n, params['Cmu'], params['nu']) 
    phi_n1 = copy.deepcopy(phi_n)

    for k in range(maxiter):
        phi_next = OrderedDict()
        phi_n1['nuT'] = getNuT(phi_n1, params['Cmu'], params['nu'])
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

########################################################
# Initial condition stuff
def MO_windshear(z,L):
    if L < 0:
        return (1-16.0*(z/L))**(-0.25)

    if L==float('inf'):
        return np.ones_like(z)

    return 1 + 5 * z/L

def init_U_ABL(z,param):
    """
    Initialize ABL profile from Monin-Obukhov theory
    """
    z0    = param['z0']
    L     = param['L']
    K     = param['kappa']
    rho   = param['rho']
    ustar = param['ustar']

    phim = MO_windshear(z,L)

    if L < 0:
        return ustar/K * (\
                np.log(z/z0) + \
                np.log( (8*phim**4) / ( (phim + 1)**2 * (phim**2 + 1))) -\
                np.pi/2 + \
                2 * np.arctan( 1/(phim)))
    else:
        return ustar/K * ( np.log(z/z0) + phim - 1)

def init_e_ABL(z,param):
    L     = param['L']
    K     = param['kappa']
    ustar = param['ustar']

    phim = MO_windshear(z,L)

    if L < 0:
        phie = 1-z/L
    elif L==float('inf'):
        phie = phim
    else:
        phie = phim - z/L

    return ustar**3/(K*z)*phie

def init_k_ABL(z,param):
    L     = param['L']
    K     = param['kappa']
    ustar = param['ustar']

    phim  = MO_windshear(z,L)

    if L < 0:
        phie = 1-z/L
    elif L==float('inf'):
        phie = phim
    else:
        phie = phim - z/L

    return 5.48 * ustar**2 * ( phie / phim )**0.5

def applyStencil(D, u, j):
    if D[0] == 0:
        return D[1]*u[j] + D[2]*u[j+1]

    if D[1] == 0:
        return D[0]*u[j-1] + D[2]*u[j+1]

    if D[2] == 0:
        return D[0]*u[j-1] + D[1]*u[j] 

    return D[0]*u[j-1] + D[1]*u[j] + D[2]*u[j+1]

def set_e_init(zvec,dz,u,k,params):
    Cmu = params['Cmu']
    e_init = np.zeros_like(u)

    Dcen   = np.array([-1,  0,  1])/(2*dz)
    N = len(e_init)

    for j in np.arange(1,N-1):
        e_init[j] = np.sqrt(Cmu* k[j]**2 *applyStencil(Dcen,u,j)**2)

    return e_init

def set_k_init(rvec,dr,u,params,k_factor=0.1):
    k_init = np.zeros_like(u)
    Dcen   = np.array([-1,  0,  1])/(2*dr)
    N = len(k_init)
    for j in np.arange(1,N-1):
        k_init[j] = applyStencil(Dcen,u,j)**2

    # scale k 
    kmax = np.max(k_init)
    C = k_factor**2 * 2 / (3 * kmax) 
    k_init *= C
    return k_init


########################################################
# Define the laminar equation system
keps_eqns        = OrderedDict()
keps_eqns['u']   = partial(advanceF, field="u")
keps_eqns['w']   = partial(advanceF, field="w")
keps_eqns['k']   = advanceTKE
keps_eqns['eps'] = advanceEPS
keps_eqns['v']   = advanceMass

# Use the same marchSystemBase in SANDWake3D_base to advance the equations
marchSystem = marchSystemBase
