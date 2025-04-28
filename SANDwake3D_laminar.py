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
