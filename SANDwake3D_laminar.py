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

def getTilde(phi_ip1, phi_i):
    """
    Get the averaged velocities for the convective term
    """
    u_ip1, u_i = phi_ip1['u'], phi_i['u']
    v_ip1, v_i = phi_ip1['v'], phi_i['v']
    w_ip1, w_i = phi_ip1['w'], phi_i['w']
    u_tilde = 0.5*(u_ip1 + u_i)
    v_tilde = 0.5*(v_ip1 + v_i)
    w_tilde = 0.5*(w_ip1 + w_i)
    return u_tilde, v_tilde, w_tilde

def RHS_u_nhalf(phi_ip1, phi_i, dx, dy, dz, params):
    """
    Go from n to n+1/2 for the u-momentum equation
    
    """
    u_ip1, u_i = phi_ip1['u'], phi_i['u']
    v_ip1, v_i = phi_ip1['v'], phi_i['v']
    w_ip1, w_i = phi_ip1['w'], phi_i['w']
    u_tilde, v_tilde, w_tilde = getTilde(phi_ip1, phi_i)

    nu = params['nu']
    
    N  = u_i.shape
    Ny = N[0]
    Nz = N[1]
    RHS = np.zeros((Ny, Nz))
    for i in range(1,Ny-1):
        for j in range(1,Nz-1):
            RHS[i,j] = 0.0
#            RHS[i,j] = delta*delta/alpha*u[i,j] + 0.5*dx*(u[i,j+1] -2.0*u[i,j] + u[i,j-1])
    return RHS
