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
        print(ab)
        print(b)
    x  = solve_banded(lu, ab, b)
    return x


def applyStencil(D, u, j):
    return D[0]*u[j-1] + D[1]*u[j] + D[2]*u[j+1]

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

def solveUMom(u_ip1, u_i, w_ip1, w_i, dx, dz, Re, UBClower, UBCupper,
              RHSforcing=None):
    N      = len(u_ip1)
    utilde = 0.5*(u_ip1 + u_i)
    wtilde = 0.5*(w_ip1 + w_i)
    # --- differentiation stencils ---
    #                  j-1  j  j+1
    Irow   = np.array([0,   1,  0])
    Dcen   = np.array([-1,  0,  1])/(2*dz)
    D2cen  = np.array([1,  -2,  1])/(dz**2)
    # -------------------------------

    if RHSforcing is None:
        forcing = np.zeros(N)
    else:
        forcing = RHSforcing
    
    matrows = []
    d       = []
    # Add the lower BC
    row, dentry = applyBC(UBClower, 'lower', dz)
    matrows.append(row)
    d.append(dentry)
    # Loop through
    for j in np.arange(1,N-1):
        LHS = utilde[j]*Irow/dx + 0.5*wtilde[j]*Dcen - 0.5*D2cen/Re
        RHS = utilde[j]/dx*u_i[j] - 0.5*wtilde[j]*applyStencil(Dcen, u_i, j) \
            + 0.5/Re*applyStencil(D2cen, u_i, j) 
        matrows.append(LHS)
        d.append(RHS)
    # Add the upper BC
    row, dentry = applyBC(UBCupper, 'upper', dz)
    matrows.append(row)
    d.append(dentry)
    
    # Solve
    matrows = np.array(matrows)
    d       = np.array(d) + forcing
    #print(d)
    u_new = solvetridiag(matrows, d)
    return u_new

def solveMass(u_ip1, u_i, w_ip1, w_i, dx, zvec, WBC): #Wlower, Wupper):
    """
    Solve mass for new W
    """
    N      = len(u_ip1)
    dz  = np.mean(np.diff(zvec)) # Assumes constant dz
    
    #                  j-1  j  j+1
    Irow   = np.array([0,   1,  0])
    Dcen   = np.array([-1,  0,  1 ])/(dz)
    D2cen  = np.array([1,  -2,  1])/(dz**2)
    d      = []
    
    # Add the lower BC
    d.append(0.0)
    # Loop through
    for j in np.arange(1,N-1):
        LHS = Dcen
        RHS = -1.0/dx*(u_ip1[j] - u_i[j])
        d.append(RHS)
    # Add the upper BC
    d.append(RHS)

    # Solve
    d       = np.array(d)
    
    w_new = np.cumsum(d)*dz 
    # Enforce the BC
    wval   = np.interp(WBC['z'], zvec, w_new)
    delta  = WBC['value'] - wval
    w_new += delta
    return w_new


def solveTemp(u_ip1, u_i, w_ip1, w_i, T_ip1, T_i, dx, dz, Re, Pr, TBClower, TBCupper):
    N      = len(u_ip1)
    utilde = 0.5*(u_ip1 + u_i)
    wtilde = 0.5*(w_ip1 + w_i)
    # --- differentiation stencils ---
    #                  j-1  j  j+1
    Irow   = np.array([0,   1,  0])
    Dcen   = np.array([-1,  0,  1])/(2*dz)
    D2cen  = np.array([1,  -2,  1])/(dz**2)
    # -------------------------------
    
    matrows = []
    d       = []
    # Add the lower BC
    row, dentry = applyBC(TBClower, 'lower', dz)
    matrows.append(row)
    d.append(dentry)
    # Loop through
    for j in np.arange(1,N-1):
        LHS = utilde[j]*Irow/dx + 0.5*wtilde[j]*Dcen - 0.5*D2cen*Pr/Re
        RHS = utilde[j]/dx*T_i[j] - 0.5*wtilde[j]*applyStencil(Dcen, T_i, j) + 0.5*Pr/Re*applyStencil(D2cen, T_i, j)
        matrows.append(LHS)
        d.append(RHS)
    # Add the upper BC
    row, dentry = applyBC(TBCupper, 'upper', dz)
    matrows.append(row)
    d.append(dentry)
    
    # Solve
    matrows = np.array(matrows)
    d       = np.array(d)
    #print(d)
    T_new = solvetridiag(matrows, d)
    return T_new

def compute_dPdx(T_ip1, T_i, dx, dz, g):
    """
    Compute -dP/dx
    """
    dTdx = (T_ip1 - T_i)/dx
    s=-1
    int_dTdx = np.cumsum(dTdx[::s])[::s]*dz
    dPdx = g*int_dTdx
    return -dPdx

def advanceSol(u_i, w_i, T_i, dx, zvec, Re, Pr, UBC, WBC, TBC,
               g=0.0, maxiter=10, tol=1.0E-8, verbose=False):
    """
    Advance the solution one step
    """
    dz  = np.mean(np.diff(zvec)) # Assumes constant dz
    u_next = u_i
    w_next = w_i
    T_next = T_i
    Ulower, Uupper = UBC[0], UBC[1]
    Tlower, Tupper = TBC[0], TBC[1]
   
    converged = lambda new, old, tol: np.linalg.norm(new-old)<=tol
    
    # Loop on this
    for k in np.arange(maxiter):
        # Save the old variables for comparison
        u_old = u_next+0.0
        w_old = w_next+0.0
        T_old = T_next+0.0
        # Compute the pressure gradient
        dPdx = compute_dPdx(T_next, T_i, dx, dz, g)
        # Advance the next iteration
        u_next = solveUMom(u_next, u_i, w_next, w_i, dx, dz, Re, Ulower, Uupper,
                           RHSforcing=dPdx)
        w_next = solveMass(u_next, u_i, w_next, w_i, dx, zvec, WBC)
        T_next = solveTemp(u_next, u_i, w_next, w_i, T_next, T_i, dx, dz, Re,
                           Pr, Tlower, Tupper)
        # Check convergence
        if verbose:
            print(k, np.linalg.norm(u_next-u_old), np.linalg.norm(w_next-w_old), np.linalg.norm(T_next-T_old))
        if converged(u_next, u_old, tol) and converged(w_next, w_old, tol) and converged(T_next, T_old, tol):
            break
    # Check if k hit maxiter:
    # -->TODO!
    
    return u_next, w_next, T_next

def marchWakeBL(uinit, winit, Tinit, xvec, zvec, Re, Pr, UBC, WBC, TBC,
                g=0, verbose=False):
    Uvec = []
    Wvec = []
    Tvec = []
    # March on each x
    for ix, x in enumerate(xvec):
        if verbose: print(f"x = {x}")
        if ix==0:
            # Set the initial values
            Uvec.append(uinit)
            Wvec.append(winit)
            Tvec.append(Tinit)
        else:
            # Advance one step
            dx =  x - xvec[ix-1]
            Unext, Wnext, Tnext = advanceSol(Uvec[ix-1], Wvec[ix-1], Tvec[ix-1],
                                             dx, zvec, Re, Pr, UBC, WBC, TBC,
                                             g=g, verbose=verbose)
            Uvec.append(Unext)
            Wvec.append(Wnext)
            Tvec.append(Tnext)
    return np.array(Uvec), np.array(Wvec), np.array(Tvec)
