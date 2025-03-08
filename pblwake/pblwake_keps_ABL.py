#!/usr/bin/env python
import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.special import expit

def double_gaussian(x, x0, x01, sigma1, A1, sigma2, A2):
    return A1 * np.exp(-(x - x0)**2 / (2 * sigma1**2)) + A2 * np.exp(-(x - x01)**2 / (2 * sigma2**2))

def triple_gaussian(x, x0, x01, x02, sigma1, A1, sigma2, A2, sigma3, A3, y_start, y_end):
    gaussian = A1 * np.exp(-(x - x0)**2 / (2 * sigma1**2)) + A2 * np.exp(-(x - x01)**2 / (2 * sigma2**2))  + A3 * np.exp(-(x - x02)**2 / (2 * sigma3**2))
    gaussian += y_end
    return gaussian

def mod_func(z,params):
    #return expit((z-z_height) * np.pi / (z[-1]-z[0]))
    z_height=params['z_height']
    width = z[-1]-z[0]
    return np.ones_like(z)
    return expit((z-z_height))

def MO_windshear(z,L):
    if L < 0:
        return (1-16.0*(z/L))**(-0.25)

    if L==float('inf'):
        return np.ones_like(z)

    return 1 + 5 * z/L

def align_profiles(U1,U2,zvec):
    """
    Aligns u2 profile with u1
    """

    def objective_function(C,zvec,mask):
        U2_shifted = U2 + C
        diff_U1_U2 = np.sum(np.abs(U1-U2_shifted)[mask]) 
        #diff_U1_U2 = np.abs(np.max(np.abs(U1)) - np.max(np.abs(U2_shifted)))
        return diff_U1_U2

    #mask = (zvec > 50) & (zvec < 200)
    mask = (zvec > 300)
    init_guess = 0
    result = minimize(objective_function,init_guess,args=(zvec,mask))
    opt_c = result.x
    return U2 + opt_c

def get_LES_profile(zvec,fileroot,velType='flat',turb='None',prof='u',loc=2):
    filename = fileroot
    filename += velType +"_"
    if turb=='None':
        filename += 'noturb_'
    else:
        filename += 'wturb_' 

    if prof=='u':
        filename += 'Uh_XZ_'
    elif prof=='w':
        filename += 'velocityz_avg_XZ_'
    elif prof=='k':
        filename += 'TKE_XZ_'
    else:
        filename += 'temperature_avg_XZ_'

    if loc == 1.0:
        filename += "1.0.dat"
    elif loc == 2.0:
        filename += "2.0.dat"
    elif loc == -2.5:
        filename += "-2.5.dat"
    elif loc == 4.0:
        filename += "4.0.dat"
    elif loc == 6.0:
        filename += "4.0.dat"
    elif loc == 8.0:
        filename += "8.0.dat"

    print("Reading data: ",filename)
    data = np.loadtxt(filename,skiprows=1)
    prof = interpolate_profile(data[:,1],data[:,0],zvec)
    return prof

def interpolate_profile(orig_U,orig_grid,zvec):
    interpolator_kind = 'cubic'
    interpolator  = interp1d(orig_grid,orig_U,kind=interpolator_kind)

    interp_u = interpolator(zvec)
    return interp_u

def smooth_profile(u,window_size = 3):
    kernel = np.ones(window_size) / window_size
    padded_u = np.pad(u,len(kernel) // 2,mode='edge')
    return np.convolve(padded_u,kernel,mode='valid')

def smooth_profile_at_x(u,window_size,x,xloc):
    kernel = np.ones(window_size) / window_size
    padded_u = np.pad(u,len(kernel) // 2,mode='edge')
    smoothed_u = np.convolve(padded_u,kernel,mode='valid')

    # Find indices within the specified smoothing range around the target x
    x_index = np.argmin(np.abs(x - xloc))
    u[max(x_index-window_size//2,0):x_index+window_size//2] = smoothed_u[max(x_index-window_size//2,0):x_index+window_size//2] 
    return  u

def init_U_tanh(z,U0,U1,delta,rdelta,zhub = 50):
    #U0 = centerline velocity
    #U1 = freestream velocity
    r = (z-zhub)**2
    return (U1-U0)/2*(np.tanh((r-rdelta)/delta)) + U0


def superimpose_on_ABL(UABL,UWAKE,z,param):
    shift = 100
    shift_ind = np.argmin(np.abs(z-shift))
    UWAKE_ALIGN = align_profiles(UABL,UWAKE,z)

    shift = 100
    shift_ind = np.argmin(np.abs(z-shift))
    counter = 0

    cutoff = 20
    cutoff_ind = np.argmin(np.abs(z-cutoff))
    for i in range(shift_ind,len(z)):
        UABL[i] = UWAKE_ALIGN[cutoff_ind+counter]
        counter += 1

    #UABL = smooth_profile(UABL,21)

    return UABL

def shift_profile(U,UABL,zshift,z):
    shift_ind = np.argmin(np.abs(z-zshift))
    U = np.roll(U,shift_ind)
    for i in range(0,shift_ind):
        U[i] = UABL[i]
    return U

def init_U_ABL(z,param):
    z0 = param['z0']
    L = param['L']
    K  = param['K']
    rho = param['rho']
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

def init_T_ABL(z,param):
    z0 = param['z0']
    L = param['L']
    K  = param['K']
    rho = param['rho']
    ustar = param['ustar']
    qw    = param['qw']
    cp     = param['cp']
    g      = param['g']
    Tw     = param['Tw']
    Tstar = np.sqrt(-qw/(rho * cp * ustar))
    phim = MO_windshear(z,L)

    if L < 0:
        return Tstar/K * ( np.log(z/z0) - 2 * np.log(0.5 * (1 + phim**(-2)))) - g/cp*(z-z0) + Tw
    else:
        if Tstar == 0:
            return np.zeros_like(z) + Tw
        return Tstar/K * ( np.log(z/z0) + phim - 1 ) - g/cp*(z-z0) + Tw

def init_linear_profile(z,val1,val2):
    return val1 + (val2-val1)/(z[-1]-z[0])*z

def init_e_ABL(z,param):
    L = param['L']
    K  = param['K']
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
    L = param['L']
    K  = param['K']
    ustar = param['ustar']

    phim = MO_windshear(z,L)

    if L < 0:
        phie = 1-z/L
    elif L==float('inf'):
        phie = phim
    else:
        phie = phim - z/L

    return 5.48 * ustar**2 * ( phie / phim )**0.5


def apply_clipping(variable,value=1e-7):
    new = np.copy(variable)
    for i in range(1,len(variable)-1):
        if variable[i] < value:
            new[i]=value
    return new

    #return np.where(variable < value ,value,variable)

def compute_dPdx(T_ip1, T_i, dx, dz, g,beta=1.0):
    """
    Compute -dP/dx
    """
    dTdx = (T_ip1 - T_i)/dx
    s=-1
    int_dTdx = np.cumsum(dTdx[::s])[::s]*dz
    dPdx = beta*g*int_dTdx
    return -dPdx #+ 0.01

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
    if D[0] == 0:
        return D[1]*u[j] + D[2]*u[j+1]

    if D[1] == 0:
        return D[0]*u[j-1] + D[2]*u[j+1]

    if D[2] == 0:
        return D[0]*u[j-1] + D[1]*u[j] 

    return D[0]*u[j-1] + D[1]*u[j] + D[2]*u[j+1]

def applyBCs_to_vec(u,bcdict_low,bcdict_up,dz):

    dentry = bcdict_low['value']
    if bcdict_low['type'] == 'dirichlet':
        u[0] = dentry
    elif bcdict_low['type'] == 'neumann':
        u[0] = u[1]-dz*dentry

    dentry = bcdict_up['value']
    if bcdict_up['type'] == 'dirichlet':
        u[-1] = dentry
    elif bcdict_up['type'] == 'neumann':
        u[-1] = dz*dentry + u[-2]

    return

def set_e_init(zvec,dz,u,k,w,params):
    Cmu = params['C_mu']
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

def evaluateGB(nut,T,params,j,Dcen):
    beta = params['beta']
    g    = params['g']
    Pr   = params['Pr']
    cp   = params['cp']
    rho  = params['rho']
    return beta * g * nut[j] / Pr * (applyStencil(Dcen,T,j) - g / cp) / rho


def solveUMom(phi_next, phi_i,phi_tilde,dx, dz, zvec,params, UBClower, UBCupper,
              RHSforcing=None):

    # Extract compontents of phi_i
    u_i = phi_i['u'] 

    # Extract compontents of phi_next
    u_ip1 = phi_next['u'] 

    # Extract compontents of phi_tilde
    utilde = phi_tilde['u'] 
    wtilde = phi_tilde['w'] 
    nut_tilde = phi_tilde['nut'] 


    # Extract parameters
    Re = params['Re']
    C_mu = params['C_mu']

    N      = len(u_ip1)
    # --- differentiation stencils ---
    #                  j-1  j  j+1
    Irow   = np.array([0,   1,  0])
    Dcen   = np.array([-1,  0,  1])/(2*dz)
    D2cen  = np.array([1,  -2,  1])/(dz**2)

    # -------------------------------
    #CLIPPING
    mod_f = mod_func(zvec,params)
    
    if RHSforcing is None:
        forcing = np.zeros(N)
    else:
        forcing = RHSforcing #+ 0.0005 
    
    matrows = []
    d       = []

    # Remove point at wall 
    row, dentry = applyBC(UBClower, 'lower', dz)
    matrows.append(row)
    d.append(dentry)

    #fix the first point off the wall for j = 0
    # matrows.append(Irow)
    # d.append(init_U_ABL(dz,params))

    for j in np.arange(1,N-1):
        LHS = utilde[j]*Irow/dx + mod_f[j]*0.5*wtilde[j]*Dcen - 0.5*( 1.0/Re + nut_tilde[j])* D2cen - 0.5*applyStencil(Dcen,nut_tilde,j)*Dcen
        RHS = utilde[j]/dx*u_i[j] - mod_f[j]*0.5*wtilde[j]*applyStencil(Dcen,u_i,j) + 0.5 * (1.0/Re + nut_tilde[j]) *applyStencil(D2cen, u_i, j) + 0.5*applyStencil(Dcen,nut_tilde,j)*applyStencil(Dcen,u_i,j) + forcing[j]

        matrows.append(LHS)
        d.append(RHS)
    # Add the upper BC
    row, dentry = applyBC(UBCupper, 'upper', dz)
    matrows.append(row)
    d.append(dentry)
    
    # Solve
    matrows = np.array(matrows)
    d       = np.array(d) 
    #print(d)
    u_new = solvetridiag(matrows, d)
    return u_new

def solveTKE(phi_next, phi_i, phi_tilde,dx, dz, zvec,params, kBClower, kBCupper):
    """
    Solve the boundary layer TKE equation
    """
    
    # Extract compontents of phi_i
    k_i = phi_i['k'] 
    u_i = phi_i['u'] 
    w_i = phi_i['w'] 

    # Extract compontents of phi_next
    k_ip1 = phi_next['k'] 
    u_ip1 = phi_next['u'] 
    w_ip1 = phi_next['w'] 

    # Extract compontents of phi_tilde
    utilde = phi_tilde['u'] 
    wtilde = phi_tilde['w'] 
    etilde = phi_tilde['e'] 
    Ttilde = phi_tilde['T'] 
    nut_tilde = phi_tilde['nut'] 

    # Extract parameters
    Re = params['Re']
    sigma_k = params['sigma_k']

    N      = len(k_ip1)
    # --- differentiation stencils ---
    #                  j-1  j  j+1
    Irow   = np.array([0,   1,  0])
    Dcen   = np.array([-1,  0,  1])/(2*dz)
    D2cen  = np.array([1,  -2,  1])/(dz**2)
    # -------------------------------

    mod_f = mod_func(zvec,params)
    
    matrows = []
    d       = []
    # Add the lower BC
    row, dentry = applyBC(kBClower, 'lower', dz)
    matrows.append(row)
    d.append(dentry)
    # Loop through

    #fix the first point off the wall for j = 0
    # matrows.append(Irow)
    # d.append(init_k_ABL(dz,params))

    for j in np.arange(1,N-1):
        LHS = utilde[j]*Irow/dx + mod_f[j]*0.5*wtilde[j]*Dcen - 0.5 * (1.0/Re + nut_tilde[j]/sigma_k)*D2cen - 0.5/sigma_k*applyStencil(Dcen,nut_tilde,j)*Dcen
        RHS = utilde[j]/dx*k_i[j]-mod_f[j]*0.5*wtilde[j]*applyStencil(Dcen,k_i,j) + 0.5 * (1.0/Re + nut_tilde[j]/sigma_k)*applyStencil(D2cen,k_i,j) + 0.5/sigma_k * applyStencil(Dcen,nut_tilde,j) * applyStencil(Dcen,k_i,j) - etilde[j] + nut_tilde[j] *  ( applyStencil(Dcen,utilde,j)**2) + evaluateGB(nut_tilde,Ttilde,params,j,Dcen)

        #RHS += nut_tilde[j] * (2*applyStencil(Dcen,wtilde,j)**2 +\
        #                      2* ((u_ip1[j] - u_i[j])/dx)**2 +\
        #                      2*((w_ip1[j] - w_i[j])/dx) * applyStencil(Dcen,utilde,j) +\
        #                      ((w_ip1[j] - w_i[j])/dx)**2 )

        matrows.append(LHS)
        d.append(RHS)
    # Add the upper BC
    row, dentry = applyBC(kBCupper, 'upper', dz)
    matrows.append(row)
    d.append(dentry)
    
    # Solve
    matrows = np.array(matrows)
    d       = np.array(d)
    k_new = solvetridiag(matrows, d)
    return k_new

def solveEps(phi_next, phi_i,phi_tilde,dx, dz, zvec,params, eBClower, eBCupper):
    """
    Solve the boundary layer EPS equation
    """
    
    # Extract compontents of phi_i
    e_i = phi_i['e'] 
    u_i = phi_i['u'] 
    w_i = phi_i['w'] 

    # Extract compontents of phi_next
    e_ip1 = phi_next['e'] 
    u_ip1 = phi_next['u'] 
    w_ip1 = phi_next['w'] 

    # Extract compontents of phi_tilde
    utilde = phi_tilde['u'] 
    ktilde = phi_tilde['k'] 
    wtilde = phi_tilde['w'] 
    etilde = phi_tilde['e'] 
    Ttilde = phi_tilde['T'] 
    nut_tilde = phi_tilde['nut'] 

    # Extract parameters
    Re = params['Re']
    sigma_e = params['sigma_e']
    C_mu = params['C_mu']
    C_1e = params['C_1e']
    C_2e = params['C_2e']
    C_3e = params['C_3e']

    N      = len(e_ip1)
    # --- differentiation stencils ---
    #                  j-1  j  j+1
    Irow   = np.array([0,   1,  0])
    Dcen   = np.array([-1,  0,  1])/(2*dz)
    D2cen  = np.array([1,  -2,  1])/(dz**2)
    # -------------------------------

    mod_f = mod_func(zvec,params)

    matrows = []
    d       = []
    # Add the lower BC
    #eBClower['value'] = 2*ktilde[1]/(Re*dz**2) #set based on k and y at the first grid point
    row, dentry = applyBC(eBClower, 'lower', dz)
    matrows.append(row)
    d.append(dentry)

    #fix the first point off the wall for j = 0
    # matrows.append(Irow)
    # d.append(init_e_ABL(dz,params))

    # Loop through
    for j in np.arange(1,N-1):

        if (etilde[j] == 0):
            T = 0.0
        else:
            T = np.maximum(ktilde[j]/etilde[j],6.0*np.sqrt(1.0/(Re*etilde[j])))
        T= np.maximum(T,1e-7)
        
        LHS = utilde[j]*Irow/dx + mod_f[j]*0.5*wtilde[j]*Dcen - 0.5 * (1.0/Re + nut_tilde[j]/sigma_e)*D2cen - 0.5/sigma_e*applyStencil(Dcen,nut_tilde,j)*Dcen + 0.5* C_2e/T * Irow
        RHS = utilde[j]/dx*e_i[j] - mod_f[j]*0.5*wtilde[j]*applyStencil(Dcen,e_i,j) + 0.5 * (1.0/Re + nut_tilde[j]/sigma_e)*applyStencil(D2cen,e_i,j) + 0.5/sigma_e * applyStencil(Dcen,nut_tilde,j) * applyStencil(Dcen,e_i,j) + C_1e/T* nut_tilde[j] * ( applyStencil(Dcen,utilde,j)**2)  - 0.5 * C_2e /T * e_i[j] + C_1e/T*(1-C_3e)*evaluateGB(nut_tilde,Ttilde,params,j,Dcen)

        #RHS += C_1e/T * nut_tilde[j] * (2 * applyStencil(Dcen,wtilde,j)**2 +\
        #                      2* ((u_ip1[j] - u_i[j])/dx)**2 +\
        #                      2*((w_ip1[j] - w_i[j])/dx) * applyStencil(Dcen,utilde,j) +\
        #                      ((w_ip1[j] - w_i[j])/dx)**2 )


        matrows.append(LHS)
        d.append(RHS)
    # Add the upper BC
    row, dentry = applyBC(eBCupper, 'upper', dz)
    matrows.append(row)
    d.append(dentry)
    
    # Solve
    matrows = np.array(matrows)
    d       = np.array(d)
    #print(d)
    e_new = solvetridiag(matrows, d)

    return e_new

def solveTemp(phi_next, phi_i, phi_tilde,dx, dz, zvec,params, TBClower, TBCupper):
    """
    Solve the boundary layer temperature equation
    """
    
    # Extract compontents of phi_i
    T_i = phi_i['T'] 

    # Extract compontents of phi_next
    T_ip1 = phi_next['T'] 

    # Extract compontents of phi_tilde
    utilde = phi_tilde['u'] 
    wtilde = phi_tilde['w'] 
    etilde = phi_tilde['e'] 
    nut_tilde = phi_tilde['nut'] 

    # Extract parameters
    Re = params['Re']
    Pr = params['Pr']
    
    N      = len(T_ip1)
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

    mod_f = mod_func(zvec,params)
    #mod_f = np.ones_like(mod_f)
    #fix the first point off the wall for j = 0
    # matrows.append(Irow)
    # d.append(init_T_ABL(dz,params))

    # Loop through
    for j in np.arange(1,N-1):
        LHS = utilde[j]*Irow/dx + mod_f[j]*0.5*wtilde[j]*Dcen - 0.5*D2cen/(Pr*Re) 
        RHS = utilde[j]/dx*T_i[j] - mod_f[j]*0.5*wtilde[j]*applyStencil(Dcen, T_i, j) + 0.5/(Re*Pr)*applyStencil(D2cen, T_i, j) 
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





def solveMass(phi_next, phi_i, dx, zvec, WBC):
    """
    Solve mass for new W
    """
    # Extract compontents of phi_i
    u_i = phi_i['u'] 
    w_i = phi_i['w'] 
    T_i = phi_i['T'] 

    # Extract compontents of phi_next
    u_ip1 = phi_next['u'] 
    w_ip1 = phi_next['w'] 
    T_ip1 = phi_next['T'] 

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
    d = np.array(d)
    
    w_new = np.cumsum(d)*dz 
    # Enforce the BC
    wval   = np.interp(WBC['z'], zvec, w_new)
    delta  = WBC['value'] - wval
    w_new += delta
    return w_new

def advanceSol(phi_i, dx, zvec, params, BCdict,
               g=0.0, maxiter=20, tol=1.0E-7, verbose=False,clipping=False):
    """
    Advance the solution one step
    """
    dz  = np.mean(np.diff(zvec)) # Assumes constant dz
    # Extract compontents of phi_i
    u_i = phi_i['u'] 
    w_i = phi_i['w'] 
    T_i = phi_i['T'] 
    k_i = phi_i['k']
    e_i = phi_i['e']
        
    u_next = u_i
    w_next = w_i
    T_next = T_i
    k_next = k_i
    e_next = e_i

    # Get the boundary conditions
    Ulower, Uupper = BCdict['u'][0], BCdict['u'][1]
    klower, kupper = BCdict['k'][0], BCdict['k'][1]
    elower, eupper = BCdict['e'][0], BCdict['e'][1]
    Tlower, Tupper = BCdict['T'][0], BCdict['T'][1]
    WBC = BCdict['w']

    converged = lambda new, old, tol: np.linalg.norm(new-old)<=tol
    
    # Loop on this
    for k in np.arange(maxiter):

        if verbose:
            print("Iteration: ",k)
        # Save the old variables for comparison
        u_old = u_next+0.0
        w_old = w_next+0.0
        k_old = k_next+0.0
        e_old = e_next+0.0
        T_old = T_next+0.0

        # Construct phi_next (at the next x_i)
        phi_next = {}
        phi_next['u'] = u_next
        phi_next['w'] = w_next
        phi_next['k'] = k_next
        phi_next['e'] = e_next
        phi_next['T'] = T_next

        # Construct phi_tilde 
        utilde = 0.5*(u_next + u_i)
        wtilde = 0.5*(w_next + w_i)
        ktilde = 0.5*(k_next + k_i)
        etilde = 0.5*(e_next + e_i)
        Ttilde = 0.5*(T_next + T_i)

        if (clipping): 
            e_i   = apply_clipping(e_i)
            e_next = apply_clipping(e_next)
            k_i   = apply_clipping(k_i)
            k_next = apply_clipping(k_next)
            etilde   = apply_clipping(etilde)
            ktilde   = apply_clipping(ktilde)

        C_mu = params['C_mu']
        Re = params['Re']
        nut_tilde = np.zeros_like(utilde)
        N=len(u_next)
        if C_mu > 0:
            nut_tilde = C_mu * ktilde**2 / etilde 
            for j in np.arange(1,N-1):
                if (etilde[j] == 0):
                    Ttime = 0.0
                else:
                    Ttime = np.maximum(ktilde[j]/etilde[j],6.0*np.sqrt(1.0/(Re*etilde[j])))
                Ttime= np.maximum(Ttime,1e-7)
                nut_tilde[j] = C_mu * ktilde[j] * Ttime
        else:
            nut_tilde = np.zeros_like(utilde)

        phi_tilde = {}
        phi_tilde['u'] = utilde
        phi_tilde['w'] = wtilde
        phi_tilde['k'] = ktilde
        phi_tilde['e'] = etilde
        phi_tilde['T'] = Ttilde
        phi_tilde['nut'] = nut_tilde

        # Compute the pressure gradient
        dPdx = compute_dPdx(T_next, T_i, dx, dz,params['g'],params['beta'])
        # Advance the next iteration
        u_next = solveUMom(phi_next, phi_i,phi_tilde,dx, dz, zvec,params, Ulower, Uupper,RHSforcing=dPdx)
        w_next = solveMass(phi_next, phi_i, dx, zvec, WBC)
        T_next = solveTemp(phi_next, phi_i,phi_tilde,dx, dz, zvec,params, Tlower, Tupper)
        k_next = solveTKE(phi_next, phi_i, phi_tilde,dx, dz, zvec,params, klower, kupper)
        e_next = solveEps(phi_next, phi_i, phi_tilde,dx, dz, zvec,params, elower, eupper)
        
        # Check convergence
        if verbose:
            print("--> Convergence: ",k, np.linalg.norm(u_next-u_old))

        # TODO: add k and eps to convergence test
        residual = np.linalg.norm(u_next - u_old) + np.linalg.norm(w_next - w_old) + np.linalg.norm(k_next - k_old) + np.linalg.norm(e_next - e_old) + np.linalg.norm(T_next - T_old) 
        if residual < 5*tol:
        #if converged(u_next, u_old, tol):
            break

    return phi_next , residual

def marchWakeBL(phi_init,numSteps, zvec, params, BCdict, g=0, dx_init = 1e-2, fixed_dx = True,clipping=False,verbose=False):
    """
    March a solution given an initial condition
    """
    varnames = [k for k, g in phi_init.items()]
    residual = 100
    previous_residual = 100
    residual_switch = 0.05
    max_dx = 100.0
    dx = dx_init
    ixvec = [] 
    xvec = [] 
    continue_flag = True
    diverged_count = 0
    
    # Initialize the solution vector
    solvec = {}
    for v in varnames:
        solvec[v] = []

    # Loop through each x
    #for ix, x in enumerate(xvec):
    ix = -1 
    while continue_flag and ix < numSteps:
        ix +=1 
        ixvec.append(ix)
        print("Working on step: ",ix, ", dx: ",dx, "prev residual: ",residual)

        if ix==0:
            # Set the initial values
            for v in varnames:
                solvec[v].append(phi_init[v]+0.0)
            xvec.append(0)
        else:
            xvec.append(xvec[-1]+dx)
            phi_i = {}
            for v in varnames:
                phi_i[v] = solvec[v][-1]
            phi_next , residual  = advanceSol(phi_i,
                                  dx, zvec, params, BCdict, 
                                  g=g, verbose=verbose,clipping=clipping)

            # Append phi_next to solvec
            for v in varnames:
                solvec[v].append(phi_next[v]+0.0)

            #residual for stationary solution 
            residual = 0
            for v in varnames:
                residual += np.linalg.norm(solvec[v][-1] - solvec[v][-2])

            if not fixed_dx:
                if (residual/previous_residual > 2.0) and residual>1e-5:
                    diverged_count += 1
                else:
                    diverged_count = 0  
                if (diverged_count >= 2): dx /= 10
                if (residual < residual_switch) and (dx < max_dx) and (residual/previous_residual  > 0.2) and (residual/previous_residual < 1.0): 
                    dx *= 2

                previous_residual = residual;

            if residual < 1e-10:
                continue_flag = False

    return np.array(solvec['u']), np.array(solvec['w']),np.array(solvec['T']),np.array(solvec['k']),np.array(solvec['e']),xvec
