#!/usr/bin/env python

import os, sys
scriptpath = os.path.dirname(os.path.realpath(__file__))
extradirs = [scriptpath, 
             os.path.join(scriptpath, 'submodules/inputdicthelper/'),
            ]
for x in extradirs: sys.path.insert(1, x)

import inputdicthelper as idh
import copy
from collections import OrderedDict
import numpy as np
import SANDwake3D_base as sdb
from functools import partial

def calc_ghat(params):
    """Calculate the gravitational acceleration vector 

    Note: this currently enforces it so that it is always pointing in
    the negative z direction

    """
    return np.array([0.0, 0.0, -np.abs(params["g"])])

def rhs_f_nhalf(field, f_n, phi_tilde, dz_nut_tilde, dx, dz, params):
    """
    Go from n to n+1/2
    """
    sigma = f"""sigma{field}"""

    u_tilde, w_tilde = phi_tilde["u"], phi_tilde["w"]
    nu_total = params["nu"] + phi_tilde["nut"] / params[sigma]
    w_total = w_tilde - dz_nut_tilde / params[sigma]

    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz
    inv_dz2 = 1.0 / (dz * dz)
    inv_half_dx = 2.0 * inv_dx

    rhs = np.empty_like(f_n)
    rhs[:, 0] = (
        u_tilde[:, 0] * f_n[:, 0] * inv_half_dx
        - w_total[:, 0] * inv_dz * (f_n[:, 1] - f_n[:, 0])
        + nu_total[:, 0] * inv_dz2 * (f_n[:, 0] - 2.0 * f_n[:, 1] + f_n[:, 2])
    )

    rhs[:, 1:-1] = (
        u_tilde[:, 1:-1] * f_n[:, 1:-1] * inv_half_dx
        - w_total[:, 1:-1] * inv_dz * 0.5 * (f_n[:, 2:] - f_n[:, :-2])
        + nu_total[:, 1:-1] * inv_dz2 * (f_n[:, 2:] - 2.0 * f_n[:, 1:-1] + f_n[:, :-2])
    )

    rhs[:, -1] = (
        u_tilde[:, -1] * f_n[:, -1] * inv_half_dx
        - w_total[:, -1] * inv_dz * (f_n[:, -1] - f_n[:, -2])
        + nu_total[:, -1] * inv_dz2 * (f_n[:, -1] - 2.0 * f_n[:, -2] + f_n[:, -3])
    )

    return rhs


def rhs_f_np1(field, f_nhalf, phi_tilde, dy_nut_tilde, dx, dy, params):
    """
    Go from n+1/2 to n+1 for the u-momentum equation
    """
    sigma = f"""sigma{field}"""

    u_tilde, v_tilde = phi_tilde["u"], phi_tilde["v"]
    nu_total = params["nu"] + phi_tilde["nut"] / params[sigma]
    v_total = v_tilde - dy_nut_tilde / params[sigma]

    inv_half_dx = 2.0 / dx
    inv_dy = 1.0 / dy
    inv_dy2 = 1.0 / (dy * dy)

    rhs = np.empty_like(f_nhalf)
    rhs[0, :] = (
        u_tilde[0, :] * f_nhalf[0, :] * inv_half_dx
        - v_total[0, :] * inv_dy * (f_nhalf[1, :] - f_nhalf[0, :])
        + nu_total[0, :]
        * inv_dy2
        * (f_nhalf[0, :] - 2.0 * f_nhalf[1, :] + f_nhalf[2, :])
    )

    rhs[1:-1, :] = (
        u_tilde[1:-1, :] * f_nhalf[1:-1, :] * inv_half_dx
        - v_total[1:-1, :] * inv_dy * 0.5 * (f_nhalf[2:, :] - f_nhalf[:-2, :])
        + nu_total[1:-1, :]
        * inv_dy2
        * (f_nhalf[2:, :] - 2.0 * f_nhalf[1:-1, :] + f_nhalf[:-2, :])
    )

    rhs[-1, :] = (
        u_tilde[-1, :] * f_nhalf[-1, :] * inv_half_dx
        - v_total[-1, :] * inv_dy * (f_nhalf[-1, :] - f_nhalf[-2, :])
        + nu_total[-1, :]
        * inv_dy2
        * (f_nhalf[-1, :] - 2.0 * f_nhalf[-2, :] + f_nhalf[-3, :])
    )
    return rhs


def rhs_f_extra_forcing(field, phi, phi_n, aux_vars, params, x, dx, dy, dz,
                        fturbines={}, fcoriolis={}):
    """
    Compute the RHS extra forcing if necessary
    """
    if field == "k":
        fk_const = params["fk_const"] if "fk_const" in params else 0.0
        return (
            aux_vars["Pk"]
            - phi["eps"]
            + aux_vars["Gb"]
            + fk_const
        )

    if field == "eps":
        nu = params["nu"]
        C1eps = params["C1eps"]
        C2eps = params["C2eps"]
        C3eps = params["C3eps_use"]
        feps_const = params["feps_const"] if "feps_const" in params else 0.0
        Tscale = time_scale(phi["k"], phi["eps"], nu)
        return (
            C1eps
            / Tscale
            * ( aux_vars["Pk"] + (1.0 - C3eps)*aux_vars["Gb"] )
            - C2eps * phi["eps"] / Tscale
            + feps_const
        )

    if field == "p":
        term1 = - (
                   aux_vars["dx_u"]**2 + aux_vars["dy_v"]**2 + aux_vars["dz_w"]**2
                  ) - 2.0*(  aux_vars["dx_v"]*aux_vars["dy_u"]
                             + aux_vars["dx_w"]*aux_vars["dz_u"]
                             + aux_vars["dy_w"]*aux_vars["dz_v"])
        term2 = ( aux_vars["dx_nut"]*(aux_vars["dyy_u"] + aux_vars["dzz_u"]) +
                  aux_vars["dy_nut"]*(aux_vars["dyy_v"] + aux_vars["dzz_v"]) +
                  aux_vars["dz_nut"]*(aux_vars["dyy_w"] + aux_vars["dzz_w"]) 
                  )
        term3 = ( aux_vars["dyx_nut"]*aux_vars["dy_u"] +
                  aux_vars["dyy_nut"]*aux_vars["dy_v"] +
                  aux_vars["dyz_nut"]*aux_vars["dy_w"]
                  )
        term4 = ( aux_vars["dzx_nut"]*aux_vars["dz_u"] +
                  aux_vars["dyz_nut"]*aux_vars["dz_v"] +
                  aux_vars["dzz_nut"]*aux_vars["dz_w"]
                  )

        g = params['g']
        beta = params['beta']
        termT  = g*beta*aux_vars["dz_T"]
        return term1 + term2 + term3 + term4 + termT
    if field in ("u"):
        rho      = 1.0  # REMINDER -- the rho's cancel out in the Poisson equation
        fturb    = fturbines['u'] if 'u' in fturbines else 0.0
        fcor     = fcoriolis['u'] if 'u' in fcoriolis else 0.0
        fx       = -1.0/rho*aux_vars["dx_p"] + fturb + fcor

        # This will be obsoleted soon!!
        if 'turbforcing' in params:
            ym       = params['ym']
            zm       = params['zm']
            turbdict = params['turbforcing']
            xturb    = turbdict['turbx']
            if np.abs(xturb-x)< (0.5*dx-1.0E-3):
                zhh  = turbdict['zhh']
                yhh  = turbdict['turby']
                R    = turbdict['turbD']*0.5
                Uinf = sdb.rotorAvgUh(ym, zm, phi_n['u'], phi_n['v'], yhh, zhh, R)
                turbADfunc = turbdict['turbfunc']
                fx +=  turbADfunc(dx, ym, zm, Uinf, phi_n, turbdict)
        return fx

    if field in ( "v"):
        fturb    = fturbines['v'] if 'v' in fturbines else 0.0
        fcor     = fcoriolis['v'] if 'v' in fcoriolis else 0.0
        rho = 1.0  # REMINDER -- the rho's cancel out in the Poisson equation 
        return -1.0/rho*aux_vars["dy_p"] + fturb + fcor
    if field in ( "w"):
        fturb    = fturbines['w'] if 'w' in fturbines else 0.0
        fcor     = fcoriolis['w'] if 'w' in fcoriolis else 0.0
        rho = 1.0  # REMINDER -- the rho's cancel out in the Poisson equation 
        return -1.0/rho*aux_vars["dz_p"] + aux_vars["B_z"] + fturb + fcor
        
    # NOTE: This conditional down here needs to be generalized
    if field in ( "T"):
        fx_const = params["fx_const"] if "fx_const" in params else 0.0
        return fx_const
    return 0


def advanceF(field, phi_np1old, phi_n, phi_tilde, aux_vars, x, dx, dy, dz,
             params, bcvar,
             fturbines={},
             fcoriolis={}):
    """
    Advance the field one full step
    """
    u_tilde, v_tilde, w_tilde = phi_tilde["u"], phi_tilde["v"], phi_tilde["w"]
    nut_tilde = phi_tilde["nut"]
    ny, nz = u_tilde.shape

    # Load parameters
    nu = params["nu"]
    sigma = f"""sigma{field}"""

    # --- differentiation stencils ---
    #                  j-1  j  j+1
    irow = np.array([0, 1, 0])
    dcen = np.array([-1, 0, 1]) * 0.5
    d2cen = np.array([1, -2, 1])
    # -------------------------------

    nu_total = nu + nut_tilde  / params[sigma]
    v_total = v_tilde - aux_vars["dy_nut"]  / params[sigma]
    w_total = w_tilde - aux_vars["dz_nut"]  / params[sigma]

    rhs_extra_forcing = rhs_f_extra_forcing(field, phi_tilde, phi_n, aux_vars, params, x, dx, dy, dz, fturbines, fcoriolis)

    # First sweep: n -> n+1/2
    # -----------------------
    f_nhalf = np.zeros((ny, nz))
    rhs_nhalf = (
        rhs_f_nhalf(field, phi_n[field], phi_tilde, aux_vars["dz_nut"], dx, dz, params)
        + rhs_extra_forcing
    )
    inv_half_dx = 2.0 / dx
    inv_dy = 1.0 / dy
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz = 1.0 / dz
    inv_dz2 = 1.0 / (dz * dz)
    row_lo, dentry_lo = sdb.applyBC(bcvar["ylo"], "lower", dy)
    row_hi, dentry_hi = sdb.applyBC(bcvar["yhi"], "upper", dy)
    if not isinstance(dentry_lo, np.ndarray):
        dentry_lo = dentry_lo*np.ones(nz)
    if not isinstance(dentry_hi, np.ndarray):
        dentry_hi = dentry_hi*np.ones(nz)
    for j in range(nz):
        lhs_nhalf = np.zeros((ny, 3))
        # == Set up the LHS matrices ==
        lhs_nhalf[1:-1, :] = (
            u_tilde[1:-1, j, np.newaxis] * inv_half_dx * irow
            + v_total[1:-1, j, np.newaxis] * inv_dy * dcen
            - nu_total[1:-1, j, np.newaxis] * inv_dy2 * d2cen
        )
        # Apply BC's
        lhs_nhalf[0, :] = row_lo
        lhs_nhalf[-1, :] = row_hi
        rhs_nhalf[0, :] = dentry_lo[j]
        rhs_nhalf[-1, :] = dentry_hi[j]
        # print(f'j = {j}\nrhs_nhalf = ',rhs_nhalf[:,j], '\nLHS = ', lhs_nhalf)
        # Solve the triadiagonal system
        f_nhalf[:, j] = sdb.solvetridiag(lhs_nhalf, rhs_nhalf[:, j])

    # print(f'f_nhalf = ',f_nhalf)
    # Second sweep: n+1/2 -> n+1
    # -----------------------
    f_np1 = np.zeros((ny, nz))
    rhs_np1 = (
        rhs_f_np1(field, f_nhalf, phi_tilde, aux_vars["dy_nut"], dx, dy, params)
        + rhs_extra_forcing
    )
    # == Set up the LHS matrices ==
    row_lo_base, dentry_lo_base = sdb.applyBC(bcvar["zlo"], "lower", dz)
    row_hi_base, dentry_hi_base = sdb.applyBC(bcvar["zhi"], "upper", dz)
    if not isinstance(dentry_lo_base, np.ndarray):
        dentry_lo_base = dentry_lo_base*np.ones(ny)
    if not isinstance(dentry_hi_base, np.ndarray):
        dentry_hi_base = dentry_hi_base*np.ones(ny)
    for i in range(ny):
        lhs_np1 = np.zeros((nz, 3))
        lhs_np1[1:-1, :] = (
            u_tilde[i, 1:-1, np.newaxis] * inv_half_dx * irow
            + w_total[i, 1:-1, np.newaxis] * inv_dz * dcen
            - nu_total[i, 1:-1, np.newaxis] * inv_dz2 * d2cen
        )
        # Apply BC's
        row_lo, dentry_lo = row_lo_base, dentry_lo_base[i]
        row_hi, dentry_hi = row_hi_base, dentry_hi_base[i]
        if i == 0:
            row_lo, dentry_lo = sdb.applyBC(
                {"type": "dirichlet", "value": phi_tilde[field][i, 0]}, "lower", dz
            )
        elif i == ny - 1:
            row_hi, dentry_hi = sdb.applyBC(
                {"type": "dirichlet", "value": phi_tilde[field][i, -1]}, "lower", dz
            )
        lhs_np1[0, :] = row_lo
        lhs_np1[-1, :] = row_hi
        rhs_np1[:, 0] = dentry_lo
        rhs_np1[:, -1] = dentry_hi
        # print(f'i = {i}\nrhs_np1 = ',rhs_np1[i,:], '\nLHS = ', lhs_np1)
        # Solve the triadiagonal system
        f_np1[i, :] = sdb.solvetridiag(lhs_np1, rhs_np1[i, :], verbose=False)
        # print('f_np1 = ',f_np1[i,:])

    return f_np1


def advanceMass(
    field,
    phi_np1old,
    phi_n,
    phi_tilde,
    aux_vars,
    x,
    dx,
    dy,
    dz,
    params,
    bcvar,
):
    """
    Advance the continuity equation one full step
    """
    u_np1, u_n = phi_np1old["u"], phi_n["u"]
    v_np1 = phi_np1old["v"]
    w_np1 = phi_np1old["w"]

    dz_w = np.gradient(w_np1, dz, axis=1, edge_order=1)
    rhs = -dy * (u_np1 - u_n) / (dx) - dy * dz_w

    # == Set up the LHS matrices ==
    _, dentry_lo = sdb.applyBC(bcvar["ylo"], "lower", dy)
    v_np1 = np.cumsum(rhs, axis=0) + dentry_lo

    return v_np1


def RHS_p_nhalf(p, dtau, dy, dz):
    """
    Go from m to m+1/2
    """
    Ny, Nz = p.shape
    RHS = np.zeros(p.shape)
    RHS[1:Ny-1, 1:Nz-1] = p[1:Ny-1, 1:Nz-1] / (dtau * 0.5) - (
        (p[1:Ny-1, 2:Nz] - 2.0 * p[1:Ny-1, 1:Nz-1] + p[1:Ny-1, 0:Nz-2]) / (dz**2)
    )
    return RHS

def RHS_p_np1(p, dtau, dy, dz):
    """
    Go from m+1/2 to m+1
    """
    Ny, Nz = p.shape
    RHS = np.zeros(p.shape)
    RHS[1:Ny-1, 1:Nz-1] = p[1:Ny-1, 1:Nz-1] / (dtau * 0.5) - (
        (p[2:Ny, 1:Nz-1] - 2.0 * p[1:Ny-1, 1:Nz-1] + p[0:Ny-2, 1:Nz-1]) / (dy**2)
    )
    return RHS

def advanceP(
    field,
    phi_np1old,
    phi_n,
    phi_tilde,
    aux_vars,
    x,
    dx,
    dy,
    dz,
    params,
    bcvar,
    fturbines={},
    fcoriolis={},
):
    """
    Advance the pressure Poisson equation one full step (in artificial time!)
    """
    p_np1, p_n = phi_np1old["p"], phi_n["p"]
    ny, nz = p_n.shape
    
    # --- differentiation stencils ---
    #                  j-1  j  j+1
    irow = np.array([0, 1, 0])
    dcen = np.array([-1, 0, 1]) * 0.5
    d2cen = np.array([1, -2, 1])
    # -------------------------------

    dtau = params['dtau']

    rhs_extra_forcing = rhs_f_extra_forcing(field, phi_tilde, phi_n, aux_vars, params, x, dx, dy, dz)

    # First sweep: m -> m+1/2
    # -----------------------
    p_nhalf = np.zeros((ny, nz))
    rhs_nhalf = (
        RHS_p_nhalf(p_n, dtau, dy, dz) + rhs_extra_forcing
    )
    #inv_half_dx = 2.0 / dx
    inv_half_dtau = 2.0 / dtau
    inv_dy = 1.0 / dy
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz = 1.0 / dz
    inv_dz2 = 1.0 / (dz * dz)
    row_lo, dentry_lo = sdb.applyBC(bcvar["ylo"], "lower", dy)
    row_hi, dentry_hi = sdb.applyBC(bcvar["yhi"], "upper", dy)
    if not isinstance(dentry_lo, np.ndarray):
        dentry_lo = dentry_lo*np.ones(nz)
    if not isinstance(dentry_hi, np.ndarray):
        dentry_hi = dentry_hi*np.ones(nz)
    for j in range(nz):
        lhs_nhalf = np.zeros((ny, 3))
        # == Set up the LHS matrices ==
        lhs_nhalf[1:-1, :] = (
            inv_half_dtau * irow + inv_dy2 * d2cen
        )
        # Apply BC's
        lhs_nhalf[0, :] = row_lo
        lhs_nhalf[-1, :] = row_hi
        rhs_nhalf[0, :] = dentry_lo[j]
        rhs_nhalf[-1, :] = dentry_hi[j]
        # Solve the triadiagonal system
        p_nhalf[:, j] = sdb.solvetridiag(lhs_nhalf, rhs_nhalf[:, j])

    # Second sweep: m+1/2 -> m+1
    # -----------------------
    p_np1 = np.zeros((ny, nz))
    rhs_np1 = (
        RHS_p_nhalf(p_nhalf, dtau, dy, dz) + rhs_extra_forcing
    )
    # == Set up the LHS matrices ==
    row_lo_base, dentry_lo_base = sdb.applyBC(bcvar["zlo"], "lower", dz)
    row_hi_base, dentry_hi_base = sdb.applyBC(bcvar["zhi"], "upper", dz)
    if not isinstance(dentry_lo_base, np.ndarray):
        dentry_lo_base = dentry_lo_base*np.ones(ny)
    if not isinstance(dentry_hi_base, np.ndarray):
        dentry_hi_base = dentry_hi_base*np.ones(ny)
    for i in range(ny):
        lhs_np1 = np.zeros((nz, 3))
        lhs_np1[1:-1, :] = (
            inv_half_dtau * irow + inv_dz2 * d2cen
        )
        # Apply BC's
        row_lo, dentry_lo = row_lo_base, dentry_lo_base[i]
        row_hi, dentry_hi = row_hi_base, dentry_hi_base[i]
        if i == 0:
            row_lo, dentry_lo = sdb.applyBC(
                {"type": "dirichlet", "value": phi_tilde[field][i, 0]}, "lower", dz
            )
        elif i == ny - 1:
            row_hi, dentry_hi = sdb.applyBC(
                {"type": "dirichlet", "value": phi_tilde[field][i, -1]}, "lower", dz
            )
        lhs_np1[0, :] = row_lo
        lhs_np1[-1, :] = row_hi
        rhs_np1[:, 0] = dentry_lo
        rhs_np1[:, -1] = dentry_hi
        # Solve the triadiagonal system
        p_np1[i, :] = sdb.solvetridiag(lhs_np1, rhs_np1[i, :], verbose=False)
        # print('f_np1 = ',f_np1[i,:])
    return p_np1


def time_scale(k, eps, nu):
    """
    Compute time scale
    """
    return np.fmax(k / eps, 6.0 * np.sqrt(nu / np.abs(eps)))


def get_nut(phi, cmu, nu):
    """
    Compute turbulent viscosity
    """
    tscale = time_scale(phi["k"], phi["eps"], nu)
    return cmu * phi["k"] * tscale


def advanceSystemKEPS(
    phi_n, x, dx, dy, dz, params, allbcs, eqnsys, maxiter=100, tol=1.0e-6, freezevar=0.0, verbose=False
):
    """
    Advance equation system 1 step in x
    """
    varlist = [v for v, g in eqnsys.items()]
    convergevars = copy.deepcopy(varlist)
        
    invdx   = 1.0/dx
    
    if "nut" not in phi_n:
        phi_n["nut"] = get_nut(phi_n, params["Cmu"], params["nu"])
    phi_n1 = copy.deepcopy(phi_n)

    dxphi = lambda phin1, phin, v, invdx: (phin1[v] - phin[v])*invdx

    # Add these sigmas so advanceF can treat everything the same
    params['sigmau'] = 1.0
    params['sigmav'] = 1.0
    params['sigmaw'] = 1.0
    
    # Create a registry of which BC functions are used in this system
    BCfuncreg = {}
    usewallmodel = False
    for v, bcgroup in allbcs.items():
        for face, bc in bcgroup.items():
            if (bc['type'] == 'bcfunc') and (bc['tag'] not in BCfuncreg):
                BCfuncreg[bc['tag']] = bc['func']
                if bc['func'] == MO_wallmodel:
                    usewallmodel = True

    # Compute turbine forces if necessary
    fturbines  = {}
    turboutput = None
    if 'turbinelist' in params:
        turbparams = params['turbinelist']
        ym         = params['ym']
        zm         = params['zm']
        fturbines, turboutput  = sdb.computeTurbineForces(x, dx, ym, zm,
                                                          phi_n, turbparams,
                                                          verbose=verbose)
    # Compute Coriolis forces if necessary
    fcoriolis  = {}
    if 'Coriolis' in params:
        coriolisparams = params['Coriolis']
        fcoriolis = sdb.computeCoriolisForces(phi_n, coriolisparams,
                                              verbose=verbose)

    # Reset convergence
    varconverged = {}
    for v in varlist: varconverged[v] = False
    
    # Loop until converged
    for k in range(maxiter):
        phi_next = OrderedDict()
        phi_n1["nut"] = get_nut(phi_n1, params["Cmu"], params["nu"])
        phi_tilde = sdb.getTildeVars(phi_n1, phi_n)

        # Auxiliary variables
        aux_vars = OrderedDict()
        aux_vars["dy_nut"], aux_vars["dz_nut"] = np.gradient(
            phi_tilde["nut"], dy, dz, edge_order=1
        )
        aux_vars["dy_u"], aux_vars["dz_u"] = np.gradient(
            phi_tilde["u"], dy, dz, edge_order=1
        )
        # These aux_vars needed for pressure Poisson
        aux_vars["dy_v"], aux_vars["dz_v"] = np.gradient(
            phi_tilde["v"], dy, dz, edge_order=1
        )
        aux_vars["dy_w"], aux_vars["dz_w"] = np.gradient(
            phi_tilde["w"], dy, dz, edge_order=1
        )
        aux_vars["dy_p"], aux_vars["dz_p"] = np.gradient(
            phi_tilde["p"], dy, dz, edge_order=1
        )
        aux_vars["dx_u"] = dxphi(phi_n1, phi_n, "u", invdx)
        aux_vars["dx_v"] = dxphi(phi_n1, phi_n, "v", invdx)
        aux_vars["dx_w"] = dxphi(phi_n1, phi_n, "w", invdx)
        aux_vars["dx_p"] = dxphi(phi_n1, phi_n, "p", invdx)
        aux_vars["dx_nut"] = dxphi(phi_n1, phi_n, "nut", invdx)
        aux_vars["dyy_u"], aux_vars["dyz_u"] = np.gradient(
            aux_vars["dy_u"], dy, dz, edge_order=1
        )
        aux_vars["dyz_u"], aux_vars["dzz_u"] = np.gradient(
            aux_vars["dz_u"], dy, dz, edge_order=1
        )
        aux_vars["dyy_v"], aux_vars["dyz_v"] = np.gradient(
            aux_vars["dy_v"], dy, dz, edge_order=1
        )
        aux_vars["dyz_w"], aux_vars["dzz_w"] = np.gradient(
            aux_vars["dz_w"], dy, dz, edge_order=1
        )
        aux_vars["dyz_v"], aux_vars["dzz_v"] = np.gradient(
            aux_vars["dz_v"], dy, dz, edge_order=1
        )
        aux_vars["dyy_w"], aux_vars["dyz_w"] = np.gradient(
            aux_vars["dy_w"], dy, dz, edge_order=1
        )

        # Nu_t cross-derivatives
        aux_vars["dyx_nut"], aux_vars["dzx_nut"] = np.gradient(
            aux_vars["dx_nut"], dy, dz, edge_order=1
        )
        aux_vars["dyy_nut"], aux_vars["dyz_nut"] = np.gradient(
            aux_vars["dy_nut"], dy, dz, edge_order=1
        )
        aux_vars["dyz_nut"], aux_vars["dzz_nut"] = np.gradient(
            aux_vars["dz_nut"], dy, dz, edge_order=1
        )

        aux_vars["dy_T"], aux_vars["dz_T"] = np.gradient(
            phi_tilde["T"], dy, dz, edge_order=1
        )
        # Compute the production term
        aux_vars["Pk"] = ( 2.0*(
                                 aux_vars["dx_u"]**2 +
                                 aux_vars["dy_v"]**2 +
                                 aux_vars["dz_w"]**2
                                ) + 
                           ( aux_vars["dy_u"] + aux_vars["dx_v"] )**2 +
                           ( aux_vars["dz_v"] + aux_vars["dy_w"] )**2 +
                           ( aux_vars["dz_u"] + aux_vars["dx_w"] )**2
                           )
        aux_vars["Pk"] *= phi_tilde["nut"]

        # Compute the bouyancy TKE term
        ghat = calc_ghat(params)
        aux_vars["Gb"] = params["beta"]*ghat[2]*phi_tilde["nut"]/params["sigmaT"]*aux_vars["dz_T"]

        # Compute the Boussinesq bouyancy TKE term
        if 'Tref_use' not in params:
            params['Tref_use'] = phi_n['T'].copy()
        if ('Tref' not in params) or (params['Tref'] is None):
            Tref = params['Tref_use']
        else:
            Tref = params['Tref']
        aux_vars["B_z"] = -ghat[2]*params["beta"]*(phi_tilde["T"] - Tref)

        # Update the boundary conditions (if necessary)
        bcdebug={}
        updatedbcs = copy.deepcopy(allbcs)
        for tag, bcfunc in BCfuncreg.items():
            # LCC: NEED TO DECIDE ON FUNCTION SIGNATURE HERE
            newbc = bcfunc(phi_tilde, aux_vars, params, dy, dz,
                           debugout=verbose)
            # Capture any debug info from the BC
            if isinstance(newbc, tuple):
                newbcvals, bcdebug[tag] = newbc[0], newbc[1]
            else:
                newbcvals = newbc
            for v, vbc in newbcvals.items():
                updatedbcs[v].update(vbc)

        # Calculate C3eps
        if usewallmodel and params['C3eps'] is None:
            params['C3eps_use'] = getCeps3(params['zlo']/params['Lnext'])
        else:
            params['C3eps_use'] = params['C3eps']
        bcdebug['C3eps_use'] = params['C3eps_use']

        # Loop over all variables
        for v in varlist:
            bcvar = updatedbcs[v] #allbcs[v]

            sigma = f"""sigma{v}"""

            if (not varconverged[v]):
                phi_next[v] = eqnsys[v](
                    v,
                    phi_n1,
                    phi_n,
                    phi_tilde,
                    aux_vars,
                    x,
                    dx,
                    dy,
                    dz,
                    params,
                    bcvar,
                    fturbines=fturbines,
                    fcoriolis=fcoriolis,
                )
            else:
                phi_next[v] = phi_n1[v].copy()
            varconverged[v], _ = sdb.convergetest({v:phi_next[v]},
                                                  {v:phi_n1[v]},
                                                  tol*freezevar, testvars=[v])

        # Test for convergence
        converged, convergedat = sdb.convergetest(phi_next, phi_n1, tol, testvars=convergevars)
        phi_n1 = copy.deepcopy(phi_next)
        if verbose>1:
            print(f"[{k}] "+ ", ".join(f"{k}: {v:0.4e}" for k, v in convergedat.items()))
        if converged:
            break

    # Print output any BC debug information 
    if verbose:
        dil = aux_vars['dx_u'] + aux_vars['dy_v'] + aux_vars['dz_w']
        bcdebug['avg_dil'] = np.linalg.norm(dil)/dil.size
        for tag, debugout in bcdebug.items():
            print(tag+': '+repr(debugout))
        if turboutput is not None:
            for turbinfo in turboutput:
                print(turbinfo)

    # Check if k hit maxiter:
    # -->TODO!
    if (k==maxiter-1):
        print('WARNING: CONVERGENCE FAILED: ')
        print(f"[{k}] "+ ", ".join(f"{k}: {v:0.4e}" for k, v in convergedat.items()))
    return phi_n1


########################################################
def phi_m(z, L):
    """
    Compute Monin-Obukhov wind shear
    """
    if L < 0:
        return (1 - 16.0 * (z / L)) ** (-0.25)

    if L == float("inf"):
        return np.ones_like(z)

    return 1 + 5 * z / L


def phi_e(z, L):
    """
    Compute Monin-Obukhov epsilon variable
    """
    phim = phi_m(z, L)

    if L < 0:
        phie = 1 - z / L
    elif L == float("inf"):
        phie = phim
    else:
        phie = phim - z / L
    return phie
    

def MO_u0(z, z0, K, L, ustar):
    """
    See equation 16 in Alinot and Masson
    """
    phim = phi_m(z, L)
    if L < 0:
        return (
            ustar
            / K
            * (
                np.log(z / z0)
                + np.log((8 * phim**4) / ((phim + 1) ** 2 * (phim**2 + 1)))
                - np.pi / 2
                + 2 * np.arctan(1 / (phim))
            )
        )
    else:
        return ustar / K * (np.log(z / z0) + phim - 1)

def MO_k0(z, L, ustar):
    """
    See equation 20 in Alinot and Masson
    """
    phim = phi_m(z, L)
    phie = phi_e(z, L)
    return 5.48 * ustar**2 * (phie / phim) ** 0.5
    

def MO_eps(z, K, L, ustar):
    phie = phi_e(z, L)
    return (ustar**3)/(K*z)*phie

def MO_Tfunc(z, z0, K, L, Tstar, g, cp):
    phim = phi_m(z, L)
    if L == float('inf'):
        dT = 0.0
    elif L < 0:
        dT = Tstar/K*( np.log(z/z0)
                       - 2.0*np.log(0.5*(1+phim**-2)) ) - g/cp*(z-z0)
    else:
        dT = Tstar/K*( np.log(z/z0)
                       + phim - 1 ) - g/cp*(z-z0)
    return dT 

# Initial condition stuff
def init_U_ABL(z, param):
    """
    Initialize ABL profile from Monin-Obukhov theory
    """
    z0 = param["z0"]
    L = param["L"]
    K = param["kappa"]
    ustar = param["ustar"]

    return MO_u0(z, z0, K, L, ustar)
    

def init_e_ABL(z, param):
    L = param["L"]
    K = param["kappa"]
    ustar = param["ustar"]
    return MO_eps(z, K, L, ustar)


def init_k_ABL(z, param):
    L = param["L"]
    ustar = param["ustar"]

    return MO_k0(z, L, ustar)

def init_T_ABL(z, param):
    g   = 0.0 #param["g"]
    cp  = param["cp"]
    L = param["L"]
    K = param["kappa"]
    Tw  = param["Tw"]
    qw  = param["qw"]
    rho = param["rho"]
    ustar = param["ustar"]
    z0  = param["z0"]
    Tw  = param["Tw"]
    Tstar = -qw/(rho*cp*ustar)    # Eq. (15), Alinot & Masson

    return Tw+MO_Tfunc(z, z0, K, L, Tstar, g, cp)


def set_e_init(zvec, dz, u, k, params):
    Cmu = params["Cmu"]
    e_init = np.zeros_like(u)

    dcen = np.array([-1, 0, 1]) / (2 * dz)
    e_init[1:-1] = np.sqrt(
        Cmu
        * k[1:-1] ** 2
        * (dcen[0] * u[:-2] + dcen[1] * u[1:-1] + dcen[2] * u[2:]) ** 2
    )

    return e_init


def set_k_init(rvec, dr, u, params, k_factor=0.1):
    k_init = np.zeros_like(u)
    dcen = np.array([-1, 0, 1]) / (2 * dr)
    k_init[1:-1] = (dcen[0] * u[:-2] + dcen[1] * u[1:-1] + dcen[2] * u[2:]) ** 2

    # scale k
    kmax = np.max(k_init)
    C = k_factor**2 * 2 / (3 * kmax)
    k_init *= C
    return k_init

def calcTref(xi, allphi, params):
    lastN = 1     # Average over the last N steps
    Tsol = allphi['T']
    if xi < lastN:
        Tslice = Tsol[0:(xi+1):,:,:]
    else:
        Tslice = Tsol[(xi-lastN+1):(xi+1),:,:] 
    Tavg = np.mean(Tslice, axis=0)
    params['Tref_use'] = Tavg 
    return 

########################################################
# Wall model stuff
# 
# See https://github.com/lawrenceccheung/AMRWind_RANSBC/blob/main/literature/Alinot-k_Eps_ABL_Stratified-2005.pdf

def MO_wallmodel(phi, aux, param, dy, dz, debugout=False):
    """
    Compute all wall model variables
    """
    useT = True if 'T' in phi.keys() else False
    
    z0  = param["z0"]
    K   = param["kappa"]
    nu  = param["nu"]
    zlo = param["zlo"]
    Cmu = param["Cmu"]

    L   = param["Lnext"] if "Lnext" in param else param["L"]
    
    nut   = get_nut(phi, Cmu, nu)[:,0]
    Uh    = np.sqrt(phi['u']**2 + phi['v']**2)
    dy_Uh, dz_Uh = np.gradient(Uh, dy, dz, edge_order=1)
    dz_u   = np.abs(dz_Uh[:,0])
    #dz_u   = np.abs(aux["dz_u"][:,0])
    ustar = np.sqrt((nu+nut)*dz_u)

    # Calculate Monin-Obukhov lengths
    uh_lo = MO_u0(zlo, z0, K, L, ustar)
    klo   = MO_k0(zlo, L, ustar)
    epslo = MO_eps(zlo, K, L, ustar)

    debugoutput = {
        'ustar':np.mean(ustar),
    }

    # break the WM velocity into U and V components
    ulo   = uh_lo*phi['u'][:,0]/Uh[:,0]
    vlo   = uh_lo*phi['v'][:,0]/Uh[:,0]
    
    # Assign the boundary conditions for each variable
    ubc = {
        'zlo':{'type':'dirichlet', 'value':ulo},
        }
    vbc = {
        'zlo':{'type':'dirichlet', 'value':vlo},
        }
    kbc = {
        'zlo':{'type':'dirichlet', 'value':klo},
        }
    ebc = {
        'zlo':{'type':'dirichlet', 'value':epslo},
        }    
    allbc = {
        'u':ubc,
        'v':vbc,
        'k':kbc,
        'eps':ebc,
    }

    # Add temperature BC if required
    if useT:
        g   = param["g"]
        cp  = param["cp"]
        Tw  = param["Tw"]
        qw  = param["qw"]
        rho = param["rho"]

        Tstar = -qw/(rho*cp*ustar)    # Eq. (15), Alinot & Masson
        dT  = MO_Tfunc(zlo, z0, K, L, Tstar, g, cp)
        Tbc = {
            'zlo':{'type':'dirichlet', 'value':(Tw + dT)},
            }
        allbc['T'] = Tbc
        
        # Calculate the next L value
        invTstar = np.array([1/x if np.abs(x)>0 else float('inf') for x in Tstar ])
        Lnext = (ustar**2)*Tw/(K*g)*invTstar  # Eq. 12 in Alinot & Masson

        # TODO: generalize to use vector valued L in the future
        param['Lnext'] = np.mean(Lnext)
        
        debugoutput['Lnext'] = np.mean(Lnext)
        debugoutput['Tstar'] = np.mean(Tstar)
    
    if debugout:
        return allbc, debugoutput
    else:
        return allbc

########################################################
def getTypicalWMBC(Uinf, TBC, uBC_y=None, veerBC=None, dTdz=None, kinf=None):
    """
    Define the "typical" wall-model boundary conditions
    """

    # Decide on the type of BC for V
    if uBC_y is None:
        utype = 'neumann'
        uval  = 0.0
    else:
        utype = 'dirichlet'
        uval  = uBC_y

    # Decide on the type of BC for V
    if veerBC is None:
        vtype = 'neumann'
        veerV = 0.0
        v_zhi = 0.0
    else:
        vtype = 'dirichlet'
        veerV = veerBC
        v_zhi = veerV[-1]

    # Decide on the type of BC for T
    if not isinstance(TBC, np.ndarray):
        T_zhi = TBC
    else:
        T_zhi = TBC[-1]


    # Defined on the zhi k BC
    if kinf is None:
        ktype = 'neumann'
        kval  = 0.0
    else:
        ktype = 'dirichlet'
        kval  = kinf
        
    ubc = {}
    ubc['ylo'] = {'type':utype,     'value':uval}
    ubc['yhi'] = {'type':utype,     'value':uval}
    ubc['zlo'] = {'type':'bcfunc',  'value':None,
                  'tag':'ZLO_WALLBC',  'func':MO_wallmodel}
    ubc['zhi'] = {'type':'dirichlet', 'value':Uinf}

    vbc = {}
    vbc['ylo'] = {'type':vtype,      'value':veerV}
    vbc['yhi'] = {'type':vtype,      'value':veerV}
    vbc['zlo'] = {'type':'bcfunc',    'value':None,
                  'tag':'ZLO_WALLBC',  'func':MO_wallmodel}  
    vbc['zhi'] = {'type':'dirichlet', 'value':v_zhi}

    wbc = {}
    wbc['ylo'] = {'type':'dirichlet', 'value':0.0}
    wbc['yhi'] = {'type':'dirichlet', 'value':0.0}
    wbc['zlo'] = {'type':'dirichlet', 'value':0.0}
    wbc['zhi'] = {'type':'neumann', 'value':0.0}

    kbc = {}
    kbc['ylo'] = {'type':'neumann', 'value':0.0}
    kbc['yhi'] = {'type':'neumann', 'value':0.0}
    kbc['zlo'] = {'type':'bcfunc',  'value':None,
                  'tag':'ZLO_WALLBC',  'func':MO_wallmodel}
    #kbc['zhi'] = {'type':'neumann', 'value':0.0}
    kbc['zhi'] = {'type':ktype, 'value':kval}

    epsbc = {}
    epsbc['ylo'] = {'type':'neumann', 'value':0.0}
    epsbc['yhi'] = {'type':'neumann', 'value':0.0}
    epsbc['zlo'] = {'type':'bcfunc',  'value':None,
                    'tag':'ZLO_WALLBC',  'func':MO_wallmodel}
    epsbc['zhi'] = {'type':'neumann', 'value':0.0}

    Tbc = {}
    Tbc['ylo'] = {'type':'dirichlet', 'value':TBC}
    Tbc['yhi'] = {'type':'dirichlet', 'value':TBC}
    Tbc['zlo'] = {'type':'bcfunc',    'value':None,
                  'tag':'ZLO_WALLBC',  'func':MO_wallmodel}
    if dTdz is None:
        Tbc['zhi'] = {'type':'dirichlet', 'value':T_zhi}
    else:
        Tbc['zhi'] = {'type':'neumann', 'value':dTdz}

    pbc = {}
    pbc['ylo'] = {'type':'neumann', 'value':0.0}
    pbc['yhi'] = {'type':'neumann', 'value':0.0}
    pbc['zlo'] = {'type':'dirichlet', 'value':0.0}
    pbc['zhi'] = {'type':'neumann', 'value':0.0}

    allbc = {'u':ubc, 'v':vbc, 'w':wbc, 'k':kbc, 'eps':epsbc, 'T':Tbc, 'p':pbc}
    return allbc

def getUVfromUhVeer(Uh, veer, format='deg'):
    """
    Calculate the U, V velocity profiles from the Uh and veer profile
    """
    Nz = len(Uh)
    U  = np.zeros(Nz)
    V  = np.zeros(Nz)
    for i in range(Nz):
        U[i] = Uh[i]*np.cos(veer[i]*np.pi/180.0)
        V[i] = Uh[i]*np.sin(veer[i]*np.pi/180.0)
    return U, V

def getCeps3(zL):
    """
    See table 1 and equation 24 from Alinot and Masson
    """
    Ce3 = {}
    Ce3[-1] = {}
    Ce3[1]  = {}

    zLdivide = {1:0.33, -1:-0.25}

    # L > 0, z/L < 0.33
    Ce3[1][-1]  = [4.181,   33.994,  -442.398,  2368.12, -6043.544,  5970.776]
    Ce3[1][1]   = [5.225,   -5.269,   5.115,   -2.406,    0.435,     0.000]
    Ce3[-1][-1] = [-0.0609, -33.672, -546.880, -3234.06, -9490.792, -11163.202]
    Ce3[-1][1]  = [1.765,   17.1346,  19.165,   11.912,   3.821,     0.492]

    zLsign = -1 if zL<0.0 else 1
    zLmag  = -1 if zL < zLdivide[zLsign] else 1

    a = Ce3[zLsign][zLmag]
    Ceps3 = 0.0
    for n in range(6):
        Ceps3 += a[n]*(zL**n) 
    return Ceps3

########################################################
# Define the keps equation system
keps_eqns = OrderedDict()
keps_eqns["u"] = advanceF
keps_eqns["w"] = advanceF
keps_eqns["k"] = advanceF
keps_eqns["eps"] = advanceF
keps_eqns["v"] = advanceMass

# Define the keps equation system with temperature
kepsT_eqns = OrderedDict()
kepsT_eqns["u"] = advanceF
kepsT_eqns["w"] = advanceF
kepsT_eqns["k"] = advanceF
kepsT_eqns["eps"] = advanceF
kepsT_eqns['T']   = advanceF
kepsT_eqns["v"] = advanceF #advanceMass
kepsT_eqns["p"] = advanceP

# Use the same marchSystemBase in SANDWake3D_base to advance the equations
marchSystem = partial(sdb.marchSystemBase, postadvfunc=calcTref)

########################################################
# Define the input parameters

# Mesh parameters
meshsubdict = [
    {'key':'dx',   'required':True, 'type':(int, float), 'default':10.0, 'validate':(lambda x: (x>0.0)), 'help':'Mesh spacing in x',},
    {'key':'xmin', 'required':True, 'type':(int, float), 'default':10.0, 'validate':None,                'help':'Minimum x location',},
    {'key':'xmax', 'required':True, 'type':(int, float), 'default':10.0, 'validate':None,                'help':'Maximum x location',},
    {'key':'dy',   'required':True, 'type':(int, float), 'default':10.0, 'validate':(lambda x: (x>0.0)), 'help':'Mesh spacing in y',},
    {'key':'ymin', 'required':True, 'type':(int, float), 'default':10.0, 'validate':None,                'help':'Minimum y location',},
    {'key':'ymax', 'required':True, 'type':(int, float), 'default':10.0, 'validate':None,                'help':'Maximum y location',},
    {'key':'dz',   'required':True, 'type':(int, float), 'default':10.0, 'validate':(lambda x: (x>0.0)), 'help':'Mesh spacing in z',},
    {'key':'zmin', 'required':True, 'type':(int, float), 'default':10.0, 'validate':(lambda x: (x>0.0)), 'help':'Minimum z height',},
    {'key':'zmax', 'required':True, 'type':(int, float), 'default':10.0, 'validate':(lambda x: (x>0.0)), 'help':'Maximum z height',},
]

Coriolisdict = [
    {'key':'latitude',  'required':True, 'type':(int, float), 'default':0.0, 'validate':(lambda x: (x>=-90.0)and(x<=90.0)), 'help':'Latitude in degrees',},
    {'key':'rotperiod', 'required':False, 'type':(int, float), 'default':86164.091, 'validate':(lambda x: (x>0.0)),         'help':'Earth rotational period',}
]

solveopts = [
    {'key':'verbose',  'required':False, 'type':int, 'default':1, 'validate':(lambda x: (x>=0)),  'help':'Solver verbosity level (0, 1, or 2)',},
    {'key':'maxiter',  'required':False, 'type':int, 'default':100, 'validate':(lambda x: (x>0)), 'help':'Number of solver iterations',},
    {'key':'tol',      'required':False, 'type':(int, float), 'default':1.0E-4, 'validate':(lambda x: (x>0.0)), 'help':'Solver convergence tolerance',},
    {'key':'freezevar','required':False, 'type':(int, float), 'default':0.0, 'validate':(lambda x: (x>=0.0)), 'help':'Freeze variable solutions',},   

]

inflowdict = [
    {'key':'initTprof',      'required':False,  'type':str,         'default':'MO',  'validate':None,             'help':'Initial temp profile',},
    {'key':'initVeerprof',   'required':False,  'type':str,         'default':'linear',  'validate':None,         'help':'Initial veer profile',},
    {'key':'veerpermeter',   'required':False,  'type':(int, float), 'default':0.0,  'validate':None, 'help':'Change in veer with elevation [deg/meter]',},
    {'key':'zeroveerheight', 'required':True,  'type':(int, float), 'default':90.0,  'validate':None, 'help':'Z height where veer is zero',},
]

turbdict = [
    {'key':'name',     'required':True,  'type':str,         'default':'',  'validate':None,         'help':'Turbine name',},
    {'key':'turbx',    'required':True, 'type':(int, float), 'default':0.0, 'validate':None,         'help':'Turbine location in x',},
    {'key':'turby',    'required':True, 'type':(int, float), 'default':0.0, 'validate':None,         'help':'Turbine location in y',},
    {'key':'zhh',      'required':True, 'type':(int, float), 'default':0.0, 'validate':(lambda x: (x>0)),         'help':'Turbine hub-height',},
    {'key':'turbD',    'required':True, 'type':(int, float), 'default':0.0, 'validate':(lambda x: (x>0)),         'help':'Turbine rotor diameter',},
    {'key':'turbnormal',  'required':False, 'type':[], 'default':[-1.0, 0.0, 0.0], 'validate':None,               'help':'Turbine rotor normal (facing upstream)',},
    {'key':'Ct',       'required':True, 'type':None, 'default':0.80, 'validate':None,                             'help':'Turbine Ct',},
    {'key':'rpm',      'required':False, 'type':None, 'default':None, 'validate':None,                             'help':'Turbine RPM',},

    {'key':'power',    'required':True, 'type':None, 'default':1000.0, 'validate':None,                           'help':'Turbine power',},
    {'key':'WS',       'required':False, 'type':None, 'default':0.0, 'validate':None,                             'help':'Wind speed table',},
    {'key':'Rdelta',   'required':True, 'type':(int, float), 'default':24.0, 'validate':(lambda x: (x>0)),        'help':'Turbine tip sharpness',},
    {'key':'aparam',   'required':False, 'type':(int, float), 'default':2.335, 'validate':(lambda x: (x>0)),        'help':'Joukowski disk a parameter',},
    {'key':'bparam',   'required':False, 'type':(int, float), 'default':4.0,   'validate':(lambda x: (x>0)),        'help':'Joukowski disk b parameter',},
    {'key':'turbfunc', 'required':False,  'type':str,        'default':'SANDwake3D_base.UnifCtADM',  'validate':None,         'help':'Turbine function type',},    
]

# Main RANS input
RANSinput = [
    {'key':'mesh', 'required':True,  'type':{}, 'default':meshsubdict, 'validate':None, 'help':'Mesh parameters',},
    {'key':'rho',  'required':True, 'type':(int, float), 'default':1.225, 'validate':(lambda x: (x>0.0)), 'help':'Density [kg/m^3]',},
    {'key':'cp',   'required':True, 'type':(int, float), 'default':1005.0, 'validate':(lambda x: (x>0.0)), 'help':'Heat capacity of air',},
    {'key':'g',    'required':True, 'type':(int, float), 'default':9.81,   'validate':(lambda x: (x>0.0)), 'help':'Gravity',},
    {'key':'nu',   'required':True, 'type':(int, float), 'default':1.5E-5, 'validate':(lambda x: (x>0.0)), 'help':'Kinematic viscosity',},

    {'key':'Cmu',    'required':True, 'type':(int, float), 'default':0.0,    'validate':(lambda x: (x>0.0)), 'help':'Calibration constant',},
    {'key':'C1eps',  'required':True, 'type':(int, float), 'default':0.0,    'validate':(lambda x: (x>0.0)), 'help':'Calibration constant',},
    {'key':'C2eps',  'required':True, 'type':(int, float), 'default':0.0,    'validate':(lambda x: (x>0.0)), 'help':'Calibration constant',},
    {'key':'C3eps',  'required':False, 'type':None,        'default':None,  'validate':None, 'help':'Calibration constant',},
    {'key':'Ck',     'required':True, 'type':(int, float), 'default':0.0,    'validate':(lambda x: (x>0.0)), 'help':'Calibration constant',},
    {'key':'sigmak',    'required':False, 'type':(int, float), 'default':1.0,  'validate':(lambda x: (x>0.0)), 'help':'Calibration constant',},
    {'key':'sigmaeps',  'required':False, 'type':(int, float), 'default':1.3,  'validate':(lambda x: (x>0.0)), 'help':'Calibration constant',},
    {'key':'sigmaT',    'required':False, 'type':(int, float), 'default':1.0,  'validate':(lambda x: (x>0.0)), 'help':'Calibration constant',},

    {'key':'dtau',      'required':False, 'type':(int, float), 'default':0.5,  'validate':(lambda x: (x>0.0)), 'help':'Pressure relaxation',},
    {'key':'ustar',     'required':True,  'type':(int, float), 'default':0.5,  'validate':(lambda x: (x>0.0)), 'help':'Inflow friction velocity',},
    {'key':'qw',        'required':True,  'type':(int, float), 'default':0.0,  'validate':None, 'help':'Surface heat flux',},
    {'key':'L',         'required':True,  'type':(int, float), 'default':0.5,  'validate':None, 'help':'Inflow Obukhov length',},
    {'key':'kappa',     'required':False,  'type':(int, float), 'default':0.42, 'validate':(lambda x: (x>0.0)), 'help':'von Karman constant',},
    {'key':'z0',        'required':True,  'type':(int, float), 'default':0.1,  'validate':(lambda x: (x>0.0)), 'help':'Surface roughness',},
    {'key':'zlo',       'required':False,  'type':None, 'default':None,  'validate':None, 'help':'Same as mesh zmin',},

    {'key':'Tw',        'required':True,  'type':(int, float), 'default':300,  'validate':(lambda x: (x>0.0)), 'help':'Surface temperature',},
    {'key':'beta',      'required':True,  'type':(int, float), 'default':1.0/300.0,  'validate':(lambda x: (x>0.0)), 'help':'Volumetric expansion',},
    {'key':'Tref',      'required':True,  'type':None,         'default':None,  'validate':None,               'help':'Reference temperature',},
    {'key':'lapserate', 'required':True,  'type':(int, float), 'default':0.0,   'validate':None,               'help':'Lapse rate',},

    # Coriolis options
    {'key':'Coriolis',       'required':False,  'type':{}, 'default':Coriolisdict, 'validate':None, 'help':'Coriolis parameters',},

    # Inflow options
    {'key':'inflow',       'required':False,  'type':{}, 'default':inflowdict, 'validate':None, 'help':'Inflow parameters',},

    # Turbine list
    {'key':'turbinelist',  'required':False,  'type':[],   'default':[], 'validate':turbdict,          'help':'List of turbines',},

    # Solver options
    {'key':'solveroptions', 'required':False,  'type':{}, 'default':solveopts, 'validate':None, 'help':'Solver options',},

    {'key':'savepklfile',  'required':False,  'type':str,   'default':'', 'validate':None,         'help':'Filename to save results',},

]

def makemesh(meshparams):
    """
    Create the y- and z-mesh, and create the x-vector
    """
    eps = 1.0E-6
    p = meshparams

    xvec = np.arange(p['xmin'], p['xmax']+eps, p['dx'])
    yvec = np.arange(p['ymin'], p['ymax']+eps, p['dy'])
    zvec = np.arange(p['zmin'], p['zmax']+eps, p['dz'])

    ym, zm = np.meshgrid(yvec, zvec, indexing='ij')
    return ym, zm, xvec, yvec, zvec

def initPhi(params, yvec, zvec):
    """
    """
    veerpermeter   = params['inflow']['veerpermeter']
    zeroveerheight = params['inflow']['zeroveerheight']
    kfactor        = params['Ck']
    
    Ny = len(yvec)
    Nz = len(zvec)

    # Set up ABL profile
    UABL = np.zeros(Nz)
    for iz, z in enumerate(zvec):
        UABL[iz] = init_U_ABL(z,params)

    # Set up veer profile
    veerProf = np.zeros(Nz)
    for i, z in enumerate(zvec):
        veerProf[i] = veerpermeter*(z-zeroveerheight)

    Uprof, Vprof = getUVfromUhVeer(UABL, veerProf)
    
    tempT = np.zeros(Nz)
    for i, z in enumerate(zvec):
        if params['inflow']['initTprof'] == 'linear':
            Tpermeter = params['lapserate']
            tempT[i] = params['Tw'] + Tpermeter*z
        else:
            tempT[i] = init_T_ABL(z, params)


    Uinit  = np.zeros((Ny, Nz))
    Vinit  = np.zeros((Ny, Nz))
    Winit  = np.zeros((Ny, Nz))
    Kinit  = np.zeros((Ny, Nz))
    Tinit  = np.zeros((Ny, Nz)) 
    EPSinit= np.zeros((Ny, Nz))
    pinit  = np.zeros((Ny, Nz))

    for iz, z in enumerate(zvec):
        Uinit[:, iz] = Uprof[iz] #SANDwake3D.init_U_ABL(z,ABLparam)
        Vinit[:, iz] = Vprof[iz]
        Winit[:, iz] = 0.0
    
        Kinit[:, iz]   = init_k_ABL(z,params)*kfactor 
        EPSinit[:, iz] = init_e_ABL(z,params)
        Tinit[:, iz] = tempT[iz]
    
    phiinit = {}
    phiinit['u'] = Uinit
    phiinit['v'] = Vinit
    phiinit['w'] = Winit
    phiinit['k'] = Kinit
    phiinit['T'] = Tinit
    phiinit['eps'] = EPSinit
    phiinit['p'] = pinit

    return phiinit, Uprof, Vprof, tempT
