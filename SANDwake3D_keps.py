#!/usr/bin/env python

import numpy as np
from scipy.linalg import solve_banded
import copy
from collections import OrderedDict
from SANDwake3D_base import *
from functools import partial


def RHS_f_nhalf(f_n, phi_np1, phi_n, phi_tilde, Dz_nuT_tilde, dx, dy, dz, params):
    """
    Go from n to n+1/2
    """
    u_tilde, w_tilde = phi_tilde["u"], phi_tilde["w"]
    nu_total = params["nu"] + phi_tilde["nuT"]
    w_total = w_tilde - Dz_nuT_tilde

    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz
    inv_dz2 = 1.0 / (dz * dz)
    inv_half_dx = 2.0 * inv_dx

    RHS = np.empty_like(f_n)
    RHS[:, 0] = (
        u_tilde[:, 0] * f_n[:, 0] * inv_half_dx
        - w_total[:, 0] * inv_dz * (f_n[:, 1] - f_n[:, 0])
        + nu_total[:, 0] * inv_dz2 * (f_n[:, 0] - 2.0 * f_n[:, 1] + f_n[:, 2])
    )

    RHS[:, 1:-1] = (
        u_tilde[:, 1:-1] * f_n[:, 1:-1] * inv_half_dx
        - w_total[:, 1:-1] * inv_dz * 0.5 * (f_n[:, 2:] - f_n[:, :-2])
        + nu_total[:, 1:-1] * inv_dz2 * (f_n[:, 2:] - 2.0 * f_n[:, 1:-1] + f_n[:, :-2])
    )

    RHS[:, -1] = (
        u_tilde[:, -1] * f_n[:, -1] * inv_half_dx
        - w_total[:, -1] * inv_dz * (f_n[:, -1] - f_n[:, -2])
        + nu_total[:, -1] * inv_dz2 * (f_n[:, -1] - 2.0 * f_n[:, -2] + f_n[:, -3])
    )

    return RHS


def RHS_f_np1(f_nhalf, phi_np1, phi_n, phi_tilde, Dy_nuT_tilde, dx, dy, dz, params):
    """
    Go from n+1/2 to n+1 for the u-momentum equation
    """
    u_tilde, v_tilde = phi_tilde["u"], phi_tilde["v"]
    nu_total = params["nu"] + phi_tilde["nuT"]
    v_total = v_tilde - Dy_nuT_tilde

    inv_half_dx = 2.0 / dx
    inv_dy = 1.0 / dy
    inv_dy2 = 1.0 / (dy * dy)

    RHS = np.empty_like(f_nhalf)
    RHS[0, :] = (
        u_tilde[0, :] * f_nhalf[0, :] * inv_half_dx
        - v_total[0, :] * inv_dy * (f_nhalf[1, :] - f_nhalf[0, :])
        + nu_total[0, :]
        * inv_dy2
        * (f_nhalf[0, :] - 2.0 * f_nhalf[1, :] + f_nhalf[2, :])
    )

    RHS[1:-1, :] = (
        u_tilde[1:-1, :] * f_nhalf[1:-1, :] * inv_half_dx
        - v_total[1:-1, :] * inv_dy * 0.5 * (f_nhalf[2:, :] - f_nhalf[:-2, :])
        + nu_total[1:-1, :]
        * inv_dy2
        * (f_nhalf[2:, :] - 2.0 * f_nhalf[1:-1, :] + f_nhalf[:-2, :])
    )

    RHS[-1, :] = (
        u_tilde[-1, :] * f_nhalf[-1, :] * inv_half_dx
        - v_total[-1, :] * inv_dy * (f_nhalf[-1, :] - f_nhalf[-2, :])
        + nu_total[-1, :]
        * inv_dy2
        * (f_nhalf[-1, :] - 2.0 * f_nhalf[-2, :] + f_nhalf[-3, :])
    )
    return RHS


def scale_nuT(phi, params, field):
    sigma = f"""sigma{field}"""
    if sigma in params:
        phi["nuT"] /= params[sigma]


def rhs_f_extra_forcing(field, phi, params, dx, dy, dz):

    if field == "k":
        sigmak = params["sigmak"]
        fk_const = params["fk_const"] if "fk_const" in params else 0.0
        Dy_U, Dz_U = np.gradient(phi["u"], dy, dz, edge_order=1)
        return sigmak * phi["nuT"] * (Dy_U * Dy_U + Dz_U * Dz_U) - phi["eps"] + fk_const
    elif field == "eps":
        nu = params["nu"]
        sigmaeps = params["sigmaeps"]
        C1eps = params["C1eps"]
        C2eps = params["C2eps"]
        C3eps = params["C3eps"]
        feps_const = params["feps_const"] if "feps_const" in params else 0.0
        Tscale = TimeScale(phi["k"], phi["eps"], nu)
        Dy_U, Dz_U = np.gradient(phi["u"], dy, dz, edge_order=1)
        return 0.0  # C1eps/Tscale*(sigmaeps*phi['nuT']*(Dy_U*Dy_U) + phi['nuT']*(Dz_U*Dz_U)) - C2eps*phi["eps"]/Tscale + feps_const
    elif field == "u" or field == "w":
        fx_const = params["fx_const"] if "fx_const" in params else 0.0
        return fx_const
    return 0


def advanceF(
    phi_np1old, phi_n, dx, dy, dz, params, bc_ylo, bc_yhi, bc_zlo, bc_zhi, field
):
    """
    Advance the field one full step
    """
    phi_tilde = getTildeVars(phi_np1old, phi_n)
    scale_nuT(phi_tilde, params, field)
    u_tilde, v_tilde, w_tilde = phi_tilde["u"], phi_tilde["v"], phi_tilde["w"]
    nuT_tilde = phi_tilde["nuT"]

    N = u_tilde.shape
    Ny = N[0]
    Nz = N[1]

    # Load parameters
    nu = params["nu"]

    # --- differentiation stencils ---
    #                  j-1  j  j+1
    Irow = np.array([0, 1, 0])
    Dcen = np.array([-1, 0, 1]) * 0.5
    D2cen = np.array([1, -2, 1])
    # -------------------------------

    # Compute some quantities related to nuT
    Dy_nuT, Dz_nuT = np.gradient(nuT_tilde, dy, dz, edge_order=1)

    nu_total = nu + nuT_tilde
    v_total = v_tilde - Dy_nuT
    w_total = w_tilde - Dz_nuT

    RHS_extra_forcing = rhs_f_extra_forcing(field, phi_tilde, params, dx, dy, dz)

    # First sweep: n -> n+1/2
    # -----------------------
    f_nhalf = np.zeros((Ny, Nz))
    RHS_nhalf = (
        RHS_f_nhalf(
            phi_n[field], phi_np1old, phi_n, phi_tilde, Dz_nuT, dx, dy, dz, params
        )
        + RHS_extra_forcing
    )
    inv_half_dx = 2.0 / dx
    inv_dy = 1.0 / dy
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz = 1.0 / dz
    inv_dz2 = 1.0 / (dz * dz)
    row_lo, dentry_lo = applyBC(bc_ylo, "lower", dy)
    row_hi, dentry_hi = applyBC(bc_yhi, "upper", dy)
    for j in range(Nz):
        LHS_nhalf = np.zeros((Ny, 3))
        # == Set up the LHS matrices ==
        LHS_nhalf[1:-1, :] = (
            u_tilde[1:-1, j, np.newaxis] * inv_half_dx * Irow
            + v_total[1:-1, j, np.newaxis] * inv_dy * Dcen
            - nu_total[1:-1, j, np.newaxis] * inv_dy2 * D2cen
        )
        # Apply BC's
        LHS_nhalf[0, :] = row_lo
        LHS_nhalf[-1, :] = row_hi
        RHS_nhalf[0, :] = dentry_lo
        RHS_nhalf[-1, :] = dentry_hi
        # print(f'j = {j}\nRHS_nhalf = ',RHS_nhalf[:,j], '\nLHS = ', LHS_nhalf)
        # Solve the triadiagonal system
        f_nhalf[:, j] = solvetridiag(LHS_nhalf, RHS_nhalf[:, j])

    # print(f'f_nhalf = ',f_nhalf)
    # Second sweep: n+1/2 -> n+1
    # -----------------------
    f_np1 = np.zeros((Ny, Nz))
    RHS_np1 = (
        RHS_f_np1(f_nhalf, phi_np1old, phi_n, phi_tilde, Dy_nuT, dx, dy, dz, params)
        + RHS_extra_forcing
    )
    # == Set up the LHS matrices ==
    row_lo_base, dentry_lo_base = applyBC(bc_zlo, "lower", dz)
    row_hi_base, dentry_hi_base = applyBC(bc_zhi, "upper", dz)
    for i in range(Ny):
        LHS_np1 = np.zeros((Nz, 3))
        LHS_np1[1:-1, :] = (
            u_tilde[i, 1:-1, np.newaxis] * inv_half_dx * Irow
            + w_total[i, 1:-1, np.newaxis] * inv_dz * Dcen
            - nu_total[i, 1:-1, np.newaxis] * inv_dz2 * D2cen
        )
        # Apply BC's
        row_lo, dentry_lo = row_lo_base, dentry_lo_base
        row_hi, dentry_hi = row_hi_base, dentry_hi_base
        if i == 0:
            row_lo, dentry_lo = applyBC(
                {"type": "dirichlet", "value": phi_tilde[field][i, 0]}, "lower", dz
            )
        elif i == Ny - 1:
            row_hi, dentry_hi = applyBC(
                {"type": "dirichlet", "value": phi_tilde[field][i, -1]}, "lower", dz
            )
        LHS_np1[0, :] = row_lo
        LHS_np1[-1, :] = row_hi
        RHS_np1[:, 0] = dentry_lo
        RHS_np1[:, -1] = dentry_hi
        # print(f'i = {i}\nRHS_np1 = ',RHS_np1[i,:], '\nLHS = ', LHS_np1)
        # Solve the triadiagonal system
        f_np1[i, :] = solvetridiag(LHS_np1, RHS_np1[i, :], verbose=False)
        # print('f_np1 = ',f_np1[i,:])
    return f_np1


def advanceMass(phi_np1old, phi_n, dx, dy, dz, params, bc_ylo, bc_yhi, bc_zlo, bc_zhi):
    """
    Advance the continuity equation one full step
    return v^(n+1)
    """
    u_np1, u_n = phi_np1old["u"], phi_n["u"]
    v_np1 = phi_np1old["v"]
    w_np1 = phi_np1old["w"]

    Dz_w = np.gradient(w_np1, dz, axis=1, edge_order=1)
    RHS = -dy * (u_np1 - u_n) / (dx) - dy * Dz_w

    # == Set up the LHS matrices ==
    row_lo, dentry_lo = applyBC(bc_ylo, "lower", dy)
    v_np1 = np.cumsum(RHS, axis=0) + dentry_lo

    return v_np1


def TimeScale(k, eps, nu):
    return np.fmax(k / eps, 6.0 * np.sqrt(nu / eps))


def getNuT(phi, Cmu, nu):
    # timescale
    # Tscale = np.fmax(phi['k']/phi['eps'], 6.0*np.sqrt(nu/phi['eps'][:,:]))
    Tscale = TimeScale(phi["k"], phi["eps"], nu)
    return Cmu * phi["k"] * Tscale


def advanceSystemKEPS(
    phi_n, dx, dy, dz, params, allbcs, eqnsys, maxiter=100, tol=1.0e-6, verbose=False
):
    """
    Advance equation system 1 step in x
    """
    varlist = [v for v, g in eqnsys.items()]

    if "nuT" not in phi_n:
        phi_n["nuT"] = getNuT(phi_n, params["Cmu"], params["nu"])
    phi_n1 = copy.deepcopy(phi_n)

    for k in range(maxiter):
        phi_next = OrderedDict()
        phi_n1["nuT"] = getNuT(phi_n1, params["Cmu"], params["nu"])
        # Loop over all variables
        for v in varlist:
            bcvar = allbcs[v]
            phi_next[v] = eqnsys[v](
                phi_n1,
                phi_n,
                dx,
                dy,
                dz,
                params,
                bcvar["ylo"],
                bcvar["yhi"],
                bcvar["zlo"],
                bcvar["zhi"],
            )
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
def MO_windshear(z, L):
    if L < 0:
        return (1 - 16.0 * (z / L)) ** (-0.25)

    if L == float("inf"):
        return np.ones_like(z)

    return 1 + 5 * z / L


def init_U_ABL(z, param):
    """
    Initialize ABL profile from Monin-Obukhov theory
    """
    z0 = param["z0"]
    L = param["L"]
    K = param["kappa"]
    rho = param["rho"]
    ustar = param["ustar"]

    phim = MO_windshear(z, L)

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


def init_e_ABL(z, param):
    L = param["L"]
    K = param["kappa"]
    ustar = param["ustar"]

    phim = MO_windshear(z, L)

    if L < 0:
        phie = 1 - z / L
    elif L == float("inf"):
        phie = phim
    else:
        phie = phim - z / L

    return ustar**3 / (K * z) * phie


def init_k_ABL(z, param):
    L = param["L"]
    K = param["kappa"]
    ustar = param["ustar"]

    phim = MO_windshear(z, L)

    if L < 0:
        phie = 1 - z / L
    elif L == float("inf"):
        phie = phim
    else:
        phie = phim - z / L

    return 5.48 * ustar**2 * (phie / phim) ** 0.5


def set_e_init(zvec, dz, u, k, params):
    Cmu = params["Cmu"]
    e_init = np.zeros_like(u)

    Dcen = np.array([-1, 0, 1]) / (2 * dz)
    e_init[1:-1] = np.sqrt(
        Cmu
        * k[1:-1] ** 2
        * (Dcen[0] * u[:-2] + Dcen[1] * u[1:-1] + Dcen[2] * u[2:]) ** 2
    )

    return e_init


def set_k_init(rvec, dr, u, params, k_factor=0.1):
    k_init = np.zeros_like(u)
    Dcen = np.array([-1, 0, 1]) / (2 * dr)
    k_init[1:-1] = (Dcen[0] * u[:-2] + Dcen[1] * u[1:-1] + Dcen[2] * u[2:]) ** 2

    # scale k
    kmax = np.max(k_init)
    C = k_factor**2 * 2 / (3 * kmax)
    k_init *= C
    return k_init


########################################################
# Define the laminar equation system
keps_eqns = OrderedDict()
keps_eqns["u"] = partial(advanceF, field="u")
keps_eqns["w"] = partial(advanceF, field="w")
keps_eqns["k"] = partial(advanceF, field="k")
keps_eqns["eps"] = partial(advanceF, field="eps")
keps_eqns["v"] = advanceMass

# Use the same marchSystemBase in SANDWake3D_base to advance the equations
marchSystem = marchSystemBase
