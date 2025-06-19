#!/usr/bin/env python

import copy
from collections import OrderedDict
from functools import partial
import numpy as np
import SANDwake3D_base as sdb


def rhs_f_nhalf(f_n, phi_tilde, dz_nut_tilde, dx, dz, params):
    """
    Go from n to n+1/2
    """
    u_tilde, w_tilde = phi_tilde["u"], phi_tilde["w"]
    nu_total = params["nu"] + phi_tilde["nut"]
    w_total = w_tilde - dz_nut_tilde

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


def rhs_f_np1(f_nhalf, phi_tilde, dy_nut_tilde, dx, dy, params):
    """
    Go from n+1/2 to n+1 for the u-momentum equation
    """
    u_tilde, v_tilde = phi_tilde["u"], phi_tilde["v"]
    nu_total = params["nu"] + phi_tilde["nut"]
    v_total = v_tilde - dy_nut_tilde

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


def scale_nut(phi, params, field):
    """
    Scale turbulent viscosity if necessary
    """
    sigma = f"""sigma{field}"""
    if sigma in params:
        phi["nut"] /= params[sigma]


def rhs_f_extra_forcing(field, phi, params, dy, dz):
    """
    Compute the RHS extra forcing if necessary
    """
    if field == "k":
        sigmak = params["sigmak"]
        fk_const = params["fk_const"] if "fk_const" in params else 0.0
        dy_u, dz_u = np.gradient(phi["u"], dy, dz, edge_order=1)
        return sigmak * phi["nut"] * (dy_u * dy_u + dz_u * dz_u) - phi["eps"] + fk_const
    if field == "eps":
        nu = params["nu"]
        sigmaeps = params["sigmaeps"]
        C1eps = params["C1eps"]
        C2eps = params["C2eps"]
        C3eps = params["C3eps"]
        feps_const = params["feps_const"] if "feps_const" in params else 0.0
        Tscale = time_scale(phi["k"], phi["eps"], nu)
        dy_u, dz_u = np.gradient(phi["u"], dy, dz, edge_order=1)
        return 0.0  # C1eps/Tscale*(sigmaeps*phi['nut']*(dy_u*dy_u) + phi['nut']*(dz_u*dz_u)) - C2eps*phi["eps"]/Tscale + feps_const
    if field in ("u", "w"):
        fx_const = params["fx_const"] if "fx_const" in params else 0.0
        return fx_const
    return 0


def advanceF(
    phi_np1old, phi_n, dx, dy, dz, params, bc_ylo, bc_yhi, bc_zlo, bc_zhi, field
):
    """
    Advance the field one full step
    """
    phi_tilde = sdb.getTildeVars(phi_np1old, phi_n)
    scale_nut(phi_tilde, params, field)
    u_tilde, v_tilde, w_tilde = phi_tilde["u"], phi_tilde["v"], phi_tilde["w"]
    nut_tilde = phi_tilde["nut"]
    ny, nz = u_tilde.shape

    # Load parameters
    nu = params["nu"]

    # --- differentiation stencils ---
    #                  j-1  j  j+1
    irow = np.array([0, 1, 0])
    dcen = np.array([-1, 0, 1]) * 0.5
    d2cen = np.array([1, -2, 1])
    # -------------------------------

    # Compute some quantities related to nut
    dy_nut, dz_nut = np.gradient(nut_tilde, dy, dz, edge_order=1)

    nu_total = nu + nut_tilde
    v_total = v_tilde - dy_nut
    w_total = w_tilde - dz_nut

    rhs_extra_forcing = rhs_f_extra_forcing(field, phi_tilde, params, dy, dz)

    # First sweep: n -> n+1/2
    # -----------------------
    f_nhalf = np.zeros((ny, nz))
    rhs_nhalf = (
        rhs_f_nhalf(phi_n[field], phi_tilde, dz_nut, dx, dz, params) + rhs_extra_forcing
    )
    inv_half_dx = 2.0 / dx
    inv_dy = 1.0 / dy
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz = 1.0 / dz
    inv_dz2 = 1.0 / (dz * dz)
    row_lo, dentry_lo = sdb.applyBC(bc_ylo, "lower", dy)
    row_hi, dentry_hi = sdb.applyBC(bc_yhi, "upper", dy)
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
        rhs_nhalf[0, :] = dentry_lo
        rhs_nhalf[-1, :] = dentry_hi
        # print(f'j = {j}\nrhs_nhalf = ',rhs_nhalf[:,j], '\nLHS = ', lhs_nhalf)
        # Solve the triadiagonal system
        f_nhalf[:, j] = sdb.solvetridiag(lhs_nhalf, rhs_nhalf[:, j])

    # print(f'f_nhalf = ',f_nhalf)
    # Second sweep: n+1/2 -> n+1
    # -----------------------
    f_np1 = np.zeros((ny, nz))
    rhs_np1 = rhs_f_np1(f_nhalf, phi_tilde, dy_nut, dx, dy, params) + rhs_extra_forcing
    # == Set up the LHS matrices ==
    row_lo_base, dentry_lo_base = sdb.applyBC(bc_zlo, "lower", dz)
    row_hi_base, dentry_hi_base = sdb.applyBC(bc_zhi, "upper", dz)
    for i in range(ny):
        lhs_np1 = np.zeros((nz, 3))
        lhs_np1[1:-1, :] = (
            u_tilde[i, 1:-1, np.newaxis] * inv_half_dx * irow
            + w_total[i, 1:-1, np.newaxis] * inv_dz * dcen
            - nu_total[i, 1:-1, np.newaxis] * inv_dz2 * d2cen
        )
        # Apply BC's
        row_lo, dentry_lo = row_lo_base, dentry_lo_base
        row_hi, dentry_hi = row_hi_base, dentry_hi_base
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


def advanceMass(phi_np1old, phi_n, dx, dy, dz, params, bc_ylo, bc_yhi, bc_zlo, bc_zhi):
    """
    Advance the continuity equation one full step
    """
    u_np1, u_n = phi_np1old["u"], phi_n["u"]
    v_np1 = phi_np1old["v"]
    w_np1 = phi_np1old["w"]

    dz_w = np.gradient(w_np1, dz, axis=1, edge_order=1)
    rhs = -dy * (u_np1 - u_n) / (dx) - dy * dz_w

    # == Set up the LHS matrices ==
    _, dentry_lo = sdb.applyBC(bc_ylo, "lower", dy)
    v_np1 = np.cumsum(rhs, axis=0) + dentry_lo

    return v_np1


def time_scale(k, eps, nu):
    """
    Compute time scale
    """
    return np.fmax(k / eps, 6.0 * np.sqrt(nu / eps))


def get_nut(phi, cmu, nu):
    """
    Compute turbulent viscosity
    """
    tscale = time_scale(phi["k"], phi["eps"], nu)
    return cmu * phi["k"] * tscale


def advanceSystemKEPS(
    phi_n, dx, dy, dz, params, allbcs, eqnsys, maxiter=100, tol=1.0e-6, verbose=False
):
    """
    Advance equation system 1 step in x
    """
    varlist = [v for v, g in eqnsys.items()]

    if "nut" not in phi_n:
        phi_n["nut"] = get_nut(phi_n, params["Cmu"], params["nu"])
    phi_n1 = copy.deepcopy(phi_n)

    for k in range(maxiter):
        phi_next = OrderedDict()
        phi_n1["nut"] = get_nut(phi_n1, params["Cmu"], params["nu"])
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
        converged, convergedat = sdb.convergetest(phi_next, phi_n1, tol)
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
def mo_windshear(z, L):
    """
    Compute Monin-Obukhov wind shear
    """
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

    phim = mo_windshear(z, L)

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

    phim = mo_windshear(z, L)

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

    phim = mo_windshear(z, L)

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


########################################################
# Define the laminar equation system
keps_eqns = OrderedDict()
keps_eqns["u"] = partial(advanceF, field="u")
keps_eqns["w"] = partial(advanceF, field="w")
keps_eqns["k"] = partial(advanceF, field="k")
keps_eqns["eps"] = partial(advanceF, field="eps")
keps_eqns["v"] = advanceMass

# Use the same marchSystemBase in SANDWake3D_base to advance the equations
marchSystem = sdb.marchSystemBase
