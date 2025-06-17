import sys
import os
import time
from datetime import timedelta

curdir = os.getcwd()
extradirs = [
    "../",
    "./",
    os.path.dirname(curdir),
]
for x in extradirs:
    sys.path.insert(1, x)

import numpy as np
import SANDwake3D_keps as SANDwake3D
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def tanhwake(y, z, zhh, turbR, delta, U1, U0=0):
    r = np.sqrt(y**2 + (z - zhh) ** 2)
    Ur = 0.5 * (U1 - U0) * (1.0 + np.tanh((r - turbR) / delta)) + U0
    return -Ur


def main():
    """Compare models."""

    # Set up mesh
    Uinf = 1.0
    nu = 1.5e-5  # m^2/s

    # mesh
    zmin = 5
    zmax = 200
    Nz = int((zmax - zmin) / 5.0 + 1)
    zvec = np.linspace(zmin, zmax, Nz)

    Ny = 51
    ymax = 125
    yvec = np.linspace(-ymax, ymax, Ny)

    # Set ABL parameters
    ABLparam = {
        "ustar": 0.4,  # 0.575652,
        "L": float("inf"),
        "kappa": 0.42,
        "rho": 1.225,
        "z0": 0.15,
        "Cmu": 5.48 ** (-2),
    }

    # This parameter defines V(z) -- m/s per meter
    veerpermeter = 0.0  # 0.02

    ym, zm = np.meshgrid(yvec, zvec, indexing="ij")
    dy = np.mean(np.diff(yvec))
    dz = np.mean(np.diff(zvec))

    turbhh = 80
    turbR = 50
    wakedef = 3.5
    rdelta = 7.5  # 10.0
    wakeU = wakedef * np.exp(-2.0 * (ym**2 + (zm - turbhh) ** 2) / turbR**2)
    wakeUr = tanhwake(ym, zm, turbhh, turbR, rdelta, wakedef)

    # Set up ABL profile
    UABL = np.zeros(Nz)
    for iz, z in enumerate(zvec):
        UABL[iz] = SANDwake3D.init_U_ABL(z, ABLparam)

    # Set up veer profile
    veerV = np.zeros(Nz)
    for i, z in enumerate(zvec):
        veerV[i] = veerpermeter * (z - turbhh)

    # Compute the wind direction
    ztop = turbhh + turbR
    zbot = turbhh - turbR
    Utop, Ubot = np.interp(ztop, zvec, UABL), np.interp(zbot, zvec, UABL)
    Vtop, Vbot = np.interp(ztop, zvec, veerV), np.interp(zbot, zvec, veerV)
    Wdirtop = np.arctan2(Vtop, Utop) * 180 / np.pi
    Wdirbot = np.arctan2(Vbot, Ubot) * 180 / np.pi

    Uhh, Vhh = np.interp(turbhh, zvec, UABL), np.interp(turbhh, zvec, UABL)
    alpha = np.log(Utop / Ubot) / np.log(ztop / zbot)

    Uinit = np.zeros((Ny, Nz))
    Vinit = np.zeros((Ny, Nz))
    Winit = np.zeros((Ny, Nz))
    Kinit = np.zeros((Ny, Nz))
    EPSinit = np.zeros((Ny, Nz))

    for iz, z in enumerate(zvec):
        Uinit[:, iz] = UABL[iz]  # SANDwake3D.init_U_ABL(z,ABLparam)
        Vinit[:, iz] = veerV[iz]
        Winit[:, iz] = 0.0
        Kinit[:, iz] = SANDwake3D.init_k_ABL(z, ABLparam) * 0.05  # 0.2
        EPSinit[:, iz] = (
            9.75e-4  # 5.0E-4 #SANDwake3D.init_e_ABL(z,ABLparam)*0.1  #5.0E-4 is good
        )

    Uinit = Uinit - wakeUr

    phiinit = {}
    phiinit["u"] = Uinit
    phiinit["v"] = Vinit
    phiinit["w"] = Winit
    phiinit["k"] = Kinit
    phiinit["eps"] = EPSinit

    # Set boundary conditions
    ubc = {}
    ubc["ylo"] = {"type": "neumann", "value": 0.0}
    ubc["yhi"] = {"type": "neumann", "value": 0.0}
    ubc["zlo"] = {"type": "dirichlet", "value": Uinit[0, 0]}
    ubc["zhi"] = {"type": "dirichlet", "value": Uinit[0, -1]}

    vbc = {}
    vbc["ylo"] = {"type": "dirichlet", "value": veerV}
    vbc["yhi"] = {"type": "dirichlet", "value": 0.0}
    vbc["zlo"] = {"type": "dirichlet", "value": 0.0}
    vbc["zhi"] = {"type": "dirichlet", "value": 0.0}

    wbc = {}
    wbc["ylo"] = {"type": "neumann", "value": 0.0}
    wbc["yhi"] = {"type": "neumann", "value": 0.0}
    wbc["zlo"] = {"type": "dirichlet", "value": 0.0}
    wbc["zhi"] = {"type": "neumann", "value": 0.0}

    kbc = {}
    kbc["ylo"] = {"type": "neumann", "value": 0.0}
    kbc["yhi"] = {"type": "neumann", "value": 0.0}
    kbc["zlo"] = {"type": "dirichlet", "value": Kinit[0, 0]}
    kbc["zhi"] = {"type": "neumann", "value": 0.0}

    epsbc = {}
    epsbc["ylo"] = {"type": "neumann", "value": 0.0}
    epsbc["yhi"] = {"type": "neumann", "value": 0.0}
    epsbc["zlo"] = {"type": "dirichlet", "value": EPSinit[0, 0]}
    epsbc["zhi"] = {"type": "neumann", "value": 0.0}

    allbc = {"u": ubc, "v": vbc, "w": wbc, "k": kbc, "eps": epsbc}

    # Set parameters
    params = {
        "nu": nu,
        "Cmu": 5.48 ** (-2),
        #'Cmu'  : 0.01,
        "C1eps": 1.76,
        "C2eps": 1.92,
        "C3eps": 0.033,
        "sigmak": 1.0,
        "sigmaeps": 1.3,
    }

    xvecplot = [0, 200, 300, 400]  # [0, 100]#[0, 100, 200, 300]
    dx = 20  # 25
    xvec = np.arange(xvecplot[0], xvecplot[-1] + 1.0e-6, dx)
    start = time.time()
    phi = SANDwake3D.marchSystem(
        phiinit,
        xvec,
        dy,
        dz,
        params,
        allbc,
        SANDwake3D.keps_eqns,
        advanceSys=SANDwake3D.advanceSystemKEPS,
        verbose=True,
        maxiter=100,
        tol=1.0e-4,
    )
    np.savez("phi.npz", **phi)

    phi_old = np.load("phi-bkp.npz")
    for k, v in phi_old.items():
        try:
            np.testing.assert_allclose(phi[k], v, rtol=1e-14, atol=1e-14)
        except AssertionError as e:
            raise ValueError(f"Arrays are not close:\n{e}")

    end = time.time() - start
    print(f"Elapsed time {timedelta(seconds=end)} (or {end:f} seconds)")


if __name__ == "__main__":
    main()
