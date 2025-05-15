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
import SANDwake3D_laminar as SANDwake3D
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def blasius(t, C0):
    """Compute Blasius solution."""
    return solve_ivp(
        lambda t, y: np.array([y[1], y[2], -0.5 * y[0] * y[2]]),
        np.array([0, t[-1]]),
        np.array([0, 0, C0[0]]),
        t_eval=t,
    )


def getUVfromBlasius(x, yvec, Uinf, nu, C0):
    """Get velocity from Blasius."""
    ReX = Uinf * x / nu
    eta = yvec * np.sqrt(Uinf / (nu * x))
    fblasius = blasius(eta, np.array([C0]))
    u = fblasius.y[1]
    v = 0.5 * (eta * fblasius.y[1] - fblasius.y[0]) * np.sqrt(nu * Uinf / x)
    return u, v


def main():
    """Compare models."""
    start = time.time()
    C0 = fsolve(lambda C0: blasius([20], C0).y[1][-1] - 1, x0=1)[0]
    print("Blasius C_F sqrt(Re) = {:.3f}".format(4 * C0))

    # Set up mesh and BL parameters
    Uinf = 1.0
    nu = 0.1

    # mesh
    Ny = 201
    ymax = 20
    yvec = np.linspace(0, ymax, Ny)

    Nz = 21
    zvec = np.linspace(-1, 1, Nz)

    dy = np.mean(np.diff(yvec))
    dz = np.mean(np.diff(zvec))

    xvecplot = [40, 50, 80]

    # Set boundary conditions
    ubc = {}
    ubc["ylo"] = {"type": "dirichlet", "value": 0.0}
    ubc["yhi"] = {"type": "dirichlet", "value": Uinf}
    ubc["zlo"] = {"type": "neumann", "value": 0.0}
    ubc["zhi"] = {"type": "neumann", "value": 0.0}

    vbc = {}
    vbc["ylo"] = {"type": "dirichlet", "value": 0.0}
    vbc["yhi"] = {"type": "dirichlet", "value": 0.0}
    vbc["zlo"] = {"type": "dirichlet", "value": 0.0}
    vbc["zhi"] = {"type": "dirichlet", "value": 0.0}

    wbc = {}
    wbc["ylo"] = {"type": "dirichlet", "value": 0.0}
    wbc["yhi"] = {"type": "dirichlet", "value": 0.0}
    wbc["zlo"] = {"type": "dirichlet", "value": 0.0}
    wbc["zhi"] = {"type": "dirichlet", "value": 0.0}

    allbc = {"u": ubc, "v": vbc, "w": wbc}

    # Set up initial conditions
    ym, zm = np.meshgrid(yvec, zvec)

    u0, v0 = getUVfromBlasius(xvecplot[0], yvec, Uinf, nu, C0)

    Uinit = np.zeros((Ny, Nz))
    Vinit = np.zeros((Ny, Nz))
    for iy in range(Ny):
        Uinit[iy, :] = u0[iy]
        Vinit[iy, :] = v0[iy]
    Winit = np.zeros((Ny, Nz))

    phiinit = {}
    phiinit["u"] = Uinit
    phiinit["v"] = Vinit
    phiinit["w"] = Winit

    # Set parameters
    params = {}
    params["nu"] = nu

    dx = 2.5
    xvec = np.arange(xvecplot[0], xvecplot[-1] + 1.0e-6, dx)
    phi = SANDwake3D.marchSystem(
        phiinit,
        xvec,
        dy,
        dz,
        params,
        allbc,
        SANDwake3D.laminar_eqns,
        verbose=True,
        maxiter=50,
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
