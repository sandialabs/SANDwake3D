import os
import sys

curdir = os.getcwd()
extradirs = [
    os.path.dirname(curdir),
]
for x in extradirs:
    sys.path.insert(1, x)

import time

import numpy as np

import SANDwake3D_keps as SANDwake3D


def get_phiinit(params):
    # Set up ABL profile
    zvec = params["zm"][0, :]
    Ny, Nz = params["zm"].shape
    UABL = np.zeros(Nz)
    for iz, z in enumerate(zvec):
        UABL[iz] = SANDwake3D.init_U_ABL(z, params)

    # Set up veer profile
    veerProf = np.zeros(Nz)
    turbhh = params["turbforcing"]["zhh"]
    for i, z in enumerate(zvec):
        veerProf[i] = params["veerpermeter"] * (z - turbhh)

    Uprof, Vprof = SANDwake3D.getUVfromUhVeer(UABL, veerProf)

    tempT = np.zeros(Nz)
    for i, z in enumerate(zvec):
        tempT[i] = params["Tw"] + params["Tpermeter"] * z

    # Compute the wind direction
    rotorD = params["turbforcing"]["turbD"]
    turbR = rotorD * 0.5
    ztop = turbhh + turbR
    zbot = turbhh - turbR
    Utop, Ubot = np.interp(ztop, zvec, Uprof), np.interp(zbot, zvec, Uprof)
    Vtop, Vbot = np.interp(ztop, zvec, Vprof), np.interp(zbot, zvec, Vprof)
    Wdirtop = np.arctan2(Vtop, Utop) * 180 / np.pi
    Wdirbot = np.arctan2(Vbot, Ubot) * 180 / np.pi

    Uhh, Vhh = np.interp(turbhh, zvec, UABL), np.interp(turbhh, zvec, UABL)
    alpha = np.log(Utop / Ubot) / np.log(ztop / zbot)

    Uinit = np.zeros((Ny, Nz))
    Vinit = np.zeros((Ny, Nz))
    Winit = np.zeros((Ny, Nz))
    Kinit = np.zeros((Ny, Nz))
    Tinit = np.zeros((Ny, Nz))
    EPSinit = np.zeros((Ny, Nz))
    pinit = np.zeros((Ny, Nz))

    for iz, z in enumerate(zvec):
        Uinit[:, iz] = Uprof[iz]
        Vinit[:, iz] = Vprof[iz]
        Winit[:, iz] = 0.0

        # Kinit[:, iz] = SANDwake3D.init_k_ABL(z, params) * np.exp(-0.0 * z) * 0.750 + 0.0
        Kinit[:, iz] = (
            SANDwake3D.init_k_ABL(z, params) * np.exp(-0.0 * z) * params["kfactor"]
        )

        EPSinit[:, iz] = SANDwake3D.init_e_ABL(z, params)
        Tinit[:, iz] = tempT[iz]

    phiinit = {}
    phiinit["u"] = Uinit
    phiinit["v"] = Vinit
    phiinit["w"] = Winit
    phiinit["k"] = Kinit
    phiinit["T"] = Tinit
    phiinit["eps"] = EPSinit
    phiinit["p"] = pinit
    return phiinit


def run_sandwake(params, phiinit):
    start_time = time.time()
    xvec = params["xvec"]
    dy = np.mean(np.diff(params["ym"][:, 0]))
    dz = np.mean(np.diff(params["zm"][0, :]))
    Uinit = phiinit["u"]
    phi = SANDwake3D.marchSystem(
        phiinit,
        xvec,
        dy,
        dz,
        params,
        SANDwake3D.getTypicalWMBC(
            phiinit["u"][0, -1],
            phiinit["T"][0],
            veerBC=phiinit["v"][0, :],
            dTdz=params["Tpermeter"],
            uBC_y=phiinit["u"][0, :],
        ),
        SANDwake3D.kepsT_eqns,
        advanceSys=SANDwake3D.advanceSystemKEPS,
        verbose=0,
        maxiter=100,
        tol=1.0e-4,
    )
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    return phi
