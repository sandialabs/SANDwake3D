import os
import sys

curdir = os.getcwd()
extradirs = [
    os.path.dirname(curdir),
]
for x in extradirs:
    sys.path.insert(1, x)


import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.backends.backend_pdf import PdfPages

import SANDwake3D_base as sdb
from run_sandwake import get_phiinit, run_sandwake
from qois import interp_qoi, get_lowws_lowti, get_interp_coords, get_medws_lowti

np.random.seed(1)


class RANS:
    def __init__(self):
        params = self.get_default_params()
        self.params = params
        self.center = [
            params["turbforcing"]["turbx"],
            params["turbforcing"]["turby"],
            params["turbforcing"]["zhh"],
        ]
        self.turbD = params["turbforcing"]["turbD"]
        self.set_xqois()
        self.cases = ["LowWSLowTI", "MedWSLowTI"]
        # self.set_test_data()
        self.read_les_data(self.cases)
        self.qoi_y, self.qoi_z = get_interp_coords(self.center)

    def set_xqois(self, dists=[4, 6]):
        xqois = []
        self.dists = dists
        for dist in dists:
            xqois.append(
                np.searchsorted(self.params["xvec"], self.center[0] + dist * self.turbD)
            )
        self.xqois = xqois

    def set_test_data(self):
        self.target_theta = [0.05, 1.1, 1.92]
        self.data = self.get_test_calib_data(self.xqois, self.target_theta)
        self.set_qoi_sigma()

    def set_qoi_sigma(self, factor=0.1):
        self.qoi_sigma = {}
        for case in self.cases:
            self.qoi_sigma[case] = [
                0.1 * np.sqrt((self.data[case][0] ** 2).sum()),
                0.1 * np.sqrt((self.data[case][1] ** 2).sum()),
            ]

    def run_rans(self, theta=[5.48 ** (-2), 1.3, 1.92, 1.0], case="LowWSLowTI"):
        params = self.get_params(case)
        params["Cmu"] = theta[0]
        params["C1eps"] = theta[1]
        params["C2eps"] = theta[2]
        params["kfactor"] = theta[3]
        phiinit = get_phiinit(params)
        phi = run_sandwake(params, phiinit)
        return phi

    def likelihood(self, theta):
        try:
            print(f"Running for theta: {theta}")
            llhood = 0
            for case in self.cases:
                phi = self.run_rans(theta, case)
                qois = self.get_qois(phi)
                for i in range(len(self.xqois)):
                    llhood += -0.5 * np.sum(
                        (qois[i] - self.data[case][i]) ** 2
                        / self.qoi_sigma[case][i] ** 2
                    )
            print(f"Current log-likelihood: {llhood}")
            return llhood
        except:
            return -np.inf

    def get_params(self, case):
        if case == "LowWSLowTI":
            return self.get_default_params()
        elif case == "MedWSLowTI":
            params = self.get_default_params()
            params["ustar"] = 0.2125
            params["z0"] = 0.00004
            params["Tpermeter"] = 0.0025
            params["L"] = 275.0
            return params
        else:
            raise ValueError

    def get_default_params(self):
        # Set Grid
        dz = 10
        zmin = 2  # 2.00
        zmax = zmin + 400
        Nz = int((zmax - zmin) / dz + 1)
        zvec = np.linspace(zmin, zmax, Nz)

        dy = 10
        ymax = 400
        Ny = int((2 * ymax) / dy + 1)
        yvec = np.linspace(-ymax, ymax, Ny)

        ym, zm = np.meshgrid(yvec, zvec, indexing="ij")
        dy = np.mean(np.diff(yvec))
        dz = np.mean(np.diff(zvec))

        # Set turbine params
        turbhh = 150.0
        rotorD = 240

        # Set parameters
        params = {
            "rho": 1.225,
            "cp": 1005,
            "g": 9.81,
            "nu": 1.5e-5,
            "Cmu": 5.48 ** (-2),
            "C1eps": 1.3,  # 1.76,
            "C2eps": 1.92,  # 3.50, #1.92,
            "C3eps": None,  # 2.0, #-1, #0.033,
            "sigmak": 1.0,
            "sigmaeps": 1.3,
            "sigmaT": 1.0,
            "dtau": 0.5,
            "ustar": 0.175,
            "qw": -0.4,  # -0.40, #-0.050,
            "L": 500,
            "kappa": 0.42,
            "z0": 0.0001,  # 0.0005,
            "zlo": zmin,
            "Tw": 300.0,
            "beta": 1.0 / 300.0,
            "Tref": None,  # 302.0,
            # Extra mesh variables needed for turbine forcing
            "ym": ym,
            "zm": zm,
            "turbforcing": {
                # "turbx": 80.0,  # Turbine x location
                "turbx": 120.0,  # Turbine x location
                "turby": 0.0,  # Turbine y location
                "zhh": turbhh,  # Turbine hub-height
                "turbD": rotorD,  # Turbine diameter
                "Ct": 0.80,
                "Rdelta": 24,  # 40,
                "turbfunc": sdb.tanhADM,
            },
            "veerpermeter": 0.05,
            "Tpermeter": 0.002,
        }

        dx = 120
        xvecplot = [0, dx, dx + 4 * rotorD, dx + 6 * rotorD]
        # dx = 80

        xvec = np.arange(xvecplot[0], xvecplot[-1] + 1.0e-6, dx)
        params["xvec"] = xvec
        params["kfactor"] = 1.0
        return params

    def read_les_data(self, cases):
        self.data = {}
        if "LowWSLowTI" in cases:
            self.data["LowWSLowTI"] = get_lowws_lowti()
        if "MedWSLowTI" in cases:
            self.data["MedWSLowTI"] = get_medws_lowti()
        self.set_qoi_sigma(0.2)

    def get_rans_qoi(self, phi, xidx):
        # Just using the full y-z plane for u
        u = phi["u"][xidx]
        return self.interp_rans_qoi(u)

    def interp_rans_qoi(self, qoi):
        return interp_qoi(
            qoi, self.params["ym"][:, 0], self.params["zm"][0, :], self.center
        )

    def plot_qoi(self, qoi):
        plt.figure()
        plt.contourf(self.params["ym"], self.params["zm"], qoi)
        plt.show()

    def get_test_calib_params(self, theta):
        params = self.get_default_params()
        params["Cmu"] = theta[0]
        params["C1eps"] = theta[1]
        params["C2eps"] = theta[2]
        return params

    def get_qois(self, phi):
        return np.array([self.get_rans_qoi(phi, xidx) for xidx in self.xqois])

    def get_test_calib_data(self, xqois, theta):
        params = self.get_test_calib_params(theta)
        phiinit = get_phiinit(params)
        phi = run_sandwake(params, phiinit)
        return self.get_qois(phi)

    def comparison_plots(self, theta, pname, case="LowWSLowTI"):
        phi = self.run_rans(theta, case)
        qois = self.get_qois(phi)
        case_limits = {
            "LowWSLowTI": [2.5, 8.5],
            "MedWSLowTI": [3.5, 12.0],
        }
        with PdfPages(pname) as pdf:
            vmin, vmax = case_limits[case]
            levels = np.linspace(vmin, vmax, 11)
            for idx in range(len(self.xqois)):
                f, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)
                cf = axs[0].contourf(self.qoi_y, self.qoi_z, qois[idx], levels=levels)
                axs[0].set_title("RANS")
                axs[0].set_xlabel("y")
                axs[0].set_ylabel("z")
                axs[1].contourf(
                    self.qoi_y, self.qoi_z, self.data[case][idx], levels=levels
                )
                axs[1].set_title("LES")
                axs[1].set_xlabel("y")
                plt.suptitle(f"Comparison at {self.dists[idx]}D")
                plt.tight_layout()
                plt.colorbar(cf, ax=axs)
                pdf.savefig(f)
        plt.close("all")


if __name__ == "__main__":
    rans = RANS()

    def cost_function(*args):
        return -rans.likelihood(*args)

    theta_mle_init = [5.48 ** (-2), 1.3, 1.92, 1.0]
    for case in rans.cases:
        rans.comparison_plots(theta_mle_init, f"initial_compare_{case}.pdf", case=case)

    MLE = sp.optimize.minimize(
        cost_function,
        theta_mle_init,
        options={"disp": True, "maxiter": 200, "gtol": 1e-3},
        method="BFGS",
    )
    theta_mle = MLE.x
    print(f"Calibrated params: {theta_mle}")
    for case in rans.cases:
        rans.comparison_plots(theta_mle, f"calibrated_compare_{case}.pdf", case=case)
