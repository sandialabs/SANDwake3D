import numpy as np
import pickle
from scipy.interpolate import RegularGridInterpolator


def get_medws_lowti():
    center = [1200, 1000, 150]
    ukey = "velocityx_avg"
    base_path = "./Calibration/Offshore_MedWS_LowTI/YZplanes/"

    # 4D downstream
    fname_4d = f"{base_path}/YZ_wake3.pkl"
    with open(fname_4d, "rb") as f:
        data_4d = pickle.load(f)
    D = 240
    assert center[0] + 4 * D == data_4d["x"][4, 0, 0]
    qoi_4d = data_4d[ukey][4]

    # 6D downstream
    fname_6d = f"{base_path}/YZ_wake4.pkl"
    with open(fname_6d, "rb") as f:
        data_6d = pickle.load(f)
    assert center[0] + 6 * D == data_6d["x"][1, 0, 0]
    qoi_6d = data_6d[ukey][1]
    y, z = data_4d["y"][0], data_4d["z"][0]
    return np.array([interp_les_qoi(q, y, z, center) for q in [qoi_4d, qoi_6d]])


def get_lowws_lowti():
    center = [2280, 1000, 150]
    ukey = "velocityx_avg"
    base_path = "./Calibration/Offshore_LowWS_LowTI/YZplanes/"

    # 4D downstream
    fname_4d = f"{base_path}/YZ_wake3.pkl"
    with open(fname_4d, "rb") as f:
        data_4d = pickle.load(f)
    D = 240
    assert center[0] + 4 * D == data_4d["x"][4, 0, 0]
    qoi_4d = data_4d[ukey][4]

    # 6D downstream
    fname_6d = f"{base_path}/YZ_wake4.pkl"
    with open(fname_6d, "rb") as f:
        data_6d = pickle.load(f)
    assert center[0] + 6 * D == data_6d["x"][1, 0, 0]
    qoi_6d = data_6d[ukey][1]
    y, z = data_4d["y"][0], data_4d["z"][0]
    return np.array([interp_les_qoi(q, y, z, center) for q in [qoi_4d, qoi_6d]])


def get_interp_coords(center=[0, 0, 0]):
    target_y = np.linspace(-360, 360, 49) + center[1]
    target_z = np.linspace(10, 400, 21)
    ty, tz = np.meshgrid(target_y, target_z, indexing="ij")
    return (ty, tz)


def interp_les_qoi(qoi, y, z, center):
    return np.flipud(interp_qoi(qoi.T, y[0, :], z[:, 0], center))


def interp_qoi(qoi, y, z, center):
    ty, tz = get_interp_coords(center)
    interp = RegularGridInterpolator((y, z), qoi)
    return interp((ty, tz))
