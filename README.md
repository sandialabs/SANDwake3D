# SANDwake 3D

A three-dimensional parabolic RANS wake modeling tool.

## Downloading and installing


```bash
git clone --recursive git@github.com:lawrenceccheung/SANDwake3D.git
```

### Required and optional libraries

SANDwake3D will run on any python3 system with numpy and scipy libraries installed.

An optional library that can be used in SANDwake3D is the [Enlighten progress bar](https://pypi.org/project/enlighten/), documentation available on [readthedocs](https://python-enlighten.readthedocs.io/en/stable/).

## Running an example problem

A couple of example wind turbine cases are available in the `examples` directory of the repo.  A sample of how to run a case is shown below:

```bash
$ cd SANDwake3D
$ cd examples/Offshore_MedWS_LowTI
$ python3 ../../SANDwake3Ddriver.py MedWS_LowTI_RANS.yaml
{'verbose': 2, 'maxiter': 250, 'tol': 0.0001, 'freezevar': 0.0}
x = 120.0 dx = 120.0
Computing forces for turbine T0
[0] u: 7.1266e+01, w: 0.0000e+00, k: 1.0392e+00, eps: 7.5205e-02, T: 2.4821e+00, v: 2.1276e+00, p: 2.2393e-03
[1] u: 1.9780e+01, w: 0.0000e+00, k: 2.9270e+00, eps: 6.6146e-02, T: 2.0012e+00, v: 1.6433e+00, p: 8.0071e-03
[2] u: 7.8596e+00, w: 0.0000e+00, k: 2.6842e+00, eps: 4.5085e-02, T: 8.0920e-01, v: 8.4327e-01, p: 4.0928e-03
...
[65] u: 1.0282e-04, w: 2.8413e-07, k: 1.2048e-05, eps: 6.4784e-07, T: 1.3626e-05, v: 1.3347e-05, p: 6.0575e-09
[66] u: 9.3651e-05, w: 2.5878e-07, k: 1.0974e-05, eps: 5.9006e-07, T: 1.2411e-05, v: 1.2157e-05, p: 5.5173e-09
ZLO_WALLBC: {'ustar': 0.18169224682764804, 'Lnext': 86.5580508341383, 'Tstar': 0.027086594981574737}
C3eps_use: 4.757802508232066
avg_dil: 0.00015375347312432375
Execution time: 24.0506 seconds
Saved to Phi_MedWS_LowTI.pkl
```
At the end of the run, the solution will be saved to the file `Phi_MedWS_LowTI.pkl`.
