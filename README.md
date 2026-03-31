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
```
