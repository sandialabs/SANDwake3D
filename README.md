# SANDwake 3D

A three-dimensional parabolic RANS wake modeling tool.

## For laminar equations

### Governing equations

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0
$$


$$
u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + w \frac{\partial u}{\partial z} = 
\nu \left( \frac{\partial^2 u}{\partial y^2} +  \frac{\partial^2 u}{\partial z^2} \right) + f_x
$$

$$
u \frac{\partial w}{\partial x} + v \frac{\partial w}{\partial y} + w \frac{\partial w}{\partial z} = \nu \left( \frac{\partial^2 w}{\partial y^2} +  \frac{\partial^2 w}{\partial z^2} \right) + f_z
$$

### Discretization

Use the following discrete operators:

$$
D_y u_{ij} = \frac{u_{i+1,j} - u_{i-1,j}}{2}
$$

$$
D_y^2 u_{ij} = u_{i+1,j} - 2 u_{i,j} + u_{i-1,j}
$$

$$
D_z u_{ij} = \frac{u_{i,j+1} -  u_{i,j-1}}{2}
$$

$$
D_z^2 u_{ij} = u_{i,j+1} - 2 u_{i,j} + u_{i,j-1}
$$

Also in the convective terms, use the averaged velocities

$$
\tilde{u} = \tilde{u}\_{ij} = \frac{u\_{ij}^{n+1} + u\_{ij}^n}{2}
$$

#### U-momentum first step

$$
\tilde{u} \frac{u_{ij}^{n+1/2} - u_{ij}^n}{\Delta x/2} + \tilde{v} \frac{D_y u_{ij}^{n+1/2}}{\Delta y}  + \tilde{w} \frac{D_z u_{ij}^n }{\Delta z} = 
\nu \left[ \frac{D_y^2 u_{ij}^{n+1/2}}{(\Delta y)^2} + \frac{D_z^2 u_{ij}^n}{(\Delta z)^2} \right] + f_x^{n}
$$

$$
\tilde{u} \frac{u_{ij}^{n+1/2}}{\Delta x/2} + \tilde{v} \frac{D_y u_{ij}^{n+1/2}}{\Delta y} + \nu \frac{D_y^2 u_{ij}^{n+1/2} }{(\Delta y)^2} =   
\tilde{u} \frac{u_{ij}^n}{\Delta x/2} - \tilde{w} \frac{D_z u_{ij}^n }{\Delta z} + \nu \frac{D_z^2 u_{ij}^n}{(\Delta z)^2} + f_x^{n}
$$

Reorganizing

$$ 
\left[\frac{ \tilde{u}}{\Delta x/2} + \tilde{v} \frac{D_y}{\Delta y} + \nu \frac{D_y^2 }{(\Delta y)^2} \right] u_{ij}^{n+1/2} =
\left[  \frac{ \tilde{u} }{\Delta x/2} - \tilde{w} \frac{D_z }{\Delta z} + \nu \frac{D_z^2}{(\Delta z)^2} \right] u_{ij}^n + f_x^n
$$

#### U-momentum second step

$$
\tilde{u} \frac{u_{ij}^{n+1} - u_{ij}^{n+1/2}}{\Delta x/2} + \tilde{v} \frac{D_y u_{ij}^{n+1/2}}{\Delta y} + \tilde{w} \frac{D_z u_{ij}^{n+1} }{\Delta z} = 
\nu \left[ \frac{D_y^2 u_{ij}^{n+1/2}}{(\Delta y)^2} + \frac{D_z^2 u_{ij}^{n+1}}{(\Delta z)^2} \right] + f_x^{n+1/2}
$$

$$
\tilde{u} \frac{u_{ij}^{n+1}}{\Delta x/2} + \tilde{w} \frac{D_z u_{ij}^{n+1} }{\Delta z} - \nu \frac{D_z^2 u_{ij}^{n+1}}{(\Delta z)^2}=
\tilde{u} \frac{u_{ij}^{n+1/2}}{\Delta x/2} - \tilde{v} \frac{D_y u_{ij}^{n+1/2}}{\Delta y} + \nu \frac{D_y^2 u_{ij}^{n+1/2}}{(\Delta y)^2} + f_x^{n+1/2}
$$

Reorganizing

$$ 
\left[\frac{\tilde{u}}{\Delta x/2} + \tilde{w} \frac{D_z }{\Delta z} - \nu \frac{D_z^2 }{(\Delta z)^2} \right] u_{ij}^{n+1}=
\left[ \frac{\tilde{u}}{\Delta x/2} - \tilde{v} \frac{D_y }{\Delta y} + \nu \frac{D_y^2 }{(\Delta y)^2} \right] u_{ij}^{n+1/2} + f_x^{n+1/2}
$$

#### W-momentum first step

$$
\tilde{u} \frac{w_{ij}^{n+1/2} - w_{ij}^n}{\Delta x/2} + \tilde{v} \frac{D_y w_{ij}^{n+1/2}}{\Delta y}  + \tilde{w} \frac{D_z w_{ij}^n }{\Delta z} = 
\nu \left[ \frac{D_y^2 w_{ij}^{n+1/2}}{(\Delta y)^2} + \frac{D_z^2 w_{ij}^n}{(\Delta z)^2} \right]
$$

Reorganizing

$$
\left[ \frac{ \tilde{u}}{\Delta x/2} + \tilde{v}\frac{D_y}{\Delta y} + \nu \frac{D_y^2 }{(\Delta y)^2} \right] w_{ij}^{n+1/2} =
\left[\frac{\tilde{u}}{\Delta x/2} - \tilde{w} \frac{D_z }{\Delta z} + \nu \frac{D_z^2}{(\Delta z)^2} \right] w_{ij}^n
$$

#### W-momentum second step

$$
\tilde{u} \frac{w_{ij}^{n+1} - w_{ij}^{n+1/2}}{\Delta x/2} + \tilde{v} \frac{D_y w_{ij}^{n+1/2}}{\Delta y} + \tilde{w} \frac{D_z w_{ij}^{n+1} }{\Delta z} = 
\nu \left[ \frac{D_y^2 w_{ij}^{n+1/2}}{(\Delta y)^2} + \frac{D_z^2 w_{ij}^{n+1}}{(\Delta z)^2} \right]
$$

Reorganizing

$$ 
\left[\frac{\tilde{u}}{\Delta x/2} + \tilde{w} \frac{D_z }{\Delta z} - \nu \frac{D_z^2 }{(\Delta z)^2} \right] w_{ij}^{n+1}=
\left[ \frac{\tilde{u}}{\Delta x/2} - \tilde{v} \frac{D_y }{\Delta y} + \nu \frac{D_y^2 }{(\Delta y)^2} \right] w_{ij}^{n+1/2}
$$

### Mass conservation


$$
\frac{u_{ij}^{n+1} - u_{ij}^{n}}{\Delta x} + \frac{D_y v_{ij}^{n+1}}{\Delta y} + \frac{D_z w_{ij}^{n+1}}{\Delta z} = 0
$$

Reorganizing

$$
D_y v_{ij}^{n+1} = -\Delta y \left[ \frac{u_{ij}^{n+1} - u_{ij}^{n}}{\Delta x} + \frac{D_z w_{ij}^{n+1}}{\Delta z}  \right]
$$

### Solution algorithm

Given solution $\phi^n$ at $x_n$, to find solution $\phi^{n+1}$ at $x_{n+1}$: 

1.  Solve U-Momentum for $u^{n+1}$:
    - First half-step
    - Second half-step
    
2.  Solve W-Momentum for $w^{n+1}$:
    - First half-step
    - Second half-step

3.  Solve Mass conservation for $v^{n+1}$

4.  Repeat steps 1-3 until solution $\phi^{n+1}$ converges
