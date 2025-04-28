# SANDwake 3D

A three-dimensional parabolic RANS wake modeling tool.

## For laminar equations

### Governing equations

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0
$$


$$
u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + w \frac{\partial u}{\partial z} = 
\nu \left( \frac{\partial^2 u}{\partial y^2} +  \frac{\partial^2 u}{\partial z^2} \right)
$$

<!--
$$
u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + w \frac{\partial v}{\partial z} =  \nu \left( \frac{\partial^2 v}{\partial y^2} +  \frac{\partial^2 v}{\partial z^2} \right)
$$
-->

$$
u \frac{\partial w}{\partial x} + v \frac{\partial w}{\partial y} + w \frac{\partial w}{\partial z} = \nu \left( \frac{\partial^2 w}{\partial y^2} +  \frac{\partial^2 w}{\partial z^2} \right)
$$

### Discretization

$$
\tilde{u} = \frac{u_{ij}^{n+1} + u_{ij}^n}{2}
$$


#### U-momentum first step

$$
\tilde{u} \frac{u_{ij}^{n+1/2} - u_{ij}^n}{\Delta x/2} + \tilde{v} \frac{D_y u_{ij}^{n+1/2}}{\Delta y}  + \tilde{w} \frac{D_z u_{ij}^n }{\Delta z} = 
\nu \left[ \frac{D_y^2 u_{ij}^{n+1/2}}{(\Delta y)^2} + \frac{D_z^2 u_{ij}^n}{(\Delta z)^2} \right]
$$

$$
\tilde{u} \frac{u_{ij}^{n+1/2}}{\Delta x/2} + \tilde{v} \frac{D_y u_{ij}^{n+1/2}}{\Delta y} + \nu \frac{D_y^2 u_{ij}^{n+1/2} }{(\Delta y)^2} =   
\tilde{u} \frac{u_{ij}^n}{\Delta x/2} - \tilde{w} \frac{D_z u_{ij}^n }{\Delta z} + \nu \frac{D_z^2 u_{ij}^n}{(\Delta z)^2}
$$

Reorganizing

$$ \left[
\frac{ \tilde{u}}{\Delta x/2} + \tilde{v} \frac{D_y}{\Delta y} + \nu \frac{D_y^2 }{(\Delta y)^2} \right] u_{ij}^{n+1/2} =
\left[  \frac{ \tilde{u} }{\Delta x/2} - \tilde{w} \frac{D_z }{\Delta z} + \nu \frac{D_z^2}{(\Delta z)^2} \right] u_{ij}^n
$$

#### U-momentum second step

$$
\tilde{u} \frac{u_{ij}^{n+1} - u_{ij}^{n+1/2}}{\Delta x/2} + \tilde{v} \frac{D_y u_{ij}^{n+1/2}}{\Delta y} + \tilde{w} \frac{D_z u_{ij}^{n+1} }{\Delta z} = 
\nu \left[ \frac{D_y^2 u_{ij}^{n+1/2}}{(\Delta y)^2} + \frac{D_z^2 u_{ij}^{n+1}}{(\Delta z)^2} \right]
$$

$$
\tilde{u} \frac{u_{ij}^{n+1}}{\Delta x/2} + \tilde{w} \frac{D_z u_{ij}^{n+1} }{\Delta z} - \nu \frac{D_z^2 u_{ij}^{n+1}}{(\Delta z)^2}=
\tilde{u} \frac{u_{ij}^{n+1/2}}{\Delta x/2} - \tilde{v} \frac{D_y u_{ij}^{n+1/2}}{\Delta y} + \nu \frac{D_y^2 u_{ij}^{n+1/2}}{(\Delta y)^2}
$$

Reorganizing

$$ \left[\frac{\tilde{u}}{\Delta x/2} + \tilde{w} \frac{D_z }{\Delta z} - \nu \frac{D_z^2 }{(\Delta z)^2} \right] u_{ij}^{n+1}=
\left[ \frac{\tilde{u}}{\Delta x/2} - \tilde{v} \frac{D_y }{\Delta y} + \nu \frac{D_y^2 }{(\Delta y)^2} \right] u_{ij}^{n+1/2}
$$
