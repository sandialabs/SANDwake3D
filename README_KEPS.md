# K-epsilon formulation

## Governing equations

### Original form

Mass

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0
$$

U-Momentum

$$
u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + w \frac{\partial u}{\partial z} = 
\frac{\partial}{\partial y} \left( (\nu + \nu_T) \frac{\partial u}{\partial y} \right) 
+
\frac{\partial}{\partial z} \left( (\nu + \nu_T) \frac{\partial u}{\partial z} \right) 
$$

W-Momentum

$$
u \frac{\partial w}{\partial x} + v \frac{\partial w}{\partial y} + w \frac{\partial w}{\partial z} = 
\frac{\partial}{\partial y} \left( (\nu + \nu_T) \frac{\partial w}{\partial y} \right) 
+
\frac{\partial}{\partial z} \left( (\nu + \nu_T) \frac{\partial w}{\partial z} \right) + g \beta (T-T_0)
$$

TKE

$$
u \frac{\partial k}{\partial x} + v \frac{\partial k}{\partial y} + w \frac{\partial k}{\partial z} = 
\nu_T \left[ \left(\frac{\partial u}{\partial y} \right)^2 + \left(\frac{\partial w}{\partial z} \right)^2 \right] - 
\epsilon + \frac{\partial}{\partial y} \left( (\nu + {\nu_T \over \sigma_k}) \frac{\partial k}{\partial y} \right) + 
\frac{\partial}{\partial z} \left( (\nu + {\nu_T\over \sigma_k} ) \frac{\partial k}{\partial z} \right) + G_B
$$

Dissipation

$$
u \frac{\partial \varepsilon}{\partial x} + v \frac{\partial \varepsilon}{\partial y} + w \frac{\partial \varepsilon}{\partial z} = 
\frac{C_{1\varepsilon}}{\mathcal T} \left[ \nu_T \left(\frac{\partial v}{\partial y} \right)^2 + \nu_T \left(\frac{\partial w}{\partial z} \right)^2 + (1-C_{3\varepsilon})G_B \right] -
\frac{C_{2\varepsilon}}{\mathcal T} \varepsilon +
\frac{\partial}{\partial y} \left( (\nu + {\nu_T \over \sigma_k}) \frac{\partial \varepsilon}{\partial y} \right) +
\frac{\partial}{\partial z} \left( (\nu + {\nu_T\over \sigma_k} ) \frac{\partial \varepsilon}{\partial z} \right) 
$$

Temperature 

_TBD_

$$
\nu_T = C_\mu k \mathcal{T}
$$

The timescsale $\mathcal{T}$ is the larger of 

$$
\mathcal{T} = \max\left( {k\over \varepsilon}, 6 \sqrt{\nu \over \varepsilon} \right)
$$

### Reorganized for implementation

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
\tilde{u} = \frac{u_{ij}^{n+1} + u_{ij}^n}{2}
$$

#### Mass

$$
\frac{\partial U}{\partial x} + \frac{\partial V}{\partial y} + \frac{\partial W}{\partial z} = 0
$$

$$
\frac{u_{ij}^{n+1} - u_{ij}^{n}}{\Delta x} + \frac{D_y v_{ij}^{n+1}}{\Delta y} + \frac{D_z w_{ij}^{n+1}}{\Delta z} = 0
$$


#### U-Momentum

$$
U \frac{\partial U}{\partial x} + V \frac{\partial U}{\partial y} + W \frac{\partial U}{\partial z} =
(\nu + \nu_T) \frac{\partial^2 U}{\partial y^2} 
+
(\nu + \nu_T) \frac{\partial^2 U}{\partial z^2} 
+
\frac{\partial \nu_T}{\partial y} \frac{\partial U}{\partial y} 
+
\frac{\partial \nu_T}{\partial z} \left( \frac{\partial U}{\partial z} \right) 
$$

$$
U \frac{\partial U}{\partial x} + 
\left( V - \frac{\partial \nu_T}{\partial y} \right) \frac{\partial U}{\partial y} + 
\left( W - \frac{\partial \nu_T}{\partial z} \right) \frac{\partial U}{\partial z} =
(\nu + \nu_T) \frac{\partial^2 U}{\partial y^2} 
+
(\nu + \nu_T) \frac{\partial^2 U}{\partial z^2} 
$$

**First half-step**

$$ 
\left[\frac{ \tilde{U}}{\Delta x/2} + \left( \tilde{V} - \frac{D_y \tilde{\nu}\_T}{\Delta y} \right) \frac{D_y}{\Delta y} - (\nu + \tilde{\nu}\_T) \frac{D_y^2 }{(\Delta y)^2} \right] U_{ij}^{n+1/2} =
\left[  \frac{ \tilde{U} }{\Delta x/2} - \left( \tilde{W} - \frac{D_z \tilde{\nu}\_T}{\Delta z} \right) \frac{D_z }{\Delta z} + (\nu + \tilde{\nu}\_T) \frac{D_z^2}{(\Delta z)^2} \right] U_{ij}^n + f_x^n
$$

**Second half-step**

$$ 
\left[\frac{\tilde{U}}{\Delta x/2} + \left( \tilde{W} - \frac{D_z \tilde{\nu}\_T}{\Delta z} \right) \frac{D_z }{\Delta z} - (\nu + \tilde{\nu}\_T) \frac{D_z^2 }{(\Delta z)^2} \right] U_{ij}^{n+1}=
\left[ \frac{\tilde{U}}{\Delta x/2} - \left( \tilde{V} - \frac{D_y \tilde{\nu}\_T}{\Delta y} \right) \frac{D_y }{\Delta y} + (\nu + \tilde{\nu}\_T) \frac{D_y^2 }{(\Delta y)^2} \right] U_{ij}^{n+1/2} + f_x^{n+1/2}
$$

#### W-Momentum

$$
U \frac{\partial W}{\partial x} + V \frac{\partial W}{\partial y} + W \frac{\partial W}{\partial z} =
(\nu + \nu_T) \frac{\partial^2 W}{\partial y^2} +
(\nu + \nu_T) \frac{\partial^2 W}{\partial z^2} +
\frac{\partial \nu_T}{\partial y} \frac{\partial W}{\partial y} +
\frac{\partial \nu_T}{\partial z} \left( \frac{\partial W}{\partial z} \right) + 
g \beta (T-T_0)
$$

$$
U \frac{\partial W}{\partial x} + 
\left( V - \frac{\partial \nu_T}{\partial y} \right) \frac{\partial W}{\partial y} + 
\left( W - \frac{\partial \nu_T}{\partial z} \right) \frac{\partial W}{\partial z} =
(\nu + \nu_T) \frac{\partial^2 W}{\partial y^2} +
(\nu + \nu_T) \frac{\partial^2 W}{\partial z^2} +
g \beta (T-T_0)
$$

**First half-step**

$$
\left[ \frac{ \tilde{U}}{\Delta x/2} + \left( \tilde{V} - \frac{D_y \tilde{\nu}\_T}{\Delta y} \right) \frac{D_y}{\Delta y} - (\nu + \tilde{\nu}\_T) \frac{D_y^2 }{(\Delta y)^2} \right] W_{ij}^{n+1/2} =
\left[\frac{\tilde{U}}{\Delta x/2} - \left( \tilde{W} - \frac{D_z \tilde{\nu}\_T}{\Delta z} \right) \frac{D_z }{\Delta z} + (\nu + \tilde{\nu}\_T) \frac{D_z^2}{(\Delta z)^2} \right] W_{ij}^n + F_z^n
$$

**Second half-step**

$$ 
\left[\frac{\tilde{U}}{\Delta x/2} + \left( \tilde{W} - \frac{D_z \tilde{\nu}\_T}{\Delta z} \right) \frac{D_z }{\Delta z} - (\nu + \tilde{\nu}\_T) \frac{D_z^2 }{(\Delta z)^2} \right] W_{ij}^{n+1}=
\left[ \frac{\tilde{U}}{\Delta x/2} - \left( \tilde{V} - \frac{D_y \tilde{\nu}\_T}{\Delta y} \right) \frac{D_y }{\Delta y} + (\nu + \tilde{\nu}\_T) \frac{D_y^2 }{(\Delta y)^2} \right] W_{ij}^{n+1/2} + F_z^{n+1/2}
$$

#### TKE

$$
U \frac{\partial k}{\partial x} + V \frac{\partial k}{\partial y} + W \frac{\partial k}{\partial z} = 
\left (\nu + {\nu_T \over \sigma_k} \right) \frac{\partial^2 k}{\partial y^2} + 
\frac{\partial}{\partial y} \left( {\nu_T \over \sigma_k}\right) \frac{\partial k}{\partial y}  + 
\left( \nu + {\nu_T\over \sigma_k} \right) \frac{\partial^2 k}{\partial z^2}  +
\frac{\partial}{\partial z} \left( {\nu_T\over \sigma_k}  \right) \frac{\partial k}{\partial z}  +
\nu_T \left[ \left(\frac{\partial U}{\partial y} \right)^2 + \left(\frac{\partial W}{\partial z} \right)^2 \right] -
\epsilon + G_B
$$

$$
U \frac{\partial k}{\partial x} + 
\left( V -\frac{\partial}{\partial y} \left( {\nu_T \over \sigma_k}\right) \right) \frac{\partial k}{\partial y} + 
\left( W -\frac{\partial}{\partial z} \left( {\nu_T\over \sigma_k}  \right) \right) \frac{\partial k}{\partial z} = 
\left (\nu + {\nu_T \over \sigma_k} \right) \frac{\partial^2 k}{\partial y^2} + 
\left( \nu + {\nu_T\over \sigma_k} \right) \frac{\partial^2 k}{\partial z^2}  +
\nu_T \left[ \left(\frac{\partial U}{\partial y} \right)^2 + \left(\frac{\partial W}{\partial z} \right)^2 \right] -
\epsilon + G_B
$$

**First half-step**

$$
\left[ \frac{ \tilde{U}}{\Delta x/2} + \left( \tilde{V} - \frac{D_y \tilde{\nu}_T}{\sigma_k \Delta y} \right) \frac{D_y}{\Delta y} - (\nu + { \tilde{\nu}_T \over \sigma_k}) \frac{D_y^2 }{(\Delta y)^2} \right] k_{ij}^{n+1/2} =
\left[\frac{\tilde{U}}{\Delta x/2} - \left( \tilde{W} - \frac{D_z \tilde{\nu}_T}{\sigma_k \Delta z} \right) \frac{D_z }{\Delta z} + (\nu + {\tilde{\nu}_T \over \sigma_k}) \frac{D_z^2}{(\Delta z)^2} \right] k_{ij}^n + F_k^n
$$

$$
F_k = \tilde{\nu}_T \left[ \left(\frac{D_y \tilde{U}}{\Delta y} \right)^2 + \left(\frac{D_z \tilde{W}}{\Delta z} \right)^2  \right] - \varepsilon + G_B
$$

**Second half-step**

$$
\left[\frac{\tilde{U}}{\Delta x/2} + \left( \tilde{W} - \frac{D_z \tilde{\nu}_T}{\sigma_k \Delta z} \right) \frac{D_z }{\Delta z} - (\nu + \frac{\tilde{\nu}_T}{\sigma_k}) \frac{D_z^2 }{(\Delta z)^2} \right] k_{ij}^{n+1}=
\left[ \frac{\tilde{U}}{\Delta x/2} - \left( \tilde{V} - \frac{D_y \tilde{\nu}_T}{\sigma_k \Delta y} \right) \frac{D_y }{\Delta y} + (\nu + \frac{\tilde{\nu}_T}{\sigma_k}) \frac{D_y^2 }{(\Delta y)^2} \right] k_{ij}^{n+1/2} + F_k^{n+1/2}
$$


#### Dissipation

$$
U \frac{\partial \varepsilon}{\partial x} + V \frac{\partial \varepsilon}{\partial y} + W \frac{\partial \varepsilon}{\partial z} = 
\frac{C_{1\varepsilon}}{\mathcal T} \left[ \nu_T \left(\frac{\partial V}{\partial y} \right)^2 + \nu_T \left(\frac{\partial W}{\partial z} \right)^2 + (1-C_{3\varepsilon})G_B \right] -
\frac{C_{2\varepsilon}}{\mathcal T} \varepsilon +
\frac{\partial}{\partial y} \left( {\nu_T \over \sigma_k} \right) \frac{\partial \varepsilon}{\partial y}  +
\left( \nu + {\nu_T \over \sigma_k} \right) \frac{\partial^2 \varepsilon}{\partial y^2}  +
\frac{\partial}{\partial z} \left( {\nu_T\over \sigma_k} \right) \frac{\partial \varepsilon}{\partial z}  + 
\left( \nu + {\nu_T\over \sigma_k} \right) \frac{\partial^2 \varepsilon}{\partial z^2} 
$$

$$
U \frac{\partial \varepsilon}{\partial x} + 
\left( V -\frac{\partial}{\partial y} \left( {\nu_T \over \sigma_k} \right) \right) \frac{\partial \varepsilon}{\partial y} + 
\left( W - \frac{\partial}{\partial z} \left( {\nu_T\over \sigma_k} \right) \right) \frac{\partial \varepsilon}{\partial z} = 
\left( \nu + {\nu_T \over \sigma_k} \right) \frac{\partial^2 \varepsilon}{\partial y^2}  +
\left( \nu + {\nu_T\over \sigma_k} \right) \frac{\partial^2 \varepsilon}{\partial z^2} +
\frac{C_{1\varepsilon}}{\mathcal T} \left[ \nu_T \left(\frac{\partial V}{\partial y} \right)^2 + \nu_T \left(\frac{\partial W}{\partial z} \right)^2 + (1-C_{3\varepsilon})G_B \right] -
\frac{C_{2\varepsilon}}{\mathcal T} \varepsilon 
$$

**First half-step**

$$
\left[ \frac{ \tilde{U}}{\Delta x/2} + \left( \tilde{V} - \frac{D_y \tilde{\nu}_T}{\sigma_k \Delta y} \right) \frac{D_y}{\Delta y} - (\nu + { \tilde{\nu}_T \over \sigma_k}) \frac{D_y^2 }{(\Delta y)^2} \right] \varepsilon_{ij}^{n+1/2} =
\left[\frac{\tilde{U}}{\Delta x/2} - \left( \tilde{W} - \frac{D_z \tilde{\nu}_T}{\sigma_k \Delta z} \right) \frac{D_z }{\Delta z} + (\nu + {\tilde{\nu}_T \over \sigma_k}) \frac{D_z^2}{(\Delta z)^2} \right] \varepsilon_{ij}^n + F_k^n
$$

**Second half-step**

$$
\left[\frac{\tilde{U}}{\Delta x/2} + \left( \tilde{W} - \frac{D_z \tilde{\nu}_T}{\sigma_k \Delta z} \right) \frac{D_z }{\Delta z} - (\nu + \frac{\tilde{\nu}_T}{\sigma_k}) \frac{D_z^2 }{(\Delta z)^2} \right] \varepsilon_{ij}^{n+1} =
\left[ \frac{\tilde{U}}{\Delta x/2} - \left( \tilde{V} - \frac{D_y \tilde{\nu}_T}{\sigma_k \Delta y} \right) \frac{D_y }{\Delta y} + (\nu + \frac{\tilde{\nu}_T}{\sigma_k}) \frac{D_y^2 }{(\Delta y)^2} \right] \varepsilon_{ij}^{n+1/2} + F_k^{n+1/2}
$$