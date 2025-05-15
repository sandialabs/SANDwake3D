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
\nu_T \left[ \left(\frac{\partial u}{\partial y} \right)^2 + \left(\frac{\partial w}{\partial z} \right)^2 \right]
- \epsilon + 
\frac{\partial}{\partial y} \left( (\nu + {\nu_T \over \sigma_k}) \frac{\partial w}{\partial y} \right) 
+
\frac{\partial}{\partial z} \left( (\nu + {\nu_T\over \sigma_k} ) \frac{\partial w}{\partial z} \right) 
+ G_B
$$

Dissipation
$$
u \frac{\partial \varepsilon}{\partial x} + v \frac{\partial \varepsilon}{\partial y} + w \frac{\partial \varepsilon}{\partial z} = 
\frac{C_{1\varepsilon}}{\mathcal T} \left[ \nu_T \left(\frac{\partial u}{\partial y} \right)^2 + \nu_T \left(\frac{\partial w}{\partial z} \right)^2 + (1-C_{3\varepsilon})G_B \right]
-
\frac{C_{2\varepsilon}}{\mathcal T} \varepsilon
+
\frac{\partial}{\partial y} \left( (\nu + {\nu_T \over \sigma_k}) \frac{\partial \varepsilon}{\partial y} \right) 
+
\frac{\partial}{\partial z} \left( (\nu + {\nu_T\over \sigma_k} ) \frac{\partial \varepsilon}{\partial z} \right) 
$$

$$
\nu_T = C_\mu k \mathcal{T}
$$

The timescsale $\mathcal{T}$ is the larger of 

$$
\mathcal{T} = \max\left( {k\over \varepsilon}, 6 \sqrt{\nu \over \varepsilon} \right)
$$

### Reorganized

Mass
$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0
$$

U-Momentum
$$
u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + w \frac{\partial u}{\partial z} =
(\nu + \nu_T) \frac{\partial^2 u}{\partial y^2} 
+
(\nu + \nu_T) \frac{\partial^2 u}{\partial z^2} 
+
\frac{\partial \nu_T}{\partial y} \frac{\partial u}{\partial y} 
+
\frac{\partial \nu_T}{\partial z} \left( \frac{\partial u}{\partial z} \right) 
$$

W-Momentum
$$
u \frac{\partial w}{\partial x} + v \frac{\partial w}{\partial y} + w \frac{\partial w}{\partial z} =
(\nu + \nu_T) \frac{\partial^2 w}{\partial y^2} 
+
(\nu + \nu_T) \frac{\partial^2 w}{\partial z^2} 
+
\frac{\partial \nu_T}{\partial y} \frac{\partial w}{\partial y} 
+
\frac{\partial \nu_T}{\partial z} \left( \frac{\partial w}{\partial z} \right)
+ g \beta (T-T_0)
$$
