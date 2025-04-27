# ADI test using heat equation

See https://en.wikipedia.org/wiki/Alternating-direction_implicit_method for more information

$$
\frac{\partial u}{\partial x} = \alpha \left( \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right) 
$$

Discretizing the equation using ADI-scheme.  For the y-derivative part, implicitly advance $u_{ij}^{n}$ to $u_{ij}^{n+1/2}$

$$
\frac{u_{ij}^{n+1/2} - u_{ij}^n}{\Delta x/2} = 
\alpha \left[ \frac{D_y^2 u_{ij}^{n+1/2}}{(\Delta y)^2} + \frac{D_z^2 u_{ij}^n}{(\Delta z)^2}
\right]
$$

Then do the z-derivative part implicitly to advance $u_{ij}^{n+1/2}$ to $u_{ij}^{n+1}$

$$
\frac{u_{ij}^{n+1} - u_{ij}^{n+1/2}}{\Delta x/2} = 
\alpha \left[ \frac{D_y^2 u_{ij}^{n+1/2}}{(\Delta y)^2} + \frac{D_z^2 u_{ij}^{n+1}}{(\Delta z)^2}
\right]
$$

Here the spatial $y$ & $z$ discretization schemes are

$$
D_y^2 u_{ij} = u_{i+1,j} - 2 u_{i,j} + u_{i-1,j}
$$

$$
D_z^2 u_{ij} = u_{i,j+1} - 2 u_{i,j} + u_{i,j-1}
$$


Rearranging the equations, for the first sweep from $n$ to $n+1/2$ we have 

$$
\left[ \frac{(\Delta z)^2}{\alpha} - \frac{\Delta x}{2} \left(\frac{\Delta z}{\Delta y}\right)^2 D_y^2 \right] u_{ij}^{n+1/2} = 
\left[ \frac{(\Delta z)^2}{\alpha} + \frac{\Delta x}{2} D_z^2 \right] u_{ij}^{n}
$$

which is a tridiagonal system over the $i$ points for every $j$.  Then for the second sweep from $n+1/2$ to $n+1$ we have 

$$
\left[ \frac{(\Delta y)^2}{\alpha} - \frac{\Delta x}{2} \left(\frac{\Delta y}{\Delta z}\right)^2 D_z^2 \right] u_{ij}^{n+1} = 
\left[ \frac{(\Delta y)^2}{\alpha} + \frac{\Delta x}{2} D_y^2 \right] u_{ij}^{n+1/2}
$$

which is a tridiagonal system over the $j$ points for every $i$.

Compare with exact solution in https://web.stanford.edu/class/math220b/handouts/heateqn.pdf.