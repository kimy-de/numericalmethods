"""
Heat equation
∂u/∂t - α ∇²u = 0
"""

from __future__ import print_function
import os
from fenics import *
import time
import matplotlib.pyplot as plt
import imageio

# Create mesh and define function space
alpha = 1 # diffusion coefficient
nx = ny = 50
domain = 1
mesh = RectangleMesh(Point(-domain, -domain), Point(domain, domain), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

T = 0.05          # final time
num_steps = 50   # number of time steps
dt = T / num_steps # time step size

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# Define initial condition  
initial_function = 'exp(-a*pow(x[0], 2) - a*pow(x[1], 2))'
initial_condition = Expression(initial_function, degree=2, a=5, t=0)
u_n = interpolate(initial_condition, V)

# Define weak form
F = u*v*dx + dt*alpha*dot(grad(u), grad(v))*dx - u_n*v*dx
a, L = lhs(F), rhs(F)
assembled_A = assemble(a)

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary

boundary_condition = [DirichletBC(V, Constant(0), boundary)]

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):
    
    # Update current time
    t += dt

    # Compute solution
    assembled_L = assemble(L)
    [bc.apply(assembled_A, assembled_L) for bc in boundary_condition]
    solve(assembled_A, u.vector(), assembled_L)

    # Plot solution
    cplt = plot(dot(u,u))
    cplt.set_cmap('coolwarm')
    cplt.set_clim(vmin=0, vmax=1)
    plt.xticks([-domain,domain])
    plt.yticks([-domain,domain])
    #plt.colorbar(cplt)
    plt.savefig('./time'+str(n)+'.jpg')
    plt.draw()
    plt.pause(0.05)
    plt.clf()

    # Update previous solution
    u_n.assign(u)

with imageio.get_writer('./heat2d.gif', mode='I') as writer:
    for i in range(num_steps):
        image = imageio.v2.imread('./time'+str(i)+'.jpg')
        writer.append_data(image)

        os.remove('./time'+str(i)+'.jpg')
