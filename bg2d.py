"""
Burgers' Equation in 2D
∂u/∂t + (u ⋅ ∇) u - ν ∇²u = 0
"""

import fenics as fe
import matplotlib.pyplot as plt
import imageio

n_points = 40

dt = 0.01 #0.025 # 1e-3*dx*dy/nu
n_time = 30 #int(1/dt)
nu = 1e-4

def main():
    pone = fe.Point(-1,-1)
    ptwo = fe.Point(1,1)
    mesh = fe.RectangleMesh(pone, ptwo, n_points, n_points)

    # Taylor-Hood Elements. The order of the function space for the pressure has
    # to be one order lower than for the velocity
    velocity_function_space = fe.VectorFunctionSpace(mesh, 'CG', 2)
 
    u_trial = fe.TrialFunction(velocity_function_space)
    v_test = fe.TestFunction(velocity_function_space)

    # Define the Boundary Condition
    def leftbot(x, on_boundary):
        #return on_boundary
        return (x[1] < -1.0 + fe.DOLFIN_EPS
                or x[0] < -1.0 + fe.DOLFIN_EPS)

    def topright(x, on_boundary):
        return (x[1] > 1.0 - fe.DOLFIN_EPS
                or x[0] > 1.0 - fe.DOLFIN_EPS)

    def top(x, on_boundary):
        return (x[1] > 1.0 - fe.DOLFIN_EPS)

    velocity_boundary_condition = [
        #fe.DirichletBC(velocity_function_space, (0.0, 0.0), top),
        fe.DirichletBC(velocity_function_space, (0.0, 0.0), leftbot)
    ]

    # Define the solution fields involved
    #u = fe.Function(velocity_function_space)
    # Define boundary condition
  
    inivstrg = 'exp(-2.0*(x[0]*x[0]+x[1]*x[1]))'
    u_ini = fe.Expression((inivstrg,inivstrg), degree=1, t=0)
    u = fe.interpolate(u_ini, velocity_function_space)
    u_next = fe.Function(velocity_function_space)

    # Weak form
    burgers_weak_form = (
        (1.0/dt) * fe.inner(u_trial - u, v_test) * fe.dx
        + fe.inner(fe.grad(u)*u, v_test) * fe.dx
        - nu * fe.inner(fe.grad(u), fe.grad(v_test)) * fe.dx
    )

    weak_form_lhs = fe.lhs(burgers_weak_form)
    weak_form_rhs = fe.rhs(burgers_weak_form)
    assembled_system_matrix = fe.assemble(weak_form_lhs)

    try:
        for t in range(n_time):
            assembled_rhs = fe.assemble(weak_form_rhs)
            [bc.apply(assembled_system_matrix, assembled_rhs) for bc in velocity_boundary_condition]
            fe.solve(
                assembled_system_matrix,
                u_next.vector(),
                assembled_rhs
            )

            # Advance in time
            u.assign(u_next)

            # Visualize interactively
            cplt = fe.plot(fe.inner(u_next,u_next))
            cplt.set_cmap('plasma')
            #plt.colorbar(cplt)
            plt.xticks([-1,1])
            plt.yticks([-1,1])
            plt.savefig('./time'+str(t)+'.jpg')
            plt.draw()
            plt.pause(0.001)
            plt.clf()
    except:
        print("non-finite values.")

    with imageio.get_writer('./bg2d.gif', mode='I') as writer:
        
        for i in range(n_time):
            try:
                image = imageio.v2.imread('./time'+str(i)+'.jpg')
                writer.append_data(image)

                os.remove('./time'+str(i)+'.jpg')
            except:
                pass


if __name__ == "__main__":
    main()