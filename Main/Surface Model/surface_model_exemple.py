from fenics import *
from dolfin import *
import numpy as np
import csv
import sys
import os
import argparse
import scipy

def adaptative_timestep(converged, nb_it, dt, dt_min, dt_max,
                        stepsize_change_ratio, t, t_stop,
                        stepsize_stop_max):
    '''
    Adapts the stepsize as function of the number of iterations of the
    solver.
    Arguments:
    - converged : bool, determines if the time step has converged.
    - nb_it : int, number of iterations
    - dt : Constant(), fenics object
    - dt_min : float, stepsize minimum value
    - stepsize_change_ration : float, stepsize change ratio
    - t : float, time
    - t_stop : float, time where adaptative time step stops
    - stepsize_stop_max : float, maximum stepsize after stop
    Returns:
    - dt : Constant(), fenics object
    '''
    while converged is False:
        dt.assign(float(dt)/stepsize_change_ratio)
        #print(float(dt))
        nb_it, converged = solver.solve()
        if float(dt) < dt_min:
            sys.exit('Error: stepsize reached minimal value')
    if t > t_stop:
        if float(dt) > stepsize_stop_max:
            dt.assign(stepsize_stop_max)
    else:
        if nb_it < 5:
            dt.assign(float(dt)*stepsize_change_ratio)
        else:
            dt.assign(float(dt)/stepsize_change_ratio)
    
    if float(dt) > dt_max:
        dt.assign(dt_max)
    return dt
Time = 144*3600
numstep = 10000000000003*Time
k = Time/numstep
dt = Constant(k)
t = 0.0
mesh = UnitIntervalMesh(1)
temp = 500
#parameters["allow_extrapolation"] = True


V = VectorFunctionSpace(mesh, "DG", 0, dim=2)

u = Function(V)
du = TrialFunction(V)
cm0, c_surf = split(u)

v0, v_surf = TestFunctions(V)
u_n = Function(V)

#u_n = interpolate(Expression(('36805096239', '5.5243559532e19'), degree=1), V)
cm0_n, c_surf_n = split(u_n)
rho = 6.3e28
n_surf = 6.9*rho**(2/3)
n_TIS = 6*rho
Pr = 0.81
flux_atom = 2.6e19 #Expression("t<50?2.6e19 : (t<98?2.6e19*(1-100/t):1e-2)", t=k, degree=0)
sigma_exc = 1.7e-21
k_B = 8.6e-5  # Boltzmann constant
nu_0d = 1e13
nu_0sb = 1e13
nu_0bs = 1e13
lambda_des = 1/(n_surf**(1/2))
lambda_abs = n_surf/n_TIS
E_D = 0.69
E_A = 1.33
E_R = 0.2
grad = Expression("0", t=k, degree=0)#-4.4e23
alpha = 110e-12
D = 1.91e-7*np.exp(-0.2/k_B/temp)
teta = c_surf / n_surf
phi_atom = (1 - Pr)*flux_atom*(1 - teta)
phi_exc = flux_atom*sigma_exc*c_surf
phi_desorb = 2*nu_0d*lambda_des**2*np.exp(-2*E_D/k_B/temp)*c_surf**(2)
phi_sb = nu_0sb*np.exp(-E_A/k_B/temp)*c_surf
phi_bs = nu_0bs*lambda_abs*np.exp(-E_R/k_B/temp)*cm0*(1-teta)
phi_diff = -D*grad

F = (cm0 - cm0_n)/dt*v0*dx + alpha*(c_surf - c_surf_n)/dt*v_surf*dx

F += -phi_atom*v_surf*dx + phi_exc*v_surf*dx + phi_desorb*v_surf*dx + phi_sb*v_surf*dx - phi_bs*v_surf*dx 

F += -phi_sb * v0*dx + phi_bs*v0*dx + phi_diff*v0*dx


set_log_level(30)
print(n_surf)
print(k, u(0.5))

#parameters['allow_extrapolation'] = False
while t < Time:

    J = derivative(F, u, du)  # Define the Jacobian

    problem = NonlinearVariationalProblem(F, u, [], J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
    #solver.parameters["newton_solver"]['maximum_iterations']=500
    solver.parameters["newton_solver"]['relative_tolerance'] = 1e-8
    nb_it, converged = solver.solve()
    dt = adaptative_timestep(converged=converged, nb_it=nb_it, dt=dt, dt_min=1e-5, dt_max=20,
                        stepsize_change_ratio=1.1, t=t, t_stop=1000000,
                        stepsize_stop_max=0.5)
    t += float(dt)
    cm0_n, c_surf_n = u_n.split()
    cm0, c_surf = u.split()
    print(t, float(dt), cm0(0.5), c_surf(0.5))
    u_n.assign(u)
    grad.t += dt
    #flux_atom.t += dt
# 36808025644.3 5.52457329233e+19
# 36807284981.5 5.52451834254e+19