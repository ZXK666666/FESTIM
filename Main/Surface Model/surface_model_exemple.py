from fenics import *
from dolfin import *
import numpy as np
import csv
import sys
import os
import argparse

numstep = 10000
Time = 100
dt = Time/numstep
k = 0.0
t = Constant(k)
mesh = UnitIntervalMesh(1)
temp = 300
parameters["allow_extrapolation"] = True

DG0 = FiniteElement('DG', interval, 0)
element = MixedElement([DG0, DG0])
V = FunctionSpace(mesh, element)

u = Function(V)
cm0, c_surf = split(u)

v0, v_surf = TestFunctions(V)
u_n = Function(V)
cm0_n, c_surf_n = split(u_n)

rho = 6.3e28
n_surf = 6.9*rho**(2/3)
print(n_surf)
n_TIS = 6*rho
Pr = 0.81
phi_atom =  5.8e19
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
grad = -1e17
alpha = 110e-12
D = 1e-17
teta = c_surf / n_surf
phi_atom = (1 - Pr)*phi_atom*(1 - teta)
phi_exc = phi_atom*sigma_exc*c_surf
phi_desorb = 2*nu_0d*lambda_des**2*np.exp(-2*E_D/k_B/temp)*c_surf**2
phi_sb = nu_0sb*np.exp(-E_A/k_B/temp)*c_surf
phi_bs = nu_0bs*lambda_abs*np.exp(-E_R/k_B/temp)*cm0*(1-teta)
phi_diff = -D*grad

F = (cm0 - cm0_n)/dt*v0*dx + (c_surf - c_surf_n)/dt*v_surf*dx

F += -phi_atom*v_surf*dx + phi_exc*v_surf*dx + phi_desorb*v_surf*dx + phi_sb*v_surf*dx - phi_bs*v_surf*dx

F += 1/alpha * (-phi_sb * v0*dx + phi_bs*v0*dx + phi_diff*v0*dx)


set_log_level(30)


parameters['allow_extrapolation'] = False
for i in range(numstep):
    
    solve(F == 0, u, [])
    u_n.assign(u)
    k += dt
    t.assign(k)
    print(u(0.5))
