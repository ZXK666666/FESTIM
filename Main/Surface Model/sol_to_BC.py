from fenics import *
from dolfin import *
import numpy as np
import csv
import sys
import os
import argparse

numstep = 50
Time = 100
dt = Time/numstep
k = 0.0
t = Constant(k)
mesh = UnitIntervalMesh(5)
mesh = UnitSquareMesh(15, 15)
#mesh = UnitCubeMesh(15, 30, 30)
n0 = FacetNormal(mesh)

print((mesh.num_cells()))
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0)


class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)


class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0)

#submesh = SubMesh(mesh, Bottom())
V = FunctionSpace(mesh, "CG", 1)
boundarymesh = BoundaryMesh(mesh, 'exterior')
submesh = SubMesh(boundarymesh, Left())
parameters['allow_extrapolation'] = True


W = VectorFunctionSpace(submesh, "CG", 1, dim=2)

c_m = Function(V)
v = TestFunction(V)
c_m_n = Function(V)

F = (c_m - c_m_n)/dt*v*dx+ 0.1*dot(grad(c_m), grad(v))*dx

u = Function(W)
cm0, c_surf = split(u)

v0, v_surf = TestFunctions(W)
u_n = Function(W)
cm0_n, c_surf_n = split(u_n)


temp = 300
rho = 6.3e28
n_surf = 6.9*rho**(2/3)
print(n_surf)
n_TIS = 6*rho
Pr = 0.81
phi_atom = Expression('5.8e19*1/(width*pow(2*3.14,0.5))*  \
               exp(-0.5*(pow(((x[1]-center)/width), 2)+pow(((x[2]-center)/width), 2)))',
               degree=2, center=0.5, width=0.15) #5.8e19
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

n02 = FacetNormal(submesh)
print(n0)
A = dot(grad(c_m_n), n02)
A = grad(c_m_n)
a = interpolate(A, FunctionSpace(mesh, "DG", 0))

gradient = project(dot(grad(c_m_n),n02), FunctionSpace(boundarymesh, "CG", 1)) #dot(grad(c_m_n), n0)
print(gradient(0, 0.5))

alpha = 110e-12
D = 1e-17
teta = c_surf / n_surf
phi_atom = (1 - Pr)*phi_atom*(1 - teta)
phi_exc = phi_atom*sigma_exc*c_surf
phi_desorb = 2*nu_0d*lambda_des**2*np.exp(-2*E_D/k_B/temp)*c_surf**2
phi_sb = nu_0sb*np.exp(-E_A/k_B/temp)*c_surf
phi_bs = nu_0bs*lambda_abs*np.exp(-E_R/k_B/temp)*cm0*(1-teta)
phi_diff = -D*gradient

F2 = (cm0 - cm0_n)/dt*v0*dx + (c_surf - c_surf_n)/dt*v_surf*dx

F2 += -phi_atom*v_surf*dx + phi_exc*v_surf*dx + phi_desorb*v_surf*dx + phi_sb*v_surf*dx - phi_bs*v_surf*dx

F2 += 1/alpha * (-phi_sb * v0*dx + phi_bs*v0*dx + phi_diff*v0*dx)


set_log_level(30)

fileu = File("Solution/u.pvd")
fileu2 = File("Solution/c_m.pvd")

for i in range(numstep):
    print(str(round(k/Time*100, 2)) + ' %        ' + str(round(t, 1)) + ' s',
      end="\r")
    solve(F2 == 0, u, [])
    cm0, c_surf = u.split()
    bcs = [DirichletBC(V, cm0, Left())]
    #parameters['allow_extrapolation'] = False
    u_n.assign(u)
    solve(F == 0, c_m, bcs)
    c_m_n.assign(c_m)
    k += dt
    t.assign(k)
    fileu << (u, k)
    fileu2 << (c_m, k)