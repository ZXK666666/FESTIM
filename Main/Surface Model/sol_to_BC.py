from fenics import *
from dolfin import *
import numpy as np
import csv
import sys
import os
import argparse

numstep = 100
Time = 100
dt = Time/numstep
k = 0.0
t = Constant(k)
mesh = UnitIntervalMesh(5)
mesh = UnitSquareMesh(15, 15)
mesh = UnitCubeMesh(15, 15, 15)

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
intmesh = UnitIntervalMesh(20)
W = FunctionSpace(submesh, "CG", 1)
u = TrialFunction(V)
u2 = TrialFunction(W)
v = TestFunction(V)
v2 = TestFunction(W)
u_n = Function(V)
u2_n = Function(W)

F = 0.01*dot(grad(u), grad(v))*dx

F2 = 0.001*dot(grad(u2), grad(v2))*dx # - Expression("2+sin(x[0]-0.5)", degree=2)*v2*dx
bc2 = [DirichletBC(W, 1.0, Bottom()), DirichletBC(W, 0.0, Top())]

u = Function(V)
u2 = Function(W)
fileu = File("Solution/u.pvd")
fileu2 = File("Solution/u2.pvd")

#parameters["allow_extrapolation"] = True
parameters['allow_extrapolation'] = False
solve(lhs(F2) == rhs(F2), u2, bc2)
bcs = [DirichletBC(V, u2, Left()), DirichletBC(V, 1.5, Right())]

solve(lhs(F) == rhs(F), u, bcs)

fileu << u
fileu2 << u2