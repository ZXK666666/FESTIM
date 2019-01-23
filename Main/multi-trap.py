from fenics import *
from dolfin import *
import numpy as np
import csv
import sys
import os
import argparse


class Ttrap():

    def save_as(self):
        '''
        - parameters : none
        - returns filedescription : string of the saving path
        '''
        valid = False
        while valid is False:
            print("Save as (.csv):")
            filedesorption = input()
            if filedesorption == '':
                filedesorption = "desorption.csv"
            if filedesorption.endswith('.csv'):
                valid = True
                try:
                    with open(filedesorption, 'r') as f:
                        print('This file already exists.'
                              ' Do you want to replace it ? (y/n)')
                    choice = input()
                    if choice == "n" or choice == "N":
                        valid = False
                    elif choice != "y" and choice != "Y":
                        valid = False
                except:
                    valid = True
            else:
                print("Please enter a file ending with the extension .csv")
                valid = False
        return filedesorption

    def export_TDS(self, filedesorption):
        '''
        - filedesorption : string, the path of the csv file.
        '''
        busy = True
        while busy is True:
            try:
                with open(filedesorption, "w+") as output:
                    busy = False
                    writer = csv.writer(output, lineterminator='\n')
                    writer.writerows(['dTt'])
                    for val in desorption:
                        writer.writerows([val])
            except:
                print("The file " + filedesorption + " is currently busy."
                      "Please close the application then press any key")
                input()
        return

    def calculate_D(self, T, E, D_0):
        '''
        Calculate the diffusion coeff at a certain temperature
        and for a specific material (subdomain)
        Arguments:
        - T : float, temperature
        - E : float, diffusion energy
        - D_0 : float, diffusion pre-exponential factor
        Returns : float, the diffusion coefficient
        '''
        coefficient = D_0 * exp(-E/k_B/T)

        return coefficient

    def update_D(self, mesh, volume_markers, materials, T):
        '''
        Iterates through the mesh and compute the value of D
        Arguments:
        - mesh : the mesh
        - volume_markers : MeshFunction that contains the subdomains
        - T : float, the temperature
        Returns : the Function D
        '''
        D = Function(V0)
        for cell in cells(mesh):
            volume_id = volume_markers[cell]
            found = False
            for material in materials:
                if volume_id == material["id"]:
                    found = True
                    D.vector()[cell.index()] = \
                        self.calculate_D(T, material['E_diff'], material['D_0'])
                    break
            if found is False:
                print('Computing D: Volume ID not found')
        return D

    def update_alpha(self, mesh, volume_markers, materials):
        '''
        Iterates through the mesh and compute the value of D
        Arguments:
        - mesh : the mesh
        - volume_markers : MeshFunction that contains the subdomains
        - materials : list, contains all the materials dictionaries

        Returns : the Function alpha
        '''
        alpha = Function(V0)
        for cell in cells(mesh):
            volume_id = volume_markers[cell]
            found = False
            for material in materials:
                if volume_id == material["id"]:
                    found = True
                    alpha.vector()[cell.index()] = material['alpha']
                    break
            if found is False:
                print('Computing alpha: Volume ID not found')
        return alpha

    def update_beta(self, mesh, volume_markers, materials):
        '''
        Iterates through the mesh and compute the value of D
        Arguments:
        - mesh : the mesh
        - volume_markers : MeshFunction that contains the subdomains
        - materials : list, contains all the materials dictionaries

        Returns : the Function beta
        '''
        beta = Function(V0)
        for cell in cells(mesh):
            volume_id = volume_markers[cell]
            found = False
            for material in materials:
                if volume_id == material["id"]:
                    found = True
                    beta.vector()[cell.index()] = material['beta']
                    break
            if found is False:
                print('Computing beta: Volume ID not found')
        return beta

    def find_energy(self, energies, level):
        #print(energies)
        for energy in energies:
            
            if energy["level"] == level:
                return energy["energy"]
                break
        
        print("Unable to find level " + str(level))
        return

    def formulation(self, traps, solutions, testfunctions, previous_solutions):
        ''' formulation takes traps as argument (list).
        Parameters:
        - traps : dict, contains the energy, density and domains
        of the traps
        - solutions : list, contains the solution fields
        - testfunctions : list, contains the testfunctions
        - previous_solutions : list, contains the previous solution fields

        Returns:
        - F : variational formulation
        '''
        

        transient_sol = ((u_1 - u_n1) / dt)*v_1*dx
        diff_sol = D*dot(grad(u_1), grad(v_1))*dx
        source_sol = - (1-r)*flux_*f*v_1*dx
        
        F = 0
        F += transient_sol + source_sol + diff_sol
        
        i = 1
        for trap in traps:
            material = trap["materials"]
            n_t = trap["density"]
            l = len(trap["energies"])
            for energy in trap["energies"]:
                level = energy["level"]
                energy = energy['energy']
                
                F += ((solutions[level+1] - previous_solutions[level+1]) / dt)*testfunctions[level+1]*dx

                def form(F, subdomain):
                    if level != l:
                        energyup = self.find_energy(trap["energies"], level + 1)
                        F += D/alpha/alpha/beta*u_1*solutions[level+1]*testfunctions[level+1]*dx(subdomain)
                        F += -D/alpha/alpha/beta*u_1*solutions[level]*testfunctions[level+1]*dx(subdomain)
                        F += v_0*exp(-energy/k_B/temp)*solutions[level+1]*testfunctions[level+1]*dx(subdomain)
                        F += - v_0*exp(-energyup/k_B/temp)*solutions[level+2]*testfunctions[level+1]*dx(subdomain)
                    else:
                            trapping_sol = D/alpha/alpha/beta*u_1*(n_t - solutions[level+1])*v_1*dx(subdomain)
                            F += trapping_sol
                            F += -D/alpha/alpha/beta*u_1*solutions[level]*testfunctions[l+1]*dx(subdomain)
                            F += v_0*exp(-energy/k_B/temp)*solutions[level+1]*testfunctions[l+1]*dx(subdomain)
                    
                    F += - v_0*exp(-energy/k_B/temp)*solutions[level+1]*v_1*dx(subdomain)
                    return F
                
                if type(material) is list:
                    for subdomain in material:
                        F = form(F, subdomain)
                else:
                    F = form(F, subdomain)
                    
                F += ((solutions[i] - previous_solutions[i]) / dt)*v_1*dx

            i += 1
        transient_n0 = ((u_2 - u_n2) / dt)*v_2*dx
        sol_to_n0 = D/alpha/alpha/beta*u_1*u_2*v_2*dx
        n1_to_n0 = - v_0*exp(-self.find_energy(trap["energies"], 1)/k_B/temp)*u_2*v_2*dx
        
        F += transient_n0 + sol_to_n0 + n1_to_n0
            
        return F

    def subdomains(self, mesh, materials):
        '''
        Iterates through the mesh and mark them
        based on their position in the domain
        Arguments:
        - mesh : the mesh
        - materials : list, contains the dictionaries of the materials
        Returns :
        - volume_markers : MeshFunction that contains the subdomains
            (0 if no domain was found)
        - measurement : the measurement dx based on volume_markers
        '''
        volume_markers = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
        for cell in cells(mesh):
            for material in materials:
                if cell.midpoint().x() >= material['borders'][0] \
                 and cell.midpoint().x() <= material['borders'][1]:
                    volume_markers[cell] = material['id']

        measurement = dx(subdomain_data=volume_markers)
        return volume_markers, measurement

    def define_traps(self, n_t):
        '''
        Create a dict corresponding to the different traps
        and containing properties.
        Arguments:
        - n_t : Function(W) or double
        Returns:
        -traps : dict corresponding to the different traps
        and containing properties.
        '''
        
        traps = [
            {
                "density": n_t,
                "materials": [1],
                "energies": [
                    {
                        "energy": 1.19,
                        "level": 1
                    },
                    {
                        "energy": 1.17,
                        "level": 2
                    }
                            ],
            }]
        return traps

    def mesh_and_refine(self, mesh_parameters):
        '''
        Mesh and refine iteratively until meeting the refinement
        conditions.
        Arguments:
        - mesh_parameters : dict, contains initial number of cells, size,
        and refinements (number of cells and position)
        Returns:
        - mesh : the refined mesh.
        '''
        print('Meshing ...')
        initial_number_of_cells = mesh_parameters["initial_number_of_cells"]
        size = mesh_parameters["size"]
        mesh = IntervalMesh(initial_number_of_cells, 0, size)
        if "refinements" in mesh_parameters:
            for refinement in mesh_parameters["refinements"]:
                nb_cells_ref = refinement["cells"]
                refinement_point = refinement["x"]
                print("Mesh size before local refinement is " + str(len(mesh.cells())))
                while len(mesh.cells()) < initial_number_of_cells + nb_cells_ref:
                    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
                    cell_markers.set_all(False)
                    for cell in cells(mesh):
                        if cell.midpoint().x() < refinement_point:
                            cell_markers[cell] = True
                    mesh = refine(mesh, cell_markers)
                print("Mesh size after local refinement is " + str(len(mesh.cells())))
                initial_number_of_cells = len(mesh.cells())
        else:
            print('No refinement parameters found')
        return mesh


class myclass(Ttrap):
    def __init__(self):
        ttrap = Ttrap()

        def define_materials():
            '''
            Create a list of dicts corresponding to the different materials
            and containing properties.
            Returns:
            -materials : list of dicts corresponding to the different materials
            and containing properties.
            '''
            materials = []
            material1 = {
                "alpha": Constant(1.1e-10),  # lattice constant ()
                "beta": Constant(6*6.3e28),  # number of solute sites per atom (6 for W)
                "density": 6.3e28,
                "borders": [0, 10e-6],
                "E_diff": 0.39,
                "D_0": 4.1e-7,
                "id": 1
            }
            materials = [material1]
            return materials

        self.__mesh_parameters = {
            "initial_number_of_cells": 50,
            "size": 10e-6,
            "refinements": [
                {
                    "cells": 1000,
                    "x": 10e-6
                }
                ,
                {
                    "cells": 100,
                    "x": 25e-9
                }
            ],
            }
        self.__mesh = ttrap.mesh_and_refine(self.__mesh_parameters)
        self.__materials = define_materials()

    def getMesh(self):
        return self.__mesh

    def getMaterials(self):
        return self.__materials

    def getMeshParameters(self):
        return self.__mesh_parameters
    # Declaration of variables
    implantation_time = 400.0
    resting_time = 50
    ramp = 8
    delta_TDS = 600
    r = 0
    flux = 2.5e19  # /6.3e28
    n_trap_3a_max = 1e-1*Constant(6.3e28)
    n_trap_3b_max = 1e-2*Constant(6.3e28)
    rate_3a = 6e-4
    rate_3b = 2e-4
    xp = 1e-6
    v_0 = 1e13  # frequency factor s-1
    k_B = 8.6e-5  # Boltzmann constant
    TDS_time = int(delta_TDS / ramp) + 1
    Time = implantation_time+resting_time+TDS_time
    num_steps = 10*int(implantation_time+resting_time+TDS_time)
    k = Time / num_steps  # time step size
    dt = Constant(k)
    t = 0  # Initialising time to 0s
    n_t = 3e-5*6.3e28


ttrap = myclass()

implantation_time = ttrap.implantation_time
resting_time = ttrap.resting_time
ramp = ttrap.ramp
delta_TDS = ttrap.delta_TDS
r = ttrap.r
flux = ttrap.flux  # /6.3e28
n_trap_3a_max = ttrap.n_trap_3a_max
n_trap_3b_max = ttrap.n_trap_3b_max
rate_3a = ttrap.rate_3a
rate_3b = ttrap.rate_3b
xp = ttrap.xp
v_0 = ttrap.v_0  # frequency factor s-1
k_B = ttrap.k_B  # Boltzmann constant
TDS_time = ttrap.TDS_time
Time = ttrap.Time
num_steps = ttrap.num_steps
k = ttrap.k # time step size
dt = ttrap.dt
t = ttrap.t  # Initialising time to 0s
n_t = ttrap.n_t

size = ttrap.getMeshParameters()["size"]

# Mesh and refinement
materials = ttrap.getMaterials()
mesh = ttrap.getMesh()

# Define function space for system of concentrations and properties
P1 = FiniteElement('P', interval, 1)
element = MixedElement([P1, P1, P1, P1, P1])
V = FunctionSpace(mesh, element)
W = FunctionSpace(mesh, 'P', 1)
V0 = FunctionSpace(mesh, 'DG', 0)

# Define and mark subdomains
volume_markers, dx = ttrap.subdomains(mesh, materials)

# BCs
print('Defining boundary conditions')


def inside(x, on_boundary):
    return on_boundary and (near(x[0], 0))


def outside(x, on_boundary):
    return on_boundary and (near(x[0], size))
# #Tritium concentration
inside_bc_c = Expression(('0', '0', '0', '0', '0'), t=0, degree=1)
bci_c = DirichletBC(V, inside_bc_c, inside)
bco_c = DirichletBC(V, inside_bc_c, outside)
bcs = [bci_c, bco_c]


# Define test functions
v_1, v_2, v_3, v_4, v_5 = TestFunctions(V)
testfunctions = [v_1, v_2, v_3, v_4, v_5]
v_trap_3 = TestFunction(W)

u = Function(V)
n_trap_3 = TrialFunction(W)  # trap 3 density


# Split system functions to access components
u_1, u_2, u_3, u_4, u_5 = split(u)
solutions = [u_1, u_2, u_3, u_4, u_5]

print('Defining initial values')
ini_u = Expression(("0", "0", "0", "0", "0"), degree=1)
u_n = interpolate(ini_u, V)
u_n1, u_n2, u_n3, u_n4, u_n5 = split(u_n)
previous_solutions = [u_n1, u_n2, u_n3, u_n4, u_n5]

ini_n_trap_3 = Expression("0", degree=1)
n_trap_3_n = interpolate(ini_n_trap_3, W)
n_trap_3_ = Function(W)

# Define expressions used in variational forms
print('Defining source terms')
center = 4.5e-9 #+ 20e-9
width = 2.5e-9
f = Expression('1/(width*pow(2*3.14,0.5))*  \
               exp(-0.5*pow(((x[0]-center)/width), 2))',
               degree=2, center=center, width=width)  # This is the tritium volumetric source term
teta = Expression('(x[0] < xp && x[0] > 0)? 1/xp : 0',
                  xp=xp, degree=1)
flux_ = Expression('t <= implantation_time ? flux : 0',
                   t=0, implantation_time=implantation_time,
                   flux=flux, degree=1)

print('Defining variational problem')
temp = Expression('t <= (implantation_time+resting_time) ? \
                  300 : 300+ramp*(t-(implantation_time+resting_time))',
                  implantation_time=implantation_time,
                  resting_time=resting_time,
                  ramp=ramp,
                  t=0, degree=2)
D = ttrap.update_D(mesh, volume_markers, materials, temp(size/2))
alpha = ttrap.update_alpha(mesh, volume_markers, materials)
beta = ttrap.update_beta(mesh, volume_markers, materials)


# Define variational problem
traps = ttrap.define_traps(n_t)
F = ttrap.formulation(traps, solutions, testfunctions, previous_solutions)

F_n3 = ((n_trap_3 - n_trap_3_n)/dt)*v_trap_3*dx
F_n3 += -(1-r)*flux_*((1 - n_trap_3_n/n_trap_3a_max)*rate_3a*f + (1 - n_trap_3_n/n_trap_3b_max)*rate_3b*teta)*v_trap_3 * dx

# Solution files
xdmf_u_1 = XDMFFile('Solution/c_sol.xdmf')
xdmf_u_2 = XDMFFile('Solution/c_trap1.xdmf')
xdmf_u_3 = XDMFFile('Solution/c_trap2.xdmf')
xdmf_u_4 = XDMFFile('Solution/c_trap3.xdmf')
xdmf_u_5 = XDMFFile('Solution/c_trap4.xdmf')
xdmf_retention = XDMFFile('Solution/retention.xdmf')
filedesorption = ttrap.save_as()

#  Time-stepping
print('Time stepping...')
total_n = 0
desorption = list()

set_log_level(30)  # Set the log level to WARNING
#set_log_level(20) # Set the log level to INFO


for n in range(num_steps):
    # Update current time
    t += k
    temp.t += k
    flux_.t += k
    if t > implantation_time:
        D = ttrap.update_D(mesh, volume_markers, materials, temp(size/2))
    print(str(round(t/Time*100, 2)) + ' %        ' + str(round(t, 1)) + ' s',
          end="\r")
    solve(F == 0, u, bcs,
          solver_parameters={"newton_solver": {"absolute_tolerance": 1e-19}})

    solve(lhs(F_n3) == rhs(F_n3), n_trap_3_, [])
    _u_1, _u_2, _u_3, _u_4, _u_5 = u.split()
    res = [_u_1, _u_2, _u_3, _u_4, _u_5]

    # Save solution to file (.xdmf)
    _u_1.rename("solute", "label")
    _u_2.rename("trap_1", "label")
    _u_3.rename("trap_2", "label")
    _u_4.rename("trap_3", "label")
    _u_5.rename("trap_4", "label")
    xdmf_u_1.write(_u_1, t)
    xdmf_u_2.write(_u_2, t)
    xdmf_u_3.write(_u_3, t)
    xdmf_u_4.write(_u_4, t)
    xdmf_u_5.write(_u_5, t)

    retention = Function(W)
    retention = project(_u_1)
    print(_u_1(size/2))
    print(_u_2(size/2))
    print(_u_3(size/2))
    print(_u_4(size/2))
    i = 1
    total_trap = 0
    for trap in traps:
        
        sol = res[i]
        total_trap += assemble(sol*dx)
        retention = project(retention + res[i], W)
        i += 1
    retention.rename("retention", "label")
    xdmf_retention.write(retention, t)

    total_sol = assemble(_u_1*dx)
    total = total_trap + total_sol
    desorption_rate = [-(total-total_n)/k, temp(size/2), t]
    total_n = total
    if t > implantation_time+resting_time:
        desorption.append(desorption_rate)

    # Update previous solutions
    u_n.assign(u)
    n_trap_3_n.assign(n_trap_3_)

ttrap.export_TDS(filedesorption)
