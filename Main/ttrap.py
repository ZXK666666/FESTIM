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
        F += transient_sol + diff_sol + source_sol 
        i = 1
        for trap in traps:
            #print(i)
            trap_density = trap['density']
            energy = trap['energy']
            material = trap['materials']
            F += ((solutions[i] - previous_solutions[i]) / dt)*testfunctions[i]*dx
            if type(trap['materials']) is list:
                for subdomain in trap['materials']:
                    F += - D/alpha/alpha/beta*u_1*(trap_density - solutions[i])*testfunctions[i]*dx(subdomain)
                    F += v_0*exp(-energy/k_B/temp)*solutions[i]*testfunctions[i]*dx(subdomain)
            else:
                subdomain = trap['materials']
                F += - D/alpha/alpha/beta*u_1*(trap_density - solutions[i])*testfunctions[i]*dx(subdomain)
                F += v_0*exp(-energy/k_B/temp)*solutions[i]*testfunctions[i]*dx(subdomain)
            F += ((solutions[i] - previous_solutions[i]) / dt)*v_1*dx
            i += 1
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

    def define_traps(self, n_trap_3_):
        '''
        Create a list of dicts corresponding to the different traps
        and containing properties.
        Arguments:
        - n_trap_3_ : Function(W), only required if extrinsic trap is
        simulated.
        Returns:
        -materials : list of dicts corresponding to the different traps
        and containing properties.
        '''
        traps = [
            {
                "energy": 0.85,
                "density": 0.5e-4*6.3e28,
                "materials": [2]
            },
            {
                "energy": 1.0,
                "density": 0.5e-4*6.3e28,
                "materials": [2]
            }
            ,
            {
                "energy": 1.65,
                "density": 9e-4*6.3e28,#n_trap_3_,
                "materials": [2]
            }
            ,
            {
                "energy": 1.85,
                "density":28e-4*6.3e28,
                "materials": [2]
            }
            ,
            {
                "energy": 2.06,
                "density": 8e-4*6.3e28,
                "materials": [2]
            }
        ]
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

    def adaptative_timestep(self, converged, nb_it, dt, dt_min, dt_max,
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

    def formulation_surface_model_1D(self, surface_parameters):
        n_surf = surface_parameters['n_surf']
        Pr = surface_parameters['Pr']
        sigma_exc = surface_parameters['sigma_exc']
        nu_0d = surface_parameters['nu_0d']
        nu_0sb = surface_parameters['nu_0sb']
        nu_0bs = surface_parameters['nu_0bs']
        lambda_des = surface_parameters['lambda_des']
        lambda_abs = surface_parameters['lambda_abs']
        E_D = surface_parameters['E_D']
        E_A = surface_parameters['E_A']
        E_R = surface_parameters['E_R']
        D_0 = surface_parameters['D_0']
        E_diff = surface_parameters['E_diff']

        teta = c_surf / n_surf
        phi_atom = (1 - Pr)*flux_atom*(1 - teta)
        phi_exc = flux_atom*sigma_exc*c_surf
        phi_desorb = 2*nu_0d*lambda_des**2*exp(-2*E_D/(k_B*temp))*c_surf**2
        phi_sb = nu_0sb*exp(-E_A/k_B/temp)*c_surf
        phi_bs = nu_0bs*lambda_abs*exp(-E_R/k_B/temp)*cm0*(1-teta)
        phi_diff = -D_0*exp(-E_diff/k_B/temp)*gradient

        F = (cm0 - cm0_n)/dt*v0*dx + (c_surf - c_surf_n)/dt*v_surf*dx

        F += -phi_atom*v_surf*dx   + phi_exc*v_surf*dx + phi_desorb*v_surf*dx + phi_sb*v_surf*dx - phi_bs*v_surf*dx 

        F += 1/alpha(0) * (-phi_sb * v0*dx + phi_bs*v0*dx + phi_diff*v0*dx)
        return F
    
    def compute_gradient_1D(self, solution):
        gradient = -assemble(dot(grad(solution), n)*ds)
        return gradient


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
                "density": 7.5e28,
                "borders": [0, 20e-9],
                "E_diff": 0.39,
                "D_0": 4.1e-7,
                "id": 1
            }
            material2 = {
                "alpha": Constant(1.1e-10),
                "beta": Constant(6*6.3e28),
                "density": 6.3e28,
                "borders": [0, 20e-4],
                "E_diff": 0.2,
                "D_0": 1.91e-7,
                "id": 2
            }
            materials = [material2]
            return materials

        def surface_parameters():
            rho = 6.3e28
            n_TIS = 6*rho
            n_surf = 6.9*rho**(2/3)
            parameters = {
                'n_surf': n_surf,
                'Pr': 0.81,
                'sigma_exc': 1.7e-21,
                'nu_0d': 1e13,
                'nu_0sb': 1e13,
                'nu_0bs': 1e13,
                'lambda_des': 1/(n_surf**(1/2)),
                'lambda_abs': n_surf/n_TIS,
                'E_D': 0.69,
                'E_A': 1.33,
                'E_R': 0.2,
                'D_0': 1.91e-7,
                'E_diff': 0.2
            }
            return parameters
        self.__surface_parameters = surface_parameters()
        self.__mesh_parameters = {
            "initial_number_of_cells": 3000,
            "size": 20e-4,
            "refinements": [
                {
                    "cells": 500,
                    "x": 2e-6
                },
                {
                    "cells": 100,
                    "x": 25e-9
                }
            ],
            }
        self.__mesh = ttrap.mesh_and_refine(self.__mesh_parameters)
        self.__materials = define_materials()

    def getSurfaceParameters(self):
        return self.__surface_parameters

    def getMesh(self):
        return self.__mesh

    def getMaterials(self):
        return self.__materials

    def getMeshParameters(self):
        return self.__mesh_parameters
    # Declaration of variables
    implantation_time = 144*3600
    resting_time = 8*3600#50
    ramp = 0.25
    delta_TDS = 700
    r = 0
    flux =  0#2.5e19  # /6.3e28
    n_trap_3a_max = 1e-1*Constant(6.3e28)
    n_trap_3b_max = 1e-2*Constant(6.3e28)
    rate_3a = 6e-4
    rate_3b = 2e-4
    xp = 1e-6
    v_0 = 1e13  # frequency factor s-1
    k_B = 8.6e-5  # Boltzmann constant
    TDS_time = int(delta_TDS / ramp) #+ 1
    Time = implantation_time+resting_time+TDS_time
    num_steps = 10000*int(implantation_time+resting_time+TDS_time)
    dT = Time / num_steps  # time step size
    t = 0  # Initialising time to 0s
    stepsize_change_ratio = 1.1
    t_stop = implantation_time + int(17*resting_time/18)
    stepsize_stop_max = 0.5
    dt_min = 1e-5
    dt_max = 1300

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
dt = Constant(ttrap.dT) # time step size
dT = ttrap.dT
t = ttrap.t  # Initialising time to 0s
stepsize_change_ratio = ttrap.stepsize_change_ratio
t_stop = ttrap.t_stop
stepsize_stop_max = ttrap.stepsize_stop_max
size = ttrap.getMeshParameters()["size"]
dt_min = ttrap.dt_min
dt_max = ttrap.dt_max

# Mesh and refinement
materials = ttrap.getMaterials()
mesh = ttrap.getMesh()
n = FacetNormal(mesh)
surface_parameters = ttrap.getSurfaceParameters()
# Define function space for system of concentrations and properties
parameters["allow_extrapolation"] = True
P1 = FiniteElement('P', interval, 1)
element = MixedElement([P1, P1, P1, P1, P1, P1])
V = FunctionSpace(mesh, element)
W = FunctionSpace(mesh, 'P', 1)
V0 = FunctionSpace(mesh, 'DG', 0)
gradient = Constant(0.0)

#Vector function space for surface model
DG0 = VectorFunctionSpace(mesh, "DG", 0, dim=2) 
surface_quantities = Function(DG0)
cm0, c_surf = split(surface_quantities)

v0, v_surf = TestFunctions(DG0)
surface_quantities_n = Function(DG0)
cm0_n, c_surf_n = split(surface_quantities_n)

# Define and mark subdomains
volume_markers, dx = ttrap.subdomains(mesh, materials)
print('enlever ça')

# BCs
print('Defining boundary conditions')


def inside(x, on_boundary):
    return on_boundary and (near(x[0], 0))


def outside(x, on_boundary):
    return on_boundary and (near(x[0], size))

# #Tritium concentration
inside_bc_c = Expression(('0', '0', '0', '0', '0', '0'), t=0, degree=1)
bci_c = DirichletBC(V, inside_bc_c, inside)
bco_c = DirichletBC(V, inside_bc_c, outside)
bcs = [bci_c, bco_c]


# Define test functions
v_1, v_2, v_3, v_4, v_5, v_6 = TestFunctions(V)
testfunctions = [v_1, v_2, v_3, v_4, v_5, v_6]
v_trap_3 = TestFunction(W)


u = Function(V)
du = TrialFunction(V)
n_trap_3 = TrialFunction(W)  # trap 3 density


# Split system functions to access components
u_1, u_2, u_3, u_4, u_5, u_6 = split(u)
solutions = [u_1, u_2, u_3, u_4, u_5, u_6]

print('Defining initial values')
ini_u = Expression(("0", "0", "0", "0", "0", "0"), degree=1)
u_n = interpolate(ini_u, V)
u_n1, u_n2, u_n3, u_n4, u_n5, u_n6 = split(u_n)
previous_solutions = [u_n1, u_n2, u_n3, u_n4, u_n5, u_n6]

ini_n_trap_3 = Expression("0", degree=1)
n_trap_3_n = interpolate(ini_n_trap_3, W)
n_trap_3_ = Function(W)

# Define expressions used in variational forms
print('Defining source terms')
center = 4.5e-9 #+ 20e-9
width = 2.5e-9
f = Expression('1/(width*pow(2*3.14,0.5))*  \
               exp(-0.5*pow(((x[0]-center)/width), 2))',
               degree=2, center=center, width=width)
teta = Expression('(x[0] < xp && x[0] > 0)? 1/xp : 0',
                  xp=xp, degree=1)
flux_ = Expression('t <= implantation_time ? flux : 0',
                   t=0, implantation_time=implantation_time,
                   flux=flux, degree=1)

print('Defining variational problem')
temp = Expression('t <= implantation_time ? \
                  500 : (t <= implantation_time + resting_time ? 300 : 300+ramp*(t-(implantation_time+resting_time)))',
                  implantation_time=implantation_time,
                  resting_time=resting_time,
                  ramp=ramp,
                  t=0, degree=2)

flux_atom = Expression('t < implantation_time? 2.6e19 : 0', t=t, implantation_time=implantation_time, degree=0)
D = ttrap.update_D(mesh, volume_markers, materials, temp(size/2))
alpha = ttrap.update_alpha(mesh, volume_markers, materials)
beta = ttrap.update_beta(mesh, volume_markers, materials)


# Define variational problem
traps = ttrap.define_traps(n_trap_3_)
F = ttrap.formulation(traps, solutions, testfunctions, previous_solutions)
F_surface = ttrap.formulation_surface_model_1D(surface_parameters)

F_n3 = ((n_trap_3 - n_trap_3_n)/dt)*v_trap_3*dx
F_n3 += -(1-r)*flux_*((1 - n_trap_3_n/n_trap_3a_max)*rate_3a*f + (1 - n_trap_3_n/n_trap_3b_max)*rate_3b*teta)*v_trap_3 * dx

# Solution files
xdmf_u_1 = XDMFFile('Solution/c_sol.xdmf')
xdmf_u_2 = XDMFFile('Solution/c_trap1.xdmf')
xdmf_u_3 = XDMFFile('Solution/c_trap2.xdmf')
xdmf_u_4 = XDMFFile('Solution/c_trap3.xdmf')
xdmf_u_5 = XDMFFile('Solution/c_trap4.xdmf')
xdmf_u_6 = XDMFFile('Solution/c_trap5.xdmf')
xdmf_retention = XDMFFile('Solution/retention.xdmf')
filedesorption = ttrap.save_as()

#  Time-stepping
print('Time stepping...')
total_n = 0
desorption = list()

set_log_level(30)  # Set the log level to WARNING
#set_log_level(20) # Set the log level to INFO

timer = Timer()  # start timer
while t < Time:
    
    print(str(round(t/Time*100, 2)) + ' %        ' + str(round(t, 1)) + ' s' +
          "    Ellapsed time so far: %s s" % round(timer.elapsed()[0], 1),
          end="\r")
    
    solve(F_surface == 0, surface_quantities, [])
    cm0, c_surf = surface_quantities.split()
    surface_model_activated = True
    if surface_model_activated is True:
        bcs = [DirichletBC(V, Expression(('cm0', '0', '0', '0', '0', '0'), \
               cm0=cm0(0.5), degree=1), inside), bco_c]
        gradient.assign(ttrap.compute_gradient_1D(u_n[0])) #TODO find a way to calculate gradient of cm(x=0)
        #print([t, float(dt), cm0(0.5), c_surf(0.5)])
    
    J = derivative(F, u, du)  # Define the Jacobian
    problem = NonlinearVariationalProblem(F, u, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["error_on_nonconvergence"] = False
    nb_it, converged = solver.solve()
    dt = ttrap.adaptative_timestep(converged=converged, nb_it=nb_it, dt=dt,
                                   stepsize_change_ratio=stepsize_change_ratio,
                                   dt_min=dt_min, dt_max=dt_max, t=t, t_stop=t_stop,
                                   stepsize_stop_max=stepsize_stop_max)

    solve(lhs(F_n3) == rhs(F_n3), n_trap_3_, [])
    _u_1, _u_2, _u_3, _u_4, _u_5, _u_6 = u.split()
    res = [_u_1, _u_2, _u_3, _u_4, _u_5, _u_6]
    # Save solution to file (.xdmf)
    _u_1.rename("solute", "label")
    _u_2.rename("trap_1", "label")
    _u_3.rename("trap_2", "label")
    _u_4.rename("trap_3", "label")
    _u_5.rename("trap_4", "label")
    _u_6.rename("trap_5", "label")
    xdmf_u_1.write(_u_1, t)
    xdmf_u_2.write(_u_2, t)
    xdmf_u_3.write(_u_3, t)
    xdmf_u_4.write(_u_4, t)
    xdmf_u_5.write(_u_5, t)
    xdmf_u_6.write(_u_6, t)
    retention = Function(W)
    retention = project(_u_1)
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
    desorption_rate = [-(total-total_n)/float(dt), temp(size/2), t]
    total_n = total
    if t > implantation_time+resting_time:
        desorption.append(desorption_rate)
    # Update previous solutions
    u_n.assign(u)
    n_trap_3_n.assign(n_trap_3_)
    surface_quantities_n.assign(surface_quantities)
    # Update current time
    t += float(dt)
    temp.t += float(dt)
    flux_.t += float(dt)
    flux_atom.t += float(dt)
    if t > implantation_time:
        D = ttrap.update_D(mesh, volume_markers, materials, temp(size/2))

ttrap.export_TDS(filedesorption)
print('\007s')