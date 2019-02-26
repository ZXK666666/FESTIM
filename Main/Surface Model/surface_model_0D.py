from scipy.integrate import odeint
import numpy as np
import csv
import matplotlib.pyplot as plt


def vectorfield(w, t, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
        w :  vector of the state variables:
                  w = [x1,y1,x2,y2]
        t :  time
        p :  vector of the parameters:
                  p = [m1,m2,k1,k2,L1,L2,b1,b2]
    """
    c_surf, cm0 = w


    # Create f = (x1',y1',x2',y2'):
    f = [(1 - Pr)*flux_atom*(1 - c_surf / n_surf) - flux_atom*sigma_exc*c_surf - 2*nu_0d*lambda_des**2*np.exp(-2*E_D/k_B/temp)*c_surf**2 - nu_0sb*np.exp(-E_A/k_B/temp)*c_surf + nu_0bs*lambda_abs*np.exp(-E_R/k_B/temp)*cm0*(1- c_surf / n_surf),
         1/alpha*nu_0sb*np.exp(-E_A/k_B/temp)*c_surf - 1/alpha*nu_0bs*lambda_abs*np.exp(-E_R/k_B/temp)*cm0*(1- c_surf / n_surf) + 1/alpha*(-1.91e-7*np.exp(-0.2/k_B/temp)*grad)]
    return f


def temperature(t):
    if t < 50:
        return 500
    else:
        return 500


# Parameter values

temp = 800
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
grad = 0#-4.4e23
alpha = 110e-12
#teta = c_surf / n_surf
#phi_atom = (1 - Pr)*flux_atom*(1 - teta)
#phi_exc = flux_atom*sigma_exc*c_surf
#phi_desorb = 2*nu_0d*lambda_des**2*np.exp(-2*E_D/k_B/temp)*c_surf**(2)
#phi_sb = nu_0sb*np.exp(-E_A/k_B/temp)*c_surf
#phi_bs = nu_0bs*lambda_abs*np.exp(-E_R/k_B/temp)*cm0*(1-teta)
#phi_diff = -D*grad

p= []

# Initial conditions
# x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
c_surf, cm0 = 0, 0

# ODE solver parameters
abserr = 1.0e12
relerr = 1.0e-15
stoptime = 100
numpoints = 100

# Pack up the parameters and initial conditions:
w0 = [c_surf, cm0]
solution = [w0]
# Define time array
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

i = 0

#Time stepping
while t[i] < stoptime:
    temp = temperature(t[i]) #update temperature
    wsol, infodict = odeint(vectorfield, w0, [t[i], t[i+1]], args=(p,),
              atol=abserr, rtol=relerr, full_output=True)
    w0 = [wsol[1][0], wsol[1][1]] #update previous solution
    solution.append(w0) #update solution field
    i += 1

plt.plot(t, np.array(solution)[:,1], label='cm0')
#plt.plot(t, np.array(solution)[:,0], label='n_surf')
plt.show()
