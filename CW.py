import time
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
D_0 = 101.9188
alpha = 2.567
k_theta = 328.645606
k_rtheta = -211.4672
k_rr = 111.70765
r_OHeq = 1
r_HHeq = 1.633
e_H = 0.41
e_O = -0.82
sigma_OO = 3.166
epsilon_OO = 0.1554
m = [15.999, 1.008]


"""Initialising sympy variables"""

x_O, x_H1, x_H2, y_O, y_H1, y_H2, z_O, z_H1, z_H2 = \
sym.symbols('x_O, x_H1, x_H2, y_O, y_H1, y_H2, z_O, z_H1, z_H2')
    
r_OH1 = sym.sqrt((x_O - x_H1)**2 + (y_O - y_H1)**2 + (z_O - z_H1)**2)
r_OH2 = sym.sqrt((x_O - x_H2)**2 + (y_O - y_H2)**2 + (z_O - z_H2)**2)
r_HH = sym.sqrt((x_H1 - x_H2)**2 + (y_H1 - y_H2)**2 + (z_H1 - z_H2)**2)
    
dr_OH1 = r_OH1 - r_OHeq
dr_OH2 = r_OH2 - r_OHeq
dr_HH = r_HH - r_HHeq
    
V_int = D_0 * ((1 - sym.exp(alpha * dr_OH1))**2 + (1 - sym.exp(alpha * dr_OH2))**2)\
+ 0.5 * k_theta * dr_HH**2 + k_rtheta * dr_HH * (dr_OH1 + dr_OH2)**2 + k_rr * dr_OH1 * dr_OH2


"""
These three are sympy equations, evaluating internal forces upon atoms.
"""

force_O = (-V_int.diff(x_O), -V_int.diff(y_O), -V_int.diff(z_O))
force_H1 = (-V_int.diff(x_H1), -V_int.diff(y_H1), -V_int.diff(z_H1))
force_H2 = (-V_int.diff(x_H2), -V_int.diff(y_H2), -V_int.diff(z_H2))

"""
These three are numpy equations of the above sympy equations.
"""

F_O = sym.lambdify((x_O, y_O, z_O, x_H1, y_H1, z_H1, x_H2, y_H2, z_H2), force_O, "numpy")
F_H1 = sym.lambdify((x_O, y_O, z_O, x_H1, y_H1, z_H1, x_H2, y_H2, z_H2), force_H1, "numpy")
F_H2 = sym.lambdify((x_O, y_O, z_O, x_H1, y_H1, z_H1, x_H2, y_H2, z_H2), force_H2, "numpy")


def scalar_dist(X):
    """Returns a scalar value for distance between points."""
    return(np.sqrt(X[:,:,0]**2 + X[:,:,1]**2 + X[:,:,2]**2))


"""
These three functions utilise the above numpy equations.
"""

def Force_O(x_O, y_O, z_O, x_H1, y_H1, z_H1, x_H2, y_H2, z_H2):
    """a"""
    return(np.array(F_O(x_O, y_O, z_O, x_H1, y_H1, z_H1, x_H2, y_H2, z_H2)))
    
def Force_H1(x_O, y_O, z_O, x_H1, y_H1, z_H1, x_H2, y_H2, z_H2):
    return(np.array(F_H1(x_O, y_O, z_O, x_H1, y_H1, z_H1, x_H2, y_H2, z_H2)))
    
def Force_H2(x_O, y_O, z_O, x_H1, y_H1, z_H1, x_H2, y_H2, z_H2):
    return(np.array(F_H2(x_O, y_O, z_O, x_H1, y_H1, z_H1, x_H2, y_H2, z_H2)))


"""
The 'accel_ext' function is self contained (calls no other functions).
"""

def accel_ext(r, m, R_c, L):
    F_ext = np.zeros_like(r)
    for i in range(len(r)):
        for j in range(3):
            s = -(r - r[i,j,:])
            u = scalar_dist(s)
            
            """Periodic boundary condition, applied
            when particles are more than L / 2 apart."""
            #print(s[s>(L/2)])
            s[s > (L/2)] -= L
            s[s < -(L/2)] += L
            #print(s[s>(L/2)])
            u = scalar_dist(s)
            
            """Periodic boundary application complete."""
            
            for k in range(len(s)):
                for l in range(3):
                    if u[k,l] < R_c:
                        if  k != i:
                            if j == 0:
                                if l == 0:
                                    V_extOO = -(e_O**2 / (4 * np.pi * u[k,l])) -\
                                    (24 * sigma_OO * epsilon_OO) *\
                                    (2 * (sigma_OO / u[k,l])**14 - (sigma_OO / u[k,l])**8)
                                    F_ext[i,j] = F_ext[i,j] - (V_extOO * s[k,l,:])
                                else:
                                    V_extOH = -(e_H * e_O / (4 * np.pi * u[k,l]))
                                    F_ext[i,j] = F_ext[i,j] - (V_extOH * s[k,l,:])
                            else:
                                if l == 0:
                                    V_extOH = -(e_H * e_O / (4 * np.pi * u[k,l]))
                                    F_ext[i,j] = F_ext[i,j] - (V_extOH * s[k,l,:])
                                else:
                                    V_extHH = -(e_H**2 / (4 * np.pi * u[k,l]))
                                    F_ext[i,j] = F_ext[i,j] - (V_extHH * s[k,l,:])
    a_ext = np.zeros_like(F_ext)
    a_ext[:,0,:] = F_ext[:,0,:] / m[0]
    a_ext[:,1,:] = F_ext[:,1,:] / m[1]
    a_ext[:,2,:] = F_ext[:,2,:] / m[1]
    return(a_ext)


"""
The 'accel' function utilises the above three functions (as well as 'accel_ext')
to return internal accelerations (and external).
"""

def accel(x, m, R_c, L):
    a = np.zeros_like(x)
    x_copy = np.copy(x)
    for i in range(len(x)):
        a[i,0,:] = Force_O(x[i,0,0], x[i,0,1], x[i,0,2], x[i,1,0],\
                            x[i,1,1], x[i,1,2], x[i,2,0], x[i,2,1], x[i,2,2]) / m[0]
        a[i,1,:] = Force_H1(x[i,0,0], x[i,0,1], x[i,0,2], x[i,1,0],\
                            x[i,1,1], x[i,1,2], x[i,2,0], x[i,2,1], x[i,2,2]) / m[1]
        a[i,2,:] = Force_H2(x[i,0,0], x[i,0,1], x[i,0,2], x[i,1,0],\
                            x[i,1,1], x[i,1,2], x[i,2,0], x[i,2,1], x[i,2,2]) / m[1]
    a = a + accel_ext(x,m,R_c,L)
    return(a)


"""
The 'verlet' function evolves the position, velocity and acceleration
over a single timestep. It utilises the 'accel' function.
"""

def verlet(x, v, a, m, f, dt, R_c, L):
    x_new = x + (dt * v) + (0.5 * a * dt ** 2)# x^(n+1) step.
    x_new[x_new > L] -= L
    x_new[x_new < 0] += L
    v_star = v + (0.5 * a * dt)# v^(n+1/2)step.
    a_new = f(x_new,m,R_c,L)# a^(n+1) step.
    v_new = v_star + (0.5 * dt * a_new)# v^(n+1) step.
    return(x_new, v_new, a_new)


"""
Main function call.
"""

def Molecular(L = 7, R_c = 5, dt = 0.0001, T = 0.05):
    """
    Description of function.
    
    Parameters
    ----------
    
    x : description of x
        make sure it's narrowly
        formatted
        
    y : same again with Important
        Terms capitalised
    
    
    Returns
    -------
    
    x_new : description
    
    y_new : description
    """
    
    """Check input data."""
    
    try:
        L = float(L)
        R_c = float(R_c)
        dt = float(dt)
        T = float(T)
    except:
        raise Exception('All inputs must be of type int, float or sting of numeric characters.')
    
    
    
    time_start = time.clock()#Variable defining start time of function.
    
    D_0 = 101.9188
    alpha = 2.567
    k_theta = 328.645606
    k_rtheta = -211.4672
    k_rr = 111.70765
    r_OHeq = 1
    r_HHeq = 1.633
    e_H = 0.41
    e_O = -0.82
    sigma_OO = 3.166
    epsilon_OO = 0.1554
    m = [15.999, 1.008]#Defines an array containing the mass values.
    
    plot0 = []#Initialises empty lists which will contain plotting data.
    plot1=[]
    plot2=[]
    plot3=[]
    plot4=[]
    plot5=[]
    plot6=[]
    plot7=[]
    
    nsteps = int(T / dt)
    r = np.array([[[1.75,1.75,1.75],[2.55, 2.35, 1.75],[0.95, 1.15, 1.75]],\
                  [[5.25,1.75,1.75],[6.05, 2.35, 1.75],[4.45, 1.15, 1.75]],\
                 [[1.75,5.25,1.75],[2.55,5.85,1.75],[0.95,4.65,1.75]],\
                 [[1.75,1.75,5.25],[2.55,2.35,5.25],[0.95,1.15,5.25]],\
                 [[5.25,5.25,1.75],[6.05,5.85,1.75],[4.45,4.65,1.75]],\
                 [[5.25,1.75,5.25],[6.05,2.35,5.25],[4.45,1.15,5.25]],\
                 [[1.75,5.25,5.25],[2.55,5.85,5.25],[0.95,4.65,5.25]],\
                  [[5.25,5.25,5.25],[6.05,5.85,5.25],[4.45,4.65,5.25]]])
    assert r.shape == (8,3,3), 'position input shape error'
    v = np.zeros_like(r)
    assert v.shape == (8,3,3), 'velocity  shape error'
    a = accel(r, m, R_c, L)
    assert a.shape == (8,3,3), 'acceleration shape error'
    
    
    """
    Preamble done.
    """
    
    time_before = time.clock() - time_start
    print("Everything up to main loop call takes ", time_before)
    
    for step in range(nsteps):
        
        plot0.append(r[0,0,0])
        plot1.append(r[1,0,0])
        plot2.append(r[2,0,0])
        plot3.append(r[3,0,0])
        plot4.append(r[4,0,0])
        plot5.append(r[5,0,0])
        plot6.append(r[6,0,0])
        plot7.append(r[7,0,0])
        #H2_z_plot.append(r[5,2,2])
        
        r,v,a = verlet(r,v,a,m,accel,dt,R_c,L)
        #print("Oxygen", r[0,0,:], "\n")
        #print("Hydrogen 1 ",r[0,1,:],"\n")
        #print("Hydrogen 2 ",r[0,2,:],"\n")
        
    
    """
    Plotting stuff.
    """
    
    t_for_graph = np.linspace(0, T, nsteps)
    
    plt.subplot(8,1,1)
    plt.plot(t_for_graph, plot0)
    plt.ylim(0,7)
    #plt.xlim(3.6,3.9)
    
    plt.subplot(8,1,2)
    plt.plot(t_for_graph, plot1)
    plt.ylim(0,7)
    #plt.xlim(2.2,2.5)
    
    plt.subplot(8,1,3)
    plt.plot(t_for_graph, plot2)
    plt.ylim(0,7)
    #plt.xlim(2.2,2.5)
    
    plt.subplot(8,1,4)
    plt.plot(t_for_graph, plot3)
    plt.ylim(0,7)
    #plt.xlim(2.2,2.5)

    plt.subplot(8,1,5)
    plt.plot(t_for_graph, plot4)
    plt.ylim(0,7)
    #plt.xlim(2.2,2.5)
    
    plt.subplot(8,1,6)
    plt.plot(t_for_graph, plot5)
    plt.ylim(0,7)
    #plt.xlim(2.2,2.5)
    
    plt.subplot(8,1,7)
    plt.plot(t_for_graph,plot6)
    plt.ylim(0,7)
    #plt.xlim(2.2,2.5)
    
    plt.subplot(8,1,8)
    plt.plot(t_for_graph, plot7)
    plt.ylim(0,7)
    #plt.xlim(2.2,2.5)

    
    
    
    plt.show()
    
    """
    #Axes3D.plot(np.array(x_for_graph),np.array(y_for_graph),np.array(z_for_graph))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(O_x_plot, O_y_plot, O_z_plot)
    ax.scatter(H1_x_plot, H1_y_plot, H1_z_plot)
    ax.scatter(H2_x_plot, H2_y_plot, H2_z_plot)
    #plt.xlim(0,7)
    #plt.ylim(0,7)
    #ax.set_zlim(0,7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    """
    time_end = time.clock() - time_start
    print("Script completed in ", time_end)
    
    return(r, v, a)