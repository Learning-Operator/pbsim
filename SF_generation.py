<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import os
import cupy as cp

G = 6.67430e-11   # m^3 kg^-1 s^-2
c = 299792458     # m/s
h_bar = 1.054571917e-34  # Joule*s

def scale_back_initial(initial_hubble, 
                       a_target, 
                       a_ref, 
                       Omega_r_ref, 
                       Omega_m_ref, 
                       Omega_de_ref):
    
    """
    Scales density parameters from their reference values at scale factor a_ref to a new scale factor a.
    
    Parameters:
        initial_hubble : float
            Hubble parameter at the reference scale factor a_ref.
        a            : float
            The desired scale factor at which to compute the densities.
        a_ref        : float, optional
            The reference scale factor for the input density parameters (default is 1).
        Omega_r_ref  : float, optional
            Radiation density parameter at a_ref (e.g., ~8.24e-5).
        Omega_m_ref  : float, optional
            Matter density parameter at a_ref (e.g., ~0.27).
        Omega_de_ref : float, optional
            Dark energy density parameter at a_ref (e.g., ~0.73).
        
    Returns:
        tuple:
            (rad_scaled, matter_scaled, DE_scaled, SFdot_0)
            where rad_scaled, matter_scaled, and DE_scaled are the energy densities at scale factor a,
            and SFdot_0 is the corresponding scale factor derivative computed via H(a)*a.
    """
    # Compute the critical density at the reference scale factor.
    rho_crit_ref = (3 * initial_hubble**2) / (8 *cp.pi * G)
    
    # Compute the actual densities at the reference scale factor.
    rho_r_ref = Omega_r_ref * rho_crit_ref
    rho_m_ref = Omega_m_ref * rho_crit_ref
    rho_de_ref = Omega_de_ref * rho_crit_ref  # dark energy is constant
    
    # Scale the densities to the new scale factor 'a'
    rad_scaled   = rho_r_ref * (a_ref / a_target)**4
    matter_scaled = rho_m_ref * (a_ref / a_target)**3
    DE_scaled = rho_de_ref  # no scaling for dark energy
    
    # Compute the Hubble parameter at scale factor a.
    H_a = cp.sqrt((8 * cp.pi * G / 3) * (rad_scaled + matter_scaled + DE_scaled)) 
    
    # Compute the derivative of the scale factor.
    SFdot_0 = H_a * a_target
    
    return rad_scaled, matter_scaled, DE_scaled, SFdot_0

        
        
        
def Expansion(Time,
              dt,
              Background_pars, 
              a_target,
              plot = False):

    G = 6.67430e-11   # m^3 kg^-1 s^-2
    c = 299792458     # m/s
    h_bar = 1.054571917e-34  # Joule*s  
        ####################################################################################################
        #  RUN TESTS ON THE ACCURACY OF THIS CODE
        #             - analytical Comparison
        #             - Convergence test
        #             - power-law test
        #             - Make sure energy is conserved
        ####################################################################################################       
        
    """
        Numerically integrates the Friedmann equation expressed in terms of the normalized scale factor 
        x = a(t)/a_i via the 4th-order Runge-Kutta method.
        
        The differential equation is:
        
            dx/dt = H_i * x * sqrt( (initial_Rad_par)/x^4 + (initial_mass_par)/x^3 + initial_DE_par )
        
        where the initial Hubble parameter is computed as:
        
            H_i = sqrt((8*pi*G/3) * (initial_Rad_par + initial_mass_par + initial_DE_par))
        
        After integrating for x(t), the full scale factor is given by:
        
            a(t) = a_i * x(t)
        
        If Initial_SF is None, we assume a_i = 1.
        
        Returns:
            - x_array: Array of normalized scale factor values.
            - t_array: Array of time values.
            - a_array: Array of full scale factor values.
    """
    
    data_amt = int(Time / dt) # to make sure that the this doesnt run to 7.5 steps
    
        
    #print(f"amt of data steps : {self.data_amt}")
    time = cp.linspace(0, Time, data_amt, dtype=cp.float64)
        
    Matter_density = cp.zeros(data_amt, dtype=cp.float64)
    Rad_density = cp.zeros(data_amt, dtype=cp.float64)
    DE_density = cp.zeros(data_amt, dtype=cp.float64)
                
    Scale_factor_dots = cp.zeros(data_amt, dtype=cp.float64)
    Scale_factors = cp.zeros(data_amt, dtype=cp.float64)
    Hubbles = cp.zeros(data_amt, dtype=cp.float64)
    
    Pars_0 = float(Background_pars[0])
    Pars_1 = float(Background_pars[1])
    Pars_2 = float(Background_pars[2])
    Pars_3 = float(Background_pars[3])
    Pars_4 = float(Background_pars[4])  
    
        
    Rho_r_a, Rho_m_a, Rho_de_a, initial_SF_dot = scale_back_initial(Background_pars[3], a_target, Background_pars[4],  Background_pars[1], Background_pars[0], Background_pars[2])
    
                                                                    # initial_hubble,    a_target,   a_ref,               Omega_r_ref,         Omega_m_ref,         Omega_de_ref
        
                                                                                            #Background_pars: [Omega_m, Omega_r, Omega_l, Ho, sf_ref] 
        
    Matter_density[0] = Rho_m_a
    Rad_density[0] = Rho_r_a
    DE_density[0] = Rho_de_a

    Scale_factors[0] = a_target
        
    Hubbles[0] = cp.sqrt((8 * cp.pi * G / 3) * (Rad_density[0] + Matter_density[0] + DE_density[0]))
    # a_dot = a(t) * H_o * sqrt( Init_mass_dp(sf_i/sf(t))^2  + Init_rad_dp(sf_i/sf(t))^2  = Init_DE_dp)
    # a = a_dot * sqrt(delta_t) as I am focussing on the radiaton dominated universe


    def f(a): 
        sf_dot = a * cp.sqrt(((8 * cp.pi * G)/3)  * (Rad_density[0] * ((a_target/a)**4)
                                    + Matter_density[0] * ((a_target/a)**3)
                                    + DE_density[0]))
        return sf_dot
        


    Scale_factor_dots[0] = f(Scale_factors[0])
        
    print("*************************************************************************")
    print(f"initial_SF_DOT is {Scale_factor_dots[0]}")
    print("*************************************************************************")

    for i in range(1, data_amt):
    
        sf = Scale_factors[i-1]
        H = Hubbles[i-1]

        sfRK1 = dt * f(sf)
        sfRK2 = dt * f(sf + 0.5 * sfRK1)
        sfRK3 = dt * f(sf + 0.5 * sfRK2)
        sfRK4 = dt * f(sf + sfRK3)
  
        Scale_factors[i] = sf + (sfRK1 + 2 * sfRK2 + 2 * sfRK3 + sfRK4)/ 6
        Scale_factor_dots[i] = f(Scale_factors[i])     
                
        Hubbles[i] =   Scale_factor_dots[i]/Scale_factors[i]       
                                                
        print(f"{i} ||| Scale_factor: {Scale_factors[i]}------ SF_dot: {Scale_factor_dots[i]} ----------- Hubble: {Scale_factor_dots[i]/Scale_factors[i]}")
        

    def plot_data():
        
        plt.figure(figsize=(10, 5))
        plt.plot(time.get(), Scale_factors.get(), label="Scale Factor (SF)", color='b')
        plt.xlabel("Time (s)")
        plt.ylabel("Scale Factor")
        plt.title("Evolution of Scale Factor ")
        plt.legend()
        plt.grid()
        
        plt.figure(figsize=(10, 5))
        plt.plot(time.get(), Scale_factor_dots.get(), label="SF Derivative (SFdot)", color='r')
        plt.xlabel("Time (s)")
        plt.ylabel("Rate of Change of Scale Factor")
        plt.title("Evolution of Scale Factor Derivative")
        plt.legend()
        plt.grid()


        plt.figure(figsize=(10, 5))
        plt.plot(time.get(), Hubbles.get(), label="Hubble Parameter over time", color='r')
        plt.xlabel("Time (s)")
        plt.ylabel("Hubble Parameter")
        plt.title("Evolution of Hubble parameter")
        plt.legend()
        plt.grid()
        plt.show()
        
    if plot == True:
        plot_data()
        
=======
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import os
import cupy as cp

G = 6.67430e-11   # m^3 kg^-1 s^-2
c = 299792458     # m/s
h_bar = 1.054571917e-34  # Joule*s

def scale_back_initial(initial_hubble, 
                       a_target, 
                       a_ref, 
                       Omega_r_ref, 
                       Omega_m_ref, 
                       Omega_de_ref):
    
    """
    Scales density parameters from their reference values at scale factor a_ref to a new scale factor a.
    
    Parameters:
        initial_hubble : float
            Hubble parameter at the reference scale factor a_ref.
        a            : float
            The desired scale factor at which to compute the densities.
        a_ref        : float, optional
            The reference scale factor for the input density parameters (default is 1).
        Omega_r_ref  : float, optional
            Radiation density parameter at a_ref (e.g., ~8.24e-5).
        Omega_m_ref  : float, optional
            Matter density parameter at a_ref (e.g., ~0.27).
        Omega_de_ref : float, optional
            Dark energy density parameter at a_ref (e.g., ~0.73).
        
    Returns:
        tuple:
            (rad_scaled, matter_scaled, DE_scaled, SFdot_0)
            where rad_scaled, matter_scaled, and DE_scaled are the energy densities at scale factor a,
            and SFdot_0 is the corresponding scale factor derivative computed via H(a)*a.
    """
    # Compute the critical density at the reference scale factor.
    rho_crit_ref = (3 * initial_hubble**2) / (8 *cp.pi * G)
    
    # Compute the actual densities at the reference scale factor.
    rho_r_ref = Omega_r_ref * rho_crit_ref
    rho_m_ref = Omega_m_ref * rho_crit_ref
    rho_de_ref = Omega_de_ref * rho_crit_ref  # dark energy is constant
    
    # Scale the densities to the new scale factor 'a'
    rad_scaled   = rho_r_ref * (a_ref / a_target)**4
    matter_scaled = rho_m_ref * (a_ref / a_target)**3
    DE_scaled = rho_de_ref  # no scaling for dark energy
    
    # Compute the Hubble parameter at scale factor a.
    H_a = cp.sqrt((8 * cp.pi * G / 3) * (rad_scaled + matter_scaled + DE_scaled)) 
    
    # Compute the derivative of the scale factor.
    SFdot_0 = H_a * a_target
    
    return rad_scaled, matter_scaled, DE_scaled, SFdot_0

        
        
        
def Expansion(Time,
              dt,
              Background_pars, 
              a_target,
              plot = False):

    G = 6.67430e-11   # m^3 kg^-1 s^-2
    c = 299792458     # m/s
    h_bar = 1.054571917e-34  # Joule*s  
        ####################################################################################################
        #  RUN TESTS ON THE ACCURACY OF THIS CODE
        #             - analytical Comparison
        #             - Convergence test
        #             - power-law test
        #             - Make sure energy is conserved
        ####################################################################################################       
        
    """
        Numerically integrates the Friedmann equation expressed in terms of the normalized scale factor 
        x = a(t)/a_i via the 4th-order Runge-Kutta method.
        
        The differential equation is:
        
            dx/dt = H_i * x * sqrt( (initial_Rad_par)/x^4 + (initial_mass_par)/x^3 + initial_DE_par )
        
        where the initial Hubble parameter is computed as:
        
            H_i = sqrt((8*pi*G/3) * (initial_Rad_par + initial_mass_par + initial_DE_par))
        
        After integrating for x(t), the full scale factor is given by:
        
            a(t) = a_i * x(t)
        
        If Initial_SF is None, we assume a_i = 1.
        
        Returns:
            - x_array: Array of normalized scale factor values.
            - t_array: Array of time values.
            - a_array: Array of full scale factor values.
    """
    
    data_amt = int(Time / dt) # to make sure that the this doesnt run to 7.5 steps
    
        
    #print(f"amt of data steps : {self.data_amt}")
    time = cp.linspace(0, Time, data_amt, dtype=cp.float64)
        
    Matter_density = cp.zeros(data_amt, dtype=cp.float64)
    Rad_density = cp.zeros(data_amt, dtype=cp.float64)
    DE_density = cp.zeros(data_amt, dtype=cp.float64)
                
    Scale_factor_dots = cp.zeros(data_amt, dtype=cp.float64)
    Scale_factors = cp.zeros(data_amt, dtype=cp.float64)
    Hubbles = cp.zeros(data_amt, dtype=cp.float64)
    
    Pars_0 = float(Background_pars[0])
    Pars_1 = float(Background_pars[1])
    Pars_2 = float(Background_pars[2])
    Pars_3 = float(Background_pars[3])
    Pars_4 = float(Background_pars[4])  
    
        
    Rho_r_a, Rho_m_a, Rho_de_a, initial_SF_dot = scale_back_initial(Background_pars[3], a_target, Background_pars[4],  Background_pars[1], Background_pars[0], Background_pars[2])
    
                                                                    # initial_hubble,    a_target,   a_ref,               Omega_r_ref,         Omega_m_ref,         Omega_de_ref
        
                                                                                            #Background_pars: [Omega_m, Omega_r, Omega_l, Ho, sf_ref] 
        
    Matter_density[0] = Rho_m_a
    Rad_density[0] = Rho_r_a
    DE_density[0] = Rho_de_a

    Scale_factors[0] = a_target
        
    Hubbles[0] = cp.sqrt((8 * cp.pi * G / 3) * (Rad_density[0] + Matter_density[0] + DE_density[0]))
    # a_dot = a(t) * H_o * sqrt( Init_mass_dp(sf_i/sf(t))^2  + Init_rad_dp(sf_i/sf(t))^2  = Init_DE_dp)
    # a = a_dot * sqrt(delta_t) as I am focussing on the radiaton dominated universe


    def f(a): 
        sf_dot = a * cp.sqrt(((8 * cp.pi * G)/3)  * (Rad_density[0] * ((a_target/a)**4)
                                    + Matter_density[0] * ((a_target/a)**3)
                                    + DE_density[0]))
        return sf_dot
        


    Scale_factor_dots[0] = f(Scale_factors[0])
        
    print("*************************************************************************")
    print(f"initial_SF_DOT is {Scale_factor_dots[0]}")
    print("*************************************************************************")

    for i in range(1, data_amt):
    
        sf = Scale_factors[i-1]
        H = Hubbles[i-1]

        sfRK1 = dt * f(sf)
        sfRK2 = dt * f(sf + 0.5 * sfRK1)
        sfRK3 = dt * f(sf + 0.5 * sfRK2)
        sfRK4 = dt * f(sf + sfRK3)
  
        Scale_factors[i] = sf + (sfRK1 + 2 * sfRK2 + 2 * sfRK3 + sfRK4)/ 6
        Scale_factor_dots[i] = f(Scale_factors[i])     
                
        Hubbles[i] =   Scale_factor_dots[i]/Scale_factors[i]       
                                                
        print(f"{i} ||| Scale_factor: {Scale_factors[i]}------ SF_dot: {Scale_factor_dots[i]} ----------- Hubble: {Scale_factor_dots[i]/Scale_factors[i]}")
        

    def plot_data():
        
        plt.figure(figsize=(10, 5))
        plt.plot(time.get(), Scale_factors.get(), label="Scale Factor (SF)", color='b')
        plt.xlabel("Time (s)")
        plt.ylabel("Scale Factor")
        plt.title("Evolution of Scale Factor ")
        plt.legend()
        plt.grid()
        
        plt.figure(figsize=(10, 5))
        plt.plot(time.get(), Scale_factor_dots.get(), label="SF Derivative (SFdot)", color='r')
        plt.xlabel("Time (s)")
        plt.ylabel("Rate of Change of Scale Factor")
        plt.title("Evolution of Scale Factor Derivative")
        plt.legend()
        plt.grid()


        plt.figure(figsize=(10, 5))
        plt.plot(time.get(), Hubbles.get(), label="Hubble Parameter over time", color='r')
        plt.xlabel("Time (s)")
        plt.ylabel("Hubble Parameter")
        plt.title("Evolution of Hubble parameter")
        plt.legend()
        plt.grid()
        plt.show()
        
    if plot == True:
        plot_data()
        
>>>>>>> 31f20d4 (Started development of the final simulation)
    return Scale_factors, Scale_factor_dots