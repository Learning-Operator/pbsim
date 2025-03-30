import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import os
from .File_naming import get_unique_filename

G = 6.67430e-11   # m^3 kg^-1 s^-2
c = 299792458     # m/s

def scale_back_initial(initial_hubble, 
                       a_target, 
                       a_ref, 
                       Omega_r_ref, 
                       Omega_m_ref, 
                       Omega_de_ref):

    # Compute the critical density at the reference scale factor.
    rho_crit_ref = (3 * initial_hubble**2) / (8 *np.pi * G) # kg/m^3
    
    # Compute the actual densities at the reference scale factor.
    rho_r_ref = Omega_r_ref * rho_crit_ref # kg/m^3
    rho_m_ref = Omega_m_ref * rho_crit_ref # kg/m^3
    rho_de_ref = Omega_de_ref * rho_crit_ref  # dark energy is constant |||| kg/m^3
    
    # Scale the densities to the new scale factor 'a'
    rad_scaled   = rho_r_ref * (a_ref / a_target)**4 # kg/m^3
    matter_scaled = rho_m_ref * (a_ref / a_target)**3 # kg/m^3
    DE_scaled = rho_de_ref  # no scaling for dark energy ||| kg/m^3
    
    # Compute the Hubble parameter at scale factor a.
    H_a = np.sqrt((8 * np.pi * G / 3) * (rad_scaled + matter_scaled + DE_scaled))  # 1/s
    
    # Compute the derivative of the scale factor.
    SFdot_0 = H_a * a_target # 1/s
    
    return rad_scaled, matter_scaled, DE_scaled, SFdot_0, H_a


def Expansion(Time,
              dt,
              Background_pars, 
              a_target,
              plot = False,
              dir = None):

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
        Inputs: 
            - Time : The duration of time to which you want to integrate over
            - dt : a  timestep of "Framerate" 
            - Background Pars: List || [Omega_m, Omega_r, Omega_l, Ho, sf_ref] 
            
        
        Returns:
            - Scale_factors: Array of full scale factor values.
            - Scale_factor_dots: Array of values of the rate of change of scale factor.
            - Plots(optional)
    """
    
    data_amt = int(Time / dt) # unitless
    
    time = np.linspace(0, Time, data_amt, dtype=np.float64) # s
        
    Matter_density = np.zeros(data_amt, dtype=np.float64) # should be kg/m^3
    Rad_density = np.zeros(data_amt, dtype=np.float64) # should be kg/m^3
    DE_density = np.zeros(data_amt, dtype=np.float64) # should be kg/m^3
                
    Scale_factor_dots = np.zeros(data_amt, dtype=np.float64) # 1/s
    Scale_factors = np.zeros(data_amt, dtype=np.float64) # unitless
    Hubbles = np.zeros(data_amt, dtype=np.float64) # 1/s
    
        
    Rho_r_a, Rho_m_a, Rho_de_a, initial_SF_dot, H0 = scale_back_initial(Background_pars[3], a_target, Background_pars[4],  Background_pars[1], Background_pars[0], Background_pars[2])
    
                                                                         # initial_hubble,    a_target,        a_ref,            Omega_r_ref,         Omega_m_ref,         Omega_de_ref
        
 
    Matter_density[0] = Rho_m_a # kg/m^3
    Rad_density[0] = Rho_r_a #kg/m^3
    DE_density[0] = Rho_de_a #kg/m^3
    Scale_factors[0] = a_target #unitless
    Hubbles[0] = H0 #1/s

    def f(a): 
        return  a * np.sqrt(((8 * np.pi * G)/3)  * (Rad_density[0] * ((a_target/a)**4)+ Matter_density[0] * ((a_target/a)**3) + Rho_de_a))
    
    Scale_factor_dots[0] = f(Scale_factors[0]) # 1/s

    for i in range(1, data_amt):
    
        sf = Scale_factors[i-1] # unitless
        H = Hubbles[i-1] # 1/s

        sfRK1 = dt * f(sf) # Unitless
        sfRK2 = dt * f(sf + 0.5 * sfRK1) # Unitless
        sfRK3 = dt * f(sf + 0.5 * sfRK2) # Unitless
        sfRK4 = dt * f(sf + sfRK3) # Unitless
  
        Scale_factors[i] = sf + (sfRK1 + 2 * sfRK2 + 2 * sfRK3 + sfRK4)/ 6 # Unitless
        Scale_factor_dots[i] = f(Scale_factors[i])      # 1/s
                
        Hubbles[i] =   Scale_factor_dots[i]/Scale_factors[i]       # 1/s
                                                
        print(f"{i} ||| Scale_factor: {Scale_factors[i]}------ SF_dot: {Scale_factor_dots[i]} ----------- Hubble: {Scale_factor_dots[i]/Scale_factors[i]}")
        

    def plot_data(dir):
        
        fig = plt.figure(figsize=(16,10))
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(time, Scale_factors, label="Scale Factor (SF)", color='b')
        
        #time_array = time
        #x_sqrt = np.linspace(time_array.min(), time_array.max(), 100)
        ## Scale the sqrt function to have a similar range as the original data
        #y_sqrt = np.sqrt(x_sqrt - time_array.min())
        ## Scale the sqrt function to match the amplitude of the original data
        #scaling_factor = Scale_factors.max() / y_sqrt.max()
        #y_sqrt = y_sqrt * scaling_factor
        #ax1.plot(x_sqrt, y_sqrt, label="Sqrt Function", color='g', linestyle='--')
        
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Scale Factor")
        ax1.set_title("Evolution of Scale Factor ")
        ax1.legend()
        ax1.grid()
        
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(time, Scale_factor_dots, label="SF Derivative (SFdot)", color='r')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Rate of Change of Scale Factor")
        ax2.set_title("Evolution of Scale Factor Derivative")
        ax2.legend()
        ax2.grid()

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(time, Hubbles, label="Hubble Parameter over time", color='r')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(time, Hubbles, label="Hubble Parameter over time", color='r')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Hubble Parameter")
        ax3.set_title("Evolution of Hubble parameter")
        ax3.legend()
        ax3.grid()
        
        file_path = get_unique_filename(output_folder = dir, output_type = 'fig', filename = "Scale_Factor_expansion", file_type = ".png")
        
        plt.savefig(file_path)
        
        plt.show()
        
        
    if plot == True:
        
        plot_data(dir)
        
    return Scale_factors, Scale_factor_dots, Hubbles