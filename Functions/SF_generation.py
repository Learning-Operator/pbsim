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

        

def get_unique_filename(directory, base_name, extension=".png"):
                """Ensure the file doesn't overwrite an existing one by appending a number if needed."""
                counter = 1
                file_path = os.path.join(directory, f"{base_name}{extension}")

                while os.path.exists(file_path):
                    file_path = os.path.join(directory, f"{base_name}_{counter}{extension}")
                    counter += 1

                return file_path

        
        
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
        
 
    Matter_density[0] = Rho_m_a
    Rad_density[0] = Rho_r_a
    DE_density[0] = Rho_de_a

    Scale_factors[0] = a_target
        
    Hubbles[0] = cp.sqrt((8 * cp.pi * G / 3) * (Rad_density[0] + Matter_density[0] + DE_density[0]))

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
        

    def plot_data(dir):
        
        fig = plt.figure(figsize=(16,10))
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(time.get(), Scale_factors.get(), label="Scale Factor (SF)", color='b')
        
        time_array = time.get()
        x_sqrt = np.linspace(time_array.min(), time_array.max(), 100)
        # Scale the sqrt function to have a similar range as the original data
        y_sqrt = np.sqrt(x_sqrt - time_array.min())
        # Scale the sqrt function to match the amplitude of the original data
        scaling_factor = Scale_factors.get().max() / y_sqrt.max()
        y_sqrt = y_sqrt * scaling_factor

        ax1.plot(x_sqrt, y_sqrt, label="Sqrt Function", color='g', linestyle='--')
        
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Scale Factor")
        ax1.set_title("Evolution of Scale Factor ")
        ax1.legend()
        ax1.grid()
        
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(time.get(), Scale_factor_dots.get(), label="SF Derivative (SFdot)", color='r')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Rate of Change of Scale Factor")
        ax2.set_title("Evolution of Scale Factor Derivative")
        ax2.legend()
        ax2.grid()

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(time.get(), Hubbles.get(), label="Hubble Parameter over time", color='r')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(time.get(), Hubbles.get(), label="Hubble Parameter over time", color='r')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Hubble Parameter")
        ax3.set_title("Evolution of Hubble parameter")
        ax3.legend()
        ax3.grid()
        
        file_name = 'Sf_expansion'
        file_extension = '.png'
        number = 1
        if not os.path.exists(dir):
            os.makedirs(dir)
        while True:
            file_path = os.path.join(dir, f"{file_name}_{number}{file_extension}")
            if not os.path.exists(file_path):
                break
            number += 1
        
        
        
        plt.savefig(file_path)
        
        plt.show()
        
        
    if plot == True:
        
        plot_data(dir)
        
    return Scale_factors, Scale_factor_dots