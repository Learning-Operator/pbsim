import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import os
import cupy as cp


# %%  
if cp.cuda.is_available():
    print("CUDA GPU is available!")
else:
    print("CUDA GPU is not available.")

# %%



class Particle:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = cp.array(position)  # Ensure position is a numpy array
        self.velocity = cp.array(velocity)  # Ensure velocity is a numpy array

    def update_position(self, scale_factor, SF_Prev):
        """Update the physical position based on the scale factor."""
        self.position = self.position *  (scale_factor/SF_Prev)

class N_Bod():
    def __init__(self, 
                 N_particles=10, 
                 Time=10.0, 
                 dt=0.01, 
                 Scalar_curvature=0, 
                 Initial_SF=None,                  
                 Masspower=None,
                 mass_range = [1,1],
                 positionxyz_range = [1,1], 
                 velocityxyz_range = [1,1]):
        '''
        Simulation parameters:
            - N_particles: Number of particles
            - Time: Total integration time (in seconds)
            - dt: Timestep (in seconds)
        Universe parameters:
            - Scalar_curvature: (currently unused; for a flat universe, set to 0)
            - Initial_SF: initial scale factor, a_i (if None, defaults to 1)
            - Initial_SFdot: initial derivative of the scale factor (unused here)
            - initial_mass_par, initial_Rad_par, initial_DE_par: density parameters for matter, radiation, dark energy
            - Masspower: placeholder for the matter power spectrum (unused)
            - w: Equation-of-state parameter (e.g. 1/3 for a radiation-dominated universe)
            - density_contrast: placeholder parameter (unused in Expansion)
        
        The class uses physical constants G, c, and h_bar, which can be set to 1 in natural units.
        '''
        self.N_particles = N_particles
        self.Time = Time
        self.dt = dt
        
        self.Scalar_curvature = Scalar_curvature
        self.Masspower = Masspower        
        self.Initial_SF = Initial_SF  # a_i, the initial scale factor
        
        self.Mass_range = mass_range
        self.position_range = positionxyz_range
        self.velocity_range = velocityxyz_range

        Particles_list = self.create_particles()       
        print("The number of particles there should be is", N_particles)
        print("The number of particles there is", len(Particles_list))
        
        # Physical constants (in SI units)
        self.G = 6.67430e-11   # m^3 kg^-1 s^-2
        self.c = 299792458     # m/s
        self.h_bar = 1.054571917e-34  # Joule*s
        
        ##############################################################
        # Set natural units if desired.
        ###############################################################
            
        self.lambda_const = 1e-52  # m^-2 (unused in expansion)
        
        # Perform the expansion integration.
        self.Sf_list, self.SF_dot_list = self.Expansion(self.Time, self.dt,self.Initial_SF)
        
        
        
        self.simulate(Particles_list, self.Time, self.dt, self.Sf_list,self.SF_dot_list )

    def scale_back_initial(self, initial_hubble, a, a_ref=1.0, Omega_r_ref=8.24e-5, Omega_m_ref=0.27, Omega_de_ref=0.73):
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
        rho_crit_ref = (3 * initial_hubble**2) / (8 *cp.pi * self.G)
    
        # Compute the actual densities at the reference scale factor.
        rho_r_ref = Omega_r_ref * rho_crit_ref
        rho_m_ref = Omega_m_ref * rho_crit_ref
        rho_de_ref = Omega_de_ref * rho_crit_ref  # dark energy is constant
    
        # Scale the densities to the new scale factor 'a'
        rad_scaled   = rho_r_ref * (a_ref / a)**4
        matter_scaled = rho_m_ref * (a_ref / a)**3
        DE_scaled    = rho_de_ref  # no scaling for dark energy
    
        # Compute the Hubble parameter at scale factor a.
        H_a = cp.sqrt((8 * cp.pi * self.G / 3) * (rad_scaled + matter_scaled + DE_scaled))
    
        # Compute the derivative of the scale factor.
        SFdot_0 = H_a * a
    
        return rad_scaled, matter_scaled, DE_scaled, SFdot_0

        
        
        
    def Expansion(self, Time, dt,
                  Initial_SF):
        
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
        self.data_amt = int(Time/dt) # to make sure that the this doesnt run to 7.5 steps
        
        #print(f"amt of data steps : {self.data_amt}")
        time = cp.linspace(0, Time, self.data_amt, dtype=cp.float64)
        
        self.Matter_density = cp.zeros(self.data_amt, dtype=cp.float64)
        self.Rad_density = cp.zeros(self.data_amt, dtype=cp.float64)
        self.DE_density = cp.zeros(self.data_amt, dtype=cp.float64)
                
        self.Scale_factor_dots = cp.zeros(self.data_amt, dtype=cp.float64)
        self.Scale_factors = cp.zeros(self.data_amt, dtype=cp.float64)
        self.Hubbles = cp.zeros(self.data_amt, dtype=cp.float64)
        
        Rho_r_a, Rho_m_a, Rho_de_a, self.initial_SF_dot = self.scale_back_initial(2.19e-18, Initial_SF, 1/1101,  8.24e-5, 0.27, 0.73)
        
        self.Matter_density[0] = Rho_m_a
        self.Rad_density[0] = Rho_r_a
        self.DE_density[0] = Rho_de_a

        self.Scale_factors[0] = Initial_SF
        
        self.Hubbles[0] = cp.sqrt((8 * cp.pi * self.G / 3) * (self.Rad_density[0] + self.Matter_density[0] + self.DE_density[0]))
        # a_dot = a(t) * H_o * sqrt( Init_mass_dp(sf_i/sf(t))^2  + Init_rad_dp(sf_i/sf(t))^2  = Init_DE_dp)
        # a = a_dot * sqrt(delta_t) as I am focussing on the radiaton dominated universe


        def f(a): 
            sf_dot = a * cp.sqrt(((8 * cp.pi * self.G)/3)  * (self.Rad_density[0] * ((self.Initial_SF/a)**4)
                                        + self.Matter_density[0] * ((self.Initial_SF/a)**3)
                                        + self.DE_density[0]))
            return sf_dot
        


        self.Scale_factor_dots[0] = f(self.Scale_factors[0])
        
        print("*************************************************************************")
        print(f"initial_SF_DOT is {self.Scale_factor_dots[0]}")
        print("*************************************************************************")

        for i in range(1, self.data_amt):
    
            sf = self.Scale_factors[i-1]
            H = self.Hubbles[i-1]

            sfRK1 = dt * f(sf)
            sfRK2 = dt * f(sf + 0.5 * sfRK1)
            sfRK3 = dt * f(sf + 0.5 * sfRK2)
            sfRK4 = dt * f(sf + sfRK3)
  
            self.Scale_factors[i] = sf + (sfRK1 + 2 * sfRK2 + 2 * sfRK3 + sfRK4)/ 6
            self.Scale_factor_dots[i] = f(self.Scale_factors[i])     
                
            self.Hubbles[i] =   self.Scale_factor_dots[i]/self.Scale_factors[i]       
                                                
            print(f"{i} ||| Scale_factor: {self.Scale_factors[i]}------ SF_dot: {self.Scale_factor_dots[i]} ----------- Hubble: {self.Scale_factor_dots[i]/self.Scale_factors[i]}")
        

        
        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        axs[0].plot(time.get(), self.Scale_factors.get(), label="Scale Factor (SF)", color='b')
        axs[0].set_ylabel("Scale Factor")
        axs[0].set_title("Evolution of Scale Factor")
        axs[0].legend()
        axs[0].grid()
        
        axs[1].plot(time.get(), self.Scale_factor_dots.get(), label="SF Derivative (SFdot)", color='r')
        axs[1].set_ylabel("Rate of Change of Scale Factor")
        axs[1].set_title("Evolution of Scale Factor Derivative")
        axs[1].legend()
        axs[1].grid()
        
        axs[2].plot(time.get(), self.Hubbles.get(), label="Hubble Parameter over time", color='g')
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Hubble Parameter")
        axs[2].set_title("Evolution of Hubble Parameter")
        axs[2].legend()
        axs[2].grid()
        
        # Adjust layout for better readability
        plt.tight_layout()
        plt.show()
        
        return self.Scale_factors, self.Scale_factor_dots



    def simulate(self, particles, Time, dt, scale_factors_list, Scale_factor_dots ):
        # Calculate the number of timesteps
        Timesteps = int(Time / dt)

        # Initialize the tensor to store position for each particle
        # Shape: (Timesteps, num_particles, 3) for position
        Positions_tensor = cp.zeros((Timesteps, len(particles), 3))

        # Run the simulation with the Runge-Kutta method
        for t in range(1, Timesteps):
            
            self.runge_kutta_step(particles, dt, scale_factors_list[t-1], Scale_factor_dots[t-1]  )
       
            
            # Store the updated positions in the tensor
            for i, particle in enumerate(particles):
                #particle.update_position(scale_factors_list[t], scale_factors_list[t-1]) #################################################################################################################################################################################################################################################################################
                Positions_tensor[t, i, :] = particle.position
                
            print(f"t = {t}")

        # Visualize the simulation as an animation
        self.animate_positions_over_time(Positions_tensor)
    

    def create_particles(self):
        particles = []

        for i in range(self.N_particles):
            particles.append(Particle(
                mass=cp.random.uniform(self.Mass_range[0], self.Mass_range[1]),
                position=[
                    cp.random.uniform(self.position_range[0], self.position_range[1]),
                    cp.random.uniform(self.position_range[0], self.position_range[1]),
                    cp.random.uniform(self.position_range[0], self.position_range[1])
                ],
                velocity=[
                    cp.random.uniform(self.velocity_range[0], self.velocity_range[1]),
                    cp.random.uniform(self.velocity_range[0], self.velocity_range[1]),
                    cp.random.uniform(self.velocity_range[0], self.velocity_range[1])
                ]
            ))
        
        return particles

    def compute_gravitational_acceleration(self, particles):
        num_particles = self.N_particles  # or len(particles)
        masses = cp.array([particle.mass for particle in particles])        # Shape: (num_particles,)
        positions = cp.array([particle.position for particle in particles])  # Shape: (num_particles, 3)

        # Compute pairwise displacement vectors (r_ij)
        pos_diff = positions[:, cp.newaxis, :] - positions[cp.newaxis, :, :]  # Shape: (num_particles, num_particles, 3)

        # Compute pairwise distances (norms of r_ij)
        distances = cp.linalg.norm(pos_diff, axis=2)  # Shape: (num_particles, num_particles)

        # Create a mask where distances > 0 to avoid division by zero
        mask = distances > 0

        # Compute force magnitudes elementwise.
        # Use cp.where to perform the division only where distance > 0,
        # otherwise set the force to zero.
        force_magnitudes = cp.where(mask, self.G * masses[None, :] / (distances**3), 0)
        # force_magnitudes now has shape (num_particles, num_particles)

        # Normalize the displacement vectors to get unit vectors.
        # For distances that are zero, set the unit vector to 0.
        unit_vectors = cp.where(mask[..., None], pos_diff / distances[..., None], 0)

        # Sum the contributions of the forces from all particles along axis 1 to get acceleration.
        accelerations = -cp.sum(force_magnitudes[..., None] * unit_vectors, axis=1)  # Shape: (num_particles, 3)

        return accelerations


    def runge_kutta_step(self, particles, dt, sf, sf_dot): ### REDO THIS PART
        num_particles = len(particles)
        positions = cp.array([p.position for p in particles])
        velocities = cp.array([p.velocity for p in particles])
        masses = cp.array([p.mass for p in particles])

        # Compute k1
        k1_v = -2 * (sf_dot/sf) * velocities - 1/(sf**3) * self.compute_gravitational_acceleration(particles)
        k1_r = velocities

        # Compute k2
        temp_positions = positions + 0.5 * dt * k1_r
        temp_velocities = velocities + 0.5 * dt * k1_v
        for i, p in enumerate(particles):
            p.position = temp_positions[i]
            p.velocity = temp_velocities[i]
            #print(i)
        k2_v = -2 * (sf_dot/sf)* velocities - 1/(sf**3) * self.compute_gravitational_acceleration(particles)
        k2_r = temp_velocities

        # Compute k3
        temp_positions = positions + 0.5 * dt * k2_r
        temp_velocities = velocities + 0.5 * dt * k2_v
        for i, p in enumerate(particles):
            p.position = temp_positions[i]
            p.velocity = temp_velocities[i]
            #print(i)
        k3_v = -2 * (sf_dot/sf)* velocities - 1/(sf**3) * self.compute_gravitational_acceleration(particles)
        k3_r = temp_velocities

        # Compute k4
        temp_positions = positions + dt * k3_r
        temp_velocities = velocities + dt * k3_v
        for i, p in enumerate(particles):
            p.position = temp_positions[i]
            p.velocity = temp_velocities[i]
            #print(i)
        k4_v = -2 * (sf_dot/sf)* velocities - 1/(sf**3) * self.compute_gravitational_acceleration(particles)
        k4_r = temp_velocities

        # Update particles using weighted sum of k1, k2, k3, and k4
        for i, particle in enumerate(particles):
            particle.velocity += dt / 6 * (k1_v[i] + 2 * k2_v[i] + 2 * k3_v[i] + k4_v[i])
            particle.position += dt / 6 * (k1_r[i] + 2 * k2_r[i] + 2 * k3_r[i] + k4_r[i])
            #print(f"complete: {i}")

            
        
    def animate_positions_over_time(self, positions_tensor):
        # Convert the positions tensor from a CuPy array to a NumPy array.
        positions_tensor_np = cp.asnumpy(positions_tensor)
    
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
        # Initialize scatter plot for each particle.
        scatters = [ax.scatter([], [], [], label=f"Particle {i}") for i in range(self.N_particles)]
    
        margin = 10  
        x_all = positions_tensor_np[:, :, 0]
        y_all = positions_tensor_np[:, :, 1]
        z_all = positions_tensor_np[:, :, 2]
        ax.set_xlim(x_all.min() - margin, x_all.max() + margin)
        ax.set_ylim(y_all.min() - margin, y_all.max() + margin)
        ax.set_zlim(z_all.min() - margin, z_all.max() + margin)
    
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Position of Particles Over Time")
    
        # Build a multi-line string with key simulation inputs.
        inputs_str = (
            f"N_particles: {self.N_particles}\n"
            f"Time: {self.Time}\n"
            f"dt: {self.dt}\n"
            f"Scalar_curvature: {self.Scalar_curvature}\n"
            f"Initial_SF: {self.Initial_SF}\n"
            f"Initial_SFdot: {self.initial_SF_dot}\n"
            f"mass_range: {self.Mass_range}\n"
            f"position_range: {self.position_range}\n"
            f"velocity_range: {self.velocity_range}"
        )
    
    # Add the input text as a text box in the upper left corner.
        ax.text2D(0.05, 0.95, inputs_str, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
        def update(frame):
            # Update each particle's scatter.
            for i, scatter in enumerate(scatters):
                scatter._offsets3d = (
                    [positions_tensor_np[frame, i, 0]],
                    [positions_tensor_np[frame, i, 1]],
                    [positions_tensor_np[frame, i, 2]]
                )
                
            xs = positions_tensor_np[frame, :, 0]
            ys = positions_tensor_np[frame, :, 1]
            zs = positions_tensor_np[frame, :, 2]
            ax.set_xlim(xs.min() - margin, xs.max() + margin)
            ax.set_ylim(ys.min() - margin, ys.max() + margin)
            ax.set_zlim(zs.min() - margin, zs.max() + margin)
            
            return scatters
    
        # Create a unique filename for saving the animation.
        directory = r'C:\Users\Kiran\Desktop\PBH project\PBH_sim'
        file_name = 'n_body'
        file_extension = '.gif'
        number = 1
        if not os.path.exists(directory):
            os.makedirs(directory)
        while True:
            file_path = os.path.join(directory, f"{file_name}_{number}{file_extension}")
            if not os.path.exists(file_path):
                break
            number += 1
    
        anim = FuncAnimation(fig, update, frames=positions_tensor_np.shape[0], interval=50, blit=False)
        anim.save(file_path, writer='pillow', fps=20)
        print(f"Animation saved as: {file_path}")
        plt.show()


import time

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper


@time_function
def main():
    universe = N_Bod(
        N_particles=100, 
        Time=5000, 
        dt= 1, 
        Scalar_curvature=0, 
        Initial_SF=1e-12, 
        mass_range=[1e27, 1e30],
        positionxyz_range=[-10000000, 1000000],
        velocityxyz_range=[-4, 4]
        
    )
    
if __name__ == "__main__":
    main()
=======
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import os
import cupy as cp


# %%  
if cp.cuda.is_available():
    print("CUDA GPU is available!")
else:
    print("CUDA GPU is not available.")

# %%



class Particle:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = cp.array(position)  # Ensure position is a numpy array
        self.velocity = cp.array(velocity)  # Ensure velocity is a numpy array

    def update_position(self, scale_factor, SF_Prev):
        """Update the physical position based on the scale factor."""
        self.position = self.position *  (scale_factor/SF_Prev)

class N_Bod():
    def __init__(self, 
                 N_particles=10, 
                 Time=10.0, 
                 dt=0.01, 
                 Scalar_curvature=0, 
                 Initial_SF=None,                  
                 Masspower=None,
                 mass_range = [1,1],
                 positionxyz_range = [1,1], 
                 velocityxyz_range = [1,1]):
        '''
        Simulation parameters:
            - N_particles: Number of particles
            - Time: Total integration time (in seconds)
            - dt: Timestep (in seconds)
        Universe parameters:
            - Scalar_curvature: (currently unused; for a flat universe, set to 0)
            - Initial_SF: initial scale factor, a_i (if None, defaults to 1)
            - Initial_SFdot: initial derivative of the scale factor (unused here)
            - initial_mass_par, initial_Rad_par, initial_DE_par: density parameters for matter, radiation, dark energy
            - Masspower: placeholder for the matter power spectrum (unused)
            - w: Equation-of-state parameter (e.g. 1/3 for a radiation-dominated universe)
            - density_contrast: placeholder parameter (unused in Expansion)
        
        The class uses physical constants G, c, and h_bar, which can be set to 1 in natural units.
        '''
        self.N_particles = N_particles
        self.Time = Time
        self.dt = dt
        
        self.Scalar_curvature = Scalar_curvature
        self.Masspower = Masspower        
        self.Initial_SF = Initial_SF  # a_i, the initial scale factor
        
        self.Mass_range = mass_range
        self.position_range = positionxyz_range
        self.velocity_range = velocityxyz_range

        Particles_list = self.create_particles()       
        print("The number of particles there should be is", N_particles)
        print("The number of particles there is", len(Particles_list))
        
        # Physical constants (in SI units)
        self.G = 6.67430e-11   # m^3 kg^-1 s^-2
        self.c = 299792458     # m/s
        self.h_bar = 1.054571917e-34  # Joule*s
        
        ##############################################################
        # Set natural units if desired.
        ###############################################################
            
        self.lambda_const = 1e-52  # m^-2 (unused in expansion)
        
        # Perform the expansion integration.
        self.Sf_list, self.SF_dot_list = self.Expansion(self.Time, self.dt,self.Initial_SF)
        
        
        
        self.simulate(Particles_list, self.Time, self.dt, self.Sf_list,self.SF_dot_list )

    def scale_back_initial(self, initial_hubble, a, a_ref=1.0, Omega_r_ref=8.24e-5, Omega_m_ref=0.27, Omega_de_ref=0.73):
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
        rho_crit_ref = (3 * initial_hubble**2) / (8 *cp.pi * self.G)
    
        # Compute the actual densities at the reference scale factor.
        rho_r_ref = Omega_r_ref * rho_crit_ref
        rho_m_ref = Omega_m_ref * rho_crit_ref
        rho_de_ref = Omega_de_ref * rho_crit_ref  # dark energy is constant
    
        # Scale the densities to the new scale factor 'a'
        rad_scaled   = rho_r_ref * (a_ref / a)**4
        matter_scaled = rho_m_ref * (a_ref / a)**3
        DE_scaled    = rho_de_ref  # no scaling for dark energy
    
        # Compute the Hubble parameter at scale factor a.
        H_a = cp.sqrt((8 * cp.pi * self.G / 3) * (rad_scaled + matter_scaled + DE_scaled))
    
        # Compute the derivative of the scale factor.
        SFdot_0 = H_a * a
    
        return rad_scaled, matter_scaled, DE_scaled, SFdot_0

        
        
        
    def Expansion(self, Time, dt,
                  Initial_SF):
        
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
        self.data_amt = int(Time/dt) # to make sure that the this doesnt run to 7.5 steps
        
        #print(f"amt of data steps : {self.data_amt}")
        time = cp.linspace(0, Time, self.data_amt, dtype=cp.float64)
        
        self.Matter_density = cp.zeros(self.data_amt, dtype=cp.float64)
        self.Rad_density = cp.zeros(self.data_amt, dtype=cp.float64)
        self.DE_density = cp.zeros(self.data_amt, dtype=cp.float64)
                
        self.Scale_factor_dots = cp.zeros(self.data_amt, dtype=cp.float64)
        self.Scale_factors = cp.zeros(self.data_amt, dtype=cp.float64)
        self.Hubbles = cp.zeros(self.data_amt, dtype=cp.float64)
        
        Rho_r_a, Rho_m_a, Rho_de_a, self.initial_SF_dot = self.scale_back_initial(2.19e-18, Initial_SF, 1/1101,  8.24e-5, 0.27, 0.73)
        
        self.Matter_density[0] = Rho_m_a
        self.Rad_density[0] = Rho_r_a
        self.DE_density[0] = Rho_de_a

        self.Scale_factors[0] = Initial_SF
        
        self.Hubbles[0] = cp.sqrt((8 * cp.pi * self.G / 3) * (self.Rad_density[0] + self.Matter_density[0] + self.DE_density[0]))
        # a_dot = a(t) * H_o * sqrt( Init_mass_dp(sf_i/sf(t))^2  + Init_rad_dp(sf_i/sf(t))^2  = Init_DE_dp)
        # a = a_dot * sqrt(delta_t) as I am focussing on the radiaton dominated universe


        def f(a): 
            sf_dot = a * cp.sqrt(((8 * cp.pi * self.G)/3)  * (self.Rad_density[0] * ((self.Initial_SF/a)**4)
                                        + self.Matter_density[0] * ((self.Initial_SF/a)**3)
                                        + self.DE_density[0]))
            return sf_dot
        


        self.Scale_factor_dots[0] = f(self.Scale_factors[0])
        
        print("*************************************************************************")
        print(f"initial_SF_DOT is {self.Scale_factor_dots[0]}")
        print("*************************************************************************")

        for i in range(1, self.data_amt):
    
            sf = self.Scale_factors[i-1]
            H = self.Hubbles[i-1]

            sfRK1 = dt * f(sf)
            sfRK2 = dt * f(sf + 0.5 * sfRK1)
            sfRK3 = dt * f(sf + 0.5 * sfRK2)
            sfRK4 = dt * f(sf + sfRK3)
  
            self.Scale_factors[i] = sf + (sfRK1 + 2 * sfRK2 + 2 * sfRK3 + sfRK4)/ 6
            self.Scale_factor_dots[i] = f(self.Scale_factors[i])     
                
            self.Hubbles[i] =   self.Scale_factor_dots[i]/self.Scale_factors[i]       
                                                
            print(f"{i} ||| Scale_factor: {self.Scale_factors[i]}------ SF_dot: {self.Scale_factor_dots[i]} ----------- Hubble: {self.Scale_factor_dots[i]/self.Scale_factors[i]}")
        

        
        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        axs[0].plot(time.get(), self.Scale_factors.get(), label="Scale Factor (SF)", color='b')
        axs[0].set_ylabel("Scale Factor")
        axs[0].set_title("Evolution of Scale Factor")
        axs[0].legend()
        axs[0].grid()
        
        axs[1].plot(time.get(), self.Scale_factor_dots.get(), label="SF Derivative (SFdot)", color='r')
        axs[1].set_ylabel("Rate of Change of Scale Factor")
        axs[1].set_title("Evolution of Scale Factor Derivative")
        axs[1].legend()
        axs[1].grid()
        
        axs[2].plot(time.get(), self.Hubbles.get(), label="Hubble Parameter over time", color='g')
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Hubble Parameter")
        axs[2].set_title("Evolution of Hubble Parameter")
        axs[2].legend()
        axs[2].grid()
        
        # Adjust layout for better readability
        plt.tight_layout()
        plt.show()
        
        return self.Scale_factors, self.Scale_factor_dots



    def simulate(self, particles, Time, dt, scale_factors_list, Scale_factor_dots ):
        # Calculate the number of timesteps
        Timesteps = int(Time / dt)

        # Initialize the tensor to store position for each particle
        # Shape: (Timesteps, num_particles, 3) for position
        Positions_tensor = cp.zeros((Timesteps, len(particles), 3))

        # Run the simulation with the Runge-Kutta method
        for t in range(1, Timesteps):
            
            self.runge_kutta_step(particles, dt, scale_factors_list[t-1], Scale_factor_dots[t-1]  )
       
            
            # Store the updated positions in the tensor
            for i, particle in enumerate(particles):
                #particle.update_position(scale_factors_list[t], scale_factors_list[t-1]) #################################################################################################################################################################################################################################################################################
                Positions_tensor[t, i, :] = particle.position
                
            print(f"t = {t}")

        # Visualize the simulation as an animation
        self.animate_positions_over_time(Positions_tensor)
    

    def create_particles(self):
        particles = []

        for i in range(self.N_particles):
            particles.append(Particle(
                mass=cp.random.uniform(self.Mass_range[0], self.Mass_range[1]),
                position=[
                    cp.random.uniform(self.position_range[0], self.position_range[1]),
                    cp.random.uniform(self.position_range[0], self.position_range[1]),
                    cp.random.uniform(self.position_range[0], self.position_range[1])
                ],
                velocity=[
                    cp.random.uniform(self.velocity_range[0], self.velocity_range[1]),
                    cp.random.uniform(self.velocity_range[0], self.velocity_range[1]),
                    cp.random.uniform(self.velocity_range[0], self.velocity_range[1])
                ]
            ))
        
        return particles

    def compute_gravitational_acceleration(self, particles):
        num_particles = self.N_particles  # or len(particles)
        masses = cp.array([particle.mass for particle in particles])        # Shape: (num_particles,)
        positions = cp.array([particle.position for particle in particles])  # Shape: (num_particles, 3)

        # Compute pairwise displacement vectors (r_ij)
        pos_diff = positions[:, cp.newaxis, :] - positions[cp.newaxis, :, :]  # Shape: (num_particles, num_particles, 3)

        # Compute pairwise distances (norms of r_ij)
        distances = cp.linalg.norm(pos_diff, axis=2)  # Shape: (num_particles, num_particles)

        # Create a mask where distances > 0 to avoid division by zero
        mask = distances > 0

        # Compute force magnitudes elementwise.
        # Use cp.where to perform the division only where distance > 0,
        # otherwise set the force to zero.
        force_magnitudes = cp.where(mask, self.G * masses[None, :] / (distances**3), 0)
        # force_magnitudes now has shape (num_particles, num_particles)

        # Normalize the displacement vectors to get unit vectors.
        # For distances that are zero, set the unit vector to 0.
        unit_vectors = cp.where(mask[..., None], pos_diff / distances[..., None], 0)

        # Sum the contributions of the forces from all particles along axis 1 to get acceleration.
        accelerations = -cp.sum(force_magnitudes[..., None] * unit_vectors, axis=1)  # Shape: (num_particles, 3)

        return accelerations


    def runge_kutta_step(self, particles, dt, sf, sf_dot): ### REDO THIS PART
        num_particles = len(particles)
        positions = cp.array([p.position for p in particles])
        velocities = cp.array([p.velocity for p in particles])
        masses = cp.array([p.mass for p in particles])

        # Compute k1
        k1_v = -2 * (sf_dot/sf) * velocities - 1/(sf**3) * self.compute_gravitational_acceleration(particles)
        k1_r = velocities

        # Compute k2
        temp_positions = positions + 0.5 * dt * k1_r
        temp_velocities = velocities + 0.5 * dt * k1_v
        for i, p in enumerate(particles):
            p.position = temp_positions[i]
            p.velocity = temp_velocities[i]
            #print(i)
        k2_v = -2 * (sf_dot/sf)* velocities - 1/(sf**3) * self.compute_gravitational_acceleration(particles)
        k2_r = temp_velocities

        # Compute k3
        temp_positions = positions + 0.5 * dt * k2_r
        temp_velocities = velocities + 0.5 * dt * k2_v
        for i, p in enumerate(particles):
            p.position = temp_positions[i]
            p.velocity = temp_velocities[i]
            #print(i)
        k3_v = -2 * (sf_dot/sf)* velocities - 1/(sf**3) * self.compute_gravitational_acceleration(particles)
        k3_r = temp_velocities

        # Compute k4
        temp_positions = positions + dt * k3_r
        temp_velocities = velocities + dt * k3_v
        for i, p in enumerate(particles):
            p.position = temp_positions[i]
            p.velocity = temp_velocities[i]
            #print(i)
        k4_v = -2 * (sf_dot/sf)* velocities - 1/(sf**3) * self.compute_gravitational_acceleration(particles)
        k4_r = temp_velocities

        # Update particles using weighted sum of k1, k2, k3, and k4
        for i, particle in enumerate(particles):
            particle.velocity += dt / 6 * (k1_v[i] + 2 * k2_v[i] + 2 * k3_v[i] + k4_v[i])
            particle.position += dt / 6 * (k1_r[i] + 2 * k2_r[i] + 2 * k3_r[i] + k4_r[i])
            #print(f"complete: {i}")

            
        
    def animate_positions_over_time(self, positions_tensor):
        # Convert the positions tensor from a CuPy array to a NumPy array.
        positions_tensor_np = cp.asnumpy(positions_tensor)
    
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
        # Initialize scatter plot for each particle.
        scatters = [ax.scatter([], [], [], label=f"Particle {i}") for i in range(self.N_particles)]
    
        margin = 10  
        x_all = positions_tensor_np[:, :, 0]
        y_all = positions_tensor_np[:, :, 1]
        z_all = positions_tensor_np[:, :, 2]
        ax.set_xlim(x_all.min() - margin, x_all.max() + margin)
        ax.set_ylim(y_all.min() - margin, y_all.max() + margin)
        ax.set_zlim(z_all.min() - margin, z_all.max() + margin)
    
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Position of Particles Over Time")
    
        # Build a multi-line string with key simulation inputs.
        inputs_str = (
            f"N_particles: {self.N_particles}\n"
            f"Time: {self.Time}\n"
            f"dt: {self.dt}\n"
            f"Scalar_curvature: {self.Scalar_curvature}\n"
            f"Initial_SF: {self.Initial_SF}\n"
            f"Initial_SFdot: {self.initial_SF_dot}\n"
            f"mass_range: {self.Mass_range}\n"
            f"position_range: {self.position_range}\n"
            f"velocity_range: {self.velocity_range}"
        )
    
    # Add the input text as a text box in the upper left corner.
        ax.text2D(0.05, 0.95, inputs_str, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
        def update(frame):
            # Update each particle's scatter.
            for i, scatter in enumerate(scatters):
                scatter._offsets3d = (
                    [positions_tensor_np[frame, i, 0]],
                    [positions_tensor_np[frame, i, 1]],
                    [positions_tensor_np[frame, i, 2]]
                )
                
            xs = positions_tensor_np[frame, :, 0]
            ys = positions_tensor_np[frame, :, 1]
            zs = positions_tensor_np[frame, :, 2]
            ax.set_xlim(xs.min() - margin, xs.max() + margin)
            ax.set_ylim(ys.min() - margin, ys.max() + margin)
            ax.set_zlim(zs.min() - margin, zs.max() + margin)
            
            return scatters
    
        # Create a unique filename for saving the animation.
        directory = r'C:\Users\Kiran\Desktop\PBH project\PBH_sim'
        file_name = 'n_body'
        file_extension = '.gif'
        number = 1
        if not os.path.exists(directory):
            os.makedirs(directory)
        while True:
            file_path = os.path.join(directory, f"{file_name}_{number}{file_extension}")
            if not os.path.exists(file_path):
                break
            number += 1
    
        anim = FuncAnimation(fig, update, frames=positions_tensor_np.shape[0], interval=50, blit=False)
        anim.save(file_path, writer='pillow', fps=20)
        print(f"Animation saved as: {file_path}")
        plt.show()


import time

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper


@time_function
def main():
    universe = N_Bod(
        N_particles=100, 
        Time=5000, 
        dt= 1, 
        Scalar_curvature=0, 
        Initial_SF=1e-12, 
        mass_range=[1e27, 1e30],
        positionxyz_range=[-10000000, 1000000],
        velocityxyz_range=[-4, 4]
        
    )
    
