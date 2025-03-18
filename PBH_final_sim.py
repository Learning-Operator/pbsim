"Final Primordia Black Hole simulatiom"

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import os
import cupy as cp
from SF_generation import Expansion, scale_back_initial
from Particles import Particle, RadiationParticle, MassParticle




"""
    Plan:
    
    
    Completed: Completed an algorithm of which integrates over the friedmann equations.
    
    """
    
"""
    ICs: 
        - List of scale factors
        - pertubation amplitude
        - N_particles
        - Time
        - Timesteps
    (Implement co-moving coordiantes)
        - sim_radius
        - Set threshold
        
    Final_output:
        - Number of primordial black holes
        - Array of masses of primordial black holes
        - Array of positions of all particles
        - Animation and gif of primordial black hole formation
    
"""
    
class EU_formation_sim():
        def __init__(self,
                     N_particles = None,
                     List_sf = None,
                     List_sf_dot = None,
                     Start_sf = None,
                     Time = float,
                     dt = float,
                     sim_rad = None,
                     delta_c = None,
                     Background_parameters = None, #[Omega_m, Omega_r, Omega_l, Ho, sf_ref]
                     dist_type = '', # Uniform, Gaussian_RF, Scale_invarient_spectrum, Adiabatic, Isocurvature
                     Mass_power = None                     
                     ):
            
            self.N_particles = N_particles
            self.List_sf = List_sf
            self.List_sf_dot = List_sf_dot
            self.Start_sf = Start_sf
            self.Time = Time
            self.dt = dt
            self.sim_rad = sim_rad
            self.delta_c = delta_c
            self.dist_type = dist_type
            self.Mass_power = Mass_power
                                
            self.G = 6.67430e-11   # m^3 kg^-1 s^-2
            self.c = 299792458     # m/s
            self.h_bar = 1.054571917e-34  # Joule*s
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
            def calc_def_pars():
                
                print("Pulling parameters from Planck_2018_TT,TE,EE+lowE+lensing+BAO")
                default_Pars = [0.3103, 0, 0.6897, 67.66, (1/(1089.80 + 1))]
                    
                print(f"setting Background parameters: {default_Pars}")
                    
                print("Omega_r being sourced from Table 1: Lahav, O., & Liddle, A. (2004). The Cosmological Parameters.")

                O_r_h = 2.47e-5
                h = 0.73
                O_r = O_r_h * h**2
                    
                print(f"""getting Omega_r
                      
                  Omega_r = {O_r_h} * {h}^2 = {O_r}
                  
                  """)
                    
                default_Pars[1] = O_r
                default_Pars[3] = default_Pars[3]/3.0857e19
                    
                return default_Pars
            
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            
            if Background_parameters is not None:
                # Fill missing values from default parameters
                self.Pars = Background_parameters

            else:
                
                default_pars = calc_def_pars()
                self.Pars = default_pars
                
                print(f"""
                      
                      #################################################################################
                      
                      HERE ARE ALL OF THE DEFAULT PARAMETERS
                      
                                    {self.Pars}
                                    
                      #################################################################################              
                      
                      """)
                
                
            if List_sf != None:
                self.List_sf = List_sf
            else:
                
                self.List_sf, self.List_sf_dot = Expansion(Time = self.Time,
                                            dt = self.dt,
                                            Background_pars = self.Pars, #[Omega_m, Omega_r, Omega_l, Ho, sf_ref]
                                            a_target = self.Start_sf,
                                            plot = True)
            
        
        
        
        def Run_simulation(self, Set_Particles_rigidly=True, Set_particles_from_density=False):
            """Run the simulation with the specified parameters"""

            if Set_Particles_rigidly == True and Set_particles_from_density == False:
                print("Ok, so you want a set amount of particles")
                print(f"Creating {self.N_particles} particles")

                Particles_list = self.Create_particles()

            elif Set_particles_from_density == True and Set_Particles_rigidly == False:
                try:
                    Rho_r_a, Rho_m_a, Rho_de_a, initial_SF_dot = scale_back_initial(
                        self.Pars[4], self.Start_sf, self.Pars[0], self.Pars[1], self.Pars[2]
                    )

                    Particles_list = self.Create_particles_from_Pars(Rho_m=Rho_m_a, Rho_r=Rho_r_a, Rho_de=Rho_de_a)
                except Exception as e:
                    print(f"Error creating particles from density: {e}")
                    Particles_list = self.Create_particles()
            else:
                print("Invalid particle creation settings. Using default.")
                Particles_list = self.Create_particles()

            
            if self.List_sf_dot is None:
                print("Warning: List_sf_dot is None. Creating default values.")
                self.List_sf_dot = np.ones_like(self.List_sf) * 0.01

            self.simulate(
                particles=Particles_list, 
                Time=self.Time, 
                dt=self.dt, 
                scale_factors_list=self.List_sf, 
                Scale_factor_dots=self.List_sf_dot
            )

        def simulate(self, particles, Time, dt, scale_factors_list, Scale_factor_dots):
            
            Timesteps = int(Time / dt)

            Positions_tensor = cp.zeros((Timesteps, len(particles), 3))

            for i, particle in enumerate(particles):
                Positions_tensor[0, i, :] = particle.position

            for t in range(1, Timesteps):
                try:
                    self.runge_kutta_step(
                        particles, dt, scale_factors_list[t-1], Scale_factor_dots[t-1]
                    )
                except Exception as e:
                    print(f"Error at timestep {t}: {e}")
                    for i, particle in enumerate(particles):
                        particle.position = Positions_tensor[t-1, i, :]

                for i, particle in enumerate(particles):
                    particle.update_position(scale_factors_list[t], scale_factors_list[t-1])
                    Positions_tensor[t, i, :] = particle.position

                if t % 10 == 0:  
                    print(f"t = {t}/{Timesteps}")

            self.animate_positions_over_time(Positions_tensor)

        def compute_gravitational_acceleration(self, particles):
            num_particles = len(particles)
            masses = cp.array([particle.mass for particle in particles])
            positions = cp.array([particle.position for particle in particles])

            # Add softening parameter to prevent singularities
            softening = 1.0e3  # Adjust based on your simulation scale

            accelerations = cp.zeros((num_particles, 3))

            for i in range(num_particles):
                pos_diff = positions - positions[i]
                distances_squared = cp.sum(pos_diff**2, axis=1)
                distances_squared = cp.where(distances_squared > 0, distances_squared + softening**2, 1e20)

                force_magnitudes = self.G * masses / (distances_squared * cp.sqrt(distances_squared))

                # Zero out self-interaction
                force_magnitudes[i] = 0

                for j in range(3):  # For each dimension
                    accelerations[i, j] = cp.sum(force_magnitudes * pos_diff[:, j])

            return accelerations

        def runge_kutta_step(self, particles, dt, sf, sf_dot):
            num_particles = len(particles)
            positions = cp.array([p.position for p in particles])
            velocities = cp.array([p.velocity for p in particles])

            # Store the original positions and velocities
            original_positions = positions.copy()
            original_velocities = velocities.copy()

            # Calculate accelerations once at the beginning
            accelerations = self.compute_gravitational_acceleration(particles)

            # Compute k1
            k1_v = -2 * (sf_dot/sf) * original_velocities + accelerations/(sf**3)
            k1_r = original_velocities

            # Update positions and velocities for k2 calculation
            temp_positions = original_positions + 0.5 * dt * k1_r
            temp_velocities = original_velocities + 0.5 * dt * k1_v

            # Update particle positions for acceleration calculation
            for i, p in enumerate(particles):
                p.position = temp_positions[i]
                p.velocity = temp_velocities[i]

            # Calculate new accelerations
            accelerations = self.compute_gravitational_acceleration(particles)

            # Compute k2
            k2_v = -2 * (sf_dot/sf) * temp_velocities + accelerations/(sf**3)
            k2_r = temp_velocities

            # Update positions and velocities for k3 calculation
            temp_positions = original_positions + 0.5 * dt * k2_r
            temp_velocities = original_velocities + 0.5 * dt * k2_v

            for i, p in enumerate(particles):
                p.position = temp_positions[i]
                p.velocity = temp_velocities[i]

            accelerations = self.compute_gravitational_acceleration(particles)

            # Compute k3
            k3_v = -2 * (sf_dot/sf) * temp_velocities + accelerations/(sf**3)
            k3_r = temp_velocities

            # Update positions and velocities for k4 calculation
            temp_positions = original_positions + dt * k3_r
            temp_velocities = original_velocities + dt * k3_v

            for i, p in enumerate(particles):
                p.position = temp_positions[i]
                p.velocity = temp_velocities[i]

            accelerations = self.compute_gravitational_acceleration(particles)

            # Compute k4
            k4_v = -2 * (sf_dot/sf) * temp_velocities + accelerations/(sf**3)
            k4_r = temp_velocities

            # Final update
            for i, particle in enumerate(particles):
                particle.velocity = original_velocities[i] + dt / 6 * (k1_v[i] + 2 * k2_v[i] + 2 * k3_v[i] + k4_v[i])
                particle.position = original_positions[i] + dt / 6 * (k1_r[i] + 2 * k2_r[i] + 2 * k3_r[i] + k4_r[i])


        
        def animate_positions_over_time(self, positions_tensor):
            positions_tensor_np = cp.asnumpy(positions_tensor)

            # Downsample for smoother animation if needed
            downsample = max(1, positions_tensor_np.shape[0] // 100)
            positions_tensor_np = positions_tensor_np[::downsample]

            # Calculate bounds and set fixed axis limits
            x_all = positions_tensor_np[:, :, 0]
            y_all = positions_tensor_np[:, :, 1]
            z_all = positions_tensor_np[:, :, 2]

            x_min, x_max = np.min(x_all), np.max(x_all)
            y_min, y_max = np.min(y_all), np.max(y_all)
            z_min, z_max = np.min(z_all), np.max(z_all)

            margin = max((x_max-x_min), (y_max-y_min), (z_max-z_min)) * 0.1

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            actual_n_particles = positions_tensor_np.shape[1]  
            colors = np.random.rand(actual_n_particles, 3)

            scatter = ax.scatter(
                positions_tensor_np[0, :, 0],
                positions_tensor_np[0, :, 1],
                positions_tensor_np[0, :, 2],
                c=colors, 
                s=10
            )

            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_zlim(z_min - margin, z_max + margin)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("3D Position of Particles Over Time")

            time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes)

            def update(frame):
                scatter._offsets3d = (
                    positions_tensor_np[frame, :, 0],
                    positions_tensor_np[frame, :, 1],
                    positions_tensor_np[frame, :, 2]
                )
                time_text.set_text(f'Frame: {frame}/{len(positions_tensor_np)-1}')
                return [scatter, time_text]
            
            def get_unique_filename(directory, base_name, extension=".gif"):
                """Ensure the file doesn't overwrite an existing one by appending a number if needed."""
                counter = 1
                file_path = os.path.join(directory, f"{base_name}{extension}")

                while os.path.exists(file_path):
                    file_path = os.path.join(directory, f"{base_name}_{counter}{extension}")
                    counter += 1

                return file_path


            anim = FuncAnimation(fig, update, frames=len(positions_tensor_np), interval=100, blit=True)

            directory = r'C:\Users\Kiran\Desktop\PBh\gifs'
            os.makedirs(directory, exist_ok=True)
            file_path = get_unique_filename(directory, "n_body_fixed", ".gif")

            try:
                anim.save(file_path, writer='pillow', fps=10)
                print(f"Animation saved as: {file_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
                plt.savefig(os.path.join(directory, "n_body_fixed_first_frame.png"))

            plt.show()
        
        
            
                
            
        def Create_particles_from_Pars(self, Rho_m, Rho_r, Rho_de):
            """Create particles based on density parameters"""
            volume = (4/3) * np.pi * self.sim_rad**3

            total_mass = Rho_m * volume
            total_rad = Rho_r * volume

            if self.Mass_power is not None:
                mass_per_particle = self.Mass_power
                N_mparticles = (int(total_mass / mass_per_particle),)
            else:
                N_particles = (10000,)

            print(f"Creating {N_particles} particles based on density parameters")

            # Call the appropriate distribution generator
            return self.Create_particles()
            
    
        def Create_particles(self):
            """Creates particles based on the specified distribution type"""
    
            if self.dist_type == 'Uniform':
                return self.generate_uniform_distribution(self.N_particles)
            elif self.dist_type == 'Gaussian_RF':
                return self.generate_gaussian_RF(self.N_particles)
            elif self.dist_type == 'Scale_invarient_spectrum':
                return self.generate_scale_invariant_spectrum(self.N_particles)
            
            elif self.dist_type == 'Adiabatic':
                positions, velocities = self.generate_adiabatic_perturbations(self.N_particles)
                return self.particles_from_arrays(positions, velocities)
            elif self.dist_type == 'Isocurvature':
                positions, particle_types = self.generate_isocurvature_perturbations(self.N_particles)
                return self.particles_from_isocurvature(positions, particle_types)
            else:
                print(f"Warning: Distribution type '{self.dist_type}' not recognized. Using Uniform distribution.")
                return self.generate_uniform_distribution(self.N_particles)


        def generate_uniform_distribution(self, N_particles):
            """Generate particles with uniform distribution within a sphere"""
            particles = []

            volume = (4/3) * np.pi * self.sim_rad**3

            critical_density = 3 * (self.Pars[3])**2 / (8 * np.pi * self.G)
            matter_density = self.Pars[0] * critical_density
            radiation_density = self.Pars[1] * critical_density

            total_matter_mass = matter_density * volume
            total_radiation_energy = radiation_density * volume

            total_particles = N_particles
            matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
            n_matter_particles = int(total_particles * matter_fraction)
            n_radiation_particles = total_particles - n_matter_particles

            mass_per_matter_particle = total_matter_mass / n_matter_particles

            # Generate spherical coordinates for better uniformity
            for _ in range(n_matter_particles):
                # Generate random points in spherical coordinates
                r = self.sim_rad * np.cbrt(np.random.random())  # Cube root for uniform volume distribution
                theta = np.arccos(2 * np.random.random() - 1)  # Uniform in cos(theta)
                phi = 2 * np.pi * np.random.random()

                # Convert to Cartesian
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)

                # Slower initial velocities for stability
                velocity = cp.array([
                    np.random.normal(0, 10),  # Reduced velocity scale
                    np.random.normal(0, 10), 
                    np.random.normal(0, 10)
                ])

                particles.append(MassParticle(mass_per_matter_particle, cp.array([x, y, z]), velocity))

            # Similar improvements for radiation particles...

            return particles

        def apply_mass_power_spectrum(self, positions, base_masses):
            """Apply mass power spectrum to a set of base particle masses"""
            n = int(np.cbrt(len(positions)))  
            grid_size = 2 * self.sim_rad

            kx = 2 * np.pi * np.fft.fftfreq(n, d=grid_size/n)
            ky = 2 * np.pi * np.fft.fftfreq(n, d=grid_size/n)
            kz = 2 * np.pi * np.fft.fftfreq(n, d=grid_size/n)

            kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
            k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

            # Avoid division by zero
            k_mag[k_mag == 0] = 1e-10


            n_s = 1.0  # Spectral index
            k_cutoff = 10.0 / self.sim_rad  # Cutoff scale

            # Implement mass power spectrum
            k_pivot = 0.05 / self.sim_rad  # Pivot scale
            power_spectrum = (k_mag/k_pivot)**(n_s-1) * np.exp(-k_mag**2/k_cutoff**2)

            random_phases = np.random.uniform(0, 2*np.pi, k_mag.shape)

            fourier_coeff = np.sqrt(power_spectrum) * np.exp(1j * random_phases)

            density_field = np.fft.ifftn(fourier_coeff).real

            perturbation_amplitude = 0.1  # Can be adjusted
            density_field = 1.0 + perturbation_amplitude * (density_field - np.mean(density_field)) / np.std(density_field)

            # Map particles to density field
            modulated_masses = np.zeros(len(positions))
            grid_indices = []

            # For each particle, find its position in the grid
            for i, pos in enumerate(positions):
                # Convert position to grid index
                ix = int((pos[0] + self.sim_rad) * n / (2 * self.sim_rad))
                iy = int((pos[1] + self.sim_rad) * n / (2 * self.sim_rad))
                iz = int((pos[2] + self.sim_rad) * n / (2 * self.sim_rad))

                ix = max(0, min(ix, n-1))
                iy = max(0, min(iy, n-1))
                iz = max(0, min(iz, n-1))

                grid_indices.append((ix, iy, iz))

            # Apply density field to masses
            for i, (ix, iy, iz) in enumerate(grid_indices):
                modulated_masses[i] = base_masses[i] * density_field[ix, iy, iz]

            return modulated_masses

        def generate_gaussian_RF(self, N_particles):
            """Generate particles with positions following a Gaussian random field"""
            n = int(np.cbrt(N_particles))  # Cubic root to get grid size
            x = np.linspace(-self.sim_rad, self.sim_rad, n)
            y = np.linspace(-self.sim_rad, self.sim_rad, n)
            z = np.linspace(-self.sim_rad, self.sim_rad, n)

            field = np.zeros((n, n, n))
            k_space = np.random.normal(0, 1, (n, n, n)) + 1j * np.random.normal(0, 1, (n, n, n))

            k_x, k_y, k_z = np.meshgrid(*[np.fft.fftfreq(n, d=2*self.sim_rad/n) for _ in range(3)])
            k_mag = np.sqrt(k_x**2 + k_y**2 + k_z**2)
            k_mag[0, 0, 0] = 1  # Avoid division by zero

            spectral_index = -1 
            power_spectrum = k_mag ** spectral_index
            power_spectrum[0, 0, 0] = 0  

            k_space *= np.sqrt(power_spectrum)

            field = np.fft.ifftn(k_space).real

            field = (field - np.mean(field)) / np.std(field)

            particles = []
            X, Y, Z = np.meshgrid(x, y, z)

            mask = X**2 + Y**2 + Z**2 <= self.sim_rad**2
            X_valid = X[mask]
            Y_valid = Y[mask]
            Z_valid = Z[mask]
            field_valid = field[mask]


            prob = np.exp(field_valid)
            prob = prob / np.sum(prob)

            indices = np.random.choice(len(X_valid), size=N_particles, p=prob)

            positions = np.array([[X_valid[i], Y_valid[i], Z_valid[i]] for i in indices])

            matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
            n_matter_particles = int(N_particles * matter_fraction)
            n_radiation_particles = N_particles - n_matter_particles

            volume = (4/3) * np.pi * self.sim_rad**3
            critical_density = 3 * (self.Pars[3])**2 / (8 * np.pi * self.G)
            matter_density = self.Pars[0] * critical_density
            radiation_density = self.Pars[1] * critical_density

            total_matter_mass = matter_density * volume
            total_radiation_energy = radiation_density * volume

            matter_positions = positions[:n_matter_particles]
            radiation_positions = positions[n_matter_particles:]

            base_matter_masses = np.ones(n_matter_particles) * (total_matter_mass / n_matter_particles)
            base_radiation_masses = np.ones(n_radiation_particles) * (total_radiation_energy / (n_radiation_particles * self.c**2))

            modulated_matter_masses = self.apply_mass_power_spectrum(matter_positions, base_matter_masses)

            for i in range(n_matter_particles):
                pos = cp.array(matter_positions[i])
                vel = cp.array([np.random.normal(0, 100), np.random.normal(0, 100), np.random.normal(0, 100)])
                particles.append(MassParticle(modulated_matter_masses[i], pos, vel))

            for i in range(n_radiation_particles):
                pos = cp.array(radiation_positions[i])
                vel = cp.array([np.random.normal(0, 1), np.random.normal(0, 1), np.random.normal(0, 1)])
                particles.append(RadiationParticle(base_radiation_masses[i], pos, vel))

            return particles

        def generate_adiabatic_perturbations(self, N_particles):
            """Generate adiabatic perturbations where all components fluctuate together"""
           
            positions = []
            for _ in range(N_particles):
                # Generate random position in a sphere
                while True:
                    x = (2 * np.random.random() - 1) * self.sim_rad
                    y = (2 * np.random.random() - 1) * self.sim_rad
                    z = (2 * np.random.random() - 1) * self.sim_rad
                    if x**2 + y**2 + z**2 <= self.sim_rad**2:
                        break
                positions.append([x, y, z])

            positions = np.array(positions)

            matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
            n_matter_particles = int(N_particles * matter_fraction)
            n_radiation_particles = N_particles - n_matter_particles

            n = int(np.cbrt(N_particles))  # Grid size

            # Create k-space grid
            kx = 2 * np.pi * np.fft.fftfreq(n, d=2*self.sim_rad/n)
            ky = 2 * np.pi * np.fft.fftfreq(n, d=2*self.sim_rad/n)
            kz = 2 * np.pi * np.fft.fftfreq(n, d=2*self.sim_rad/n)

            kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
            k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

            k_mag[k_mag == 0] = 1e-10

            n_s = 0.965  # Planck 2018 value

            power_spectrum = k_mag**n_s

            random_phases = np.random.uniform(0, 2*np.pi, k_mag.shape)

            fourier_coeff = np.sqrt(power_spectrum) * np.exp(1j * random_phases)

            density_field = np.fft.ifftn(fourier_coeff).real

            perturbation_amplitude = 0.01  # Small for early universe
            density_field = 1.0 + perturbation_amplitude * (density_field - np.mean(density_field)) / np.std(density_field)

            particle_densities = np.zeros(len(positions))
            for i, pos in enumerate(positions):
                # Convert position to grid index
                ix = int((pos[0] + self.sim_rad) * n / (2 * self.sim_rad))
                iy = int((pos[1] + self.sim_rad) * n / (2 * self.sim_rad))
                iz = int((pos[2] + self.sim_rad) * n / (2 * self.sim_rad))

                # Clamp indices to valid range
                ix = max(0, min(ix, n-1))
                iy = max(0, min(iy, n-1))
                iz = max(0, min(iz, n-1))

                particle_densities[i] = density_field[ix, iy, iz]

            velocities = np.zeros_like(positions)

            # Calculate base masses
            volume = (4/3) * np.pi * self.sim_rad**3
            critical_density = 3 * (self.Pars[3])**2 / (8 * np.pi * G)
            matter_density = self.Pars[0] * critical_density
            radiation_density = self.Pars[1] * critical_density

            total_matter_mass = matter_density * volume
            total_radiation_energy = radiation_density * volume

            # Base masses
            base_matter_mass = total_matter_mass / n_matter_particles
            base_radiation_mass = total_radiation_energy / (n_radiation_particles * c**2)

            # Create particles
            particles = []

            for i in range(n_matter_particles):
                pos = cp.array(positions[i])

                hubble_param = self.Pars[3] * np.sqrt(self.Pars[0]/self.Start_sf[0]**3 + self.Pars[1]/self.Start_sf[0]**4 + self.Pars[2])
                peculiar_velocity = np.random.normal(0, 100, 3) * (particle_densities[i] - 1.0)
                vel = cp.array(peculiar_velocity)

                mass = base_matter_mass * particle_densities[i]

                particles.append(MassParticle(mass, pos, vel))

            # Radiation particles (remaining particles)
            for i in range(n_matter_particles, N_particles):
                pos = cp.array(positions[i])
                # Random direction at speed of light
                vel_direction = np.random.normal(0, 1, 3)
                vel_direction = vel_direction / np.linalg.norm(vel_direction)
                vel = cp.array(vel_direction * c)

                # Mass modulated by density (for radiation, this affects number density)
                mass = base_radiation_mass * particle_densities[i]

                particles.append(RadiationParticle(mass, pos, vel))

            return particles


        def generate_scale_invariant_spectrum(self):
            """Generate particles with positions following a scale-invariant power spectrum P(k) ∝ k^1"""
            n = int(np.cbrt(self.N_particles))  # Cubic root to get grid size
            x = np.linspace(-self.sim_rad, self.sim_rad, n)
            y = np.linspace(-self.sim_rad, self.sim_rad, n)
            z = np.linspace(-self.sim_rad, self.sim_rad, n)

            # Create a 3D random field
            k_space = np.random.normal(0, 1, (n, n, n)) + 1j * np.random.normal(0, 1, (n, n, n))

            # Create k-space grid
            kx = np.fft.fftfreq(n, d=2*self.sim_rad/n)
            ky = np.fft.fftfreq(n, d=2*self.sim_rad/n)
            kz = np.fft.fftfreq(n, d=2*self.sim_rad/n)

            kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
            k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

            # Avoid division by zero at k=0
            k_mag[0, 0, 0] = 1e-10

            power_spectrum = k_mag.copy()

            power_spectrum[0, 0, 0] = 0

            k_space *= np.sqrt(power_spectrum)

            density_field = np.fft.ifftn(k_space).real

            density_field = (density_field - np.mean(density_field)) / np.std(density_field)

            perturbation_amplitude = 0.01  # Typical for early universe
            density_contrast = 1.0 + perturbation_amplitude * density_field

            # Create particles
            particles = []

            # Get positions from the density field
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

            # Keep only points within the sphere
            mask = X**2 + Y**2 + Z**2 <= self.sim_rad**2
            X_valid = X[mask]
            Y_valid = Y[mask]
            Z_valid = Z[mask]
            density_valid = density_contrast[mask]

            prob = np.maximum(density_valid, 0)  # Ensure non-negative probabilities
            prob = prob / np.sum(prob)

            indices = np.random.choice(len(X_valid), size=self.N_particles, p=prob)

            positions = np.array([[X_valid[i], Y_valid[i], Z_valid[i]] for i in indices])
            densities = np.array([density_valid[i] for i in indices])

            matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
            n_matter_particles = int(self.N_particles * matter_fraction)
            n_radiation_particles = self.N_particles - n_matter_particles

            volume = (4/3) * np.pi * self.sim_rad**3
            critical_density = 3 * (self.Pars[3])**2 / (8 * np.pi * G)
            matter_density = self.Pars[0] * critical_density
            radiation_density = self.Pars[1] * critical_density

            total_matter_mass = matter_density * volume
            total_radiation_energy = radiation_density * volume

            # Create matter particles
            for i in range(n_matter_particles):
                pos = cp.array(positions[i])

                hubble_param = self.Pars[3] * np.sqrt(self.Pars[0]/self.Start_sf[0]**3 + self.Pars[1]/self.Start_sf[0]**4 + self.Pars[2])
                peculiar_velocity = np.random.normal(0, 100, 3) * (densities[i] - 1.0)
                vel = cp.array(peculiar_velocity)

                mass = (total_matter_mass / n_matter_particles) * densities[i]

                particles.append(MassParticle(mass, pos, vel))

            for i in range(n_matter_particles, self.N_particles):
                pos = cp.array(positions[i])

                vel_direction = np.random.normal(0, 1, 3)
                vel_direction = vel_direction / np.linalg.norm(vel_direction)
                vel = cp.array(vel_direction * c)

                # "Effective mass" for radiation (E = mc²)
                mass = (total_radiation_energy / (n_radiation_particles * c**2)) * densities[i]

                particles.append(RadiationParticle(mass, pos, vel))

            return particles

        def generate_isocurvature_perturbations(self):
            """Generate isocurvature perturbations where component ratios vary but total energy density is constant"""
            n = int(np.cbrt(self.N_particles))  # Cubic root to get grid size
            x = np.linspace(-self.sim_rad, self.sim_rad, n)
            y = np.linspace(-self.sim_rad, self.sim_rad, n)
            z = np.linspace(-self.sim_rad, self.sim_rad, n)

           
            k_space_matter = np.random.normal(0, 1, (n, n, n)) + 1j * np.random.normal(0, 1, (n, n, n))
            k_space_radiation = np.random.normal(0, 1, (n, n, n)) + 1j * np.random.normal(0, 1, (n, n, n))

            kx = np.fft.fftfreq(n, d=2*self.sim_rad/n)
            ky = np.fft.fftfreq(n, d=2*self.sim_rad/n)
            kz = np.fft.fftfreq(n, d=2*self.sim_rad/n)

            kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
            k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

            k_mag[0, 0, 0] = 1e-10

            n_iso = 2.0  

            power_spectrum = k_mag**n_iso
            power_spectrum[0, 0, 0] = 0  # No DC component

            k_space_matter *= np.sqrt(power_spectrum)
            k_space_radiation *= np.sqrt(power_spectrum)

            density_field_matter = np.fft.ifftn(k_space_matter).real
            density_field_radiation = np.fft.ifftn(k_space_radiation).real

            density_field_matter = (density_field_matter - np.mean(density_field_matter)) / np.std(density_field_matter)
            density_field_radiation = (density_field_radiation - np.mean(density_field_radiation)) / np.std(density_field_radiation)


            critical_density = 3 * (self.Pars[3])**2 / (8 * np.pi * self.G)
            matter_density = self.Pars[0] * critical_density
            radiation_density = self.Pars[1] * critical_density

            # Balance matter and radiation perturbations
            total_energy_density = matter_density + radiation_density
            matter_weight = matter_density / total_energy_density
            radiation_weight = radiation_density / total_energy_density

            combined_field = density_field_matter * matter_weight - density_field_radiation * radiation_weight

            iso_amplitude = 0.01  # Typical for early universe


            matter_contrast = 1.0 + iso_amplitude * combined_field
            radiation_contrast = 1.0 - iso_amplitude * combined_field * (matter_weight/radiation_weight)

            particles = []

            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            mask = X**2 + Y**2 + Z**2 <= self.sim_rad**2

            X_valid = X[mask]
            Y_valid = Y[mask]
            Z_valid = Z[mask]


            
            matter_contrast_valid = matter_contrast[mask]
            radiation_contrast_valid = radiation_contrast[mask]

            matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
            n_matter_particles = int(self.N_particles * matter_fraction)
            n_radiation_particles = self.N_particles - n_matter_particles

            matter_prob = np.maximum(matter_contrast_valid, 0)
            matter_prob = matter_prob / np.sum(matter_prob)
            matter_indices = np.random.choice(len(X_valid), size=n_matter_particles, p=matter_prob)

            radiation_prob = np.maximum(radiation_contrast_valid, 0)
            radiation_prob = radiation_prob / np.sum(radiation_prob)
            radiation_indices = np.random.choice(len(X_valid), size=n_radiation_particles, p=radiation_prob)

            volume = (4/3) * np.pi * self.sim_rad**3
            total_matter_mass = matter_density * volume
            total_radiation_energy = radiation_density * volume

            for i, idx in enumerate(matter_indices):
                pos = cp.array([X_valid[idx], Y_valid[idx], Z_valid[idx]])

                peculiar_velocity = np.random.normal(0, 100, 3) * (matter_contrast_valid[idx] - 1.0)
                vel = cp.array(peculiar_velocity)

                mass = (total_matter_mass / n_matter_particles) * matter_contrast_valid[idx]

                particles.append(MassParticle(mass, pos, vel))

            for i, idx in enumerate(radiation_indices):
                pos = cp.array([X_valid[idx], Y_valid[idx], Z_valid[idx]])

                vel_direction = np.random.normal(0, 1, 3)
                vel_direction = vel_direction / np.linalg.norm(vel_direction)
                vel = cp.array(vel_direction * self.c)

                mass = (total_radiation_energy / (n_radiation_particles * self.c**2)) * radiation_contrast_valid[idx]

                particles.append(RadiationParticle(mass, pos, vel))

            return particles    





sim = EU_formation_sim(
    N_particles=500,
    Start_sf=1e-12,
    Time=2000000000000.0,
    dt=100000000000,  # Reduce time step for better stability
    sim_rad=1.0e6,
    delta_c=0.45,
    dist_type='Gaussian_RF'
)
sim.Run_simulation(Set_Particles_rigidly=True)
    
    