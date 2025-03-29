"Final Primordia Black Hole simulatiom"

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import os
import cupy as cp
from functions.SF_generation import Expansion, scale_back_initial
from functions.File_naming import get_unique_filename
from functions.Particles import Particle, RadiationParticle, MassParticle
from functions.Distributions import generate_uniform_distribution, \
                          generate_gaussian_RF, \
                          generate_scale_invariant_spectrum, \
                          generate_adiabatic_perturbations, \
                          generate_isocurvature_perturbations #, \
                          #particles_from_arrays, \
                          #particles_from_isocurvature
  
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
                     N_Matter_particles = None,
                     N_Radiation_particles = None,
                     List_sf = None,
                     List_sf_dot = None,
                     Start_sf = None,
                     Time = float,
                     dt = float,
                     sim_rad = None,
                     delta_c = None,
                     Background_parameters = None, #[Omega_m, Omega_r, Omega_l, Ho, sf_ref]
                     dist_type = '', # Uniform, Gaussian_RF, Scale_invarient_spectrum, Adiabatic, Isocurvature
                     Mass_power = None,
                     softener = None               
                     ):
            
            self.N_Matter_particles = N_Matter_particles
            self.N_Radiation_particles = N_Radiation_particles
            self.List_sf = List_sf
            self.List_sf_dot = List_sf_dot
            self.Start_sf = Start_sf
            self.Time = Time
            self.dt = dt
            self.sim_rad = sim_rad
            self.delta_c = delta_c
            self.dist_type = dist_type
            self.Mass_power = Mass_power
            self.softener = softener     
                         
            self.G = 6.67430e-11   # m^3 kg^-1 s^-2
            self.c = 299792458     # m/s
            self.h_bar = 1.054571917e-34  # Joule*s
            
            self.output_dir = 'PBsim\\output'
            os.makedirs(self.output_dir , exist_ok= True)
                    
            def calc_def_pars(): # Remove this later and change it
                
                print("Pulling parameters from Planck_2018_TT,TE,EE+lowE+lensing+BAO")
                print(f"Parameters being sourced: [Omega_m, Omega_r, Omega_l, Ho, sf_ref]")
                default_Pars = [0.3103, 0, 0.6897, 67.66, (1/(1089.80 + 1))]
                    
                print("Omega_r is being sourced from Table 1: Lahav, O., & Liddle, A. (2004). The Cosmological Parameters.")

                O_r_h = 2.47e-5
                h = 0.73
                O_r = O_r_h * h**2
                    
                default_Pars[1] = O_r
                default_Pars[3] = default_Pars[3]/3.0857e19
                    
                return default_Pars
                        
            if Background_parameters is not None:
                # Fill missing values from default parameters
                self.Pars = Background_parameters
            else: 
                default_pars = calc_def_pars() # change this, utilize a Rag agent.
                self.Pars = default_pars
                print(f" Parameters: {self.Pars} ")
                    
            if List_sf != None:
                self.List_sf = List_sf
            else:
                self.List_sf, self.List_sf_dot = Expansion(Time = self.Time,
                                            dt = self.dt,
                                            Background_pars = self.Pars, #[Omega_m, Omega_r, Omega_l, Ho, sf_ref]
                                            a_target = self.Start_sf,
                                            plot = True,
                                            dir= self.output_dir)
            
        
        
        
        def Run_simulation(self, Set_Particles_rigidly=True, Set_particles_from_density=False):
            """Run the simulation with the specified parameters"""

            if Set_Particles_rigidly == True and Set_particles_from_density == False:
                print("Ok, so you want a set amount of particles")
                #print(f"Creating {self.N_particles} particles")

                self.Particles_list = self.Create_particles()

            elif Set_particles_from_density == True and Set_Particles_rigidly == False:
                try:
                    Rho_r_a, Rho_m_a, Rho_de_a, initial_SF_dot = scale_back_initial( self.Pars[4], self.Start_sf, self.Pars[0], self.Pars[1], self.Pars[2])

                    self.Particles_list = self.Create_particles_from_Pars(Rho_m=Rho_m_a, Rho_r=Rho_r_a, Rho_de=Rho_de_a)
                    
                except Exception as e:
                    print(f"Error creating particles from density: {e}")
                    self.Particles_list = self.Create_particles()
            else:
                print("Invalid particle creation settings. Using default.")
                self.Particles_list = self.Create_particles()

            self.simulate(
                particles=self.Particles_list, 
                Time=self.Time, 
                dt=self.dt, 
                pars = self.Pars,
                scale_factors_list=self.List_sf, 
                Scale_factor_dots=self.List_sf_dot
            )

        def simulate(self, particles, Time, dt, scale_factors_list, Scale_factor_dots, pars):
            
            Timesteps = int(Time / dt)

            Positions_tensor = cp.zeros((Timesteps, len(particles), 3))

            for i, particle in enumerate(particles):
                Positions_tensor[0, i, :] = particle.position

            for t in range(1, int(Time)):
                try:
                    self.runge_kutta_step( particles, dt, scale_factors_list[t-1], Scale_factor_dots[t-1])
                    
                except Exception as e:
                    print(f"Error at timestep {t}: {e}")
                    break

                for i, particle in enumerate(particles):
                    particle.update_position(scale_factors_list[t-1], scale_factors_list[t-1])
                    Positions_tensor[t-1, i, :] = particle.position
                
                print(f"t = {t}")

            self.animate_positions_over_time(Positions_tensor)
        

            
        def Create_particles_from_Pars(self, Rho_m, Rho_r, Rho_de, N_Rad_Particles, N_Mass_Particles):
            """Create particles based on density parameters"""
            volume = (4/3) * np.pi * self.sim_rad**3



            total_mass = Rho_m * volume
            total_rad = Rho_r * volume
            
            
            if N_Rad_Particles is not None:
                Rad_per_Particles = total_rad/N_Rad_Particles  
            else:
                N_radparticles = (10000,)
                Rad_per_Particles = total_rad/N_Rad_Particles  
        
                
            if N_Mass_Particles is not None:
                Mass_Per_particle = total_mass/N_Mass_Particles
            else:
                N_mparticles = (10000,)
                Mass_Per_particle = total_mass/N_Mass_Particles
            

            # Call the appropriate distribution generator
            return self.Create_particles()
            
    
        def Create_particles(self):
            
            """Creates particles based on the specified distribution type"""
    
            if self.dist_type == 'Uniform' or self.dist_type == "uniform":
                return generate_uniform_distribution(N_matter_particles = self.N_Matter_particles,N_radiation_particles = self.N_Radiation_particles, Pars= self.Pars, sim_rad= self.sim_rad, mass_velocity_range= [-100,100], output_fold = self.output_dir) # This needs to be user set!!!!
            elif self.dist_type == 'Gaussian_RF':
                return generate_gaussian_RF(N_Matter_particles=self.N_Matter_particles,N_Radiation_particles=self.N_Radiation_particles, Pars=self.Pars, sim_rad= self.sim_rad, output_fold= self.output_dir)
            elif self.dist_type == 'Scale_invarient_spectrum':
                return generate_scale_invariant_spectrum(N_particles= self.N_particles, sim_rad= self.sim_rad, Pars=self.Pars, output_fold = self.output_dir)
            
            #elif self.dist_type == 'Adiabatic':
            #    positions, velocities = generate_adiabatic_perturbations(N_particles=self.N_particles, Pars= self.Pars, sim_rad=self.sim_rad,Sf_start=self.List_sf[0])
            #    return particles_from_arrays(positions, velocities)
            
            #elif self.dist_type == 'Isocurvature':
            #    positions, particle_types = generate_isocurvature_perturbations(self.N_particles, Pars= self.Pars, sim_rad=self.sim_rad)
            #    return particles_from_isocurvature(positions, particle_types)
            
            else:
                print(f"Warning: Distribution type '{self.dist_type}' not recognized. Using Uniform distribution.")
                return generate_uniform_distribution(self.N_particles, Pars= self.Pars, sim_rad= self.sim_rad, mass_velocity_range= [-100,100], output_fold = self.output_dir) # This needs to be user set!!!!



        def compute_gravitational_acceleration(self,num_particles, positions, masses):             
            
            #print(f"Number of particles: {num_particles}")
            

            #print(f"Masses shape: {masses.shape}")
            #print(f"Positions shape: {positions.shape}")

            softening = self.softener  # how can i dynamically tweak this based on number of particles and sim size?

            accelerations = cp.zeros((num_particles, 3))

            for i in range(num_particles):
                
                pos_diff = positions - positions[i]
                
                #print(f"pos_diff shape:{pos_diff.shape}")
                
                mask = cp.ones(len(pos_diff), dtype=bool)
                mask[i] = False
            
                
                distances_squared = cp.sum(pos_diff[mask]**2, axis=1) + softening**2
                                
                #print(f"after sum of squared value shape: {distances_squared.shape}")
                
                epsilon = 1e-10
                distances_squared = cp.maximum(distances_squared, epsilon)
                
                inv_distances_cubed = 1.0 / (distances_squared**1.5)
                
                #print(f"position dif {pos_diff}")
                
                #print(f"masked poisition difference: {pos_diff[mask]}")
                
                
                acc_components = (
                     masses[mask] * inv_distances_cubed[:, cp.newaxis] * pos_diff[mask]
                )
                
                #print(f"acc components {acc_components}")
                
                accelerations[i] = -self.G * cp.sum(acc_components, axis=0)



            return accelerations


        def runge_kutta_step(self, particles, dt, sf, sf_dot):
        
            num_particles = len(particles)
            masses = cp.array([p.mass for p in particles])[:, cp.newaxis]  # Ensure correct shape
            positions = cp.array([p.position for p in particles])
            velocities = cp.array([p.velocity for p in particles])

            def acceleration(position, masses):
                return self.compute_gravitational_acceleration(num_particles=num_particles, positions=position, masses=masses)

            # Compute RK4 coefficients
            k1_v = dt * acceleration(positions, masses)
            k1_r = dt * velocities

            k2_v = dt * acceleration(positions + 0.5 * k1_r, masses)
            k2_r = dt * (velocities + 0.5 * k1_v)

            k3_v = dt * acceleration(positions + 0.5 * k2_r, masses)
            k3_r = dt * (velocities + 0.5 * k2_v)

            k4_v = dt * acceleration(positions + k3_r, masses)
            k4_r = dt * (velocities + k3_v)

            # Update positions and velocities
            new_positions = positions + (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6
            new_velocities = velocities + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6

            # Update particle attributes
            for i, p in enumerate(particles):
                p.position = new_positions[i]
                p.velocity = new_velocities[i]
            

                

            # original_positions = positions.copy()
            # original_velocities = velocities.copy()

            
            # accelerations = self.compute_gravitational_acceleration(particles, sf)
            
            # k1_v = accelerations
            # k1_r = original_velocities

            # temp_positions = original_positions + 0.5 * dt *  k1_r
            # temp_velocities = original_velocities + 0.5 * dt * k1_v

            # for i, p in enumerate(particles):
            #     p.position = temp_positions[i]
            #     p.velocity = temp_velocities[i]


            # accelerations = self.compute_gravitational_acceleration(particles, sf)
            
            # k2_v = accelerations
            # k2_r = temp_velocities

            # # Update positions and velocities for k3 calculation
            # temp_positions = original_positions + 0.5 * dt * k2_r
            # temp_velocities = original_velocities + 0.5 * dt * k2_v

            # for i, p in enumerate(particles):
            #     p.position = temp_positions[i]
            #     p.velocity = temp_velocities[i]

            # accelerations = self.compute_gravitational_acceleration(particles, sf)

            # # Compute k3
            # k3_v = accelerations
            # k3_r = temp_velocities

            # # Update positions and velocities for k4 calculation
            # temp_positions = original_positions + dt * k3_r
            # temp_velocities = original_velocities + dt * k3_v

            # for i, p in enumerate(particles):
            #     p.position = temp_positions[i]
            #     p.velocity = temp_velocities[i]

            # accelerations = self.compute_gravitational_acceleration(particles, sf)

            # # Compute k4
            # k4_v = accelerations
            # k4_r = temp_velocities

            # # Final update
            # for i, particle in enumerate(particles):
                
            #     particle.velocity = original_velocities[i] + dt / 6 * (k1_v[i] + 2 * k2_v[i] + 2 * k3_v[i] + k4_v[i])
                
            #     particle.position = original_positions[i] + dt / 6 * (k1_r[i] + 2 * k2_r[i] + 2 * k3_r[i] + k4_r[i])
                
                 #print(particle.position)
                


        
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

            colors = []
            for particle in self.Particles_list:  
                if isinstance(particle, MassParticle):
                    colors.append([0, 0, 1])  # Blue for mass particles
                elif isinstance(particle, RadiationParticle):
                    colors.append([1, 0, 0])  # Red for radiation particles
                else:
                    colors.append([0, 1, 0])  # Green for any other type (fallback)
            
            colors = np.array(colors)
        
            scatter = ax.scatter(
                positions_tensor_np[0, :, 0],
                positions_tensor_np[0, :, 1],
                positions_tensor_np[0, :, 2],
                c=colors, 
                s=3
            )

            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_zlim(z_min - margin, z_max + margin)

            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")
            ax.set_zlabel("Z (meters)")
            ax.set_title("3D Position of Particles Over Time")

            time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes)

            def update(frame):
                scatter._offsets3d = (
                    positions_tensor_np[frame, :, 0],
                    positions_tensor_np[frame, :, 1],
                    positions_tensor_np[frame, :, 2]
                )
                #ax.view_init(elev=30, azim=frame)  # Adjust elevation and increment as needed
                time_text.set_text(f'Frame: {frame}/{len(positions_tensor_np)-1}')
                return [scatter, time_text]
            

            anim = FuncAnimation(fig, update, frames=len(positions_tensor_np), interval=100, blit=False)
                            
            file_path = get_unique_filename(output_folder = self.output_dir, output_type = 'gif', filename = "n_body", file_type = ".gif")

            try:
                anim.save(file_path, writer='pillow', fps=10)
                print(f"Animation saved as: {file_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
                plt.savefig(os.path.join(file_path, "n_body_fixed_first_frame.png"))

            plt.show()



sim = EU_formation_sim(
    N_Matter_particles=3000,
    N_Radiation_particles=1,
    softener = 1000, # 
    Start_sf=1,
    Time=30.0,
    dt=  1,  # Reduce time step for better stability
    sim_rad=10000000000.0,
    delta_c=0.45,
    dist_type='Gaussian_RF',
    
)
sim.Run_simulation(Set_Particles_rigidly=True)
 
 
 
 #Todo:
    #- implementing co-moving coordinates.
    #- incorporate aspect of CMBagent or some sort of agent code to create a RAG agent 
        #(This is to source parameters, so i can simply prompt it to get the appropriate parameter values)
        #  Analytics?
        #  possibly smooth it and try to incorporate other types fo parameters and values

    # Find a way to tweak softening param in compute grav
    # Create a PBH formation check
    # Develop agents for resutls analysis
 
 
    
    