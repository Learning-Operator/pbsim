"Final Primordia Black Hole simulatiom"

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import os
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
                     delta_c = None,
                     Background_parameters = None, #[Omega_m, Omega_r, Omega_l, Ho, sf_ref]
                     dist_type = '', # Uniform, Gaussian_RF, Scale_invarient_spectrum, Adiabatic, Isocurvature
                     Mass_power = None,
                     softener = None,
                     ):
            
            self.N_Matter_particles = N_Matter_particles
            self.N_Radiation_particles = N_Radiation_particles
            self.List_sf = List_sf
            self.List_sf_dot = List_sf_dot
            self.Start_sf = Start_sf
            self.Time = Time
            self.dt = dt
            self.delta_c = delta_c
            self.dist_type = dist_type
            self.Mass_power = Mass_power
            self.softener = softener     
                         
            self.G = 6.67430e-11   # m^3 kg^-1 s^-2
            self.c = 299792458     # m/s
            
            self.output_dir = 'PBsim\\output'
            os.makedirs(self.output_dir , exist_ok= True)
                    
            def calc_def_pars(): # Remove this later and change it
                            #[Omega_m, Omega_r, Omega_l, Ho, sf_ref]
                default_Pars = [0.3103, 0, 0.6897, 67.66, (1/(1089.80 + 1))] # All except 0, is from the Planck collab
                O_r_h = 2.47e-5 # Being sourced from Table 1: Lahav, O., & Liddle, A. (2004). The Cosmological Parameters.
                h = 0.73  # being sourced from Table 1: Lahav, O., & Liddle, A. (2004). The Cosmological Parameters.
                O_r = O_r_h * h**2
                default_Pars[1] = O_r # unitless
                default_Pars[3] = default_Pars[3] * 1/3.0857e20 # (Meters/Km)/(Meters/MPC)
                    
                return default_Pars
                        
            if Background_parameters is not None:
                self.Pars = Background_parameters
            else: 
                default_pars = calc_def_pars() # change this, utilize a Rag agent.
                self.Pars = default_pars
                
                    
            if List_sf != None:
                self.List_sf = List_sf
            else:
                self.List_sf, self.List_sf_dot, Hubbles = Expansion(Time = self.Time, # s
                                            dt = self.dt, # s
                                            Background_pars = self.Pars, #[Omega_m, Omega_r, Omega_l, Ho, sf_ref]
                                            a_target = self.Start_sf, # unitless
                                            plot = True,
                                            dir= self.output_dir)
                
                
            Rho_r_a, Rho_m_a, Rho_de_a, _, H_a  = scale_back_initial( self.Pars[4], self.Start_sf, (1/(1089.80 + 1)) ,self.Pars[0], self.Pars[1], self.Pars[2])
            
            debug_list = [Rho_r_a, Rho_m_a, Rho_de_a, _, H_a]
            
            print("*****************************************************************************************************************************************************")            
            print("*****************************************************************************************************************************************************")
            print(f" Parameters: {debug_list} ")
            print("*****************************************************************************************************************************************************")
            print("*****************************************************************************************************************************************************")
            
            rho_c = (3 * H_a**2)/(8 * np.pi * self.G)
            
            nu_Pars = [ Rho_m_a/rho_c, Rho_r_a/rho_c, 0.6897, H_a, Start_sf]
            
            self.sim_rad = self.calculate_sim_rad(a_start= self.Start_sf, Pars= nu_Pars)

            
            
        def calculate_sim_rad(self, a_start, Pars):
            c = 299792458  # Speed of light in m/s
    
            Omega_m, Omega_r, Omega_l, H0, _ = Pars

            Omega_m = float(Omega_m) #unitless
            Omega_r = float(Omega_r) #unitless
            Omega_l = float(Omega_l) #unitless
            H0 = float(H0) # 1/s

            horizon_scale = c / (a_start * H0) # m/s/s == m

            factor = 5.0  
            
            sim_rad = horizon_scale * factor
            
            return sim_rad # m

        def Run_simulation(self, Set_Particles_rigidly=True, Set_particles_from_density=False):
            """Run the simulation with the specified parameters"""

            if Set_Particles_rigidly == True and Set_particles_from_density == False:

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
            
            Timesteps = int(Time / dt) # unitless
            Positions_tensor = np.zeros((Timesteps, len(particles), 3)) #unitless x num_particles --> m, m, m

            for i, particle in enumerate(particles):
                Positions_tensor[0, i, :] = particle.position # m, m, m

            total_steps = len(scale_factors_list)

            for t in range(total_steps): # s
                
                self.runge_kutta_step( particles, dt, scale_factors_list[t-1], Scale_factor_dots[t-1], t=t)
                    

                for i, particle in enumerate(particles):
                    Positions_tensor[t-1, i, :] = particle.position # m m m
                
                print(f"t = {t}")

            #for _ in range(20):
            #    self.plot_particle_phase_space(Positions_tensor, particle_index=np.random.randint(0, len(particles)), output_fold= self.output_dir )
            
            
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


        def compute_gravitational_acceleration(self,num_particles, positions, masses, t):             
            
            softening = self.softener  # how can i dynamically tweak this based on number of particles and sim size? ||| meters

            accelerations = np.zeros((num_particles, 3)) # m/s^2

            for i in range(num_particles): 
                
                pos_diff = positions - positions[i] # m m m
                
                posdif_2 = positions[i] - positions # m m m
                
                print(f"{t}")
                
                mask = np.ones(len(pos_diff), dtype=bool)
                mask[i] = False
            
                distances_squared = np.sum(posdif_2[mask]**2, axis=1) + softening**2 # m^2 m^2 m^2
                
                inv_distances = 1.0 / (distances_squared**1.5) # 1/m^3
                   
                acc_components = (masses[mask] * inv_distances[:, np.newaxis] * pos_diff[mask]) * (-1)  # kg m / m^3 ||| kg/m^2
                                
                accelerations[i] = self.G * np.sum(acc_components, axis=0) # m/s^2
                print(accelerations[i])
            return accelerations


        def runge_kutta_step(self, particles, dt, sf, sf_dot, t):
        
            num_particles = len(particles) # unitless
            masses = np.array([p.mass for p in particles])[:, np.newaxis]  # kg
            # positions = np.array([p.position for p in particles]) # m m m 
            # velocities = np.array([p.velocity for p in particles]) # m/s m/s m/s

            phys_positions = np.array([p.position for p in particles])  # m
            phys_velocities = np.array([p.velocity for p in particles])  # m/s
            
            comoving_positions = phys_positions / sf
            
            Hubble = sf_dot/sf
            hubble_flow = Hubble * phys_positions
            peculiar_velocities = phys_velocities - hubble_flow
            
            comoving_velocities = sf * peculiar_velocities

            def acceleration(position, masses,velocity, t):
                
                r = position * sf
                
                acc = self.compute_gravitational_acceleration(num_particles=num_particles, positions=r, masses=masses, t=t)
                
                comoving_acc = acc / sf 
                
                Hubble_drag = -2 * Hubble * velocity
                         
                return comoving_acc + Hubble_drag
            

            # Compute RK4 coefficients
            k1_v = dt * acceleration(comoving_positions, masses,velocity=comoving_velocities,t=t) # m/s, m/s, m/s
            k1_r = dt * comoving_velocities # m m m

            k2_v = dt * acceleration(comoving_positions + 0.5 * k1_r, masses,velocity=comoving_velocities,t=t) # m/s, m/s, m/s
            k2_r = dt * (comoving_velocities + 0.5 * k1_v) # m m m

            k3_v = dt * acceleration(comoving_positions + 0.5 * k2_r, masses,velocity=comoving_velocities,t=t) # m/s, m/s, m/s
            k3_r = dt * (comoving_velocities + 0.5 * k2_v)# m m m

            k4_v = dt * acceleration(comoving_positions + k3_r, masses,velocity=comoving_velocities,t=t) # m/s, m/s, m/s
            k4_r = dt * (comoving_velocities + k3_v) # m m m

            # Update positions and velocities
            new_comoving_positions  = comoving_positions + (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6 # m, m, m
            new_comoving_velocities  = comoving_velocities + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6 # m/s, m/s, m/s
            
            new_phys_positions = new_comoving_positions * sf
            
            new_hubble_flow = Hubble * new_phys_positions
            new_peculiar_velocities = new_comoving_velocities / sf
            new_phys_velocities = new_peculiar_velocities + new_hubble_flow
            
            # Update particle attributes
            for i, p in enumerate(particles):
                p.position = new_phys_positions[i] # m, m, m
                p.velocity = new_phys_velocities[i] # m/s, m/s, m/s
                
        def plot_particle_phase_space(self, positions_tensor, particle_index, output_fold):
            projections = ['XY', 'XZ', 'YZ']
            axes_labels = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            for i, (proj, labels) in enumerate(zip(projections, axes_labels)):
                x = positions_tensor[:, particle_index, i % 3]
                y = positions_tensor[:, particle_index, (i + 1) % 3]

                axes[i].plot(x, y, 'bo-', linewidth=0.5, markersize=2, label=f"Particle {particle_index}")
                axes[i].set_xlabel(f"{labels[0]} position (m)")
                axes[i].set_ylabel(f"{labels[1]} position (m)")
                axes[i].set_title(f"Phase Space Trajectory ({proj}) of Particle {particle_index}")
                axes[i].grid(True)
                axes[i].legend()

            file_name = f'Particle_trajectory_{particle_index}'
            file_extension = '.png'
            file_path = get_unique_filename(output_folder=output_fold, output_type='fig', filename=file_name, file_type=file_extension)

            plt.savefig(file_path)



        
        def animate_positions_over_time(self, positions_tensor):
            downsample = max(1, positions_tensor.shape[0] // 100)
            positions_tensor = positions_tensor[::downsample]

            # Calculate bounds and set fixed axis limits
            x_all = positions_tensor[:, :, 0]
            y_all = positions_tensor[:, :, 1]
            z_all = positions_tensor[:, :, 2]

            x_min, x_max = np.min(x_all), np.max(x_all)
            y_min, y_max = np.min(y_all), np.max(y_all)
            z_min, z_max = np.min(z_all), np.max(z_all)

            margin = max((x_max-x_min), (y_max-y_min), (z_max-z_min)) * 0.1

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            colors = []
            for particle in self.Particles_list:  
                if isinstance(particle, MassParticle):
                    colors.append([0, 0, 1])  
                elif isinstance(particle, RadiationParticle):
                    colors.append([1, 0, 0])  
                else:
                    colors.append([0, 1, 0]) 
            
            colors = np.array(colors)
        
            scatter = ax.scatter(
                positions_tensor[0, :, 0],
                positions_tensor[0, :, 1],
                positions_tensor[0, :, 2],
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
                    positions_tensor[frame, :, 0],
                    positions_tensor[frame, :, 1],
                    positions_tensor[frame, :, 2]
                )
                #ax.view_init(elev=30, azim=frame)  # Adjust elevation and increment as needed
                time_text.set_text(f'Frame: {frame}/{len(positions_tensor)-1}')
                return [scatter, time_text]
            

            anim = FuncAnimation(fig, update, frames=len(positions_tensor), interval=100, blit=False)
                            
            file_path = get_unique_filename(output_folder = self.output_dir, output_type = 'gif', filename = "n_body", file_type = ".gif")

            try:
                anim.save(file_path, writer='pillow', fps=10)
                print(f"Animation saved as: {file_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
                plt.savefig(os.path.join(file_path, "n_body_fixed_first_frame.png"))

            plt.show()



sim = EU_formation_sim(
    N_Matter_particles=300,
    N_Radiation_particles=1,
    softener = 0.01, #  
    Start_sf=1e-12,
    Time=500.0,
    dt=  1,  # Reduce time step for better stability
    delta_c=0.45,
    dist_type='Gaussian_RF',
    # With a spectral index of 0.96
    
)

#sim.Run_simulation(Set_Particles_rigidly=True)
 
 
 
 #Todo:
    #- implementing co-moving coordinates.
    #- incorporate aspect of CMBagent or some sort of agent code to create a RAG agent 
        #(This is to source parameters, so i can simply prompt it to get the appropriate parameter values)
        #  Analytics?
        #  possibly smooth it and try to incorporate other types fo parameters and values

    # Find a way to tweak softening param in compute grav
    # Create a PBH formation check
    # Develop agents for resutls analysis
 
 
    
    