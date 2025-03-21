import cupy as cp
from Particles import Particle, MassParticle, RadiationParticle

G = 6.67430e-11   # m^3 kg^-1 s^-2
c = 299792458     # m/s
h_bar = 1.054571917e-34  # Joule*s



def generate_uniform_distribution(self, N_particles, Pars, sim_rad, mass_velocity_range):
    """Generate particles with uniform distribution within a sphere"""
    particles = []

    volume = (4/3) * cp.pi * sim_rad**3

    critical_density = 3 * (Pars[3])**2 / (8 * cp.pi * G) # kg/m^3
    
    matter_density = Pars[0] * critical_density
    radiation_density = Pars[1] * critical_density # kg/m^3
    
    total_matter_mass = matter_density * volume
    total_radiation_energy = radiation_density * volume # Kg
    
    total_radiation_energy = total_radiation_energy * c**2
    
    # [Omega_m, Omega_r, Omega_l, Ho, sf_ref]
 
    total_particles = N_particles
    matter_fraction = Pars[0] / (Pars[0] + Pars[1])
    n_matter_particles = int(total_particles * matter_fraction)
    n_radiation_particles = total_particles - n_matter_particles
    

    mass_per_matter_particle = total_matter_mass / n_matter_particles
    Energy_per_Radiation_particle =  total_radiation_energy / n_radiation_particles

    # Generate spherical coordinates for better uniformity
    for _ in range(n_matter_particles):

        r_mass = sim_rad * cp.cbrt(cp.random.random()) 
        theta_mass = cp.arccos(2 * cp.random.random() - 1)  
        phi_mass = 2 * cp.pi * cp.random.random()

        # Convert to Cartesian
        x_mass = r_mass * cp.sin(theta_mass) * cp.cos(phi_mass)
        y_mass = r_mass * cp.sin(theta_mass) * cp.sin(phi_mass)
        z_mass = r_mass * cp.cos(theta_mass)

        velocity_mass = cp.array([
            cp.random.uniform(mass_velocity_range[0], mass_velocity_range[1]),  
            cp.random.uniform(mass_velocity_range[0], mass_velocity_range[1]), 
            cp.random.uniform(mass_velocity_range[0], mass_velocity_range[1])
        ])
        
        r_Rad = sim_rad * cp.cbrt(cp.random.random()) 
        theta_Rad = cp.arccos(2 * cp.random.random() - 1)  
        phi_Rad = 2 * cp.pi * cp.random.random()
        
        particles.append(MassParticle(mass = mass_per_matter_particle, position= cp.array([x_mass, y_mass, z_mass]), velocity = velocity_mass))

    for _ in range(n_radiation_particles):
            # Convert to Cartesian
            x_Rad = r_Rad * cp.sin(theta_Rad) * cp.cos(phi_Rad)
            y_Rad = r_Rad * cp.sin(theta_Rad) * cp.sin(phi_Rad)
            z_Rad = r_Rad * cp.cos(theta_Rad)
            
            velocity_Rad = cp.array([
                cp.random.uniform(-1, 1),  
                cp.random.uniform(-1, 1), 
                cp.random.uniform(-1, 1)
            ])
            
            velocity_Rad /= cp.linalg.norm(velocity_Rad)
            
            velocity_Rad = c * velocity_Rad

            particles.append(RadiationParticle(energy= Energy_per_Radiation_particle, position= cp.array([x_Rad , y_Rad , z_Rad ], velocity = velocity_Rad)))


    return particles

def apply_mass_power_spectrum(self, positions, base_masses, Pars, sim_rad):
    """Apply mass power spectrum to a set of base particle masses"""
    
    n = int(cp.cbrt(len(positions)))  
    grid_size = sim_rad

    kx = 2 * cp.pi * cp.fft.fftfreq(n, d=grid_size/n)
    ky = 2 * cp.pi * cp.fft.fftfreq(n, d=grid_size/n)
    kz = 2 * cp.pi * cp.fft.fftfreq(n, d=grid_size/n)

    kx_grid, ky_grid, kz_grid = cp.meshgrid(kx, ky, kz, indexing='ij')
    
    r = cp.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    
    spherical_mask = r <= sim_rad

    r[r == 0] = 1e-10


    n_s = 1.0  # Spectral index
    
    k_cutoff = 10.0 / sim_rad  

    # Implement mass power spectrum
    k_pivot = 0.05 / sim_rad  # Pivot scale
    
    power_spectrum = (r/k_pivot)**(n_s-1) * cp.exp(-r**2/k_cutoff**2)

    random_phases = cp.random.uniform(0, 2*cp.pi, r.shape)

    fourier_coeff = cp.sqrt(power_spectrum) * cp.exp(1j * random_phases)

    density_field = cp.fft.ifftn(fourier_coeff).real

    perturbation_amplitude = 0.1  # Can be adjusted
    density_field = 1.0 + perturbation_amplitude * (density_field - cp.mean(density_field)) / cp.std(density_field)

    # Map particles to density field
    modulated_masses = cp.zeros(len(positions))
    grid_indices = []

    # For each particle, find its position in the grid
    for i, pos in enumerate(positions):
        # Convert position to grid index
        ix = int((pos[0] + sim_rad) * n / (2 * sim_rad))
        iy = int((pos[1] + sim_rad) * n / (2 * sim_rad))
        iz = int((pos[2] + sim_rad) * n / (2 * sim_rad))

        ix = max(0, min(ix, n-1))
        iy = max(0, min(iy, n-1))
        iz = max(0, min(iz, n-1))

        grid_indices.append((ix, iy, iz))

    # Apply density field to masses
    for i, (ix, iy, iz) in enumerate(grid_indices):
        modulated_masses[i] = base_masses[i] * density_field[ix, iy, iz]

    return modulated_masses

def generate_gaussian_RF(self, N_particles, Pars, sim_rad):
    """Generate particles with positions following a Gaussian random field"""
    n = int(cp.cbrt(N_particles))  # Cubic root to get grid size
    x = cp.linspace(-sim_rad, sim_rad, n)
    y = cp.linspace(-sim_rad, sim_rad, n)
    z = cp.linspace(-sim_rad, sim_rad, n)

    field = cp.zeros((n, n, n))
    k_space = cp.random.normal(0, 1, (n, n, n)) + 1j * cp.random.normal(0, 1, (n, n, n))

    k_x, k_y, k_z = cp.meshgrid(*[cp.fft.fftfreq(n, d=2*sim_rad/n) for _ in range(3)])
    k_mag = cp.sqrt(k_x**2 + k_y**2 + k_z**2)
    k_mag[0, 0, 0] = 1  # Avoid division by zero

    spectral_index = -1 
    power_spectrum = k_mag ** spectral_index
    power_spectrum[0, 0, 0] = 0  

    k_space *= cp.sqrt(power_spectrum)

    field = cp.fft.ifftn(k_space).real

    field = (field - cp.mean(field)) / cp.std(field)

    particles = []
    X, Y, Z = cp.meshgrid(x, y, z)

    mask = X**2 + Y**2 + Z**2 <= sim_rad**2
    X_valid = X[mask]
    Y_valid = Y[mask]
    Z_valid = Z[mask]
    field_valid = field[mask]


    prob = cp.exp(field_valid)
    prob = prob / cp.sum(prob)

    indices = cp.random.choice(len(X_valid), size=N_particles, p=prob)

    positions = cp.array([[X_valid[i], Y_valid[i], Z_valid[i]] for i in indices])

    matter_fraction = Pars[0] / (Pars[0] + Pars[1])
    n_matter_particles = int(N_particles * matter_fraction)
    n_radiation_particles = N_particles - n_matter_particles

    volume = (4/3) * cp.pi * sim_rad**3
    critical_density = 3 * (Pars[3])**2 / (8 * cp.pi * G)
    matter_density = Pars[0] * critical_density
    radiation_density = Pars[1] * critical_density

    total_matter_mass = matter_density * volume
    total_radiation_energy = radiation_density * volume

    matter_positions = positions[:n_matter_particles]
    radiation_positions = positions[n_matter_particles:]

    base_matter_masses = cp.ones(n_matter_particles) * (total_matter_mass / n_matter_particles)
    base_radiation_masses = cp.ones(n_radiation_particles) * (total_radiation_energy / (n_radiation_particles * c**2))

    modulated_matter_masses = apply_mass_power_spectrum(matter_positions, base_matter_masses)

    for i in range(n_matter_particles):
        pos = cp.array(matter_positions[i])
        vel = cp.array([cp.random.normal(0, 100), cp.random.normal(0, 100), cp.random.normal(0, 100)])
        particles.append(MassParticle(modulated_matter_masses[i], pos, vel))

    for i in range(n_radiation_particles):
        pos = cp.array(radiation_positions[i])
        vel = cp.array([cp.random.normal(0, 1), cp.random.normal(0, 1), cp.random.normal(0, 1)])
        particles.append(RadiationParticle(base_radiation_masses[i], pos, vel))

    return particles

def generate_adiabatic_perturbations(self, N_particles, Pars, sim_rad):
    """Generate adiabatic perturbations where all components fluctuate together"""
    
    positions = []
    for _ in range(N_particles):
        # Generate random position in a sphere
        while True:
            x = (2 * cp.random.random() - 1) * sim_rad
            y = (2 * cp.random.random() - 1) * sim_rad
            z = (2 * cp.random.random() - 1) * sim_rad
            if x**2 + y**2 + z**2 <= sim_rad**2:
                break
        positions.append([x, y, z])

    positions = cp.array(positions)

    matter_fraction = Pars[0] / (Pars[0] + Pars[1])
    n_matter_particles = int(N_particles * matter_fraction)
    n_radiation_particles = N_particles - n_matter_particles

    n = int(cp.cbrt(N_particles))  # Grid size

    # Create k-space grid
    kx = 2 * cp.pi * cp.fft.fftfreq(n, d=2*sim_rad/n)
    ky = 2 * cp.pi * cp.fft.fftfreq(n, d=2*sim_rad/n)
    kz = 2 * cp.pi * cp.fft.fftfreq(n, d=2*sim_rad/n)

    kx_grid, ky_grid, kz_grid = cp.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = cp.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

    k_mag[k_mag == 0] = 1e-10

    n_s = 0.965  # Planck 2018 value

    power_spectrum = k_mag**n_s

    random_phases = cp.random.uniform(0, 2*cp.pi, k_mag.shape)

    fourier_coeff = cp.sqrt(power_spectrum) * cp.exp(1j * random_phases)

    density_field = cp.fft.ifftn(fourier_coeff).real

    perturbation_amplitude = 0.01  # Small for early universe
    density_field = 1.0 + perturbation_amplitude * (density_field - cp.mean(density_field)) / cp.std(density_field)

    particle_densities = cp.zeros(len(positions))
    for i, pos in enumerate(positions):
        # Convert position to grid index
        ix = int((pos[0] + sim_rad) * n / (2 * sim_rad))
        iy = int((pos[1] + sim_rad) * n / (2 * sim_rad))
        iz = int((pos[2] + sim_rad) * n / (2 * sim_rad))

        # Clamp indices to valid range
        ix = max(0, min(ix, n-1))
        iy = max(0, min(iy, n-1))
        iz = max(0, min(iz, n-1))

        particle_densities[i] = density_field[ix, iy, iz]

    velocities = cp.zeros_like(positions)

    # Calculate base masses
    volume = (4/3) * cp.pi * sim_rad**3
    critical_density = 3 * (Pars[3])**2 / (8 * cp.pi * G)
    matter_density = Pars[0] * critical_density
    radiation_density = Pars[1] * critical_density

    total_matter_mass = matter_density * volume
    total_radiation_energy = radiation_density * volume

    # Base masses
    base_matter_mass = total_matter_mass / n_matter_particles
    base_radiation_mass = total_radiation_energy / (n_radiation_particles * c**2)

    # Create particles
    particles = []

    for i in range(n_matter_particles):
        pos = cp.array(positions[i])

        hubble_param = Pars[3] * cp.sqrt(Pars[0]/Start_sf[0]**3 + Pars[1]/Start_sf[0]**4 + Pars[2])
        peculiar_velocity = cp.random.normal(0, 100, 3) * (particle_densities[i] - 1.0)
        vel = cp.array(peculiar_velocity)

        mass = base_matter_mass * particle_densities[i]

        particles.append(MassParticle(mass, pos, vel))

    # Radiation particles (remaining particles)
    for i in range(n_matter_particles, N_particles):
        pos = cp.array(positions[i])
        # Random direction at speed of light
        vel_direction = cp.random.normal(0, 1, 3)
        vel_direction = vel_direction / cp.linalg.norm(vel_direction)
        vel = cp.array(vel_direction * c)

        # Mass modulated by density (for radiation, this affects number density)
        mass = base_radiation_mass * particle_densities[i]

        particles.append(RadiationParticle(mass, pos, vel))

    return particles


def generate_scale_invariant_spectrum(self, Pars, sim_rad, N_particles):
    """Generate particles with positions following a scale-invariant power spectrum P(k) ∝ k^1"""
    n = int(cp.cbrt(N_particles))  # Cubic root to get grid size
    x = cp.linspace(-sim_rad, sim_rad, n)
    y = cp.linspace(-sim_rad, sim_rad, n)
    z = cp.linspace(-sim_rad, sim_rad, n)

    # Create a 3D random field
    k_space = cp.random.normal(0, 1, (n, n, n)) + 1j * cp.random.normal(0, 1, (n, n, n))

    # Create k-space grid
    kx = cp.fft.fftfreq(n, d=2*sim_rad/n)
    ky = cp.fft.fftfreq(n, d=2*sim_rad/n)
    kz = cp.fft.fftfreq(n, d=2*sim_rad/n)

    kx_grid, ky_grid, kz_grid = cp.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = cp.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

    # Avoid division by zero at k=0
    k_mag[0, 0, 0] = 1e-10

    power_spectrum = k_mag.copy()

    power_spectrum[0, 0, 0] = 0

    k_space *= cp.sqrt(power_spectrum)

    density_field = cp.fft.ifftn(k_space).real

    density_field = (density_field - cp.mean(density_field)) / cp.std(density_field)

    perturbation_amplitude = 0.01  # Typical for early universe
    density_contrast = 1.0 + perturbation_amplitude * density_field

    # Create particles
    particles = []

    # Get positions from the density field
    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

    # Keep only points within the sphere
    mask = X**2 + Y**2 + Z**2 <= sim_rad**2
    X_valid = X[mask]
    Y_valid = Y[mask]
    Z_valid = Z[mask]
    density_valid = density_contrast[mask]

    prob = cp.maximum(density_valid, 0)  # Ensure non-negative probabilities
    prob = prob / cp.sum(prob)

    indices = cp.random.choice(len(X_valid), size=N_particles, p=prob)

    positions = cp.array([[X_valid[i], Y_valid[i], Z_valid[i]] for i in indices])
    densities = cp.array([density_valid[i] for i in indices])

    matter_fraction = Pars[0] / (Pars[0] + Pars[1])
    n_matter_particles = int(N_particles * matter_fraction)
    n_radiation_particles = N_particles - n_matter_particles

    volume = (4/3) * cp.pi * sim_rad**3
    critical_density = 3 * (Pars[3])**2 / (8 * cp.pi * G)
    matter_density = Pars[0] * critical_density
    radiation_density = Pars[1] * critical_density

    total_matter_mass = matter_density * volume
    total_radiation_energy = radiation_density * volume

    # Create matter particles
    for i in range(n_matter_particles):
        pos = cp.array(positions[i])

        hubble_param = Pars[3] * cp.sqrt(Pars[0]/Start_sf[0]**3 + Pars[1]/Start_sf[0]**4 + Pars[2])
        peculiar_velocity = cp.random.normal(0, 100, 3) * (densities[i] - 1.0)
        vel = cp.array(peculiar_velocity)

        mass = (total_matter_mass / n_matter_particles) * densities[i]

        particles.append(MassParticle(mass, pos, vel))

    for i in range(n_matter_particles, N_particles):
        pos = cp.array(positions[i])

        vel_direction = cp.random.normal(0, 1, 3)
        vel_direction = vel_direction / cp.linalg.norm(vel_direction)
        vel = cp.array(vel_direction * c)

        # "Effective mass" for radiation (E = mc²)
        mass = (total_radiation_energy / (n_radiation_particles * c**2)) * densities[i]

        particles.append(Radiationparticle(mass, pos, vel))

    return particles

def generate_isocurvature_perturbations(self, Pars, sim_rad, N_particles):
    """Generate isocurvature perturbations where component ratios vary but total energy density is constant"""
    n = int(cp.cbrt(N_particles))  # Cubic root to get grid size
    x = cp.linspace(-sim_rad, sim_rad, n)
    y = cp.linspace(-sim_rad, sim_rad, n)
    z = cp.linspace(-sim_rad, sim_rad, n)

    
    k_space_matter = cp.random.normal(0, 1, (n, n, n)) + 1j * cp.random.normal(0, 1, (n, n, n))
    k_space_radiation = cp.random.normal(0, 1, (n, n, n)) + 1j * cp.random.normal(0, 1, (n, n, n))

    kx = cp.fft.fftfreq(n, d=2*sim_rad/n)
    ky = cp.fft.fftfreq(n, d=2*sim_rad/n)
    kz = cp.fft.fftfreq(n, d=2*sim_rad/n)

    kx_grid, ky_grid, kz_grid = cp.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = cp.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

    k_mag[0, 0, 0] = 1e-10

    n_iso = 2.0  

    power_spectrum = k_mag**n_iso
    power_spectrum[0, 0, 0] = 0  # No DC component

    k_space_matter *= cp.sqrt(power_spectrum)
    k_space_radiation *= cp.sqrt(power_spectrum)

    density_field_matter = cp.fft.ifftn(k_space_matter).real
    density_field_radiation = cp.fft.ifftn(k_space_radiation).real

    density_field_matter = (density_field_matter - cp.mean(density_field_matter)) / cp.std(density_field_matter)
    density_field_radiation = (density_field_radiation - cp.mean(density_field_radiation)) / cp.std(density_field_radiation)


    critical_density = 3 * (Pars[3])**2 / (8 * cp.pi * G)
    matter_density = Pars[0] * critical_density
    radiation_density = Pars[1] * critical_density

    # Balance matter and radiation perturbations
    total_energy_density = matter_density + radiation_density
    matter_weight = matter_density / total_energy_density
    radiation_weight = radiation_density / total_energy_density

    combined_field = density_field_matter * matter_weight - density_field_radiation * radiation_weight

    iso_amplitude = 0.01  # Typical for early universe


    matter_contrast = 1.0 + iso_amplitude * combined_field
    radiation_contrast = 1.0 - iso_amplitude * combined_field * (matter_weight/radiation_weight)

    particles = []

    X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')
    mask = X**2 + Y**2 + Z**2 <= sim_rad**2

    X_valid = X[mask]
    Y_valid = Y[mask]
    Z_valid = Z[mask]


    
    matter_contrast_valid = matter_contrast[mask]
    radiation_contrast_valid = radiation_contrast[mask]

    matter_fraction = Pars[0] / (Pars[0] + Pars[1])
    n_matter_particles = int(N_particles * matter_fraction)
    n_radiation_particles = N_particles - n_matter_particles

    matter_prob = cp.maximum(matter_contrast_valid, 0)
    matter_prob = matter_prob / cp.sum(matter_prob)
    matter_indices = cp.random.choice(len(X_valid), size=n_matter_particles, p=matter_prob)

    radiation_prob = cp.maximum(radiation_contrast_valid, 0)
    radiation_prob = radiation_prob / cp.sum(radiation_prob)
    radiation_indices = cp.random.choice(len(X_valid), size=n_radiation_particles, p=radiation_prob)

    volume = (4/3) * cp.pi * sim_rad**3
    total_matter_mass = matter_density * volume
    total_radiation_energy = radiation_density * volume

    for i, idx in enumerate(matter_indices):
        pos = cp.array([X_valid[idx], Y_valid[idx], Z_valid[idx]])

        peculiar_velocity = cp.random.normal(0, 100, 3) * (matter_contrast_valid[idx] - 1.0)
        vel = cp.array(peculiar_velocity)

        mass = (total_matter_mass / n_matter_particles) * matter_contrast_valid[idx]

        particles.append(MassParticle(mass, pos, vel))

    for i, idx in enumerate(radiation_indices):
        pos = cp.array([X_valid[idx], Y_valid[idx], Z_valid[idx]])

        vel_direction = cp.random.normal(0, 1, 3)
        vel_direction = vel_direction / cp.linalg.norm(vel_direction)
        vel = cp.array(vel_direction * c)

        mass = (total_radiation_energy / (n_radiation_particles * c**2)) * radiation_contrast_valid[idx]

        particles.append(RadiationParticle(mass, pos, vel))

    return particles    