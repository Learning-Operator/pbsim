<<<<<<< HEAD

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import os
import cupy as cp
from SF_generation import Expansion, scale_back_initial


class Particle:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = cp.array(position)  # Ensure position is a numpy array
        self.velocity = cp.array(velocity)  # Ensure velocity is a numpy array

    def update_position(self, scale_factor, SF_Prev):
        """Update the physical position based on the scale factor."""
        self.position = self.position *  (scale_factor/SF_Prev)

G = 6.67430e-11   # m^3 kg^-1 s^-2
c = 299792458     # m/s
h_bar = 1.054571917e-34  # Joule*s

class MassParticle(Particle):
    """Mass particle representing dark matter or baryonic matter"""
    def __init__(self, mass, position, velocity):
        super().__init__(mass, position, velocity)
        self.type = "mass"

class RadiationParticle(Particle):
    """Radiation particle representing photons or neutrinos"""
    def __init__(self, mass, position, velocity):
        super().__init__(mass, position, velocity)
        self.type = "radiation"
        # Radiation particles move at or near the speed of light
        vel_magnitude = c
        # Normalize velocity direction and set to speed of light
        if cp.sum(velocity**2) > 0:
            velocity = velocity / cp.sqrt(cp.sum(velocity**2)) * vel_magnitude
        self.velocity = velocity



def generate_uniform_distribution(self, N_particles):
    """Generate particles with uniform distribution within a sphere"""
    particles = []
    
    # Calculate total mass and radiation energy
    volume = (4/3) * np.pi * self.sim_rad**2
    
    critical_density = 3 * (self.Pars[3])**2 / (8 * np.pi * G)
    matter_density = self.Pars[0] * critical_density
    radiation_density = self.Pars[1] * critical_density
    
    total_matter_mass = matter_density * volume
    total_radiation_energy = radiation_density * volume
    
    # Determine particle counts based on density ratios
    total_particles = N_particles[0]
    matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
    n_matter_particles = int(total_particles * matter_fraction)
    n_radiation_particles = total_particles - n_matter_particles
    
    # Create matter particles
    mass_per_matter_particle = total_matter_mass / n_matter_particles
    for _ in range(n_matter_particles):
        # Generate random position in a sphere
        while True:
            x = (2 * np.random.random() - 1) * self.sim_rad
            y = (2 * np.random.random() - 1) * self.sim_rad
            z = (2 * np.random.random() - 1) * self.sim_rad
            
            if x**2 + y**2 + z**2 <= self.sim_rad**2:
                break
        
        # Initial velocity (small random values)
        velocity = cp.array([np.random.normal(0, 100), np.random.normal(0, 100), np.random.normal(0, 100)])
        
        particles.append(MassParticle(mass_per_matter_particle, cp.array([x, y, z]), velocity))
    
    # Create radiation particles
    # For radiation, we use "effective mass" based on E=mc²
    effective_mass_per_radiation_particle = total_radiation_energy / (n_radiation_particles * c**2)
    for _ in range(n_radiation_particles):
        while True:
            x = (2 * np.random.random() - 1) * self.sim_rad
            y = (2 * np.random.random() - 1) * self.sim_rad
            z = (2 * np.random.random() - 1) * self.sim_rad
            
            if x**2 + y**2 + z**2 <= self.sim_rad**2:
                break
        
        # Random direction at speed of light
        velocity = cp.array([np.random.normal(0, 1), np.random.normal(0, 1), np.random.normal(0, 1)])
        
        particles.append(RadiationParticle(effective_mass_per_radiation_particle, cp.array([x, y, z]), velocity))
    
    return particles

def apply_mass_power_spectrum(self, positions, base_masses):
    """Apply mass power spectrum to a set of base particle masses"""
    # Create a power spectrum in k-space
    n = int(np.cbrt(len(positions)))  # Cube root for 3D grid
    grid_size = 2 * self.sim_rad
    
    # Create k-space grid
    kx = 2 * np.pi * np.fft.fftfreq(n, d=grid_size/n)
    ky = 2 * np.pi * np.fft.fftfreq(n, d=grid_size/n)
    kz = 2 * np.pi * np.fft.fftfreq(n, d=grid_size/n)
    
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    
    # Avoid division by zero
    k_mag[k_mag == 0] = 1e-10
    
    # Apply power spectrum: P(k) ∝ k^n * exp(-k^2/k_cutoff^2)
    # For primordial black holes, often n ≈ 1 (blue spectrum)
    n_s = 1.0  # Spectral index
    k_cutoff = 10.0 / self.sim_rad  # Cutoff scale
    
    # Implement mass power spectrum
    # P(k) = A * (k/k_pivot)^(n_s-1) * exp(-k^2/k_cutoff^2)
    k_pivot = 0.05 / self.sim_rad  # Pivot scale
    power_spectrum = (k_mag/k_pivot)**(n_s-1) * np.exp(-k_mag**2/k_cutoff**2)
    
    # Generate random phases
    random_phases = np.random.uniform(0, 2*np.pi, k_mag.shape)
    
    # Create complex Fourier coefficients
    fourier_coeff = np.sqrt(power_spectrum) * np.exp(1j * random_phases)
    
    # Transform to real space
    density_field = np.fft.ifftn(fourier_coeff).real
    
    # Normalize field to have mean of 1 and standard deviation based on desired perturbation amplitude
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
        
        # Clamp indices to valid range
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
    # Create a mesh grid
    n = int(np.cbrt(N_particles[0]))  # Cubic root to get grid size
    x = np.linspace(-self.sim_rad, self.sim_rad, n)
    y = np.linspace(-self.sim_rad, self.sim_rad, n)
    z = np.linspace(-self.sim_rad, self.sim_rad, n)
    
    # Create a 3D Gaussian random field
    field = np.zeros((n, n, n))
    # Using Fourier transform to generate the field
    k_space = np.random.normal(0, 1, (n, n, n)) + 1j * np.random.normal(0, 1, (n, n, n))
    
    # Apply power spectrum (could be scale-invariant or any other)
    k_x, k_y, k_z = np.meshgrid(*[np.fft.fftfreq(n, d=2*self.sim_rad/n) for _ in range(3)])
    k_mag = np.sqrt(k_x**2 + k_y**2 + k_z**2)
    k_mag[0, 0, 0] = 1  # Avoid division by zero
    
    # Power spectrum: P(k) ~ k^n where n is the spectral index
    spectral_index = -1  # Scale-invariant spectrum has n = 1
    power_spectrum = k_mag ** spectral_index
    power_spectrum[0, 0, 0] = 0  # Remove DC component
    
    # Apply power spectrum to k-space field
    k_space *= np.sqrt(power_spectrum)
    
    # Transform back to real space
    field = np.fft.ifftn(k_space).real
    
    # Normalize field
    field = (field - np.mean(field)) / np.std(field)
    
    # Create particles based on field values
    particles = []
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Keep only points within the sphere
    mask = X**2 + Y**2 + Z**2 <= self.sim_rad**2
    X_valid = X[mask]
    Y_valid = Y[mask]
    Z_valid = Z[mask]
    field_valid = field[mask]
    
    # Sample positions based on field values (higher density where field is higher)
    # Normalize field values to get probabilities
    prob = np.exp(field_valid)
    prob = prob / np.sum(prob)
    
    # Sample indices based on probabilities
    indices = np.random.choice(len(X_valid), size=N_particles[0], p=prob)
    
    # Get positions
    positions = np.array([[X_valid[i], Y_valid[i], Z_valid[i]] for i in indices])
    
    # Calculate matter and radiation fractions
    matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
    n_matter_particles = int(N_particles[0] * matter_fraction)
    n_radiation_particles = N_particles[0] - n_matter_particles
    
    # Calculate base masses
    volume = (4/3) * np.pi * self.sim_rad**3
    critical_density = 3 * (self.Pars[3])**2 / (8 * np.pi * G)
    matter_density = self.Pars[0] * critical_density
    radiation_density = self.Pars[1] * critical_density
    
    total_matter_mass = matter_density * volume
    total_radiation_energy = radiation_density * volume
    
    # Base masses
    matter_positions = positions[:n_matter_particles]
    radiation_positions = positions[n_matter_particles:]
    
    base_matter_masses = np.ones(n_matter_particles) * (total_matter_mass / n_matter_particles)
    base_radiation_masses = np.ones(n_radiation_particles) * (total_radiation_energy / (n_radiation_particles * c**2))
    
    # Apply mass power spectrum
    modulated_matter_masses = self.apply_mass_power_spectrum(matter_positions, base_matter_masses)
    
    # Create particles
    for i in range(n_matter_particles):
        pos = cp.array(matter_positions[i])
        # Random small velocity
        vel = cp.array([np.random.normal(0, 100), np.random.normal(0, 100), np.random.normal(0, 100)])
        particles.append(MassParticle(modulated_matter_masses[i], pos, vel))
    
    for i in range(n_radiation_particles):
        pos = cp.array(radiation_positions[i])
        # Random direction at speed of light
        vel = cp.array([np.random.normal(0, 1), np.random.normal(0, 1), np.random.normal(0, 1)])
        particles.append(RadiationParticle(base_radiation_masses[i], pos, vel))
    
    return particles

def generate_adiabatic_perturbations(self, N_particles):
    """Generate adiabatic perturbations where all components fluctuate together"""
    # For adiabatic perturbations, matter and radiation fluctuate together
    # Generate base positions in a grid or uniform distribution
    positions = []
    for _ in range(N_particles[0]):
        # Generate random position in a sphere
        while True:
            x = (2 * np.random.random() - 1) * self.sim_rad
            y = (2 * np.random.random() - 1) * self.sim_rad
            z = (2 * np.random.random() - 1) * self.sim_rad
            if x**2 + y**2 + z**2 <= self.sim_rad**2:
                break
        positions.append([x, y, z])
    
    positions = np.array(positions)
    
    # Calculate matter and radiation fractions
    matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
    n_matter_particles = int(N_particles[0] * matter_fraction)
    n_radiation_particles = N_particles[0] - n_matter_particles
    
    # Create a density field for perturbations (adiabatic = same for all components)
    n = int(np.cbrt(N_particles[0]))  # Grid size
    
    # Create k-space grid
    kx = 2 * np.pi * np.fft.fftfreq(n, d=2*self.sim_rad/n)
    ky = 2 * np.pi * np.fft.fftfreq(n, d=2*self.sim_rad/n)
    kz = 2 * np.pi * np.fft.fftfreq(n, d=2*self.sim_rad/n)
    
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    
    # Avoid division by zero
    k_mag[k_mag == 0] = 1e-10
    
    # ΛCDM primordial power spectrum: P(k) ∝ k^(n_s)
    n_s = 0.965  # Planck 2018 value
    
    # Implement power spectrum
    power_spectrum = k_mag**n_s
    
    # Generate random phases
    random_phases = np.random.uniform(0, 2*np.pi, k_mag.shape)
    
    # Create complex Fourier coefficients
    fourier_coeff = np.sqrt(power_spectrum) * np.exp(1j * random_phases)
    
    # Transform to real space
    density_field = np.fft.ifftn(fourier_coeff).real
    
    # Normalize field to get desired perturbation amplitude
    perturbation_amplitude = 0.01  # Small for early universe
    density_field = 1.0 + perturbation_amplitude * (density_field - np.mean(density_field)) / np.std(density_field)
    
    # Map particles to grid to get density at each position
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
    
    # Calculate velocities from potential gradient
    # In linear theory, v ∝ ∇Φ ∝ ∇(δ/k²)
    # Simplified: we'll use small velocities aligned with density gradient
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
    
    # Matter particles (first n_matter_particles)
    for i in range(n_matter_particles):
        pos = cp.array(positions[i])
        # Velocity depends on growth factor, scale factor, etc.
        # For simplicity, we use a scaling of the Hubble flow
        hubble_param = self.Pars[3] * np.sqrt(self.Pars[0]/self.Start_sf[0]**3 + self.Pars[1]/self.Start_sf[0]**4 + self.Pars[2])
        peculiar_velocity = np.random.normal(0, 100, 3) * (particle_densities[i] - 1.0)
        vel = cp.array(peculiar_velocity)
        
        # Mass modulated by density
        mass = base_matter_mass * particle_densities[i]
        
        particles.append(MassParticle(mass, pos, vel))
    
    # Radiation particles (remaining particles)
    for i in range(n_matter_particles, N_particles[0]):
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
    # Create a mesh grid
    n = int(np.cbrt(self.N_particles[0]))  # Cubic root to get grid size
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
    
    # Scale-invariant spectrum: P(k) ∝ k^1
    # This is specifically for the primordial universe
    # For scale-invariant fluctuations, we use P(k) ∝ k^1 (Harrison-Zeldovich spectrum)
    power_spectrum = k_mag.copy()
    
    # Set k=0 mode to 0 (no DC component)
    power_spectrum[0, 0, 0] = 0
    
    # Apply power spectrum to k-space field
    k_space *= np.sqrt(power_spectrum)
    
    # Transform to real space
    density_field = np.fft.ifftn(k_space).real
    
    # Normalize field (mean=0, std=1)
    density_field = (density_field - np.mean(density_field)) / np.std(density_field)
    
    # Create the density contrast with a specific amplitude
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
    
    # Sample positions with probability proportional to density
    prob = np.maximum(density_valid, 0)  # Ensure non-negative probabilities
    prob = prob / np.sum(prob)
    
    indices = np.random.choice(len(X_valid), size=self.N_particles[0], p=prob)
    
    # Positions for all particles
    positions = np.array([[X_valid[i], Y_valid[i], Z_valid[i]] for i in indices])
    densities = np.array([density_valid[i] for i in indices])
    
    # Determine matter and radiation particles
    matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
    n_matter_particles = int(self.N_particles[0] * matter_fraction)
    n_radiation_particles = self.N_particles[0] - n_matter_particles
    
    # Calculate volumes and densities
    volume = (4/3) * np.pi * self.sim_rad**3
    critical_density = 3 * (self.Pars[3])**2 / (8 * np.pi * G)
    matter_density = self.Pars[0] * critical_density
    radiation_density = self.Pars[1] * critical_density
    
    total_matter_mass = matter_density * volume
    total_radiation_energy = radiation_density * volume
    
    # Create matter particles
    for i in range(n_matter_particles):
        pos = cp.array(positions[i])
        
        # Velocity based on growing mode solutions (∝ √a)
        # For scale-invariant spectrum, velocity ∝ k
        hubble_param = self.Pars[3] * np.sqrt(self.Pars[0]/self.Start_sf[0]**3 + self.Pars[1]/self.Start_sf[0]**4 + self.Pars[2])
        peculiar_velocity = np.random.normal(0, 100, 3) * (densities[i] - 1.0)
        vel = cp.array(peculiar_velocity)
        
        # Mass modulated by density
        mass = (total_matter_mass / n_matter_particles) * densities[i]
        
        particles.append(MassParticle(mass, pos, vel))
    
    # Create radiation particles
    for i in range(n_matter_particles, self.N_particles[0]):
        pos = cp.array(positions[i])
        
        # For radiation, velocity is always c in a random direction
        vel_direction = np.random.normal(0, 1, 3)
        vel_direction = vel_direction / np.linalg.norm(vel_direction)
        vel = cp.array(vel_direction * c)
        
        # "Effective mass" for radiation (E = mc²)
        mass = (total_radiation_energy / (n_radiation_particles * c**2)) * densities[i]
        
        particles.append(RadiationParticle(mass, pos, vel))
    
    return particles

def generate_isocurvature_perturbations(self):
    """Generate isocurvature perturbations where component ratios vary but total energy density is constant"""
    # Create a mesh grid
    n = int(np.cbrt(self.N_particles[0]))  # Cubic root to get grid size
    x = np.linspace(-self.sim_rad, self.sim_rad, n)
    y = np.linspace(-self.sim_rad, self.sim_rad, n)
    z = np.linspace(-self.sim_rad, self.sim_rad, n)
    
    # Create two separate random fields - one for matter, one for radiation
    # This is crucial for isocurvature - components have different fluctuations
    k_space_matter = np.random.normal(0, 1, (n, n, n)) + 1j * np.random.normal(0, 1, (n, n, n))
    k_space_radiation = np.random.normal(0, 1, (n, n, n)) + 1j * np.random.normal(0, 1, (n, n, n))
    
    # Create k-space grid
    kx = np.fft.fftfreq(n, d=2*self.sim_rad/n)
    ky = np.fft.fftfreq(n, d=2*self.sim_rad/n)
    kz = np.fft.fftfreq(n, d=2*self.sim_rad/n)
    
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    
    # Avoid division by zero at k=0
    k_mag[0, 0, 0] = 1e-10
    
    # For isocurvature, often blue-tilted spectrum is used
    # P(k) ∝ k^n_iso
    n_iso = 2.0  # Blue spectrum, can be adjusted
    
    power_spectrum = k_mag**n_iso
    power_spectrum[0, 0, 0] = 0  # No DC component
    
    # Apply power spectrum
    k_space_matter *= np.sqrt(power_spectrum)
    k_space_radiation *= np.sqrt(power_spectrum)
    
    # Transform to real space
    density_field_matter = np.fft.ifftn(k_space_matter).real
    density_field_radiation = np.fft.ifftn(k_space_radiation).real
    
    # Normalize fields
    density_field_matter = (density_field_matter - np.mean(density_field_matter)) / np.std(density_field_matter)
    density_field_radiation = (density_field_radiation - np.mean(density_field_radiation)) / np.std(density_field_radiation)
    
    # For isocurvature: ensure total density is constant (δρ_total = 0)
    # This means δρ_matter = -δρ_radiation (scaled by energy densities)
    
    # Calculate matter and radiation energy densities
    critical_density = 3 * (self.Pars[3])**2 / (8 * np.pi * G)
    matter_density = self.Pars[0] * critical_density
    radiation_density = self.Pars[1] * critical_density
    
    # Balance matter and radiation perturbations
    total_energy_density = matter_density + radiation_density
    matter_weight = matter_density / total_energy_density
    radiation_weight = radiation_density / total_energy_density
    
    # Combined field ensures total density remains constant
    combined_field = density_field_matter * matter_weight - density_field_radiation * radiation_weight
    
    # Set perturbation amplitude
    iso_amplitude = 0.01  # Typical for early universe
    
    # Create density contrasts
    # For isocurvature: components have opposite fluctuations
    matter_contrast = 1.0 + iso_amplitude * combined_field
    radiation_contrast = 1.0 - iso_amplitude * combined_field * (matter_weight/radiation_weight)
    
    # Create particles
    particles = []
    
    # Positions on grid inside the sphere
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    mask = X**2 + Y**2 + Z**2 <= self.sim_rad**2
    
    # Valid positions inside sphere
    X_valid = X[mask]
    Y_valid = Y[mask]
    Z_valid = Z[mask]
    
    # Get density contrasts at valid positions
    matter_contrast_valid = matter_contrast[mask]
    radiation_contrast_valid = radiation_contrast[mask]
    
    # Determine numbers of particles
    matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
    n_matter_particles = int(self.N_particles[0] * matter_fraction)
    n_radiation_particles = self.N_particles[0] - n_matter_particles
    
    # Sample matter positions proportional to matter density
    matter_prob = np.maximum(matter_contrast_valid, 0)
    matter_prob = matter_prob / np.sum(matter_prob)
    matter_indices = np.random.choice(len(X_valid), size=n_matter_particles, p=matter_prob)
    
    # Sample radiation positions proportional to radiation density
    radiation_prob = np.maximum(radiation_contrast_valid, 0)
    radiation_prob = radiation_prob / np.sum(radiation_prob)
    radiation_indices = np.random.choice(len(X_valid), size=n_radiation_particles, p=radiation_prob)
    
    # Calculate total masses
    volume = (4/3) * np.pi * self.sim_rad**3
    total_matter_mass = matter_density * volume
    total_radiation_energy = radiation_density * volume
    
    # Create matter particles
    for i, idx in enumerate(matter_indices):
        pos = cp.array([X_valid[idx], Y_valid[idx], Z_valid[idx]])
        
        # Velocity - for isocurvature, entropy perturbation drives velocity
        peculiar_velocity = np.random.normal(0, 100, 3) * (matter_contrast_valid[idx] - 1.0)
        vel = cp.array(peculiar_velocity)
        
        # Mass reflects local density
        mass = (total_matter_mass / n_matter_particles) * matter_contrast_valid[idx]
        
        particles.append(MassParticle(mass, pos, vel))
    
    # Create radiation particles
    for i, idx in enumerate(radiation_indices):
        pos = cp.array([X_valid[idx], Y_valid[idx], Z_valid[idx]])
        
        # Radiation velocity - always c in a random direction
        vel_direction = np.random.normal(0, 1, 3)
        vel_direction = vel_direction / np.linalg.norm(vel_direction)
        vel = cp.array(vel_direction * c)
        
        # "Effective mass" reflects local radiation density
        mass = (total_radiation_energy / (n_radiation_particles * c**2)) * radiation_contrast_valid[idx]
        
        particles.append(RadiationParticle(mass, pos, vel))
    
    return particles














=======

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import os
import cupy as cp
from SF_generation import Expansion, scale_back_initial


class Particle:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = cp.array(position)  # Ensure position is a numpy array
        self.velocity = cp.array(velocity)  # Ensure velocity is a numpy array

    def update_position(self, scale_factor, SF_Prev):
        """Update the physical position based on the scale factor."""
        self.position = self.position *  (scale_factor/SF_Prev)

G = 6.67430e-11   # m^3 kg^-1 s^-2
c = 299792458     # m/s
h_bar = 1.054571917e-34  # Joule*s

class MassParticle(Particle):
    """Mass particle representing dark matter or baryonic matter"""
    def __init__(self, mass, position, velocity):
        super().__init__(mass, position, velocity)
        self.type = "mass"

class RadiationParticle(Particle):
    """Radiation particle representing photons or neutrinos"""
    def __init__(self, mass, position, velocity):
        super().__init__(mass, position, velocity)
        self.type = "radiation"
        # Radiation particles move at or near the speed of light
        vel_magnitude = c
        # Normalize velocity direction and set to speed of light
        if cp.sum(velocity**2) > 0:
            velocity = velocity / cp.sqrt(cp.sum(velocity**2)) * vel_magnitude
        self.velocity = velocity



def generate_uniform_distribution(self, N_particles):
    """Generate particles with uniform distribution within a sphere"""
    particles = []
    
    # Calculate total mass and radiation energy
    volume = (4/3) * np.pi * self.sim_rad**2
    
    critical_density = 3 * (self.Pars[3])**2 / (8 * np.pi * G)
    matter_density = self.Pars[0] * critical_density
    radiation_density = self.Pars[1] * critical_density
    
    total_matter_mass = matter_density * volume
    total_radiation_energy = radiation_density * volume
    
    # Determine particle counts based on density ratios
    total_particles = N_particles[0]
    matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
    n_matter_particles = int(total_particles * matter_fraction)
    n_radiation_particles = total_particles - n_matter_particles
    
    # Create matter particles
    mass_per_matter_particle = total_matter_mass / n_matter_particles
    for _ in range(n_matter_particles):
        # Generate random position in a sphere
        while True:
            x = (2 * np.random.random() - 1) * self.sim_rad
            y = (2 * np.random.random() - 1) * self.sim_rad
            z = (2 * np.random.random() - 1) * self.sim_rad
            
            if x**2 + y**2 + z**2 <= self.sim_rad**2:
                break
        
        # Initial velocity (small random values)
        velocity = cp.array([np.random.normal(0, 100), np.random.normal(0, 100), np.random.normal(0, 100)])
        
        particles.append(MassParticle(mass_per_matter_particle, cp.array([x, y, z]), velocity))
    
    # Create radiation particles
    # For radiation, we use "effective mass" based on E=mc²
    effective_mass_per_radiation_particle = total_radiation_energy / (n_radiation_particles * c**2)
    for _ in range(n_radiation_particles):
        while True:
            x = (2 * np.random.random() - 1) * self.sim_rad
            y = (2 * np.random.random() - 1) * self.sim_rad
            z = (2 * np.random.random() - 1) * self.sim_rad
            
            if x**2 + y**2 + z**2 <= self.sim_rad**2:
                break
        
        # Random direction at speed of light
        velocity = cp.array([np.random.normal(0, 1), np.random.normal(0, 1), np.random.normal(0, 1)])
        
        particles.append(RadiationParticle(effective_mass_per_radiation_particle, cp.array([x, y, z]), velocity))
    
    return particles

def apply_mass_power_spectrum(self, positions, base_masses):
    """Apply mass power spectrum to a set of base particle masses"""
    # Create a power spectrum in k-space
    n = int(np.cbrt(len(positions)))  # Cube root for 3D grid
    grid_size = 2 * self.sim_rad
    
    # Create k-space grid
    kx = 2 * np.pi * np.fft.fftfreq(n, d=grid_size/n)
    ky = 2 * np.pi * np.fft.fftfreq(n, d=grid_size/n)
    kz = 2 * np.pi * np.fft.fftfreq(n, d=grid_size/n)
    
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    
    # Avoid division by zero
    k_mag[k_mag == 0] = 1e-10
    
    # Apply power spectrum: P(k) ∝ k^n * exp(-k^2/k_cutoff^2)
    # For primordial black holes, often n ≈ 1 (blue spectrum)
    n_s = 1.0  # Spectral index
    k_cutoff = 10.0 / self.sim_rad  # Cutoff scale
    
    # Implement mass power spectrum
    # P(k) = A * (k/k_pivot)^(n_s-1) * exp(-k^2/k_cutoff^2)
    k_pivot = 0.05 / self.sim_rad  # Pivot scale
    power_spectrum = (k_mag/k_pivot)**(n_s-1) * np.exp(-k_mag**2/k_cutoff**2)
    
    # Generate random phases
    random_phases = np.random.uniform(0, 2*np.pi, k_mag.shape)
    
    # Create complex Fourier coefficients
    fourier_coeff = np.sqrt(power_spectrum) * np.exp(1j * random_phases)
    
    # Transform to real space
    density_field = np.fft.ifftn(fourier_coeff).real
    
    # Normalize field to have mean of 1 and standard deviation based on desired perturbation amplitude
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
        
        # Clamp indices to valid range
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
    # Create a mesh grid
    n = int(np.cbrt(N_particles[0]))  # Cubic root to get grid size
    x = np.linspace(-self.sim_rad, self.sim_rad, n)
    y = np.linspace(-self.sim_rad, self.sim_rad, n)
    z = np.linspace(-self.sim_rad, self.sim_rad, n)
    
    # Create a 3D Gaussian random field
    field = np.zeros((n, n, n))
    # Using Fourier transform to generate the field
    k_space = np.random.normal(0, 1, (n, n, n)) + 1j * np.random.normal(0, 1, (n, n, n))
    
    # Apply power spectrum (could be scale-invariant or any other)
    k_x, k_y, k_z = np.meshgrid(*[np.fft.fftfreq(n, d=2*self.sim_rad/n) for _ in range(3)])
    k_mag = np.sqrt(k_x**2 + k_y**2 + k_z**2)
    k_mag[0, 0, 0] = 1  # Avoid division by zero
    
    # Power spectrum: P(k) ~ k^n where n is the spectral index
    spectral_index = -1  # Scale-invariant spectrum has n = 1
    power_spectrum = k_mag ** spectral_index
    power_spectrum[0, 0, 0] = 0  # Remove DC component
    
    # Apply power spectrum to k-space field
    k_space *= np.sqrt(power_spectrum)
    
    # Transform back to real space
    field = np.fft.ifftn(k_space).real
    
    # Normalize field
    field = (field - np.mean(field)) / np.std(field)
    
    # Create particles based on field values
    particles = []
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Keep only points within the sphere
    mask = X**2 + Y**2 + Z**2 <= self.sim_rad**2
    X_valid = X[mask]
    Y_valid = Y[mask]
    Z_valid = Z[mask]
    field_valid = field[mask]
    
    # Sample positions based on field values (higher density where field is higher)
    # Normalize field values to get probabilities
    prob = np.exp(field_valid)
    prob = prob / np.sum(prob)
    
    # Sample indices based on probabilities
    indices = np.random.choice(len(X_valid), size=N_particles[0], p=prob)
    
    # Get positions
    positions = np.array([[X_valid[i], Y_valid[i], Z_valid[i]] for i in indices])
    
    # Calculate matter and radiation fractions
    matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
    n_matter_particles = int(N_particles[0] * matter_fraction)
    n_radiation_particles = N_particles[0] - n_matter_particles
    
    # Calculate base masses
    volume = (4/3) * np.pi * self.sim_rad**3
    critical_density = 3 * (self.Pars[3])**2 / (8 * np.pi * G)
    matter_density = self.Pars[0] * critical_density
    radiation_density = self.Pars[1] * critical_density
    
    total_matter_mass = matter_density * volume
    total_radiation_energy = radiation_density * volume
    
    # Base masses
    matter_positions = positions[:n_matter_particles]
    radiation_positions = positions[n_matter_particles:]
    
    base_matter_masses = np.ones(n_matter_particles) * (total_matter_mass / n_matter_particles)
    base_radiation_masses = np.ones(n_radiation_particles) * (total_radiation_energy / (n_radiation_particles * c**2))
    
    # Apply mass power spectrum
    modulated_matter_masses = self.apply_mass_power_spectrum(matter_positions, base_matter_masses)
    
    # Create particles
    for i in range(n_matter_particles):
        pos = cp.array(matter_positions[i])
        # Random small velocity
        vel = cp.array([np.random.normal(0, 100), np.random.normal(0, 100), np.random.normal(0, 100)])
        particles.append(MassParticle(modulated_matter_masses[i], pos, vel))
    
    for i in range(n_radiation_particles):
        pos = cp.array(radiation_positions[i])
        # Random direction at speed of light
        vel = cp.array([np.random.normal(0, 1), np.random.normal(0, 1), np.random.normal(0, 1)])
        particles.append(RadiationParticle(base_radiation_masses[i], pos, vel))
    
    return particles

def generate_adiabatic_perturbations(self, N_particles):
    """Generate adiabatic perturbations where all components fluctuate together"""
    # For adiabatic perturbations, matter and radiation fluctuate together
    # Generate base positions in a grid or uniform distribution
    positions = []
    for _ in range(N_particles[0]):
        # Generate random position in a sphere
        while True:
            x = (2 * np.random.random() - 1) * self.sim_rad
            y = (2 * np.random.random() - 1) * self.sim_rad
            z = (2 * np.random.random() - 1) * self.sim_rad
            if x**2 + y**2 + z**2 <= self.sim_rad**2:
                break
        positions.append([x, y, z])
    
    positions = np.array(positions)
    
    # Calculate matter and radiation fractions
    matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
    n_matter_particles = int(N_particles[0] * matter_fraction)
    n_radiation_particles = N_particles[0] - n_matter_particles
    
    # Create a density field for perturbations (adiabatic = same for all components)
    n = int(np.cbrt(N_particles[0]))  # Grid size
    
    # Create k-space grid
    kx = 2 * np.pi * np.fft.fftfreq(n, d=2*self.sim_rad/n)
    ky = 2 * np.pi * np.fft.fftfreq(n, d=2*self.sim_rad/n)
    kz = 2 * np.pi * np.fft.fftfreq(n, d=2*self.sim_rad/n)
    
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    
    # Avoid division by zero
    k_mag[k_mag == 0] = 1e-10
    
    # ΛCDM primordial power spectrum: P(k) ∝ k^(n_s)
    n_s = 0.965  # Planck 2018 value
    
    # Implement power spectrum
    power_spectrum = k_mag**n_s
    
    # Generate random phases
    random_phases = np.random.uniform(0, 2*np.pi, k_mag.shape)
    
    # Create complex Fourier coefficients
    fourier_coeff = np.sqrt(power_spectrum) * np.exp(1j * random_phases)
    
    # Transform to real space
    density_field = np.fft.ifftn(fourier_coeff).real
    
    # Normalize field to get desired perturbation amplitude
    perturbation_amplitude = 0.01  # Small for early universe
    density_field = 1.0 + perturbation_amplitude * (density_field - np.mean(density_field)) / np.std(density_field)
    
    # Map particles to grid to get density at each position
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
    
    # Calculate velocities from potential gradient
    # In linear theory, v ∝ ∇Φ ∝ ∇(δ/k²)
    # Simplified: we'll use small velocities aligned with density gradient
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
    
    # Matter particles (first n_matter_particles)
    for i in range(n_matter_particles):
        pos = cp.array(positions[i])
        # Velocity depends on growth factor, scale factor, etc.
        # For simplicity, we use a scaling of the Hubble flow
        hubble_param = self.Pars[3] * np.sqrt(self.Pars[0]/self.Start_sf[0]**3 + self.Pars[1]/self.Start_sf[0]**4 + self.Pars[2])
        peculiar_velocity = np.random.normal(0, 100, 3) * (particle_densities[i] - 1.0)
        vel = cp.array(peculiar_velocity)
        
        # Mass modulated by density
        mass = base_matter_mass * particle_densities[i]
        
        particles.append(MassParticle(mass, pos, vel))
    
    # Radiation particles (remaining particles)
    for i in range(n_matter_particles, N_particles[0]):
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
    # Create a mesh grid
    n = int(np.cbrt(self.N_particles[0]))  # Cubic root to get grid size
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
    
    # Scale-invariant spectrum: P(k) ∝ k^1
    # This is specifically for the primordial universe
    # For scale-invariant fluctuations, we use P(k) ∝ k^1 (Harrison-Zeldovich spectrum)
    power_spectrum = k_mag.copy()
    
    # Set k=0 mode to 0 (no DC component)
    power_spectrum[0, 0, 0] = 0
    
    # Apply power spectrum to k-space field
    k_space *= np.sqrt(power_spectrum)
    
    # Transform to real space
    density_field = np.fft.ifftn(k_space).real
    
    # Normalize field (mean=0, std=1)
    density_field = (density_field - np.mean(density_field)) / np.std(density_field)
    
    # Create the density contrast with a specific amplitude
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
    
    # Sample positions with probability proportional to density
    prob = np.maximum(density_valid, 0)  # Ensure non-negative probabilities
    prob = prob / np.sum(prob)
    
    indices = np.random.choice(len(X_valid), size=self.N_particles[0], p=prob)
    
    # Positions for all particles
    positions = np.array([[X_valid[i], Y_valid[i], Z_valid[i]] for i in indices])
    densities = np.array([density_valid[i] for i in indices])
    
    # Determine matter and radiation particles
    matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
    n_matter_particles = int(self.N_particles[0] * matter_fraction)
    n_radiation_particles = self.N_particles[0] - n_matter_particles
    
    # Calculate volumes and densities
    volume = (4/3) * np.pi * self.sim_rad**3
    critical_density = 3 * (self.Pars[3])**2 / (8 * np.pi * G)
    matter_density = self.Pars[0] * critical_density
    radiation_density = self.Pars[1] * critical_density
    
    total_matter_mass = matter_density * volume
    total_radiation_energy = radiation_density * volume
    
    # Create matter particles
    for i in range(n_matter_particles):
        pos = cp.array(positions[i])
        
        # Velocity based on growing mode solutions (∝ √a)
        # For scale-invariant spectrum, velocity ∝ k
        hubble_param = self.Pars[3] * np.sqrt(self.Pars[0]/self.Start_sf[0]**3 + self.Pars[1]/self.Start_sf[0]**4 + self.Pars[2])
        peculiar_velocity = np.random.normal(0, 100, 3) * (densities[i] - 1.0)
        vel = cp.array(peculiar_velocity)
        
        # Mass modulated by density
        mass = (total_matter_mass / n_matter_particles) * densities[i]
        
        particles.append(MassParticle(mass, pos, vel))
    
    # Create radiation particles
    for i in range(n_matter_particles, self.N_particles[0]):
        pos = cp.array(positions[i])
        
        # For radiation, velocity is always c in a random direction
        vel_direction = np.random.normal(0, 1, 3)
        vel_direction = vel_direction / np.linalg.norm(vel_direction)
        vel = cp.array(vel_direction * c)
        
        # "Effective mass" for radiation (E = mc²)
        mass = (total_radiation_energy / (n_radiation_particles * c**2)) * densities[i]
        
        particles.append(RadiationParticle(mass, pos, vel))
    
    return particles

def generate_isocurvature_perturbations(self):
    """Generate isocurvature perturbations where component ratios vary but total energy density is constant"""
    # Create a mesh grid
    n = int(np.cbrt(self.N_particles[0]))  # Cubic root to get grid size
    x = np.linspace(-self.sim_rad, self.sim_rad, n)
    y = np.linspace(-self.sim_rad, self.sim_rad, n)
    z = np.linspace(-self.sim_rad, self.sim_rad, n)
    
    # Create two separate random fields - one for matter, one for radiation
    # This is crucial for isocurvature - components have different fluctuations
    k_space_matter = np.random.normal(0, 1, (n, n, n)) + 1j * np.random.normal(0, 1, (n, n, n))
    k_space_radiation = np.random.normal(0, 1, (n, n, n)) + 1j * np.random.normal(0, 1, (n, n, n))
    
    # Create k-space grid
    kx = np.fft.fftfreq(n, d=2*self.sim_rad/n)
    ky = np.fft.fftfreq(n, d=2*self.sim_rad/n)
    kz = np.fft.fftfreq(n, d=2*self.sim_rad/n)
    
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    
    # Avoid division by zero at k=0
    k_mag[0, 0, 0] = 1e-10
    
    # For isocurvature, often blue-tilted spectrum is used
    # P(k) ∝ k^n_iso
    n_iso = 2.0  # Blue spectrum, can be adjusted
    
    power_spectrum = k_mag**n_iso
    power_spectrum[0, 0, 0] = 0  # No DC component
    
    # Apply power spectrum
    k_space_matter *= np.sqrt(power_spectrum)
    k_space_radiation *= np.sqrt(power_spectrum)
    
    # Transform to real space
    density_field_matter = np.fft.ifftn(k_space_matter).real
    density_field_radiation = np.fft.ifftn(k_space_radiation).real
    
    # Normalize fields
    density_field_matter = (density_field_matter - np.mean(density_field_matter)) / np.std(density_field_matter)
    density_field_radiation = (density_field_radiation - np.mean(density_field_radiation)) / np.std(density_field_radiation)
    
    # For isocurvature: ensure total density is constant (δρ_total = 0)
    # This means δρ_matter = -δρ_radiation (scaled by energy densities)
    
    # Calculate matter and radiation energy densities
    critical_density = 3 * (self.Pars[3])**2 / (8 * np.pi * G)
    matter_density = self.Pars[0] * critical_density
    radiation_density = self.Pars[1] * critical_density
    
    # Balance matter and radiation perturbations
    total_energy_density = matter_density + radiation_density
    matter_weight = matter_density / total_energy_density
    radiation_weight = radiation_density / total_energy_density
    
    # Combined field ensures total density remains constant
    combined_field = density_field_matter * matter_weight - density_field_radiation * radiation_weight
    
    # Set perturbation amplitude
    iso_amplitude = 0.01  # Typical for early universe
    
    # Create density contrasts
    # For isocurvature: components have opposite fluctuations
    matter_contrast = 1.0 + iso_amplitude * combined_field
    radiation_contrast = 1.0 - iso_amplitude * combined_field * (matter_weight/radiation_weight)
    
    # Create particles
    particles = []
    
    # Positions on grid inside the sphere
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    mask = X**2 + Y**2 + Z**2 <= self.sim_rad**2
    
    # Valid positions inside sphere
    X_valid = X[mask]
    Y_valid = Y[mask]
    Z_valid = Z[mask]
    
    # Get density contrasts at valid positions
    matter_contrast_valid = matter_contrast[mask]
    radiation_contrast_valid = radiation_contrast[mask]
    
    # Determine numbers of particles
    matter_fraction = self.Pars[0] / (self.Pars[0] + self.Pars[1])
    n_matter_particles = int(self.N_particles[0] * matter_fraction)
    n_radiation_particles = self.N_particles[0] - n_matter_particles
    
    # Sample matter positions proportional to matter density
    matter_prob = np.maximum(matter_contrast_valid, 0)
    matter_prob = matter_prob / np.sum(matter_prob)
    matter_indices = np.random.choice(len(X_valid), size=n_matter_particles, p=matter_prob)
    
    # Sample radiation positions proportional to radiation density
    radiation_prob = np.maximum(radiation_contrast_valid, 0)
    radiation_prob = radiation_prob / np.sum(radiation_prob)
    radiation_indices = np.random.choice(len(X_valid), size=n_radiation_particles, p=radiation_prob)
    
    # Calculate total masses
    volume = (4/3) * np.pi * self.sim_rad**3
    total_matter_mass = matter_density * volume
    total_radiation_energy = radiation_density * volume
    
    # Create matter particles
    for i, idx in enumerate(matter_indices):
        pos = cp.array([X_valid[idx], Y_valid[idx], Z_valid[idx]])
        
        # Velocity - for isocurvature, entropy perturbation drives velocity
        peculiar_velocity = np.random.normal(0, 100, 3) * (matter_contrast_valid[idx] - 1.0)
        vel = cp.array(peculiar_velocity)
        
        # Mass reflects local density
        mass = (total_matter_mass / n_matter_particles) * matter_contrast_valid[idx]
        
        particles.append(MassParticle(mass, pos, vel))
    
    # Create radiation particles
    for i, idx in enumerate(radiation_indices):
        pos = cp.array([X_valid[idx], Y_valid[idx], Z_valid[idx]])
        
        # Radiation velocity - always c in a random direction
        vel_direction = np.random.normal(0, 1, 3)
        vel_direction = vel_direction / np.linalg.norm(vel_direction)
        vel = cp.array(vel_direction * c)
        
        # "Effective mass" reflects local radiation density
        mass = (total_radiation_energy / (n_radiation_particles * c**2)) * radiation_contrast_valid[idx]
        
        particles.append(RadiationParticle(mass, pos, vel))
    
    return particles














>>>>>>> 31f20d4 (Started development of the final simulation)
#[Omega_m, Omega_r, Omega_l, Ho, sf_ref]