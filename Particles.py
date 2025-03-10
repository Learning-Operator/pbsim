<<<<<<< HEAD
import cupy as cp

G = 6.67430e-11   # m^3 kg^-1 s^-2
c = 299792458     # m/s
h_bar = 1.054571917e-34  # Joule*s

class Particle:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = cp.array(position)  # Ensure position is a numpy array
        self.velocity = cp.array(velocity)  # Ensure velocity is a numpy array

    def update_position(self, scale_factor, SF_Prev):
        """Update the physical position based on the scale factor."""
        self.position = self.position * (scale_factor/SF_Prev)

class MassParticle(Particle):
    def __init__(self, mass, position, velocity):
        super().__init__(mass, position, velocity)
        self.type = "mass"

class RadiationParticle(Particle):
    def __init__(self, mass, position, velocity):
        super().__init__(mass, position, velocity)
        self.type = "radiation"
        # Radiation particles move at or near the speed of light
        vel_magnitude = c
        # Normalize velocity direction and set to speed of light
        if cp.sum(velocity**2) > 0:
            velocity = velocity / cp.sqrt(cp.sum(velocity**2)) * vel_magnitude
=======
import cupy as cp

G = 6.67430e-11   # m^3 kg^-1 s^-2
c = 299792458     # m/s
h_bar = 1.054571917e-34  # Joule*s

class Particle:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = cp.array(position)  # Ensure position is a numpy array
        self.velocity = cp.array(velocity)  # Ensure velocity is a numpy array

    def update_position(self, scale_factor, SF_Prev):
        """Update the physical position based on the scale factor."""
        self.position = self.position * (scale_factor/SF_Prev)

class MassParticle(Particle):
    def __init__(self, mass, position, velocity):
        super().__init__(mass, position, velocity)
        self.type = "mass"

class RadiationParticle(Particle):
    def __init__(self, mass, position, velocity):
        super().__init__(mass, position, velocity)
        self.type = "radiation"
        # Radiation particles move at or near the speed of light
        vel_magnitude = c
        # Normalize velocity direction and set to speed of light
        if cp.sum(velocity**2) > 0:
            velocity = velocity / cp.sqrt(cp.sum(velocity**2)) * vel_magnitude
>>>>>>> 31f20d4 (Started development of the final simulation)
        self.velocity = velocity