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
        
        self.mass = mass
        self.position =position
        self.velocity = velocity


class RadiationParticle(Particle):
    def __init__(self, energy, position, velocity):
        super().__init__(position, velocity)
        
        # E = hf
        # c = λf
        # m = hf/c^2

        self.type = "radiation"
        self.energy = energy
        self.position = cp.array(position)
        self.velocity = cp.array(velocity) # Have to make sure that the magnitude of all vectors is c
        
        self.mass = self.energy/(c**2)
        
        
        
        




