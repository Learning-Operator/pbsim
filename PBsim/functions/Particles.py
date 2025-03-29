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
        
        self.position = cp.array(position)
        
        self.position_physical = cp.array(position)
        self.velocity_physical = cp.array(velocity)

        self.position_comoving = cp.array(position)
        self.velocity_comoving = cp.array(velocity)
        
    def transform_to_comoving(self, phys_pos, Sf):
        return phys_pos/Sf
    
    def transform_to_actual(self,com_pos, Sf):
        return com_pos * Sf
        

class RadiationParticle(Particle):
    def __init__(self, energy, position, velocity):
        
        # E = hf
        # c = λf
        # m = hf/c^2

        self.type = "radiation"
        self.energy = energy
        
        self.position = cp.array(position)
        self.velocity = cp.array(velocity)
        
        self.position_physical = cp.array(position)
        self.velocity_physical = cp.array(velocity)
        
        self.position_comoving = cp.array(position)
        self.velocity_comoving = cp.array(velocity)
        
        self.mass = self.energy/(c**2)
        
    def transform_to_comoving(self, phys_pos, Sf):
        return phys_pos/Sf

    def transform_to_actual(self,com_pos, Sf):
        return com_pos * Sf
        
        




