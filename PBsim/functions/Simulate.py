import numpy as np
from functions.Particles import PseudoMassParticle

class Node:
    def __init__(self, half_len, center, particle_array):        
        self.half_len      = half_len
        self.center        = np.array(center, dtype=float)
        self.particle_array= particle_array
        
        # will be filled by build_tree()
        self.mass          = 0.0
        self.center_of_mass= np.zeros(3)
        self.children      = [None]*8

class Oct_tree:
    def __init__(self, Particles, particle_choice, root_settings):
        """
        root_settings = (half_len, center, particle_array)
        """
        self.Particles_array  = Particles
        self.Target_Particle  = particle_choice
        half_len, center, arr = root_settings
        
        # make root node
        self.root = Node(half_len, center, arr)
        # build the tree and collect all the 'external' mass nodes
        self.external_nodes = self.build_tree(self.root, [])

    def initialize_root(self, root_settings):
        half_len, center, arr = root_settings
        return Node(half_len, center, arr)

    def build_tree(self, node, external_nodes):
        # if only one or zero in this node, stop
        if len(node.particle_array) <= 1:
            return external_nodes

        # compute COM & total mass
        positions = np.array([p.position for p in node.particle_array])
        masses    = np.array([p.mass     for p in node.particle_array])
        node.mass = masses.sum()
        node.center_of_mass = np.average(positions, axis=0, weights=masses)

        # prepare 8 buckets
        quarter = node.half_len/2.0
        offsets = np.array([
            [-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
            [ 1,-1,-1],[ 1,-1,1],[ 1,1,-1],[ 1,1,1]
        ], dtype=float)*quarter

        children_particles = [[] for _ in range(8)]
        for p in node.particle_array:
            # compute octant index as integer
            idx = 0
            idx |= (p.position[0] > node.center[0]) * 1
            idx |= (p.position[1] > node.center[1]) * 2
            idx |= (p.position[2] > node.center[2]) * 4
            children_particles[int(idx)].append(p)

        # recurse / collect externals
        for i, plist in enumerate(children_particles):
            if not plist:
                continue

            child_center = node.center + offsets[i]
            child = Node(quarter, child_center, plist)
            node.children[i] = child

            # compute its mass and COM
            cp = np.array([p.position for p in plist])
            cm = np.array([p.mass     for p in plist])
            child.mass = cm.sum()
            child.center_of_mass = np.average(cp, axis=0, weights=cm)

            # if the target lives here, go deeper
            if self.Target_Particle in plist:
                self.build_tree(child, external_nodes)
            else:
                # otherwise treat it as one pseudo–particle
                external_nodes.append(
                    PseudoMassParticle(position=child.center_of_mass,
                                       mass=child.mass)
                )

        return external_nodes
