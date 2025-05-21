import numpy as np

class Body:
    def __init__(self, pos, vel, mass, radius):
        """Initialize a body with position, velocity, mass and radius."""
        self.pos = np.array(pos, dtype=np.float32)  # 2D position vector
        self.vel = np.array(vel, dtype=np.float32)  # 2D velocity vector
        self.acc = np.zeros(2, dtype=np.float32)    # 2D acceleration vector
        self.mass = np.float32(mass)
        self.radius = np.float32(radius)

    def update(self, dt):
        """Update position and velocity using current acceleration."""
        self.vel += self.acc * dt
        self.pos += self.vel * dt
