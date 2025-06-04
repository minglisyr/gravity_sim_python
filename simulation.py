import numpy as np
from body import Body
from quadtree import QuadTree
import os
from concurrent.futures import ThreadPoolExecutor

class Simulation:
    def __init__(self, n_bodies=100000):
        self.dt = 0.05  # Match Rust timestep
        self.frame = 0
        
        # Initialize arrays for better performance
        self.positions = np.zeros((n_bodies, 2), dtype=np.float32)
        self.velocities = np.zeros((n_bodies, 2), dtype=np.float32)
        self.accelerations = np.zeros((n_bodies, 2), dtype=np.float32)
        self.masses = np.zeros(n_bodies, dtype=np.float32)
        self.radii = np.zeros(n_bodies, dtype=np.float32)
        
        # Create bodies with proper initialization
        self._initialize_bodies(n_bodies)
        
        # Initialize quadtree with parameters matching Rust
        self.quadtree = QuadTree(theta=1.0, epsilon=1.0, leaf_capacity=16)
        
        # Convert bodies to list format for compatibility
        self.bodies = [Body(self.positions[i].copy(), self.velocities[i].copy(), 
                           self.masses[i], self.radii[i]) 
                      for i in range(n_bodies)]
        
        # Initialize thread pool for parallel computation
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())

    def _compute_forces_chunk(self, chunk_range):
        """Compute forces for a chunk of bodies."""
        start_idx, end_idx = chunk_range
        forces = np.zeros((end_idx - start_idx, 2), dtype=np.float32)
        for i in range(start_idx, end_idx):
            forces[i - start_idx] = self.quadtree.compute_force(self.bodies[i], i)
        return forces

    def step(self) -> None:
        """Advance the simulation by one timestep."""
        self.frame += 1
        
        # Update quadtree
        self.quadtree.build(self.bodies)
        
        # Calculate forces using parallel processing
        n_bodies = len(self.bodies)
        n_threads = os.cpu_count() or 1
        chunk_size = max(1, n_bodies // n_threads)
        
        # Create chunks for parallel processing
        chunks = []
        for start in range(0, n_bodies, chunk_size):
            end = min(start + chunk_size, n_bodies)
            chunks.append((start, end))
        
        # Process chunks in parallel using thread pool
        futures = [self.executor.submit(self._compute_forces_chunk, chunk) 
                  for chunk in chunks]
        accelerations = np.concatenate([future.result() for future in futures])
        
        # Update all bodies using vectorized operations
        self.velocities += accelerations * self.dt
        self.positions += self.velocities * self.dt
        
        # Update bodies list
        for i in range(n_bodies):
            self.bodies[i].pos = self.positions[i].copy()
            self.bodies[i].vel = self.velocities[i].copy()
            self.bodies[i].acc = accelerations[i]

        # Handle collisions
        self._handle_collisions()

    def _initialize_bodies(self, n: int) -> None:
        """Initialize bodies in a uniform disc formation, matching Rust implementation."""
        # Constants (same as Rust)
        inner_radius = 25.0
        outer_radius = np.sqrt(n) * 5.0
        center_mass = 1e6

        # Set central massive body
        self.positions[0] = np.array([0.0, 0.0], dtype=np.float32)
        self.velocities[0] = np.array([0.0, 0.0], dtype=np.float32)
        self.masses[0] = center_mass
        self.radii[0] = inner_radius
        
        # Create remaining bodies with consistent random state
        np.random.seed(0)
        angles = np.random.random(n-1) * 2 * np.pi
        t = inner_radius / outer_radius
        r = np.random.random(n-1) * (1.0 - t * t) + t * t
        r_sqrt = np.sqrt(r)
        
        # Calculate positions (vectorized)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        self.positions[1:, 0] = cos_angles * outer_radius * r_sqrt
        self.positions[1:, 1] = sin_angles * outer_radius * r_sqrt
        
        # Set initial velocities perpendicular to positions
        self.velocities[1:, 0] = sin_angles
        self.velocities[1:, 1] = -cos_angles
        
        # Set masses and radii
        self.masses[1:] = 1.0
        self.radii[1:] = np.cbrt(self.masses[1:])
        
        # Sort bodies by distance from center (excluding central body)
        distances = np.sum(self.positions[1:] * self.positions[1:], axis=1)
        sort_indices = np.argsort(distances) + 1  # +1 to skip central body
        
        self.positions[1:] = self.positions[sort_indices]
        self.velocities[1:] = self.velocities[sort_indices]
        self.masses[1:] = self.masses[sort_indices]
        self.radii[1:] = self.radii[sort_indices]
        
        # Adjust velocities for Keplerian orbits
        total_mass = np.cumsum(self.masses)
        for i in range(1, n):
            pos_mag = np.linalg.norm(self.positions[i])
            if pos_mag > 0:  # Skip central body
                v_scale = np.sqrt(total_mass[i-1] / pos_mag)
                self.velocities[i] *= v_scale

    def _handle_collisions(self):
        """Handle collisions between bodies (matching Rust implementation)."""
        for i in range(len(self.bodies)):
            pos_i = self.positions[i]
            vel_i = self.velocities[i]
            radius_i = self.radii[i]
            mass_i = self.masses[i]

            for j in range(i + 1, len(self.bodies)):
                pos_j = self.positions[j]
                vel_j = self.velocities[j]
                radius_j = self.radii[j]
                mass_j = self.masses[j]

                d = pos_j - pos_i
                r = radius_i + radius_j

                # Check if bodies are colliding
                if np.dot(d, d) > r * r:
                    continue

                v = vel_j - vel_i
                d_dot_v = np.dot(d, v)
                
                if d_dot_v >= 0.0 and not np.array_equal(d, np.zeros(2)):
                    # Elastic collision response
                    weight1 = mass_j / (mass_i + mass_j)
                    weight2 = mass_i / (mass_i + mass_j)
                    
                    # Move bodies apart
                    d_mag = np.linalg.norm(d)
                    tmp = d * (r / d_mag - 1.0)
                    self.positions[i] -= weight1 * tmp
                    self.positions[j] += weight2 * tmp
                    continue

                d_mag_sq = np.dot(d, d)
                v_mag_sq = np.dot(v, v)
                r_sq = r * r

                # Conservation of momentum
                discriminant = d_dot_v * d_dot_v - v_mag_sq * (d_mag_sq - r_sq)
                t = (d_dot_v + np.sqrt(max(0.0, discriminant))) / v_mag_sq if v_mag_sq > 0 else 0.0

                # Backup positions to collision time
                self.positions[i] -= self.velocities[i] * t
                self.positions[j] -= self.velocities[j] * t

                # Update positions and velocities after collision
                d = self.positions[j] - self.positions[i]
                d_mag_sq = np.dot(d, d)
                
                if d_mag_sq > 0:
                    # Apply collision impulse
                    weight1 = mass_j / (mass_i + mass_j)
                    weight2 = mass_i / (mass_i + mass_j)
                    
                    d_dot_v = np.dot(d, v)
                    tmp = d * (1.5 * d_dot_v / d_mag_sq)
                    self.velocities[i] += tmp * weight1
                    self.velocities[j] -= tmp * weight2

                # Move bodies to post-collision positions
                self.positions[i] += self.velocities[i] * t
                self.positions[j] += self.velocities[j] * t

                # Keep bodies list in sync
                self.bodies[i].pos = self.positions[i].copy()
                self.bodies[i].vel = self.velocities[i].copy()
                self.bodies[j].pos = self.positions[j].copy()
                self.bodies[j].vel = self.velocities[j].copy()

    def get_body_positions(self) -> np.ndarray:
        """Return positions of all bodies for rendering."""
        return self.positions.copy()

    def get_body_radii(self) -> np.ndarray:
        """Return radii of all bodies for rendering."""
        return self.radii.copy()
