import numpy as np
from body import Body
from quadtree import QuadTree
from typing import List, Tuple
import numpy.typing as npt
import multiprocessing as mp
from functools import partial
import os

def _barnes_hut_force(pos: np.ndarray, node_pos: np.ndarray, node_mass: float, 
                    node_size: float, theta: float, epsilon: float) -> np.ndarray:
    """Calculate force on a body from a node using Barnes-Hut approximation."""
    r = node_pos - pos
    r_mag_sq = np.sum(r * r) + epsilon
    r_mag = np.sqrt(r_mag_sq)
    
    # If node is far enough away, use approximation
    if node_size / r_mag < theta:
        G = 1.0  # Gravitational constant
        force_magnitude = G * node_mass / (r_mag_sq * r_mag)
        return force_magnitude * r
    return None

def _force_calculation_worker(body_chunk: Tuple[int, int], positions: np.ndarray, 
                            masses: np.ndarray, node_data: List[Tuple[np.ndarray, float, float]], 
                            theta: float, epsilon: float) -> np.ndarray:
    """Worker function for parallel force calculation using Barnes-Hut algorithm."""
    start_idx, end_idx = body_chunk
    chunk_size = end_idx - start_idx
    forces = np.zeros((chunk_size, 2), dtype=np.float32)
    
    for i in range(start_idx, end_idx):
        total_force = np.zeros(2, dtype=np.float32)
        pos = positions[i]
        
        # Process each node in the quadtree
        for node_pos, node_mass, node_size in node_data:
            # Skip empty nodes and self
            if node_mass > 0:
                force = _barnes_hut_force(pos, node_pos, node_mass, node_size, theta, epsilon)
                if force is not None:
                    total_force += force
        
        forces[i - start_idx] = total_force
    
    return forces
    
    return forces

class Simulation:
    def __init__(self, n_bodies=1000):
        self.dt = 0.05
        self.frame = 0
        self.bodies = self._initialize_bodies(n_bodies)
        self.quadtree = QuadTree(theta=0.5, epsilon=1e-4, leaf_capacity=16)
        
        # Initialize multiprocessing
        self.n_processes = max(os.cpu_count() - 1, 1)  # Leave one core free
        self.pool = mp.Pool(processes=self.n_processes)
        
        # Pre-allocate arrays for better performance
        self.positions = np.zeros((n_bodies, 2), dtype=np.float32)
        self.velocities = np.zeros((n_bodies, 2), dtype=np.float32)
        self.masses = np.zeros(n_bodies, dtype=np.float32)
        self.forces = np.zeros((n_bodies, 2), dtype=np.float32)

    def _initialize_bodies(self, n: int) -> List[Body]:
        """Initialize bodies in a disc formation with optimized parameters."""
        # Pre-allocate arrays for vectorized operations
        radii = np.random.random(n) * 300 * np.sqrt(np.random.random(n))
        angles = np.random.random(n) * 2 * np.pi
        masses = np.random.pareto(2.5, n) * 0.1
        
        # Compute positions
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        
        # Compute velocities (Keplerian orbits)
        speeds = np.sqrt(2.0 / np.maximum(radii, 10)) * 20
        vxs = -speeds * np.sin(angles)
        vys = speeds * np.cos(angles)
        
        # Create bodies using vectorized computations
        body_radii = (masses ** 0.5) * 2
        bodies = []
        
        for i in range(n):
            bodies.append(Body(
                [xs[i], ys[i]],
                [vxs[i], vys[i]],
                masses[i],
                body_radii[i]
            ))
        
        return bodies    def step(self) -> None:
        """Advance the simulation by one timestep using parallel Barnes-Hut algorithm."""
        # Update arrays with current body states
        for i, body in enumerate(self.bodies):
            self.positions[i] = body.pos
            self.masses[i] = body.mass

        # Build quadtree
        self.quadtree.build(self.bodies)
        
        # Extract node data for parallel processing
        node_data = []
        tree = self.quadtree
        for node in tree.get_nodes():  # You'll need to implement this in QuadTree
            if node.mass > 0:
                node_data.append((node.com, node.mass, node.size))

        # Divide work among processes
        n_bodies = len(self.bodies)
        chunk_size = max(n_bodies // self.n_processes, 1)
        chunks = [(i, min(i + chunk_size, n_bodies)) 
                 for i in range(0, n_bodies, chunk_size)]

        # Calculate forces in parallel
        worker_func = partial(_force_calculation_worker,
                            positions=self.positions,
                            masses=self.masses,
                            node_data=node_data,
                            theta=self.quadtree.theta,
                            epsilon=self.quadtree.epsilon)
        
        force_chunks = self.pool.map(worker_func, chunks)
        
        # Combine force chunks and update bodies
        start_idx = 0
        for forces in force_chunks:
            chunk_size = forces.shape[0]
            self.forces[start_idx:start_idx + chunk_size] = forces
            start_idx += chunk_size

        # Update accelerations and positions using vectorized operations
        accelerations = self.forces / self.masses[:, np.newaxis]
        
        # Update all bodies with new accelerations
        for i, body in enumerate(self.bodies):
            body.acc = accelerations[i]
            body.update(self.dt)
        
        self.frame += 1

    def get_body_positions(self) -> np.ndarray:
        """Return positions of all bodies for rendering."""
        return self.positions.copy()

    def get_body_radii(self) -> np.ndarray:
        """Return radii of all bodies for rendering."""
        return np.array([body.radius for body in self.bodies])

    def __del__(self):
        """Clean up multiprocessing resources."""
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()
