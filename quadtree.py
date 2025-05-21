import numpy as np
from body import Body
from typing import List, Optional

class QuadNode:
    __slots__ = ['center', 'size', 'mass', 'com', 'children', 'body_indices', 'is_leaf']
    
    def __init__(self, center: np.ndarray, size: float):
        self.center = center
        self.size = size
        self.mass = 0.0
        self.com = np.zeros(2, dtype=np.float32)  # Center of mass
        self.children = [None] * 4  # NW, NE, SW, SE
        self.body_indices: List[int] = []  # Store indices instead of body objects
        self.is_leaf = True

class QuadTree:
    def __init__(self, theta: float = 0.5, epsilon: float = 1e-4, leaf_capacity: int = 16):
        self.theta = theta
        self.epsilon = epsilon
        self.leaf_capacity = leaf_capacity        
        self.root: Optional[QuadNode] = None
        self.bodies: List[Body] = []
        self.positions: Optional[np.ndarray] = None
        self.masses: Optional[np.ndarray] = None
        self.nodes: List[QuadNode] = []  # All nodes in the quadtree
        
    def build(self, bodies: List[Body]) -> None:
        if not self.bodies:
            return

        self.bodies = bodies
        n_bodies = len(bodies)
        
        # Pre-compute positions and masses arrays for faster access
        self.positions = np.array([b.pos for b in bodies], dtype=np.float32)
        self.masses = np.array([b.mass for b in bodies], dtype=np.float32)

        # Find bounds
        min_pos = np.min(self.positions, axis=0)
        max_pos = np.max(self.positions, axis=0)
        center = (min_pos + max_pos) / 2
        size = max(max_pos[0] - min_pos[0], max_pos[1] - min_pos[1]) * 1.1

        # Clear previous nodes and create root
        self.nodes = []
        root = QuadNode(center, size)
        root.body_indices = list(range(n_bodies))
        self.nodes.append(root)
        
        # Build tree in parallel for large datasets
        self._subdivide_parallel(root, 0, n_bodies)
        
        # Propagate mass and center of mass upward
        self._propagate_mass()

    def _subdivide(self, node: QuadNode) -> None:
        if len(node.body_indices) <= self.leaf_capacity:
            # Compute center of mass for leaf node
            node.mass = np.sum(self.masses[node.body_indices])
            if node.mass > 0:
                node.com = np.average(self.positions[node.body_indices], 
                                    weights=self.masses[node.body_indices],
                                    axis=0)
            return

        # Create child nodes
        node.is_leaf = False
        half_size = node.size / 2

        for i in range(4):
            dx = half_size * (1 if i in [1, 3] else -1)
            dy = half_size * (1 if i in [0, 1] else -1)
            child_center = node.center + np.array([dx, dy], dtype=np.float32)
            node.children[i] = QuadNode(child_center, node.size/2)

        # Distribute bodies to children
        for idx in node.body_indices:
            pos = self.positions[idx]
            quadrant = self._get_quadrant(pos, node.center)
            node.children[quadrant].body_indices.append(idx)

        # Clear parent node's body indices
        node.body_indices = []

        # Recursively subdivide children
        node.mass = 0
        node.com = np.zeros(2, dtype=np.float32)
        
        for child in node.children:
            if child.body_indices:
                self._subdivide(child)
                node.mass += child.mass
                node.com += child.mass * child.com

        if node.mass > 0:
            node.com /= node.mass

    def _subdivide_parallel(self, node: QuadNode, start: int, size: int) -> None:
        """Subdivide a node using parallel computation for large enough partitions."""
        if size <= self.leaf_capacity:
            # Small enough for leaf node - compute center of mass
            if size > 0:
                positions = self.positions[node.body_indices[start:start + size]]
                masses = self.masses[node.body_indices[start:start + size]]
                node.mass = np.sum(masses)
                if node.mass > 0:
                    node.com = np.average(positions, weights=masses, axis=0)
            return

        # Divide into quadrants in parallel for large partitions
        # Calculate center splits
        center = node.center
        ranges = []
        quad_indices = [[] for _ in range(4)]

        # Sort bodies into quadrants
        for i in range(start, start + size):
            idx = node.body_indices[i]
            pos = self.positions[idx]
            quadrant = self._get_quadrant(pos, center)
            quad_indices[quadrant].append(idx)

        # Process each quadrant
        node.is_leaf = False
        next_free = len(self.nodes)
        node.children = next_free
        half_size = node.size / 2

        # Create child nodes and process recursively
        for i in range(4):
            bodies = quad_indices[i]
            if bodies:
                dx = half_size * (1 if i in [1, 3] else -1)
                dy = half_size * (1 if i in [0, 1] else -1)
                child_center = node.center + np.array([dx, dy], dtype=np.float32)
                
                new_node = QuadNode(child_center, node.size/2)
                new_node.body_indices = bodies
                self.nodes.append(new_node)
                
                # Process child node recursively
                self._subdivide_parallel(new_node, 0, len(bodies))
            else:
                # Empty quadrant
                empty_node = QuadNode(node.center, node.size/2)
                self.nodes.append(empty_node)

    def _get_quadrant(self, pos: np.ndarray, center: np.ndarray) -> int:
        if pos[0] >= center[0]:
            return 1 if pos[1] >= center[1] else 3
        return 0 if pos[1] >= center[1] else 2
        
    def get_nodes(self) -> List[QuadNode]:
        """Get all nodes in the quadtree."""
        if self.root is None:
            return []
            
        nodes = []
        stack = [self.root]
        
        while stack:
            node = stack.pop()
            nodes.append(node)
            
            if not node.is_leaf:
                for child in node.children:
                    if child is not None and child.mass > 0:
                        stack.append(child)
        
        return nodes

    def compute_force(self, body: Body, body_idx: int) -> np.ndarray:
        if self.root is None:
            return np.zeros(2, dtype=np.float32)
        
        force = np.zeros(2, dtype=np.float32)
        self._compute_force_recursive(self.root, body_idx, force)
        return force

    def _compute_force_recursive(self, node: QuadNode, body_idx: int, force: np.ndarray) -> None:
        if node.is_leaf and not node.body_indices:
            return

        # Vector from body to node's center of mass
        r = node.com - self.positions[body_idx]
        r_sq = np.sum(r * r) + self.epsilon
        r_mag = np.sqrt(r_sq)

        # If node is a leaf or is far enough away
        if node.is_leaf or (node.size / r_mag < self.theta):
            if r_mag > 0:  # Avoid self-interaction
                # Gravitational force = G * m1 * m2 * r / |r|^3
                G = 1.0  # Gravitational constant
                force_mag = G * self.masses[body_idx] * node.mass / (r_sq * r_mag)
                force += force_mag * r
            return

        # Otherwise, recursively compute forces from children
        for child in node.children:
            if child and child.mass > 0:
                self._compute_force_recursive(child, body_idx, force)

    def _collect_nodes(self, node: QuadNode, nodes: List[tuple]) -> None:
        """Collect node boundaries for visualization."""
        if node is None:
            return
            
        # Add current node's boundary
        half_size = node.size / 2
        min_x = node.center[0] - half_size
        min_y = node.center[1] - half_size
        nodes.append((
            (min_x, min_y),
            node.size,
            not node.is_leaf
        ))

        # Recursively collect child nodes
        if not node.is_leaf:
            for child in node.children:
                if child is not None:
                    self._collect_nodes(child, nodes)

    def get_boundaries(self) -> List[tuple]:
        """Get all node boundaries for visualization.
        Returns:
            List of tuples: (min_corner, size, has_children)
        """
        nodes = []
        if self.root:
            self._collect_nodes(self.root, nodes)
        return nodes

    def _propagate_mass(self):
        """Propagate mass and center of mass upward through the tree."""
        for node in reversed(self.nodes):
            if not node.is_leaf:
                # Get child nodes
                children = []
                for i in range(4):
                    if node.children + i < len(self.nodes):
                        children.append(self.nodes[node.children + i])
                
                # Calculate total mass and center of mass from children
                total_mass = 0.0
                weighted_pos = np.zeros(2, dtype=np.float32)
                for child in children:
                    total_mass += child.mass
                    weighted_pos += child.com * child.mass

                node.mass = total_mass
                if total_mass > 0:
                    node.com = weighted_pos / total_mass
