# Barnes-Hut N-Body Simulation (Python Implementation)

This is a Python implementation of the Barnes-Hut algorithm for N-body gravitational simulations, based on the [original Rust implementation](https://github.com/DeadlockCode/barnes-hut). The Barnes-Hut algorithm optimizes N-body simulations by approximating distant groups of bodies as single masses, reducing the computational complexity from O(n²) to O(n log n).

## Features
- Fast N-body gravitational simulation using the Barnes-Hut algorithm
- Quadtree-based space partitioning for efficient force calculations
- Visual representation using Pygame
- Particle system initialized in a galaxy-like disc formation

## Requirements
- Python 3.13 or higher
- NumPy 2.2.6 or higher
- Pygame 2.6.1 or higher

## Installation

1. Clone the repository
2. Create and activate a Python virtual environment (recommended)
3. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Running the Simulation

To run the simulation:
```powershell
python main.py
```

## Controls
- Space: Pause/unpause the simulation
- Escape: Exit the simulation

## Project Structure
- `main.py`: Entry point and visualization code
- `simulation.py`: Core simulation logic and physics calculations
- `quadtree.py`: Implementation of the Barnes-Hut quadtree algorithm
- `body.py`: Definition of the Body class representing particles

## Implementation Details

### Barnes-Hut Algorithm
The simulation uses the Barnes-Hut algorithm to approximate gravitational forces between bodies. It works by:
1. Dividing space into a quadtree
2. Treating distant groups of bodies as single point masses
3. Using a threshold angle (θ) to determine when approximation is acceptable

### Physics
- Bodies interact through gravitational forces
- Force calculations use Newton's law of universal gravitation
- Positions and velocities are updated using velocity Verlet integration

## Performance
The implementation can efficiently handle thousands of bodies in real-time, with performance primarily determined by:
- Number of bodies
- Quadtree theta parameter (controls accuracy vs. speed trade-off)
- Display resolution and frame rate cap

## Credits
This implementation is based on the original Rust implementation by DeadlockCode, as shown in the video [How to make HUGE N-Body Simulations](https://youtu.be/nZHjD3cI-EU).