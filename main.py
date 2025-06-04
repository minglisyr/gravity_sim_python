import pygame
import numpy as np
from simulation import Simulation
import time
import threading
from dataclasses import dataclass, field
from typing import Optional
import pygame.gfxdraw
import hsluv

@dataclass
class RenderState:
    positions: np.ndarray
    velocities: np.ndarray  # Added velocities for color calculation
    radii: np.ndarray
    show_quadtree: bool = False
    paused: bool = False
    scale: float = 1.0
    depth_range: tuple = (0, 16)  # Match Rust's quadtree depth visualization
    offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    
class Renderer:
    def __init__(self, window_size):
        self.window_size = window_size
        self.center = np.array([window_size[0]/2, window_size[1]/2])
        self.scale = 1.0
        self.offset = np.array([0.0, 0.0])
        self.show_quadtree = False
        self.background_color = (0, 0, 0)
        self.body_color = (255, 255, 255)
        self.drag_start = None
        self.state_lock = threading.Lock()
        self.current_state: Optional[RenderState] = None
        self.initialized = False  # Track if we've done initial positioning
        
    def update_state(self, positions, velocities, radii):
        with self.state_lock:
            self.current_state = RenderState(
                positions=positions.copy(),
                velocities=velocities.copy(),
                radii=radii.copy(),
                show_quadtree=self.show_quadtree,
                scale=self.scale,
                depth_range=self.depth_range if hasattr(self, 'depth_range') else (0, 16),
                offset=self.offset
            )
            
    def handle_input(self, event): 
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 2:  # Middle mouse button
                self.drag_start = np.array(pygame.mouse.get_pos())
            elif event.button == 4:  # Mouse wheel up
                self.scale *= 1.1
            elif event.button == 5:  # Mouse wheel down
                self.scale /= 1.1
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 2:
                self.drag_start = None
        elif event.type == pygame.MOUSEMOTION:
            if self.drag_start is not None:
                current_pos = np.array(pygame.mouse.get_pos())
                delta = (current_pos - self.drag_start) / self.scale
                self.offset += delta
                self.drag_start = current_pos
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                self.show_quadtree = not self.show_quadtree

    def _get_velocity_color(self, velocity: np.ndarray) -> tuple:
        """Convert velocity to color using HSLuv color space (matching Rust implementation)."""
        speed = np.linalg.norm(velocity)
        if speed == 0:
            return (255, 255, 255)  # White for stationary bodies
            
        # Calculate hue from velocity direction (angle)
        angle = np.arctan2(velocity[1], velocity[0])
        hue = (angle + np.pi) * 180 / np.pi  # Convert to degrees [0, 360)
        
        # Calculate saturation and lightness from speed
        # Use log scale for better visualization
        saturation = min(100.0, 50.0 + np.log10(speed) * 20.0)
        lightness = max(20.0, min(80.0, 60.0 - np.log10(speed) * 10.0))
        
        # Convert HSLuv to RGB
        rgb = hsluv.hsluv_to_rgb([hue, saturation, lightness])
        return tuple(int(x * 255) for x in rgb)

def main():
    # Initialize Pygame
    pygame.init()

    # Constants
    WINDOW_SIZE = (900, 900)
    BACKGROUND_COLOR = (0, 0, 0)

    # Set up the display
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Barnes-Hut N-body Simulation")


    # Use 1,000 bodies to test performance
    n_bodies = 300
    sim = Simulation(n_bodies=n_bodies)
    
    # Create renderer
    renderer = Renderer(WINDOW_SIZE)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Barnes-Hut N-body Simulation")
    
    # Set up FPS tracking
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    fps_update_interval = 0.5
    last_fps_update = time.time()
    fps_display = "FPS: 0"
    frame_count = 0
    
    # Set up simulation thread
    from queue import Queue, Empty
    
    state_queue = Queue(maxsize=1)  # Only keep latest state
    simulation_running = True
    paused = False
    
    def simulation_loop():
        last_update = time.time()
        physics_time = 0
        
        while simulation_running:
            if not paused:
                # Update simulation
                physics_start = time.time()
                sim.step()
                physics_time = (time.time() - physics_start) * 1000
                
                # Update state at most 60 times per second
                current_time = time.time()
                if current_time - last_update >= 1/60:
                    # Try to update state without blocking
                    try:
                        # Get state including velocities
                        positions = sim.get_body_positions()
                        velocities = sim.velocities  # Access velocities directly from sim
                        radii = sim.get_body_radii()
                        state_queue.put_nowait((positions, velocities, radii, physics_time))
                        last_update = current_time
                    except Queue.Full:
                        pass  # Skip update if renderer is behind
            else:
                time.sleep(1/60)  # Don't burn CPU while paused
    
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()

    # Main game loop
    running = True

    # Main render loop
    while running:
        frame_start_time = time.time()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                simulation_running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False
                    simulation_running = False
            
            # Handle renderer input (zoom, pan, etc.)
            renderer.handle_input(event)

        # Try to get latest simulation state
        try:
            positions, velocities, radii, physics_time = state_queue.get_nowait()
            renderer.update_state(positions, velocities, radii)  # Update renderer state with new data
            state_queue.task_done()
        except Empty:
            # If no new state, use last known values
            if renderer.current_state:
                positions = renderer.current_state.positions
                velocities = renderer.current_state.velocities
                radii = renderer.current_state.radii
            else:
                positions = np.array([])
                velocities = np.array([])
                radii = np.array([])
            physics_time = 0

        # Clear screen
        screen.fill(BACKGROUND_COLOR)

            # Draw bodies using the current render state
        with renderer.state_lock:
            if renderer.current_state:
                state = renderer.current_state
                # Transform positions based on view state
                view_scale = state.scale
                screen_center = np.array([WINDOW_SIZE[0]/2, WINDOW_SIZE[1]/2])
                
                # Center the view on the first frame
                if renderer.current_state is not None and not hasattr(renderer, 'initialized'):
                    min_pos = np.min(state.positions, axis=0)
                    max_pos = np.max(state.positions, axis=0)
                    center_pos = (min_pos + max_pos) / 2
                    size = max(max_pos[0] - min_pos[0], max_pos[1] - min_pos[1])
                    renderer.scale = min(WINDOW_SIZE[0], WINDOW_SIZE[1]) / (size * 2.2)
                    renderer.offset = -center_pos
                    renderer.initialized = True
                
                # Apply transformations to all positions at once
                screen_positions = (state.positions + renderer.offset) * view_scale + screen_center
                
                # Draw quadtree if enabled
                if state.show_quadtree and hasattr(sim, 'quadtree'):
                    boundaries = sim.quadtree.get_boundaries()
                    if boundaries:
                        min_depth = min(depth for _, _, depth in boundaries)
                        max_depth = max(depth for _, _, depth in boundaries)
                        
                        for (min_corner, size, depth) in boundaries:
                            # Transform quadtree coordinates to screen space
                            screen_pos = (np.array(min_corner) + renderer.offset) * view_scale + screen_center
                            screen_size = size * view_scale
                            
                            # Calculate color based on depth (matching Rust HSLuv coloring)
                            t = (depth - min_depth) / max((max_depth - min_depth), 1)
                            
                            # Map depth to HSLuv color space (match Rust implementation)
                            start_h = -100.0
                            end_h = 80.0
                            h = start_h + (end_h - start_h) * t  # Hue range from -100 to 80
                            s = 100.0  # Full saturation
                            l = t * 100.0  # Lightness increases with depth
                            
                            # Convert HSLuv to RGB
                            rgb = hsluv.hsluv_to_rgb([h, s, l])
                            color = tuple(int(x * 255) for x in rgb)
                            
                            # Draw rectangle with anti-aliasing for smoother appearance
                            pygame.gfxdraw.rectangle(screen, 
                                (int(screen_pos[0]), int(screen_pos[1]),
                                 int(screen_size), int(screen_size)), 
                                color)
                            # Draw outline slightly darker
                            outline = tuple(int(x * 0.8) for x in color)
                            pygame.draw.rect(screen, outline,
                                          (int(screen_pos[0]), int(screen_pos[1]),
                                           int(screen_size), int(screen_size)), 1)
                
                # Draw bodies with velocity-based colors
                min_radius = 3.0  # Minimum visible radius
                for pos, vel, radius in zip(screen_positions, state.velocities, state.radii):
                    # Get color based on velocity
                    color = renderer._get_velocity_color(vel)
                    
                    # Scale body size by mass (use log scale)
                    display_radius = np.log1p(radius) * view_scale
                    screen_radius = max(display_radius, min_radius)
                    
                    # Scale central body differently (it has much larger mass)
                    if radius > 20:  # Central body
                        screen_radius = max(10 * view_scale, min_radius)
                        color = (255, 255, 255)  # White for central body
                    
                    # Draw only if in screen bounds (with some margin)
                    margin = max(screen_radius, 50)  # Allow partially visible bodies
                    if (-margin <= pos[0] <= WINDOW_SIZE[0] + margin and 
                        -margin <= pos[1] <= WINDOW_SIZE[1] + margin):
                        # Use anti-aliased circles for better appearance
                        if screen_radius < 3:
                            # Small bodies: filled circle with anti-aliasing
                            pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), 2, color)
                            pygame.gfxdraw.aacircle(screen, int(pos[0]), int(pos[1]), 2, color)
                        else:
                            # Larger bodies: filled circle with outline
                            radius = max(int(screen_radius), 1)
                            pygame.gfxdraw.filled_circle(screen, int(pos[0]), int(pos[1]), radius, color)
                            pygame.gfxdraw.aacircle(screen, int(pos[0]), int(pos[1]), radius, color)

        # Update and display FPS
        frame_count += 1
        current_time = time.time()
        if current_time - last_fps_update > fps_update_interval:
            fps = frame_count / (current_time - last_fps_update)
            fps_display = f"FPS: {fps:.1f}"
            if not paused:
                fps_display += f" | Physics: {physics_time:.1f}ms | Bodies: {n_bodies:,}"
            frame_count = 0
            last_fps_update = current_time

        # Render FPS text
        fps_surface = font.render(fps_display, True, (255, 255, 0))
        screen.blit(fps_surface, (10, 10))

        # Update display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    # Clean up
    pygame.quit()


if __name__ == "__main__":
    main()
