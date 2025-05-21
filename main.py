import pygame
import numpy as np
from simulation import Simulation

def main():
    # Initialize Pygame
    pygame.init()

    # Constants
    WINDOW_SIZE = (900, 900)
    CENTER = np.array([WINDOW_SIZE[0]/2, WINDOW_SIZE[1]/2])
    BACKGROUND_COLOR = (0, 0, 0)
    BODY_COLOR = (255, 255, 255)

    # Set up the display
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Barnes-Hut N-body Simulation")

    # Create simulation
    sim = Simulation(n_bodies=1000)

    # Main game loop
    running = True
    paused = False
    clock = pygame.time.Clock()

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not paused:
            # Update simulation
            sim.step()

        # Clear screen
        screen.fill(BACKGROUND_COLOR)

        # Get body positions and radii
        positions = sim.get_body_positions()
        radii = sim.get_body_radii()

        # Draw bodies
        for pos, radius in zip(positions, radii):
            screen_pos = pos + CENTER
            pygame.draw.circle(screen, BODY_COLOR, screen_pos.astype(int), max(int(radius), 1))

        # Update display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    # Clean up
    pygame.quit()


if __name__ == "__main__":
    main()
