import numpy as np
import pygame


def draw_arrow(screen, at, angle, length, color):
    pygame.draw.line(
        screen,
        color,
        (int(at[0]), int(at[1])),
        (
            int(at[0] + length * np.cos(angle)),
            int(at[1] + length * np.sin(angle)),
        ),
    )
    pygame.draw.line(
        screen,
        color,
        (
            int(at[0] + length * np.cos(angle)),
            int(at[1] + length * np.sin(angle)),
        ),
        (
            int(at[0] + length * np.cos(angle + np.pi / 8)),
            int(at[1] + length * np.sin(angle + np.pi / 8)),
        ),
    )
    pygame.draw.line(
        screen,
        color,
        (
            int(at[0] + length * np.cos(angle)),
            int(at[1] + length * np.sin(angle)),
        ),
        (
            int(at[0] + length * np.cos(angle - np.pi / 8)),
            int(at[1] + length * np.sin(angle - np.pi / 8)),
        ),
    )


def plot_forces(screen, sim):
    for acc, pos in zip(sim.accelerations.cpu().numpy(), sim.particles.cpu().numpy()):
        if acc[0] == 0 and acc[1] == 0:
            continue
        angle = np.arctan2(acc[1], acc[0])
        lenght = 20
        draw_arrow(screen, pos, angle, lenght, (0, 0, 0))
