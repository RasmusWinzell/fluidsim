import numpy as np
import pygame
import torch


def calc_density(screen, sim, scale=16):
    scale = int(scale)

    sizeX = screen.get_width() // scale
    sizeY = screen.get_height() // scale

    xs = torch.arange(0, sizeX, device=sim.device) * scale
    ys = torch.arange(0, sizeY, device=sim.device) * scale
    grid = torch.stack(torch.meshgrid(xs, ys), dim=-1).reshape(-1, 2)

    array = (sim.calc_density(grid) - sim.target_density).reshape(sizeX, sizeY)

    # array = np.zeros((sx, sy))  # Replace this with your array

    # for x in range(sx):
    #     for y in range(sy):
    #         # array[x, y] = (
    #         #     calc_prop(torch.tensor([x, y], device=sim.device), props, sim).cpu().numpy()
    #         # )
    #         array[x, y] = (
    #             sim.calc_density(torch.tensor([x, y], device=sim.device)).cpu().numpy()
    #         )

    return array


def plot_density(screen, density, sim):
    array = density

    std = torch.std(array)

    array = array / std

    neg_array = torch.min(array, torch.zeros_like(array))
    pos_array = torch.max(array, torch.zeros_like(array))
    rarray = 1 / (1 + neg_array * neg_array)
    garray = 1 / (1 + array * array)
    barray = 1 / (1 + pos_array * pos_array)

    array = torch.dstack((rarray, garray, barray))  # Add two more dimensions

    array = (array * 255).cpu().numpy().astype(np.uint8)  # Convert to 8-bit integer

    # Create a new Pygame surface
    surface = pygame.Surface(array.shape[:2])

    # Copy the array to the surface
    pygame.surfarray.blit_array(surface, array)
    surface = pygame.transform.scale(surface, (screen.get_width(), screen.get_height()))

    # Display the surface (replace (0, 0) with the desired position)
    screen.blit(surface, (0, 0))
