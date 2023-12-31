import time

import numpy as np
import pygame
import torch

from profiling import Profiler

profiler = Profiler()

# from sim import Simulation
import sims.sim as sim
from visualize.density import calc_density, plot_density
from visualize.gradviz import plot_forces

pygame.init()
screen = pygame.display.set_mode((400, 400), pygame.RESIZABLE)

pygame.display.set_caption("Visualizer")

sim = sim.Simulation(
    1000,
    2,
    device=torch.device("mps"),
    x1=20,
    x2=screen.get_width() - 20,
    y1=20,
    y2=screen.get_height() - 20,
)

# sim.particles = torch.tensor([[200.0, 100.0], [150.0, 100.0]], device=sim.device)


def spawn_particles():
    sim.particles = torch.rand(sim.amount, 2, device=sim.device) * torch.tensor(
        [screen.get_width(), screen.get_height()],
        dtype=torch.float32,
        device=sim.device,
    )
    sim.velocities = torch.zeros_like(sim.particles, dtype=torch.float32)


spawn_particles()

particle_radius = 3
particle_color = (0, 0, 255)

viz_density = False
steps_per_frame = 20

mouse_radius = 100
show_mouse_circle = False
last_scroll_time = None
apply_force = 0
force_strength = 9e2

scale_frames = 0
scale = 1

running = True
count = 0
last_time = time.perf_counter()
avg_fps = None
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            sim.set_bounds(0, event.w, 0, event.h)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                sim.pressure_multiplier *= 1.1
                print("pressure_multiplier:", sim.pressure_multiplier)
            elif event.key == pygame.K_DOWN:
                sim.pressure_multiplier /= 1.1
                print("pressure_multiplier:", sim.pressure_multiplier)
            elif event.key == pygame.K_LEFT:
                sim.target_density /= 1.1
            elif event.key == pygame.K_RIGHT:
                sim.target_density *= 1.1
            elif event.key == pygame.K_SPACE:
                sim.simulate_pressure = not sim.simulate_pressure
            elif event.key == pygame.K_g:
                sim.simulate_gravity = not sim.simulate_gravity
            elif event.key == pygame.K_r:
                spawn_particles()
            elif event.key == pygame.K_w:
                scale_frames += 1000
                scale = pow(1.1, 1 / 1000)
            elif event.key == pygame.K_s:
                scale_frames += 1000
                scale = pow(1 / 1.1, 1 / 1000)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                print("Mouse position:", event.pos)
                apply_force = 1
            if event.button == 3:
                apply_force = -1
            show_mouse_circle = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                print("Mouse position:", event.pos)
                apply_force = 0
            if event.button == 3:
                apply_force = 0
            show_mouse_circle = False
        elif event.type == pygame.MOUSEWHEEL:
            if event.y > 0:
                mouse_radius *= 1.1
            elif event.y < 0:
                mouse_radius /= 1.1
            last_scroll_time = time.perf_counter()
            show_mouse_circle = True

    if (
        last_scroll_time is not None
        and time.perf_counter() - last_scroll_time > 0.5
        and apply_force == 0
    ):
        show_mouse_circle = False
        last_scroll_time = None

    screen.fill((0, 0, 0))

    dt = time.perf_counter() - last_time
    last_time = time.perf_counter()
    fps = steps_per_frame / dt
    if avg_fps is None:
        avg_fps = fps
    else:
        avg_fps = avg_fps * 0.9 + fps * 0.1

    if count % (avg_fps // steps_per_frame) == 0:
        print(
            "FPS:",
            avg_fps,
        )

    if viz_density:
        if count % 10 == 0:
            density = calc_density(screen, sim, scale=4)
        plot_density(screen, density, sim)
        plot_forces(screen, sim)

    if apply_force != 0:
        sim.apply_force(
            force_strength * apply_force,
            torch.tensor(
                pygame.mouse.get_pos(), dtype=torch.float32, device=sim.device
            ),
            mouse_radius,
        )

    profiler.start("step")
    for _ in range(steps_per_frame):
        if scale_frames > 0:
            scale_frames -= 1
            sim.scale_system(scale)
        sim.step(dt)
    profiler.stop("step")

    profiler.start("draw")

    for particle, speed in zip(
        sim.particles.detach().cpu().numpy(),
        torch.norm(sim.velocities, dim=1).detach().cpu().numpy(),
    ):
        particle = np.round(particle).astype(int)
        sp = 1 / (speed * 2 + 1)
        color = (255 - sp * 255, 0, sp * 255)
        pygame.draw.circle(
            screen, color, particle, int(sim.smoothing_radius**2 * 0.01)
        )

    profiler.stop("draw")

    if show_mouse_circle:
        pygame.draw.circle(
            screen,
            (255, 255, 255),
            pygame.mouse.get_pos(),
            mouse_radius,
            width=1,
        )

    count += 1
    pygame.display.flip()

from profiling.present import SourcePresenter
from profiling.present.metrics import FPS, Name

presenter = SourcePresenter(
    profiler, metrics=[Name(), FPS()], format="{} {}", leaves_only=True
)

presenter.present()

pygame.quit()
