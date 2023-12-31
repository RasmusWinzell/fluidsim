import sys
import time
from array import array

import moderngl
import numpy as np
import pygame
import torch

import sims.sim as sim
from presentation.image_sim import ImageSim
from profiling import Profiler
from visualize.particleGL import particleGL

profiler = Profiler()

pygame.init()

pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

screen = pygame.display.set_mode(
    (1200, 600), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
)
display = pygame.Surface((800, 600))
ctx = moderngl.create_context()
print("Using GLSL version:", ctx.info["GL_VERSION"])

clock = pygame.time.Clock()

quad_buffer = ctx.buffer(
    data=array(
        "f",
        [
            # position (x, y), uv coords (x, y)
            -1.0,
            1.0,
            0.0,
            0.0,  # topleft
            1.0,
            1.0,
            1.0,
            0.0,  # topright
            -1.0,
            -1.0,
            0.0,
            1.0,  # bottomleft
            1.0,
            -1.0,
            1.0,
            1.0,  # bottomright
        ],
    )
)


vert_shader = """
#version 330 core

in vec2 vert;
in vec2 texcoord;
out vec2 uvs;

void main() {
    uvs = texcoord;
    gl_Position = vec4(vert, 0.0, 1.0);
}
"""

frag_shader = """
#version 330 core

uniform sampler2D tex;
uniform float time;

in vec2 uvs;
out vec4 f_color;

void main() {
    vec2 sample_pos = vec2(uvs.x + sin(uvs.y * 10 + time * 0.01) * 0.1, uvs.y);
    f_color = vec4(texture(tex, sample_pos).rg, texture(tex, sample_pos).b * 1.5, 1.0);
}
"""

program = ctx.program(vertex_shader=vert_shader, fragment_shader=frag_shader)
render_object = ctx.vertex_array(program, [(quad_buffer, "2f 2f", "vert", "texcoord")])


def surf_to_texture(surf):
    tex = ctx.texture(surf.get_size(), 4)
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex.swizzle = "BGRA"
    tex.write(surf.get_view("1"))
    return tex


sim = sim.Simulation(
    3000,
    2,
    device=torch.device("mps"),
    x1=20,
    x2=screen.get_width() - 20,
    y1=20,
    y2=screen.get_height() - 20,
)
sim.set_pressure_multiplier(6e4)


def spawn_particles():
    sim.particles = torch.rand(sim.amount, 2, device=sim.device) * torch.tensor(
        [screen.get_width(), screen.get_height()],
        dtype=torch.float32,
        device=sim.device,
    )
    sim.velocities = torch.zeros_like(
        sim.particles, dtype=torch.float32, device=sim.device
    )


particles = torch.zeros_like(sim.particles, dtype=torch.float32)

spawn_particles()

particles_vis = particleGL(ctx, particles.cpu().numpy())

screen_size = torch.tensor([screen.get_width(), screen.get_height()], device=sim.device)

idxs = (torch.rand(int(sim.amount * 0.1), device=sim.device) * (sim.amount - 1)).long()

particles = (sim.particles / screen_size * 2 - 1).detach().cpu().numpy()

particle_radius = 3
particle_color = (0, 0, 255)

viz_density = False
steps_per_frame = 20
target_fps = 30

mouse_radius = 100
show_mouse_circle = False
last_scroll_time = None
apply_force = 0
force_strength = 9e2

scale_frames = 0
scale = 1

form_image = False
form_factor = 0
form_step = 0.01
avg = 0
last_avg = 0
sim_factor = 1

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
            elif event.key == pygame.K_i:
                form_image = not form_image
                print("Form image:", form_image)
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

    ctx.clear(1.0, 1.0, 1.0)
    screen.fill((0, 0, 0))

    dt = time.perf_counter() - last_time
    last_time = time.perf_counter()
    fps = steps_per_frame / dt
    if avg_fps is None:
        avg_fps = fps
    else:
        avg_fps = avg_fps * 0.9 + fps * 0.1

    if count % (avg_fps // steps_per_frame) == 0:
        pass
        # print(
        #     "FPS:",
        #     avg_fps,
        #     "SPF:",
        #     steps_per_frame,
        # )

    if apply_force != 0:
        sim.apply_force(
            force_strength * apply_force,
            torch.tensor(
                pygame.mouse.get_pos(), dtype=torch.float32, device=sim.device
            ),
            mouse_radius,
        )

    steps_per_frame = int(avg_fps / target_fps) + 1
    udt = 3 * dt / steps_per_frame
    # print("update dt:", udt)
    sim.set_dt(udt)

    for _ in range(steps_per_frame):
        if scale_frames > 0:
            scale_frames -= 1
            sim.scale_system(scale)
        profiler.start("step")
        if form_image:
            form_factor = min(1, form_factor + form_step * (1 - form_factor))

        else:
            form_factor = max(0, form_factor - form_step * (1.1 - form_factor))

        if form_factor > 0:
            if form_image:
                sim.gravity = (1 - form_factor) * sim.gravity_og
                sim.pressure_multiplier = (
                    max(0.0, (1 - form_factor) ** 0.8) * sim.pressure_multiplier_og
                )
            else:
                sim.gravity = sim.gravity_og
                sim.pressure_multiplier = (1 - form_factor) * sim.pressure_multiplier_og
            sim.step(dt)
        else:
            sim.step(dt)

        profiler.stop("step")

    profiler.start("draw")
    # particles = (sim.particles / screen_size * 2 - 1).detach().cpu().numpy()
    profiler.start("render")
    particles_vis.render()
    profiler.stop("render")
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
