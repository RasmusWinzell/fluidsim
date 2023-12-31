import time

import pygame
import torch
from pygame_widgets.slider import Slider


class SimulationControl:
    def __init__(self, simulation):
        self.sim = simulation

        self.apply_force = 0
        self.force_strength = 9e2
        self.force_radius = 100
        self.last_scroll = time.perf_counter()
        self.show_force_circle = False

    def draw(self, surf, screen_size):
        self.force_strength_slider = Slider(
            surf,
            screen_size[0] - 200,
            20,
            180,
            20,
            min=0,
            max=1e3,
            step=1,
            initial=self.force_strength,
            colour=(0, 0, 0),
            handleColour=(0, 0, 0),
            handleRadius=10,
            handleBorder=0,
        )

    def handle_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.apply_force = 1
            if event.button == 3:
                self.apply_force = -1
            self.show_force_circle = True
            print("Mouse position:", event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button in (1, 3):
                self.apply_force = 0
            self.show_force_circle = False
        elif event.type == pygame.MOUSEWHEEL:
            if event.y > 0:
                mouse_radius *= 1.1
            elif event.y < 0:
                mouse_radius /= 1.1
            self.last_scroll = time.perf_counter()
            self.show_force_circle = True

    def update(self):
        self.force_strength = self.force_strength_slider.getValue()
        if time.perf_counter() - self.last_scroll > 0.5 and self.apply_force == 0:
            self.show_force_circle = False
        if self.apply_force != 0:
            self.sim.apply_force(
                self.force_strength * self.apply_force,
                torch.tensor(
                    pygame.mouse.get_pos(), dtype=torch.float32, device=self.sim.device
                ),
                self.force_radius,
            )

    def set_force_strength(self, val):
        print("set force strength")
        self.force_strength = val
