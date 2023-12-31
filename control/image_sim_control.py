import time

import pygame


class ImageSimControl:
    def __init__(self, image_sim, sim):
        self.queued_events = []
        self.targetting_image = False
        self.duration = 5.0
        self.strength = 1.0
        self.set_sims(sim, image_sim)

        self.start = None
        self.in_transition = False
        self.progress = 0.0
        self.curr_strength = 0.0
        self.curr_duration = 0.0
        self.start_colors = self.sim.colors
        self.target_colors = None
        self.start_particle_size = self.sim.particle_size
        self.target_particle_size = None
        self.curr_colors = self.start_colors
        self.curr_particle_size = self.start_particle_size

    def set_sims(self, sim, image_sim):
        self.sim = sim
        self.image_sim = image_sim
        if self.targetting_image:
            self.queue_event(False, self.duration / 2, self.strength)
            self.queue_event(None, self.duration, self.strength, overwrite=False)
            self.queue_event(True, self.duration / 2, self.strength, overwrite=False)

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_i:
                self.queue_event(
                    not self.targetting_image, self.duration, self.strength
                )
                return True
        return False

    def queue_event(self, to_image, duration, strength, overwrite=True):
        if overwrite:
            self.in_transition = False
            self.queued_events = []
        self.queued_events.append((to_image, duration, strength))

    def update(self):
        if self.start is None or not self.in_transition:
            if len(self.queued_events) != 0:
                (
                    self.targetting_image,
                    self.curr_duration,
                    self.curr_strength,
                ) = self.queued_events.pop(0)
                self.start = time.time()
                self.in_transition = True
                if self.targetting_image is True:
                    self.target_colors = self.image_sim.sampled_colors
                    self.target_particle_size = self.image_sim.particle_size
                elif self.targetting_image is False:
                    self.target_colors = self.sim.colors
                    self.target_particle_size = self.sim.particle_size
        if self.start is None:
            return
        t = time.time()
        self.progress = min(1.0, (t - self.start) / self.curr_duration)
        if self.targetting_image is not None:
            self.update_visuals()
            self.update_sim()
            self.update_image_sim()
        if self.progress == 1.0:
            self.in_transition = False
            if self.targetting_image is not None:
                self.start_colors = self.target_colors
                self.start_particle_size = self.target_particle_size

    def update_visuals(self):
        self.curr_colors = (
            self.start_colors * (1 - self.progress) + self.target_colors * self.progress
        )
        self.curr_particle_size = (
            self.start_particle_size * (1 - self.progress)
            + self.target_particle_size * self.progress
        )

    def update_sim(self):
        if self.targetting_image:
            # transition to image sim
            self.sim.gravity = (1 - self.progress) * self.sim.gravity_og
            self.sim.pressure_multiplier = (
                1 - self.progress
            ) * self.sim.pressure_multiplier_og
        else:
            # transition to sim
            self.sim.gravity = self.sim.gravity_og
            self.sim.pressure_multiplier = (
                self.progress * self.sim.pressure_multiplier_og
            )

    def update_image_sim(self):
        if self.targetting_image:
            # transition to image sim
            self.image_sim.strength = self.curr_strength * self.progress
