import cv2
import pygame


class Slide:
    def __init__(self, background_path, sim_image_path):
        self._background_path = background_path
        self._sim_image_path = sim_image_path
        self._sim_image = None
        self._background = None

    @property
    def background(self):
        if self._background is None and self._background_path is not None:
            self._background = pygame.image.load(self._background_path)
        return self._background

    @property
    def sim_image(self):
        if self._sim_image is None and self._sim_image_path is not None:
            self._sim_image = pygame.image.load(self._sim_image_path)
        return self._sim_image

    def preload(self):
        self.sim_image
        self.background
