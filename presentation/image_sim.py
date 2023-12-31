# import image with matplotlib
import cv2
import numpy as np
import pygame
import torch


class ImageSim:
    def __init__(self, sim, image):
        self.sim = sim
        self.load_image(image)
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.strength = 1.0
        self.resize_image(self.img, 100, 100)
        self.velocities = torch.zeros_like(sim.velocities, device=sim.device)

    def load_image(self, image):
        self.image_path = image
        if isinstance(image, str):
            self.img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGRA2RGBA)
        elif isinstance(image, pygame.Surface):
            color = pygame.surfarray.array3d(image)
            alpha = pygame.surfarray.array_alpha(image)
            self.img = np.dstack((color, alpha))
            self.img = np.transpose(self.img, (1, 0, 2))

        if self.img.shape[2] == 3:
            self.img = np.concatenate(
                (self.img, np.ones((self.img.shape[0], self.img.shape[1], 1))), axis=2
            )
        if np.max(self.img) > 1:
            self.img = self.img / 255.0

    def non_transparent(self):
        return np.where(self.img[:, :, 3] != 0)

    def resize_image(self, img, width, height):
        new_img = np.zeros((height, width, 4))
        yy = np.linspace(0, img.shape[0] - 1, height + 1)
        xx = np.linspace(0, img.shape[1] - 1, width + 1)
        for i in range(height):
            for j in range(width):
                y1 = int(yy[i])
                y2 = int(yy[i + 1])
                x1 = int(xx[j])
                x2 = int(xx[j + 1])
                area = img[y1:y2, x1:x2, :]
                colors = area[:, :, :3]
                weight = area[:, :, 3]
                new_img[i, j, :3] = np.sum(colors * weight[:, :, None], axis=(0, 1)) / (
                    np.sum(weight) + 1e-6
                )
                new_img[i, j, 3] = np.mean(weight) > 0.5
        return new_img

    def sample_image(self):
        non_transparent = np.where(self.img[:, :, 3] != 0)
        num_non_transparent = len(non_transparent[0])

        img = self.img
        scale = 1
        width = self.width
        height = self.height
        while (num_non_transparent := np.count_nonzero(img[:, :, 3])) > self.sim.amount:
            new_scale = np.sqrt(self.sim.amount / num_non_transparent)
            scale *= new_scale
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = self.resize_image(self.img, new_width, new_height)

        non_transparent = np.where(img[:, :, 3] != 0)
        num_non_transparent = len(non_transparent[0])
        non_transparent = np.array(non_transparent).T
        inital_samples = min(num_non_transparent, self.sim.amount)
        sampled_indexes = np.random.choice(
            num_non_transparent, inital_samples, replace=False
        )
        extra_samples = abs(self.sim.amount - inital_samples)
        sampled_indexes = np.concatenate(
            (sampled_indexes, np.random.choice(num_non_transparent, extra_samples))
        )
        self.sampled_pixels = non_transparent[sampled_indexes]
        self.sampled_colors = img[
            self.sampled_pixels[:, 0], self.sampled_pixels[:, 1], :3
        ]
        # convert to torch
        self.sampled_pixels = (
            torch.tensor(self.sampled_pixels, device=self.sim.device) / scale
        )
        # flip dimensions
        self.sampled_pixels = self.sampled_pixels.flip(1)
        # normalize
        screen_scale = min(
            (self.sim.bounds[1][0] - self.sim.bounds[0][0]) / self.width,
            (self.sim.bounds[1][1] - self.sim.bounds[0][1]) / self.height,
        )
        self.sampled_pixels = self.sampled_pixels * screen_scale
        self.total_scale = screen_scale / scale
        # center image
        sim_center = torch.mean(self.sim.bounds.float(), dim=0)
        image_center = (
            torch.tensor([self.width, self.height], device=self.sim.device)
            / 2
            * screen_scale
        )
        offset = sim_center - image_center
        self.sampled_pixels += offset + self.total_scale

        self.particle_size = self.total_scale * 2  # np.sqrt(2)

        return self.total_scale

    def attract(self):
        diffs = self.sampled_pixels - self.sim.particles
        abs_diff = torch.abs(diffs)
        normalized_diffs = torch.sign(diffs) * torch.sqrt(1 + abs_diff)

        target_velocities = (
            normalized_diffs * self.strength * (torch.sigmoid(abs_diff) - 0.5)
        )
        self.sim.velocities += (1.01) * target_velocities - self.velocities
        self.velocities = target_velocities


if __name__ == "__main__":
    # read image
    imgsim = ImageSim(None, "oscar.jpg")
    imgsim.sample_image_stratified(10)
