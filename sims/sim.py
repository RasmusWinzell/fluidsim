import math

import numpy as np
import torch


# @torch.jit.script
class Simulation:
    """A fluid dynamics simulation using the Smoothed Particle Hydrodynamics method."""

    def __init__(
        self,
        amount: int,
        dimensions: int,
        device: torch.device = torch.device("cpu"),
        x1: float = 0,
        x2: float = 1000,
        y1: float = 0,
        y2: float = 500,
        color: tuple = (0.0, 0.0, 1.0),
        particle_size: float = 10,
    ) -> None:
        self.device = get_device() if device is None else device

        self.amount = amount
        self.dimensions = dimensions
        self.color = np.array(color)
        self.particle_size = particle_size
        self.bounds = torch.zeros((2, 2), device=self.device)
        self.set_bounds(x1, x2, y1, y2)
        self.dt = torch.tensor(2 / 100, device=self.device)
        self.dt2 = self.dt * self.dt
        self.gravity_og = torch.tensor([0.0, 50.0], device=device) * self.dt2
        self.gravity = self.gravity_og
        self.smoothing_radius = 20
        self.kernel_scale = 6 / (math.pi * pow(self.smoothing_radius, 4))
        self.kernel_gradient_scale = -12 / (math.pi * pow(self.smoothing_radius, 4))
        self.simulate_pressure = True
        self.simulate_gravity = True

        self.particles = torch.zeros((10, dimensions), device=self.device)
        self.predicted_positions = self.particles
        self.velocities = torch.zeros((10, dimensions), device=self.device)
        self.accelerations = torch.zeros((10, dimensions), device=self.device)
        self.densities = torch.zeros(10, device=self.device)
        self.shared_pressure_mask = torch.zeros((10, 10), device=self.device)

        self.setup_particles(amount, dimensions)

        self.target_density = torch.tensor(1.5e-2, device=self.device)
        self.pressure_multiplier_og = torch.tensor(8e5, device=self.device)
        self.pressure_multiplier = self.pressure_multiplier_og

    def setup_particles(self, amount: int, dimensions: int) -> None:
        """Set up the particles in a grid."""

        self.amount = amount
        self.dimensions = dimensions
        self.particles = torch.zeros(
            (amount, dimensions), dtype=torch.float32, device=self.device
        )
        self.predicted_positions = self.particles
        self.velocities = torch.zeros(
            (amount, dimensions), dtype=torch.float32, device=self.device
        )
        self.accelerations = torch.zeros((amount, dimensions), device=self.device)
        self.rnd_offset = (
            2 * torch.rand_like(self.particles, device=self.device) - 1
        ) * 1e-6
        self.densities = torch.zeros(amount, device=self.device)
        self.shared_pressure_mask = torch.ones(
            (amount, amount), device=self.device
        ) - torch.eye(amount, device=self.device)
        self.spawn_particles()
        self.colors = np.zeros((amount, 3))
        self.colors[:, :] = self.color
        pass

    def spawn_particles(self):
        self.particles = torch.rand(self.amount, 2, device=self.device) * self.bounds[1]
        self.velocities = torch.zeros_like(
            self.particles, dtype=torch.float32, device=self.device
        )

    def set_dt(self, dt):
        self.dt = torch.tensor(dt, device=self.device)
        self.dt2 = self.dt * self.dt
        self.gravity = torch.tensor([0.0, 50.0], device=self.device) * self.dt2

    def set_pressure_multiplier(self, multiplier):
        self.pressure_multiplier_og = torch.tensor(multiplier, device=self.device)
        self.pressure_multiplier = self.pressure_multiplier_og

    def set_bounds(self, x1: float, x2: float, y1: float, y2: float):
        self.bounds = torch.tensor([[x1, y1], [x2, y2]], device=self.device)
        # area = torch.prod(self.bounds[1] - self.bounds[0])
        # self.target_density = self.amount / area

    def scale_system(self, scale: float) -> None:
        self.smoothing_radius *= scale
        self.target_density /= scale

    def step(self, dt2: float = 1, strength=None) -> None:
        """Step the simulation forward by dt seconds."""

        if self.simulate_gravity:
            self.velocities += self.gravity
        self.predicted_positions = self.particles + self.velocities

        # calc_density
        diffs = self.predicted_positions.unsqueeze(
            1
        ) - self.predicted_positions.unsqueeze(0)
        dists = torch.norm(diffs, dim=-1)

        # smoothing_kernel
        kernel_vals = torch.nn.functional.relu(self.smoothing_radius - dists)

        # smoothing_kernel_gradient
        slopes = kernel_vals * self.kernel_gradient_scale

        kernel_vals *= kernel_vals * self.kernel_scale

        self.densities = torch.sum(kernel_vals, dim=-1)

        # calc_pressure_force
        masked_dists = self.shared_pressure_mask / dists
        torch.nan_to_num_(masked_dists, nan=0.0, posinf=1.0, neginf=1.0)
        dirs = masked_dists.unsqueeze(-1) * diffs
        pressures = self.density_to_pressure(self.densities)
        shared_pressures = pressures.unsqueeze(1) + pressures.unsqueeze(0)
        k1 = shared_pressures * slopes
        k2 = k1 / self.densities.unsqueeze(0)
        kernel_vals = k2.unsqueeze(-1) * dirs

        pressure_forces = torch.sum(kernel_vals, dim=1)

        self.accelerations = pressure_forces / self.densities[:, None] * -self.dt2
        if self.simulate_pressure:
            self.velocities += self.accelerations

        self.particles += self.velocities

        self.new_particles = torch.clamp(self.particles, self.bounds[0], self.bounds[1])
        dv = self.particles - self.new_particles
        self.velocities -= dv + self.rnd_offset

        self.particles = self.new_particles

    def density_to_pressure(self, density: torch.Tensor) -> torch.Tensor:
        """Convert density to pressure."""
        density_diff = density - self.target_density
        density_diff *= self.pressure_multiplier
        return density_diff

    def apply_force(
        self, strength: float, location: torch.Tensor, radius: float
    ) -> None:
        diffs = location - self.particles
        dists = torch.norm(diffs, dim=-1)
        kernel_vals = torch.nn.functional.relu(radius - dists)
        scale = strength / pow(radius, 3)
        kernel_vals *= scale
        self.velocities += kernel_vals[:, None] * diffs


@torch.jit.script
def calc_kernels(shared_pressures, densities, slopes, dirs):
    k1 = shared_pressures * slopes
    k2 = k1 / densities.unsqueeze(0)
    kernel_vals = k2.unsqueeze(-1) * dirs
    return kernel_vals


def __del__(self):
    del self.particles
    del self.predicted_positions
    del self.velocities
    del self.accelerations
    del self.densities
    del self.shared_pressure_mask


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
