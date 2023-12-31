import math

import torch


class Simulation:
    """A fluid dynamics simulation using the Smoothed Particle Hydrodynamics method."""

    def __init__(
        self,
        amount: int,
        dimensions: int,
        device: torch.device | None = None,
        x1=0,
        x2=1000,
        y1=0,
        y2=500,
    ) -> None:
        self.device = get_device() if device is None else device

        self.amount = amount
        self.set_bounds(x1, x2, y1, y2)
        self.gravity = torch.tensor([0, 50], device=device)
        self.smoothing_radius = 30
        self.simulate_pressure = True
        self.simulate_gravity = True

        self.setup_particles(amount, dimensions)

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
        self.densities = torch.zeros(amount, device=self.device)
        self.shared_pressure_mask = torch.ones(
            (amount, amount), device=self.device
        ) - torch.eye(amount, device=self.device)
        self.target_density = (
            self.smoothing_kernel(torch.tensor([0.0], device=self.device)) * 0.5
        )
        self.pressure_multiplier = torch.tensor(3e6, device=self.device)

        grid_size = math.ceil(math.sqrt(amount))
        x0, y0 = self.bounds[0]
        x1, y1 = self.bounds[1] / 2

        # Set up the particles in a grid
        for i in range(amount):
            self.particles[i] = torch.tensor(
                [
                    x0 + ((x1 - x0) / (1 + grid_size)) * (i % grid_size),
                    y0 + ((y1 - y0) / (1 + grid_size)) * (i // grid_size),
                ],
                device=self.device,
            )

    def set_bounds(self, x1, x2, y1, y2):
        self.bounds = torch.tensor([[x1, y1], [x2, y2]], device=self.device)
        # area = torch.prod(self.bounds[1] - self.bounds[0])
        # self.target_density = self.amount / area

    def step(self, dt: float) -> None:
        """Step the simulation forward by dt seconds."""

        self.velocities = torch.zeros_like(self.velocities)
        if self.simulate_gravity:
            self.velocities += self.gravity * dt
        self.predicted_positions = self.particles + self.velocities * dt
        self.update_densities()

        pressure_forces = self.calc_pressure_force2(self.predicted_positions)
        # TODO: dividing by densities unncecessary? (see end of calc_pressure_force2, multiplying by densities)
        self.accelerations = -pressure_forces / self.densities[:, None]
        if self.simulate_pressure:
            self.velocities += self.accelerations * dt

        # for i, particle in enumerate(self.particles):
        #     pressure_force = self.calc_pressure_force(particle)
        #     self.accelerations[i] = pressure_force / self.densities[i]
        #     self.velocities[i] += self.accelerations[i] * dt

        self.particles += self.velocities * dt

        self.particles = torch.max(self.particles, self.bounds[0])
        self.particles = torch.min(self.particles, self.bounds[1])

        # self.particles += torch.rand_like(self.particles) * 1e-2

    def calc_density(self, sample_points: torch.Tensor) -> torch.Tensor:
        """Calculate the density at a point."""

        sample_points = sample_points.reshape(-1, self.dimensions)
        diffs = sample_points.unsqueeze(1) - self.predicted_positions.unsqueeze(0)
        dists = torch.norm(diffs, dim=-1)
        kernel_vals = self.smoothing_kernel(dists)
        return torch.sum(kernel_vals, dim=-1)

    def smoothing_kernel(self, dist: torch.Tensor) -> torch.Tensor:
        """Calculate the kernel function."""
        scale = 6 / (math.pi * pow(self.smoothing_radius, 4))
        value = torch.max(torch.zeros_like(dist), self.smoothing_radius - dist)
        # TODO: this is same as in smoothing_kernel_gradient
        return value * value * scale

    def smoothing_kernel_gradient(self, dist: torch.Tensor) -> torch.Tensor:
        """Calculate the gradient of the kernel function."""
        scale = -12 / (math.pi * pow(self.smoothing_radius, 4))
        return torch.max(torch.zeros_like(dist), self.smoothing_radius - dist) * scale

    def update_densities(self):
        self.densities = self.calc_density(self.predicted_positions)

    def density_to_pressure(self, density: torch.Tensor) -> torch.Tensor:
        """Convert density to pressure."""
        density_diff = density - self.target_density
        density_diff *= self.pressure_multiplier
        return density_diff

    def calc_pressure_force(self, sample_point: torch.Tensor) -> torch.Tensor:
        """Calculate the gradient at a point."""
        dists = torch.norm(sample_point - self.predicted_positions, dim=1)
        dirs1 = (self.predicted_positions - sample_point).T
        dirs2 = dirs1 / dists
        dirs2 = torch.where(torch.isnan(dirs2), torch.zeros_like(dirs2), dirs2)
        slopes = self.smoothing_kernel_gradient(dists)
        kernel_vals = (
            self.density_to_pressure(self.densities)
            * dirs2
            * slopes
            / self.densities
            * (dists > 0)
            * (dists < self.smoothing_radius)
        )
        force = torch.sum(kernel_vals, dim=1)
        return force

    def calc_pressure_force2(self, sample_point: torch.Tensor) -> torch.Tensor:
        """Calculate the gradient at a point."""
        # TODO: many same lines as in calc_density2
        sample_point = sample_point.reshape(-1, 2)
        diffs = sample_point.unsqueeze(1) - self.predicted_positions.unsqueeze(0)
        dists = torch.norm(diffs, dim=-1)
        # dirs = self.shared_pressure_mask[:, :, None] * diffs / dists[:, :, None]
        dirs = diffs / dists[:, :, None]
        dirs[torch.eye(self.amount, device=self.device, dtype=bool)] = 0
        dirs = torch.where(torch.isnan(dirs), torch.zeros_like(dirs), dirs)
        slopes = self.smoothing_kernel_gradient(dists)
        pressures = self.density_to_pressure(self.densities)
        shared_pressures = (pressures[None, :] + pressures[:, None]) * 0.5
        shared_pressures[torch.eye(shared_pressures.shape[0]).bool()] = 0
        kernel_vals = (shared_pressures * slopes / self.densities[None, :])[
            :, :, None
        ] * dirs

        forces = torch.sum(kernel_vals, dim=1)
        return forces


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
