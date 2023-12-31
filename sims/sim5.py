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
        self.kernel_scale = 6 / (math.pi * pow(self.smoothing_radius, 4))
        self.kernel_gradient_scale = -12 / (math.pi * pow(self.smoothing_radius, 4))
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
        self.target_density = self.smoothing_radius**2 * self.kernel_scale * 0.5
        self.pressure_multiplier = torch.tensor(1e5, device=self.device)

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

        if self.simulate_gravity:
            self.velocities += self.gravity * dt
        self.predicted_positions = self.particles + self.velocities * dt

        # calc_density
        diffs = self.predicted_positions.unsqueeze(
            1
        ) - self.predicted_positions.unsqueeze(0)
        dists = torch.norm(diffs, dim=-1)

        # smoothing_kernel
        value = torch.nn.functional.relu(self.smoothing_radius - dists)
        kernel_vals = value * value * self.kernel_scale

        # smoothing_kernel_gradient
        slopes = value * self.kernel_gradient_scale
        self.densities = torch.sum(kernel_vals, dim=-1)

        # calc_pressure_force
        masked_dists = self.shared_pressure_mask / dists
        torch.nan_to_num_(masked_dists, nan=0.0, posinf=0.0, neginf=0.0)
        dirs = masked_dists.unsqueeze(-1) * diffs
        pressures = self.density_to_pressure(self.densities)
        shared_pressures = pressures.unsqueeze(1) + pressures.unsqueeze(0)
        k1 = shared_pressures * slopes
        k2 = k1 / self.densities.unsqueeze(0)
        kernel_vals = k2.unsqueeze(-1) * dirs

        pressure_forces = torch.sum(kernel_vals, dim=1)

        # pressure_forces = self.calc_pressure_force2(self.predicted_positions)
        # TODO: dividing by densities unncecessary? (see end of calc_pressure_force2, multiply)
        self.accelerations = -pressure_forces / self.densities[:, None]
        if self.simulate_pressure:
            self.velocities += self.accelerations * dt

        self.particles += self.velocities * dt

        self.particles = torch.max(self.particles, self.bounds[0])
        self.particles = torch.min(self.particles, self.bounds[1])

    def density_to_pressure(self, density: torch.Tensor) -> torch.Tensor:
        """Convert density to pressure."""
        density_diff = density - self.target_density
        density_diff *= self.pressure_multiplier
        return density_diff


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
