import torch
from tqdm import tqdm

import sims.sim as sim
from profiling.profilers import AdvancedProfiler, Profiler

N = 1000
# 0.0015
# 0.0017

profiler = AdvancedProfiler()
sim.Simulation = profiler.profile(sim.Simulation)

simulation = sim.Simulation(N, 2, device="mps")
simulation.particles = torch.ones((N, 2), device=simulation.device) * 499.0
simulation.step(1 / 100)  # warmup

profiler2 = Profiler()

for i in tqdm(range(1000)):
    profiler2.start("step")
    simulation.step(1 / 100)
    profiler2.stop("step")


from profiling.present import SourcePresenter
from profiling.present.metrics import FPS

presenter = SourcePresenter(profiler)
result = presenter.present()

presenter2 = SourcePresenter(profiler2, metrics=[FPS()], format="{}", leaves_only=True)
result2 = presenter2.present()

from sims import store

result = result + "\n" + result2
# store.save_sim(sim, result)
