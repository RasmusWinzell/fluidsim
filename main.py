import cv2
import moderngl
import pygame
import pygame_widgets
import torch
from pygame_widgets.slider import Slider

from control.image_sim_control import ImageSimControl
from control.presentation_control import PresentationControl
from control.simulation_control import SimulationControl
from presentation import Presentation
from presentation.image_sim import ImageSim
from sims import Simulation
from visualize.particleGL import particleGL
from visualize.planeGL import PlaneGL


# main loop
# sim
# sim_renderer
# background renderer
# ui
# presentation stuff
class Main:
    def __init__(self):
        self.running = False
        self.image_sims = {}

        self.show_controls = True
        self.sim = None
        self.sim_renderer = None
        self.background_renderer = None
        self.image_sim = None
        self.presentation = None
        self.presentation_control = None
        self.image_sim_control = None
        self.sim_control = None

        self.event_handlers = []

        self.setup_context()
        self.setup_simulation()
        self.setup_particle_renderer()
        self.setup_presentation()
        self.setup_image_sim()
        self.setup_background_renderer()
        # Link controls
        self.setup_presentation_control()
        self.setup_image_sim_control()
        self.setup_simulation_control()
        self.setup_control_ui()

    def setup_context(self, screen_size=(1200 - 133, 600)):
        pygame.init()

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

        self.set_screen_size(screen_size)

        ctx = moderngl.create_context()
        print("Using GLSL version:", ctx.info["GL_VERSION"])
        self.ctx = ctx

    def set_screen_size(self, size):
        self.size = size
        return pygame.display.set_mode(
            size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        )

    def setup_simulation(self):
        self.sim = Simulation(
            3000,
            2,
            device=torch.device("mps"),
            x2=self.size[0],
            y2=self.size[1],
        )
        self.sim.set_pressure_multiplier(6e4)

    def setup_particle_renderer(self):
        self.sim_renderer = particleGL(self.ctx, self.sim)

    def setup_presentation(self):
        self.presentation = Presentation(load_only=False)

    def setup_background_renderer(self):
        if self.presentation is None:
            self.background_renderer = None
            return
        image = self.presentation.current_slide.background
        if image is None:
            self.background_renderer = None
            return
        self.background_renderer = PlaneGL(self.ctx, False, self.size)
        self.background_renderer.draw_image(image)
        self.background_renderer.update()

    def setup_image_sim(self):
        if self.presentation is None:
            self.image_sim = None
            return
        image = self.presentation.current_slide.sim_image
        if image is None:
            self.image_sim = None
            return
        if image in self.image_sims:
            self.image_sim = self.image_sims[image]
            return
        self.image_sim = ImageSim(self.sim, image)
        self.image_sim.sample_image()
        self.image_sims[image] = self.image_sim

    def setup_presentation_control(self):
        if self.presentation is None:
            return

        def on_changed():
            self.setup_background_renderer()
            self.setup_image_sim()
            self.setup_image_sim_control()

        self.presentation_control = PresentationControl(self.presentation, on_changed)
        self.event_handlers.append(self.presentation_control)

    def setup_image_sim_control(self):
        if self.image_sim is None:
            return
        if self.image_sim_control is not None:
            self.image_sim_control.set_sims(self.sim, self.image_sim)
        else:
            self.image_sim_control = ImageSimControl(self.image_sim, self.sim)
            self.event_handlers.append(self.image_sim_control)
        self.sim_renderer.render(self.image_sim.sampled_colors)

    def setup_simulation_control(self):
        self.sim_control = SimulationControl(self.sim)
        self.event_handlers.append(self.sim_control)

    def setup_control_ui(self):
        self.control_surface = PlaneGL(self.ctx, True, self.size)
        self.presentation_control.draw(self.control_surface, self.size)
        self.sim_control.draw(self.control_surface, self.size)

    def main_loop(self):
        self.running = True
        while self.running:
            events = pygame.event.get()
            remaining_events = self.handle_events(events)

            self.ctx.clear(1.0, 1.0, 1.0)

            self.update()

            pygame_widgets.update(events)
            pygame.display.flip()

    def handle_events(self, events):
        remaining_events = []
        for event in events:
            handled = False
            if event.type == pygame.QUIT:
                self.running = False
                handled = True
            elif event.type == pygame.VIDEORESIZE:
                self.set_screen_size(event.size)
                self.update_sizes()
                handled = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.show_controls = not self.show_controls
                    handled = True
            for handler in self.event_handlers:
                if handler.handle_events(event):
                    handled = True
                    break
            if not handled:
                remaining_events.append(event)

    def update_sizes(self):
        pass

    def update(self):
        self.update_background()
        self.update_image_sim()
        self.update_particles()
        if self.show_controls:
            self.update_controls()

    def update_background(self):
        if self.background_renderer is None:
            return
        self.background_renderer.render()

    def update_image_sim(self):
        if self.image_sim is None:
            return
        self.image_sim_control.update()

    def update_particles(self):
        for _ in range(10):
            if self.image_sim is not None and self.image_sim_control.targetting_image:
                self.image_sim.attract()
            self.sim.step()
        if self.image_sim_control is not None:
            self.sim_renderer.render(
                self.image_sim_control.curr_colors,
                self.image_sim_control.curr_particle_size,
            )
        else:
            self.sim_renderer.render()

    def update_controls(self):
        # self.control_surface.fill((0, 0, 0, 0))
        self.sim_control.update()
        self.control_surface.update()
        self.control_surface.render()


if __name__ == "__main__":
    import cv2

    img = cv2.imread("images/blixten.png", cv2.IMREAD_UNCHANGED)

    main = Main()
    main.main_loop()
