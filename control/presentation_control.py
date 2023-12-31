import pygame
from pygame_widgets.button import Button


class PresentationControl:
    def __init__(self, presentation, on_changed=None):
        self.presentation = presentation
        self.on_changed = on_changed

    def draw(self, surf, screen_size):
        screen_x, screen_y = screen_size
        button_width = 80
        button_height = 30
        inactive_colour = (50, 50, 50)
        pressed_colour = (200, 200, 200)
        text_colour = (255, 255, 255)
        font_size = 20
        margin = 20
        self.prev_btn = Button(
            surf,
            screen_x - button_width * 3,
            screen_y - button_height,
            button_width,
            button_height,
            text="Prev",
            fontSize=font_size,
            textColour=text_colour,
            margin=margin,
            inactiveColour=inactive_colour,
            pressedColour=pressed_colour,
            onClick=lambda: self.prev(),
        )
        self.next_btn = Button(
            surf,
            screen_x - button_width * 2,
            screen_y - button_height,
            button_width,
            button_height,
            text="Next",
            fontSize=font_size,
            textColour=text_colour,
            margin=margin,
            inactiveColour=inactive_colour,
            pressedColour=pressed_colour,
            onClick=lambda: self.next(),
        )
        self.reload_btn = Button(
            surf,
            screen_x - button_width * 1,
            screen_y - button_height,
            button_width,
            button_height,
            text="Reload",
            fontSize=font_size,
            textColour=text_colour,
            margin=margin,
            inactiveColour=inactive_colour,
            pressedColour=pressed_colour,
            onClick=lambda: self.reload(),
        )

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                self.next()
                return True
            elif event.key == pygame.K_LEFT:
                self.prev()
                return True
            elif event.key == pygame.K_r:
                self.reload()
                return True
        return False

    def notify_changed(self):
        if self.on_changed is not None:
            self.on_changed()

    def next(self):
        self.presentation.next_slide()
        self.notify_changed()

    def prev(self):
        self.presentation.prev_slide()
        self.notify_changed()

    def reload(self):
        self.presentation.reload()
        self.notify_changed()
