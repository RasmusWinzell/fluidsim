from array import array

import cv2
import moderngl
import numpy as np
import pygame

quad = [
    # position (x, y), uv coords (x, y)
    -1.0,
    1.0,
    0.0,
    0.0,  # topleft
    1.0,
    1.0,
    1.0,
    0.0,  # topright
    -1.0,
    -1.0,
    0.0,
    1.0,  # bottomleft
    1.0,
    -1.0,
    1.0,
    1.0,  # bottomright
]

vert_shader = """
#version 330 core

in vec2 vert;
in vec2 texcoord;
out vec2 uvs;

void main() {
    uvs = texcoord;
    gl_Position = vec4(vert, 0.0, 1.0);
}
"""

frag_shader = """
#version 330 core

uniform sampler2D tex;

in vec2 uvs;
out vec4 f_color;

void main() {
    f_color = texture(tex, uvs);
}
"""


class PlaneGL(pygame.Surface):
    """Renders an image to the screen."""

    def __init__(self, ctx: moderngl.Context, transparent, *args, **kwargs):
        super().__init__(*args, **kwargs, flags=pygame.SRCALPHA)
        self.ctx = ctx
        self.transparent = transparent
        self.setup()
        self.update()

    def setup(self):
        self.quad_buffer = self.ctx.buffer(data=array("f", quad))
        self.program = self.ctx.program(
            vertex_shader=vert_shader, fragment_shader=frag_shader
        )
        self.render_object = self.ctx.vertex_array(
            self.program, [(self.quad_buffer, "2f 2f", "vert", "texcoord")]
        )
        # img = cv2.imread("images/blixten.png", cv2.IMREAD_UNCHANGED)
        # self.tex = self.ctx.texture(img.shape[1::-1], img.shape[2], img)
        self.tex = self.ctx.texture(self.get_size(), 4)
        # self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        # self.tex.swizzle = "RGBA"

    def update(self):
        # Create texture
        r = np.array(self.get_view("R"), dtype=np.uint8, copy=False)
        g = np.array(self.get_view("G"), dtype=np.uint8, copy=False)
        b = np.array(self.get_view("B"), dtype=np.uint8, copy=False)
        a = np.array(self.get_view("A"), dtype=np.uint8, copy=False)
        data = np.transpose(np.stack((r, g, b, a), axis=2), (1, 0, 2))
        data = data.copy(order="C")

        self.tex.write(data)

    def render(self):
        self.tex.use(0)
        self.program["tex"] = 0
        if self.transparent:
            self.ctx.enable_only(moderngl.BLEND)
            self.ctx.blend_func = moderngl.DEFAULT_BLENDING
        else:
            self.ctx.disable(moderngl.BLEND)
        self.render_object.render(mode=moderngl.TRIANGLE_STRIP)

    def draw_image(self, img: pygame.Surface):
        """draw image in center of surface as large as possible"""
        scale = min(
            self.get_width() / img.get_width(), self.get_height() / img.get_height()
        )
        scaled_img = pygame.transform.scale(
            img, (int(img.get_width() * scale), int(img.get_height() * scale))
        )
        self.blit(scaled_img, scaled_img.get_rect(center=self.get_rect().center))
