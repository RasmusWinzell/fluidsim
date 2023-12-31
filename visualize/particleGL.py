import moderngl
import numpy as np

vertex_shader = """
#version 330

in vec2 in_position;

in vec3 in_color;

uniform float scale;

out vec3 color;

void main() {
    gl_Position = vec4(in_position.x, -in_position.y, 0.0, 1.0);

    // Set the point size
    gl_PointSize = scale;

    // Calculate a random color based on the vertex index
    color = in_color;
}
"""
fragment_shader = """
#version 330

in vec3 color;
out vec4 outColor;

void main() {
    // Calculate the distance from the center of the point
    // gl_PointCoord is available when redering points. It's basically an uv coordinate.
    float dist = step(length(gl_PointCoord.xy - vec2(0.5)), 0.5);

    // .. an use to render a circle!
    outColor = vec4(color, dist);
}
"""


class particleGL:
    def __init__(self, ctx, sim):
        self.ctx = ctx
        self.sim = sim
        self.program = ctx.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader
        )
        positions = sim.particles.detach().cpu().numpy()
        self.program["scale"].value = 10.0
        self.pos_buffer = self.ctx.buffer(positions.astype("f4"))
        init_color = np.zeros((positions.shape[0], 3))
        init_color[:, 2] = 1.0
        self.color_buffer = self.ctx.buffer(init_color.astype("f4"))
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.pos_buffer, "2f", "in_position"),
                (self.color_buffer, "3f", "in_color"),
            ],
        )

    @property
    def scale(self):
        return self.program["scale"].value

    @scale.setter
    def scale(self, value):
        self.program["scale"].value = value

    def render(self, colors=None, scale=None):
        positions = (
            (self.sim.particles / self.sim.bounds[1] * 2 - 1).detach().cpu().numpy()
        )
        self.pos_buffer.write(positions.astype("f4"))
        if colors is not None:
            self.color_buffer.write(colors.astype("f4"))
        if scale is not None:
            self.scale = scale

        self.ctx.enable_only(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.ctx.blend_func = moderngl.DEFAULT_BLENDING

        self.vao.render(mode=moderngl.POINTS)
