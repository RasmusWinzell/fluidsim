import os
import re

from pdf2image import convert_from_path

from . import pptx_interface
from .slide import Slide


class Presentation:
    IMAGE_DIR = os.getcwd() + "/presentation/slide_images/"
    SIM_IMAGE_FORMAT = r"sim(\d+).png"
    PDF_FORMAT = r"slides.pdf"
    SLIDE_FORMAT = r"slide(\d+).png"

    def __init__(
        self,
        name="presentation",
        filepath=None,
        image_dir=IMAGE_DIR,
        preload=True,
        load_only=False,
    ):
        self.spec = (name,)
        if filepath is not None:
            self.spec += (filepath,)
        self.image_dir = image_dir
        self.reload(preload, load_only)
        self._slide_index = 0

    @property
    def slides(self):
        return self._slides

    @property
    def num_slides(self):
        return len(self.slides)

    @property
    def current_slide(self) -> Slide:
        return self.slides[self.slide_index]

    @property
    def slide_index(self):
        return self._slide_index

    @slide_index.setter
    def slide_index(self, value):
        self._slide_index = min(self.num_slides - 1, max(0, value))

    def next_slide(self):
        self.slide_index += 1
        return self.current_slide

    def prev_slide(self):
        self.slide_index -= 1
        return self.current_slide

    def reload(self, preload=True, load_only=False):
        if load_only:
            slide_images = self.load_slides()
            sim_images = self.load_sim_images()
        else:
            slide_images = self.export_slides()
            sim_images = self.export_sim_images()
        self._slides = []
        for i, slide_image in enumerate(slide_images):
            sim_image = sim_images.get(i, None)
            slide = Slide(slide_image, sim_image)
            if preload:
                slide.preload()
            self._slides.append(slide)
        print("Loaded {} slides".format(self.num_slides))

    def load_sim_images(self):
        images = {}
        for file in os.listdir(self.image_dir):
            if slide_nr := re.match(Presentation.SIM_IMAGE_FORMAT, file):
                images[int(slide_nr.group(1)) - 1] = os.path.join(self.image_dir, file)
        return images

    def load_slides(self):
        images = []
        for file in os.listdir(self.image_dir):
            if slide_nr := re.match(Presentation.SLIDE_FORMAT, file):
                images.append(
                    (int(slide_nr.group(1)) - 1, os.path.join(self.image_dir, file))
                )
        return [image for _, image in sorted(images)]

    def export_sim_images(self):
        pptx_interface.export_sim_images(*self.spec)

        images = {}
        for file in os.listdir(pptx_interface.DATA_DIR):
            if slide_nr := re.match(Presentation.SIM_IMAGE_FORMAT, file):
                src_path = os.path.join(pptx_interface.DATA_DIR, file)
                dst_path = os.path.join(self.image_dir, file)
                # Move file to destination
                os.rename(
                    src_path,
                    dst_path,
                )
                images[int(slide_nr.group(1)) - 1] = dst_path
        return images

    def export_slides(self):
        pptx_interface.hide_sim_images(*self.spec)
        pptx_interface.export_slides(*self.spec)
        pptx_interface.show_sim_images(*self.spec)

        images = []
        for file in os.listdir(pptx_interface.DATA_DIR):
            if re.match(Presentation.PDF_FORMAT, file):
                src_path = os.path.join(pptx_interface.DATA_DIR, file)
                pages = convert_from_path(src_path)
                for i, page in enumerate(pages):
                    dst_path = os.path.join(self.image_dir, "slide{}.png".format(i + 1))
                    page.save(dst_path, "PNG")
                    images.append(dst_path)
        return images
