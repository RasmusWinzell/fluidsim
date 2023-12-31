import os

import applescript

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/scripts/"
DATA_DIR = os.path.expanduser("~/Library/Containers/com.microsoft.Powerpoint/Data/")


def load_script(filename: str):
    print(filename.ljust(20), end=": ")
    full_path = SCRIPT_DIR + filename
    try:
        res = open(full_path).read()
        print("OK")
    except FileNotFoundError:
        print("ERROR")
        return None
    return res


print("Loading PowerPoint scripts...")
EXPORT_SIM_IMAGES_SCRIPT = load_script("export_sim_images")
EXPORT_SLIDES_SCRIPT = load_script("export_slides")
SHOW_SIM_IMAGES_SCRIPT = load_script("show_sim_images")
HIDE_SIM_IMAGES_SCRIPT = load_script("hide_sim_images")


def run_script(script: str, *args):
    applescript.AppleScript(script.format(*args)).run()


def export_sim_images(
    presentation_name: str = "presentation",
    presentation_path: str = "presentation.pptx",
):
    run_script(EXPORT_SIM_IMAGES_SCRIPT, presentation_name, presentation_path)


def export_slides(
    presentation_name: str = "presentation",
    presentation_path: str = "presentation.pptx",
):
    run_script(EXPORT_SLIDES_SCRIPT, presentation_name, presentation_path)


def show_sim_images(
    presentation_name: str = "presentation",
    presentation_path: str = "presentation.pptx",
):
    run_script(SHOW_SIM_IMAGES_SCRIPT, presentation_name, presentation_path)


def hide_sim_images(
    presentation_name: str = "presentation",
    presentation_path: str = "presentation.pptx",
):
    run_script(HIDE_SIM_IMAGES_SCRIPT, presentation_name, presentation_path)
