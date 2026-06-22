import imageio.v2 as imageio
import os
import glob

def create_gif(naming_con, input_folder, dout, duration, step):
    """
    Create a GIF from images in the input folder.

    Parameters:
    - naming_con: Naming convention that figures start with.
    - input_folder: Folder containing the images to compile.
    - dout: Path for the output GIF file.
    - total_duration: Total duration (in seconds) for the entire GIF.
    - step: Only include every nth file from the folder.
    """
    # Collect image file paths from the input folder
    images = []
    file_list = sorted(glob.glob(f"{input_folder}*.png"))
    selected_files = file_list[::step]  # Select every nth file

    for file_name in selected_files:
        images.append(imageio.imread(file_name))

    # Create the GIF with per-frame duration
    imageio.mimsave(f"{dout}{naming_con}.gif", images, duration=duration, loop=1)

# naming_con = "20260414_D2"
# input_folder = "/home/robbyfrost/Figures/MW/truck/20260414/"
# dout = input_folder
# duration = 300
# step = 1

# create_gif(naming_con, input_folder, dout, duration, step)
# print(f"Output to: {dout}{naming_con}.gif")