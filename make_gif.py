import imageio
import os

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
    file_list = sorted(os.listdir(input_folder))
    selected_files = file_list[::step]  # Select every nth file
    for file_name in selected_files:
        if file_name.startswith(naming_con):
            file_path = os.path.join(input_folder, file_name)
            print(file_path)
            images.append(imageio.imread(file_path))
    # Create the GIF with per-frame duration
    imageio.mimsave(dout, images, duration=duration, loop=0)

naming_con = "refl_worms"
input_folder = "/data/arrcwx/robbyfrost/cm1_output/cm1_supercell_75m_turb/storm_1min_out/"
dout = input_folder
duration = 100
step = 1

create_gif(naming_con, input_folder, dout, duration, step)