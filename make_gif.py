import imageio
import os

# Input settings
des_elev = 5
input_folder = "/home/robbyfrost/Analysis/TurbTor_Lidar/figures/ARRC_Truck/2024/09/24/"
os.makedirs(input_folder, exist_ok=True)
output_gif = f"GIF_vr_vort_PPI_el{des_elev}_19z.gif"
dout = input_folder + output_gif
duration = 200
step = 1 # only include figures every step

def create_gif(input_folder, dout, duration, step):
    """
    Create a GIF from images in the input folder.

    Parameters:
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
        if file_name.startswith('vr_vortz_PPI_el5_2024092419'):
            file_path = os.path.join(input_folder, file_name)
            images.append(imageio.imread(file_path))
    
    # Create the GIF with per-frame duration
    imageio.mimsave(dout, images, duration=duration, loop=0)

# run function
create_gif(input_folder, dout, duration, step)
print(f"GIF saved at {dout}")