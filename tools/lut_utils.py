import numpy as np
from ColorLib import output
from pathlib import Path
import os

# More points near the end
def base_e_out(start, end, count, scale=2.5):
    return start + (end - start) * (1 - np.exp(-scale * np.linspace(0, 1, count)))


# More points near the start
def base_e_in(start, end, count, scale=2.5):
    return start + (end - start) * (np.exp(-scale * np.linspace(0, 1, count)))


def normalize_phase(phase):
    # Convert input to a NumPy array if it isn't one
    phase = np.asarray(phase)
    
    # Calculate the sum across the specified axis
    total_sum = np.sum(phase, axis=0)
    
    # Handle potential division by zero (if total_sum is 0, we leave the phase unchanged)
    if np.any(total_sum == 0):
        print("Warning: Sum of phase is zero, normalization skipped.")
        return phase
    
    # Normalize the phase values
    pdf = phase / total_sum
    
    # Handle any NaN values that may appear due to division by zero or invalid input
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    
    return pdf


def build_lut(settings, aux_data):
    image_size = settings['column_count'] + len(aux_data) # One column for absorption cross-section, another for scattering cross-section.
    image_size_x = image_size
    density_block_size = int(np.floor((image_size) / settings['med_ior_rows']))

    if 'rows' in settings:
        density_block_size = settings['rows'] # Row override
        image_size_x = density_block_size

    image = np.zeros((image_size_x, image_size, 3), dtype=np.float32)
    return image, density_block_size

def read_lut(molecule, mode, flavor, choice=None):
    current_dir = Path(__file__).parent
    path = current_dir.parent / 'output' / molecule / mode / flavor
    
    # Find all the image files in the directory
    images = sorted(path.glob('*.exr'), key=os.path.getmtime, reverse=True)  # Sorted by most recent first
    
    if not images:
        raise FileNotFoundError(f"No images found in directory: {path}")
    
    # If the user specifies a file name directly
    if isinstance(choice, str):
        specified_file = path / choice
        if not specified_file.exists():
            raise FileNotFoundError(f"Specified file '{choice}' does not exist in directory: {path}")
        chosen_image = specified_file
    # If the user specifies an index (1 for the first file, 2 for the second, etc.)
    elif isinstance(choice, int):
        if choice < 1 or choice > len(images):
            raise IndexError(f"Choice out of range. There are {len(images)} images available.")
        chosen_image = images[choice - 1]  # choice is 1-based index
    # If no choice is given, default to the most recent image
    else:
        chosen_image = images[0]
    
    # Read the chosen image
    phase, metadata, custom_metadata = output.read_exr(str(chosen_image))

    for key, value in metadata.items():
        clean_val = output.clean_value(value)
        metadata[key] = clean_val

    for key, value in custom_metadata.items():
        clean_val = output.clean_value(value)
        custom_metadata[key] = clean_val

    if 'aux_columns' in custom_metadata and custom_metadata['aux_columns'].strip():
        aux = custom_metadata['aux_columns'].split(", ")
        height, width, _ = phase.shape

        # Convert aux_columns into an array instead of a dictionary
        aux_columns = np.zeros((height, len(aux), 3))  # Create a 3D array for aux_columns

        for i, column in enumerate(aux):
            aux_columns[:, i, :] = phase[:, width - len(aux) + i, :]  # Adjust the indexing

        phase = phase[:, :-len(aux), :]  # Exclude the aux columns from the phase
    else:
        # Handle case when aux_columns is missing or blank
        aux_columns = None  # Or handle this based on your specific requirement

    # Continue with phase processing
    pdf = normalize_phase(phase)


    return phase, custom_metadata, pdf, aux_columns, metadata


