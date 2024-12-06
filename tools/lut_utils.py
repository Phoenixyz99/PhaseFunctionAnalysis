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

def reverse_base_e_in(start, end, count, scale, result):
    t = -np.log((result - start) / (end - start)) / scale
    index = int(round(t * (count - 1)))
    return max(0, min(count - 1, index))

#TODO: To be refactored
def normalize_phase(phase):
    phase = np.asarray(phase)
    total_sum = np.sum(phase, axis=0)

    if np.any(total_sum == 0): # Skip values that are erroneous
        return phase

    pdf = phase / total_sum
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    
    return pdf


#TODO: To be refactored
def build_lut(settings, aux_data):
    image_size = settings['column_count'] + len(aux_data)
    image_size_x = image_size
    density_block_size = int(np.floor((image_size) / settings['med_ior_rows']))

    if 'rows' in settings:
        density_block_size = settings['rows'] # Row override
        image_size_x = density_block_size

    image = np.zeros((image_size_x, image_size, 3), dtype=np.float32)
    return image, density_block_size


#TODO: To be refactored
def read_lut(molecule, mode, flavor, choice=None):
    current_dir = Path(__file__).parent
    path = current_dir.parent / 'output' / molecule / mode / flavor

    images = sorted(path.glob('*.exr'), key=os.path.getmtime, reverse=True)
    
    if not images:
        raise FileNotFoundError(f"No images found in directory: {path}")
    
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

        aux_columns = np.zeros((height, len(aux), 3))

        for i, column in enumerate(aux):
            aux_columns[:, i, :] = phase[:, width - len(aux) + i, :]

        phase = phase[:, :-len(aux), :]  # Exclude the aux columns from the phase
    else:

        aux_columns = None 

    pdf = normalize_phase(phase)

    return phase, custom_metadata, pdf, aux_columns, metadata


#TODO: To be refactored
def get_aux_column(aux_columns, custom_metadata, key):
    """
    Retrieve a specific column from aux_columns based on the key in custom_metadata['aux_columns'].

    Args:
        aux_columns (numpy.ndarray): The auxiliary columns data.
        custom_metadata (dict): The metadata dictionary containing 'aux_columns'.
        key (str): The key to specify which auxiliary column to retrieve.

    Returns:
        numpy.ndarray: The specific auxiliary column matching the key.
    
    Raises:
        ValueError: If the key is not found in custom_metadata['aux_columns'].
    """
    if aux_columns is None:
        raise ValueError("Auxiliary columns are not available.")
    
    if 'aux_columns' not in custom_metadata or not custom_metadata['aux_columns'].strip():
        raise ValueError("'aux_columns' is not defined in custom_metadata.")
    
    aux_keys = custom_metadata['aux_columns'].split(", ")
    if key not in aux_keys:
        raise ValueError(f"The key '{key}' is not in custom_metadata['aux_columns']: {aux_keys}")
    
    index = aux_keys.index(key)
    return aux_columns[:, index, :]