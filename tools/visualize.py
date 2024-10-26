import lut_utils as utils
import ColorLib.output as output
import numpy as np


# Step 1. Read exr file
# Step 2. Ask the user to pick a medium IOR
# Step 3. Ask the user to pick a row (number density)
# Step 4. Plot the phase function or export an EXR image based on the user's request

def visualize_phase(molecule_name, mode, flavor, choice):
    phase, custom_metadata, _, aux_columns, _ = utils.read_lut(molecule_name, mode, flavor, choice)

    # phase = (H, W, (R, G, B)) ###MAIN IMAGE
    # custom_metadata = dict[]
    # aux_columns = (H, W, (R, G, B))

    # 1. Determine how many medium IOR blocks there are 
    # 1.5. Determine the medium IORs for each block # WARNING
    # 2. Ask the user which medium IOR they wish to visualize. ex. "Here's a list of med_iors: 1.001, 1.002, etc. Which one do you want?"
    # 3. Determine how many Ndensity arrays there are (rows in the block)
    # 3.5. Determine the number densities of each row # WARNING
    # 4. Ask the same question as 2. but with Number Density
    # 5. Ask if the user wants to plot a graph or save an EXR file

    theta = np.linspace(0, np.pi, phase.shape[1])

    # Dimensions
    phase.shape[0] # Height
    phase.shape[1] # Width
    phase.shape[2] # RGB (This will always be 3)

    #Ignore these, do not remove them
    aux = np.array([item.strip() for item in custom_metadata['aux_columns'].split(',')])
    image, density_block_size = utils.build_lut(custom_metadata, aux)

    density_block_size # Size of each medium IOR block (height)

    if mode == "molecule_mode":
        # Given the number of number density rows, calculate the exact number density for each row.
        # This means going to utils.base_e_in, and solving it for the index of the arrays.
        # AKA, given an array index (row 4), give the exact number density as an output.

        custom_metadata['starting_ndensity'] # Etc.
    else:
        # Do the same here with droplet radius reather than number density
        
        custom_metadata['starting_radius'] # Etc.

    output.plot(yaxis=intensity, xaxis=theta, xlabel="Theta", ylabel="Intensity", scale="logy")

    return



def initialize_vis(args):
    molecule_name = args.name
    mode = args.mode
    flavor = args.flavor

    while molecule_name is None:
        molecule_name = input("Please enter the exact molecule name (str): ")

        if not isinstance(molecule_name, str):
            print("Molecule name must be a string.")
    
    while mode is None:
        if input("Would you like to evaluate molecules, rather than spheres? (Y/N): ") == "Y":
            settings_data = settings_data['molecule_mode']
            mode = 'molecule_mode'
        else:
            settings_data = settings_data['droplet_mode']
            mode = 'droplet_mode'

    while choice is None:
        print("Please enter the exact name of the .exr file you wish to evaluate.")
        choice = input("Or, enter an int representing the number of files since the recently created file. (1 is most recent): ")

        if not isinstance(choice, str) or not isinstance(choice, int):
            print("Invalid entry.")
            choice = None

    visualize_phase(molecule_name, mode, flavor, choice)
    return

    