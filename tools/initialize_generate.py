from .data import read
from tools import generate_lut

# Main

def initialize_lut_gen(args):

    molecule_name = args.name
    settings_profile = args.profile
    mode = args.mode
    settings_data = {}
    ior_data = None

    if molecule_name is not None:
        ior_data = read.read_ior(molecule_name)

    while ior_data is None:
        molecule_name = input("Please enter the exact molecule name (str): ")

        if not isinstance(molecule_name, str):
            print("Molecule name must be a string.")
        else:
            ior_data = read.read_ior(molecule_name)
        

    settings_data = read.read_settings()['lut_gen']

    if mode is None or mode not in settings_data:
        if input("Would you like to evaluate molecules, rather than spheres? (Saying Y will estimate IOR change with number density) (Y/N): ") == "Y":
            settings_data = settings_data['molecule_mode']
            mode = 'molecule_mode'
        else:
            settings_data = settings_data['droplet_mode']
            mode = 'droplet_mode'
    else:
        settings_data = settings_data[mode]

    if settings_profile is None or settings_profile not in settings_data:
        print(f"There are {len(settings_data) - 1} profiles.")
        profiles = [key for key in settings_data if key != 'template']
        print(f"Available profiles: {profiles}")

        while settings_profile not in profiles:
            settings_profile = input("Enter the exact profile name you wish to use: ")
            if settings_profile not in profiles:
                print("Settings profile not found. Please try again.")

    settings_data[settings_profile]['Profile'] = settings_profile
    generate_lut.image_loop(settings_data[settings_profile], ior_data, mode, molecule_name)

    return


# Aux

toollist = {'chop', 'integrate'}


def initialize_aux_gen(args):
    molecule_name = args.name
    mode = args.mode
    choice = args.choice
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
 
    if args.generate_command == "cdf":
        while flavor is None:
            if input("Would you like to evaluate the chopped phase function, rather than the normal one? (Y/N): ") == "Y":
                flavor = 'Phase'
            else:
                flavor = 'Chopped_Phase'
        
        generate_lut.cdf_loop(flavor, mode, molecule_name, choice)
    elif args.generate_command == "modify":

        tool = args.tool
        while tool is None:
            tool = input("Please input the tool you wish to use: ")

            if tool not in toollist:
                print("Invalid tool!")
                tool = None


        if args.tool == "chop":
            params = args.params
            while params is None:
                params = str(input('Please enter the width of the diffraction peak, followed by a comma and space, and then the degrees after the peak to use in the extrapolation (ex. "10, 5"): ')).split(", ")
            params = params.split(", ")

            generate_lut.chop_loop(mode, molecule_name, choice, params)
        else:
            SyntaxError(f"Tool {args.tool} not found!")

    return