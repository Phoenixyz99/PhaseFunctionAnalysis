import argparse
import logging
import os
from tools import initialize_generate, visualize

log_file_path = os.path.join(os.path.dirname(__file__), 'log.log')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

#logging.debug("Debug message from main.py")
#logging.info("Info message from main.py")
#logging.warning("Warning message from main.py")
#logging.error("Error message from main.py")
#logging.critical("Critical message from main.py")

def main(args):
    if args.module == "generate":
        args.mode = None
        
        if args.generate_command in ["lut", "cdf", "modify"]:
            
            if args.molecule is not None:
                args.mode = 'molecule_mode'
            elif args.droplet is not None:
                args.mode = 'droplet_mode'

            if args.generate_command != "lut":
                if args.creation_date is not None:
                    args.choice = args.creation_date
                elif args.file_name is not None:
                    args.choice = args.file_name
                
                if args.generate_command == "cdf":
                    initialize_generate.initialize_aux_gen(args)
                elif args.generate_command == "modify":
                    initialize_generate.initialize_aux_gen(args)
            
            initialize_generate.initialize_lut_gen(args)

        else:
            ValueError(f"{args.generate_command} not found!")

    elif args.module == "visualize":
        args.mode = None

        if args.molecule is not None:
            args.mode = 'molecule_mode'
        elif args.droplet is not None:
            args.mode = 'droplet_mode'

        visualize.initialize_vis(args)

        
            

# generate lut [molecule] [-p Profilename] [-d -m]
# generate chop [molecule] [file]
# generate cdf [molecule] [file] [-d or -m] [-c to sample chopped LUT]
# If [file] is an int, chose the latest, second latest, etc. file in the LUT folder. Otherwise, file name of the EXR.
# -d and -m for droplet or molecule modes.

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("module", choices=["generate", "visualize"], help="Specify which tool to use")
    subparsers = parser.add_subparsers(dest="generate_command", help="Subcommand under 'generate'")
    subparservis = parser.add_subparsers(dest="visualize_command", help="Subcommand under 'visualize'")

    # LUT subparser
    lut_parser = subparsers.add_parser("lut", help="Generate LUT")
    lut_parser.add_argument("name", nargs='?', const=None, default=None, help="Name of the molecule")
    
    # Exclusive to LUT
    lut_parser.add_argument("-p", "--profile", type=str, help="Settings profile")
    lut_mode_group = lut_parser.add_mutually_exclusive_group()
    lut_mode_group.add_argument("-m", "--molecule", action="store_true", help="Set mode to molecule_mode")
    lut_mode_group.add_argument("-d", "--droplet", action="store_true", help="Set mode to droplet_mode")



    # CDF subparser
    cdf_parser = subparsers.add_parser("cdf", help="Generate CDF")
    cdf_parser.add_argument("name", nargs='?', const=None, default=None, help="Name of the molecule")
    
    # Exclusive to CDF
    cdf_parser.add_argument("-p", "--profile", type=str, help="Settings profile")
    cdf_parser.add_argument("-f", "--flavor", type=str, help="Phase or Chopped_Phase")
    cdf_mode_group = cdf_parser.add_mutually_exclusive_group()
    cdf_mode_group.add_argument("-m", "--molecule", action="store_true", help="Set mode to molecule_mode")
    cdf_mode_group.add_argument("-d", "--droplet", action="store_true", help="Set mode to droplet_mode")

    cdf_mode_file = cdf_parser.add_mutually_exclusive_group()
    cdf_mode_file.add_argument("-c", "--creation_date", type=int, help="Enter an integer to chose from the latest generated LUTs (1 being latest)")
    cdf_mode_file.add_argument("-n", "--file_name", type=str, help="Enter the exact .exr file name")



    # Phase Modifier subparser
    modify_parser = subparsers.add_parser("modify", help="Modify an LUT")
    modify_parser.add_argument("name", nargs='?', const=None, default=None, help="Name of the molecule")
    
    # Exclusive to Phase Modifier
    modify_parser.add_argument("-f", "--flavor", type=str, help="Phase or Chopped_Phase")
    modify_parser.add_argument("-t", "--tool", type=str, help="Select the tool to use [Chop, Integrate]")
    modify_parser.add_argument("-p", "--params", type=str, help="Tool parameters")
    modify_group = modify_parser.add_mutually_exclusive_group(required=True)
    modify_group.add_argument("-m", "--molecule", action="store_true", help="Set mode to molecule_mode")
    modify_group.add_argument("-d", "--droplet", action="store_true", help="Set mode to droplet_mode")

    modify_group_file = modify_parser.add_mutually_exclusive_group(required=True)
    modify_group_file.add_argument("-c", "--creation_date", type=int, help="Enter an integer to chose from the latest generated LUTs (1 being latest)")
    modify_group_file.add_argument("-n", "--file_name", type=str, help="Enter the exact .exr file name")



    # Vis subparser
    vis_parser = subparservis.add_argument("name", nargs='?', const=None, default=None, help="Name of the molecule")
    vis_parser.add_argument("-f", "--flavor", type=str, help="Phase or Chopped_Phase")
    vis_parser.add_argument("-p", "--params", type=str, help="Tool parameters")

    # Exclusive to Vis
    vismodify_group_file = vis_parser.add_mutually_exclusive_group(required=True)
    vismodify_group_file.add_argument("-c", "--creation_date", type=int, help="Enter an integer to chose from the latest generated LUTs (1 being latest)")
    vismodify_group_file.add_argument("-n", "--file_name", type=str, help="Enter the exact .exr file name")

    vismodify_group = vis_parser.add_mutually_exclusive_group(required=True)
    vismodify_group.add_argument("-m", "--molecule", action="store_true", help="Set mode to molecule_mode")
    vismodify_group.add_argument("-d", "--droplet", action="store_true", help="Set mode to droplet_mode")



    args = parser.parse_args()
    main(args)