import argparse
import logging
import os
from tools import initialize_tools, visualize

# TODO: MAJOR# Move the phase function to its own class.

# TODO: Use a text-based approach, like ORCA uses
# TODO: Allow command-line accessibility from the whole OS (again, like ORCA)

# TODO: Allow output data settings
# TODO: Allow graph settings
# TODO: Allow logs and log settings
# TODO: Add regular logging

# Configure logging
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

"""
Main CLI Tool for generating, visualizing, and rendering LUTs.

Usage:
    main.py generate [lut|cdf|modify] [options]
    main.py visualize [options]
    main.py render [scene|diffuse] [options]

Modules:
    generate    Tools for generating LUTs and related files.
    visualize   Tools for visualizing LUTs.
    render      Tools for rendering scenes or diffuse maps.

Common Arguments:
    name                Name of the molecule.
    -f, --flavor        Flavor of the LUT (e.g., Phase, Chopped_Phase).
    -m, --molecule      Set mode to molecule_mode.
    -d, --droplet       Set mode to droplet_mode.
    -c, --creation_date Choose from the latest LUTs (1 being latest).
    -n, --file_name     Exact .exr file name.

Examples:
    main.py generate lut water -m
    main.py visualize water -f Phase -m -c 1
    main.py render scene water -f Phase -m -n file.exr
"""

def add_common_arguments(parser):
    """
    Adds common arguments to the parser.
    These arguments are shared across multiple commands
    to avoid code duplication.
    """
    parser.add_argument("name", nargs='?', const=None, default=None, help="Name of the molecule")
    parser.add_argument("-f", "--flavor", type=str, help="Flavor of the LUT (e.g., Phase, Chopped_Phase)")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("-m", "--molecule", action="store_true", help="Set mode to molecule_mode")
    mode_group.add_argument("-d", "--droplet", action="store_true", help="Set mode to droplet_mode")
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument("-c", "--creation_date", type=int, help="Choose from the latest LUTs (1 being latest)")
    file_group.add_argument("-n", "--file_name", type=str, help="Exact .exr file name")

def main(args):
    """
    Main function to handle the parsed arguments and execute
    the appropriate functions based on the module and commands.
    """
    if args.module == "generate":
        # Handle 'generate' module commands
        args.mode = None

        if args.generate_command in ["lut", "cdf", "modify"]:
            if args.molecule:
                args.mode = 'molecule_mode'
            elif args.droplet:
                args.mode = 'droplet_mode'

            if args.generate_command != "lut":
                if args.creation_date is not None:
                    args.choice = args.creation_date
                elif args.file_name is not None:
                    args.choice = args.file_name

                if args.generate_command == "cdf":
                    initialize_tools.initialize_aux_gen(args)
                elif args.generate_command == "modify":
                    initialize_tools.initialize_aux_gen(args)
            else:
                initialize_tools.initialize_lut_gen(args)
        else:
            ValueError(f"{args.generate_command} not found!")

    elif args.module == "visualize":
        # Handle 'visualize' module
        args.mode = None

        if args.molecule:
            args.mode = 'molecule_mode'
        elif args.droplet:
            args.mode = 'droplet_mode'

        visualize.initialize_vis(args)

    elif args.module == "render":
        # Handle 'render' module commands
        if args.render_command in ["scene", "diffuse"]:
            args.mode = None

            if args.molecule:
                args.mode = 'molecule_mode'
            elif args.droplet:
                args.mode = 'droplet_mode'

            # Call the appropriate render function (to be implemented)
            if args.render_command == "scene":
                # Example: render_scene(args)
                pass
            elif args.render_command == "diffuse":
                # Example: render_diffuse(args)
                pass
        else:
            ValueError(f"{args.render_command} not found!")

    else:
        ValueError(f"Unknown module {args.module}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main CLI Tool")
    subparsers = parser.add_subparsers(dest="module", required=True, help="Module to use (generate, visualize, render)")

    # Generate subparser and its commands
    generate_parser = subparsers.add_parser("generate", help="Generate tools")
    generate_subparsers = generate_parser.add_subparsers(dest="generate_command", required=True, help="Generation command")

    # LUT command under 'generate' module
    lut_parser = generate_subparsers.add_parser("lut", help="Generate LUT")
    lut_parser.add_argument("name", nargs='?', const=None, default=None, help="Name of the molecule")
    lut_parser.add_argument("-p", "--profile", type=str, help="Settings profile")
    lut_mode_group = lut_parser.add_mutually_exclusive_group()
    lut_mode_group.add_argument("-m", "--molecule", action="store_true", help="Set mode to molecule_mode")
    lut_mode_group.add_argument("-d", "--droplet", action="store_true", help="Set mode to droplet_mode")

    # CDF command under 'generate' module
    cdf_parser = generate_subparsers.add_parser("cdf", help="Generate CDF")
    add_common_arguments(cdf_parser)
    cdf_parser.add_argument("-p", "--profile", type=str, help="Settings profile")

    # Modify command under 'generate' module
    modify_parser = generate_subparsers.add_parser("modify", help="Modify an LUT")
    add_common_arguments(modify_parser)
    modify_parser.add_argument("-t", "--tool", type=str, help="Select the tool to use [Chop, Integrate]")
    modify_parser.add_argument("-p", "--params", type=str, help="Tool parameters")

    # Visualize subparser
    visualize_parser = subparsers.add_parser("visualize", help="Visualize tools")
    add_common_arguments(visualize_parser)
    visualize_parser.add_argument("-p", "--params", type=str, help="Tool parameters")

    # Render subparser and its commands
    render_parser = subparsers.add_parser("render", help="Render tools")
    render_subparsers = render_parser.add_subparsers(dest="render_command", required=True, help="Render command")

    # Scene command under 'render' module
    scene_parser = render_subparsers.add_parser("scene", help="Render scene")
    add_common_arguments(scene_parser)

    # Diffuse command under 'render' module
    diffuse_parser = render_subparsers.add_parser("diffuse", help="Render diffuse")
    add_common_arguments(diffuse_parser)

    args = parser.parse_args()
    main(args)
