import argparse
import logging
import os
from tools import generate_lut

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
    if args.module == "generate_lut":
            args.mode = None

            if args.molecule:
                args.mode = 'molecule_mode'
            elif args.droplet:
                args.mode = 'droplet_mode'

            generate_lut.initialize_lut_gen(args)

# generate_lut:
# py main.py generate_lut MoleculeName -m/-d -p ProfileName
# python main.py generate_lut MoleculeName --molecule/--droplet --profile ProfileName
#
# Where:
# MoleculeName is the exact name of the molecule
# -p [ProfileName] Denotes the settings profile where [ProfileName] is the exact settings profile.
# -m Denotes molecule mode.
# -d Denotes Droplet mode.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("module", choices=["generate_lut"], help="Specify which tool to use")

    parser.add_argument("name", nargs='?', const=None, default=None, help="Name of the molecule")
    parser.add_argument("-p", "--profile", type=str, help="Settings profile")

    # Args for generate_lut
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("-m", "--molecule", action="store_true", help="Set mode to molecule_mode")
    mode_group.add_argument("-d", "--droplet", action="store_true", help="Set mode to droplet_mode")

    args = parser.parse_args()
    main(args)