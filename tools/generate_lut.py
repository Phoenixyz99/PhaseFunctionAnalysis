# Generate phase LUT and CDF-1 LUT
# Two modes: Droplet mode, molecule mode
# Metadata:
# Mode, min and max parameters, molecule data
# Format: 1800 angles + Absorption cross section + Scattering cross section + Extinction coefficient
# Changeable format to allow for phase to be phase * cross section * 4pi or phase * absorption cross section, etc.

from .data import read

def initialize_lut_gen():
    molecule_name = input("Please enter the exact molecule name (str):")
    read.read_ior(molecule_name)