# Generate phase LUT and CDF-1 LUT
# Take mie at molecule ior, generate an image LUT in an EXR, where each column is an angle, and each row is a particle size (for droplets) or differing IOR (from Lorentz-Lorenz)
# See importance.py under \OLD\ for CDF-1

# Two modes: Droplet mode, molecule mode
# Metadata:
# Mode, min and max parameters, molecule data
# Format: 1800 angles + Absorption cross section + Scattering cross section + Extinction coefficient
# Changeable format to allow for phase to be phase + cross section * 4pi or phase + absorption cross section, etc.

# Take IOR, extrapolar/interpolate
# Find number density. For gas, get pressure and temperature. For solid/liquid, density and molar mass. Give users an option.
# Find polarizability. Solve loretnz-Lorenz for polarizability using given ior data.
# Use Lorentz-Lorenz to find the resulting n for a new number density.
# Medium IOR and number density changes per row. 

from .data import read

def initialize_lut_gen():
    molecule_name = input("Please enter the exact molecule name (str):")
    read.read_ior(molecule_name)