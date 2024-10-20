import pandas as pd
import json
import os
import numpy as np

def read_ior(molecule_name):
    info = _get_info(molecule_name)
    if info is None:
        return None
    density, pressure, temperature, citation = info

    file_path = os.path.join(os.path.dirname(__file__), "molecules", f"{molecule_name}.csv")

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"CSV file for {molecule_name} not found!")
        return None

    n_start = None
    k_start = None
    for i, line in enumerate(lines):
        if line.strip() == 'wl,n':
            n_start = i + 1
        elif line.strip() == 'wl,k':
            k_start = i + 1
            break

    if n_start is None:
        print("Could not find section header for wl,n")
        return None

    # Read n data
    if k_start is not None:
        n_data = pd.read_csv(file_path, skiprows=n_start, nrows=k_start - n_start - 1, header=None)
    else:
        n_data = pd.read_csv(file_path, skiprows=n_start, header=None)
    
    n_data = n_data[pd.to_numeric(n_data[0], errors='coerce').notna()]
    wl_n = n_data[0].astype(float).values
    n_values = n_data[1].astype(float).values

    # Read k data if available
    if k_start is not None:
        k_data = pd.read_csv(file_path, skiprows=k_start, header=None)
        k_data = k_data[pd.to_numeric(k_data[0], errors='coerce').notna()]
        wl_k = k_data[0].astype(float).values
        k_values = k_data[1].astype(float).values
    else:
        # If no k data, return an array of ones with the same length as n_data
        wl_k = wl_n  # Assume wavelengths are the same
        k_values = np.zeros_like(n_values)

    return (wl_n * 1E-6, n_values, wl_k * 1E-6, k_values, density, pressure, temperature, citation)


def _get_info(molecule_name):
    file_path = os.path.join(os.path.dirname(__file__), 'data.json')

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"{file_path} file not found!")
        return None
    except json.JSONDecodeError:
        print(f"{file_path} is not a valid JSON file!")
        return None

    for molecule in data.get('molecules', []):
        if molecule['name'].lower() == molecule_name.lower():
            citation = molecule.get('citation', None)
            pressure = molecule.get('pressure', None)
            temperature = molecule.get('temperature', None)
            ndensity = molecule.get('ndensity', None)

            if ndensity is None and (pressure is None or temperature is None):
                print(f"Invalid data for {molecule_name}: Either 'pressure' and 'temperature' or 'ndensity' is required.")
                return None

            return ndensity, pressure, temperature, citation

    print(f"data.json entry for {molecule_name} not found!")
    return None


def read_settings():

    file_path = os.path.join(os.path.dirname(__file__), "settings.json")

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"The file at {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")