import pandas as pd
import json
from ColorLib import output

def read_ior(molecule_name):
    """"""

    info = _get_info(molecule_name)
    if info is None:
        return
    diameter, citation = info

    try:
        df = pd.read_csv(fr"molecules\{molecule_name}.csv", header=None)
    except FileNotFoundError:
        print(f"CSV file for {molecule_name} not found!")
        return

    # Assuming k follows the blank line after n
    split_index = df[df.isna().any(axis=1)].index[0]

    # n
    df_n = df.iloc[:split_index].dropna()
    wl_n = df_n[0].values.astype(float)
    n_values = df_n[1].values.astype(float)

    # k
    df_k = df.iloc[split_index+1:].dropna()
    wl_k = df_k[0].values.astype(float)
    k_values = df_k[1].values.astype(float)

    return wl_n, n_values, wl_k, k_values, diameter, citation 

def _get_info(molecule_name):

    """Gets the info for the molecule from the data.json file.
    
    Args:
        molecule_name: Name of the molecule (string).
    
    Returns:
        None: if the data.json entry is not found.
        float, str: Diameter of the molecule in Angstroms and the citation(s) for the IOR data.
    """

    global data
    with open(r"data.json", 'r') as file:
        data = json.load(file)

    for molecule in data['molecules']:
        if molecule['name'].lower() == molecule_name.lower():
            return molecule['diameter'], molecule['citation']

    print(f"data.json entry for {molecule_name} not found!")
    return