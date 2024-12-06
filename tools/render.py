# Render interactive mode with cube
# Render cross section to analyze bouncing around low density areas
# See runcuda.py under \OLD\

import tools.lut_utils as utils
import ColorLib.output as output

def render_diffuse(mode, molecule_name, choice):
    phase, custom_metadata, pdf, aux_columns, _ = utils.read_lut(molecule_name, mode, "Phase", choice)
    cdf_phase, cdf_custom_metadata, cdf_pdf, cdf_aux_columns, _ = utils.read_lut(molecule_name, mode, "CDF", choice)


    #TODO: To be refactored

    if pdf is None:
        NotImplementedError("The LUT read does not feature a valid PDF or a valid PDF could not be derived from auxiliary columns.")  

    if "ext" not in custom_metadata["aux_columns"]:
        ValueError("Extinction must be in the aux columns!")

    if "scat" in custom_metadata["aux_columns"] and "scat" not in custom_metadata["normalization"]:
        true_phase = pdf  * utils.get_aux_column(aux_columns, custom_metadata, "scat")
    elif "scat" not in custom_metadata["aux_columns"] and "scat" in custom_metadata["normalization"]:
        true_phase = phase
    else:
        ValueError("The phase function must be normalized to the scattering cross section, or an aux column must contain the scattering cross section!")

    extinction = utils.get_aux_column(aux_columns, custom_metadata, "ext")
    true_phase

    return