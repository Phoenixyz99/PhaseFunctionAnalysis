import numpy as np
from ColorLib import cieobserver, output
from scipy import interpolate, constants
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path
from datetime import datetime
from tools import create_phase as create
from tools import lut_utils as utils



def dipole_radius(new_Ndensity, ior_array, material_data): 
    """Given a new number density, ior, and material data containing pressure, temperature, or number density, use the

    Clausius-Mossotti relation to approximate the number density"""
    old_Ndensity, pressure, temperature = material_data

    if old_Ndensity is None:
        old_Ndensity = (pressure * constants.Avogadro)/(constants.R * temperature)

    ior_array = ior_array.reshape(1, -1)  # (1, 1000)
    new_Ndensity = new_Ndensity.reshape(-1, 1)  # (180, 1)

    polarizability = 3 * constants.epsilon_0 * ((ior_array**2 - 1) / (ior_array**2 + 2)) * 1 / old_Ndensity
    clausius_mossotti = (new_Ndensity * polarizability) / (3 * constants.epsilon_0) # Equivalent to (n**2 - 1) / (n**2 + 2)

    if np.any(np.abs(clausius_mossotti) < 1e-10):
        raise ValueError("The calculated IOR from the Clausius-Mossotti relation is exceedingly small!")

    radius_array = (polarizability / (4 * np.pi * constants.epsilon_0) * (1 / clausius_mossotti))**(1 / 3)
    new_ior_array = ((-1 - 2 * clausius_mossotti)/(clausius_mossotti - 1))**0.5

    return radius_array, new_ior_array  # (180, 1000)



def build_arrays(settings, mode, density_block_size, ior_data):

    epsilon = settings['epsilon']

    wl_n, n_values, wl_k, k_values, Ndensity, pressure, temperature, citation = ior_data

    # We use in smoothing to increase the sampling of smaller sizes/densities/iors, which is typically multiple orders of magnitude smaller.
    med_ior_array = utils.base_e_in(settings['starting_med_ior'] + epsilon, settings['ending_med_ior'], settings['med_ior_rows'], settings['med_ior_scale'])
    wavelength_array = np.linspace(3.8E-7, 7.8E-7, num=settings['wavelength_count'])

    if np.any(wavelength_array < wl_n.min()) or np.any(wavelength_array > wl_n.max()):
        print("Warning! Wavelength array contains values beyond the available data range.")

    # Interpolate/exterpolate ior
    cs_n = interpolate.CubicSpline(wl_n, n_values, bc_type="not-a-knot", extrapolate=True)
    ior_array = cs_n(wavelength_array)

    cs_k = interpolate.CubicSpline(wl_k, k_values, bc_type="not-a-knot", extrapolate=True)
    abs_array = cs_k(wavelength_array)

    material_data = Ndensity, pressure, temperature

    if mode == 'molecule_mode': # If molecule_mode, assume a point dipole and use the clausius mossati relation to find the change in ior with number density.
        density_array = utils.base_e_in(settings['starting_ndensity'] + epsilon, settings['ending_ndensity'], density_block_size, settings['ndensity_scale'])

        if (Ndensity is None and temperature is None and pressure is None):
            raise ValueError("The data chosen does not feature a number density, temperature, and pressure!")

        radius_array, ior_array = dipole_radius(density_array, ior_array, material_data)
    else: # Otherwise, process a range of molecule sizes rather than number densities.
        radius_array = utils.base_e_in(settings['starting_radius'] + epsilon, settings['ending_radius'], density_block_size, settings['size_scale'])

    return radius_array, wavelength_array, ior_array, abs_array, med_ior_array, material_data



def process_wavelengths(args):
    settings, aux, radius, wavelength_array_list, ior_array_list, abs_array_list, medium_ior = args

    radius_is_list = False
    if isinstance(radius, list):
        radius_is_list = True
    
    wavelength_array = np.array(wavelength_array_list)
    ior_array = np.array(ior_array_list)
    abs_array = np.array(abs_array_list)

    xyz_phase = np.zeros((settings['angle_count'], 3))
    xyz_aux = np.zeros((len(aux), 3))
    
    for l, wavelength in enumerate(wavelength_array):
        n = ior_array[l]  # Sample the ior at the specific wavelength
        k = abs_array[l]
        # Compute phase function and cross-section using mie

        radius_sample = radius
        if radius_is_list == True:
            radius_sample = radius[l]

        probability_distribution, theta, auxiliary_data = create.mie(settings, complex(n, -k), medium_ior, radius_sample, wavelength)
        qext, qscat, qback, g = auxiliary_data

        radiusSquaredpi = np.pi * radius_sample**2
        scat_cross_section = qscat * radiusSquaredpi
        ext_cross_section = qext * radiusSquaredpi

        normalization_case_table = {
            "one": 1,
            "scat": scat_cross_section,
            "ext": ext_cross_section,
            "qsca": qscat,
            "qext": qext,
            "bohren": 4 * np.pi * ((2 * np.pi * radius_sample)/wavelength)**2,
            "wiscombe": np.pi * ((2 * np.pi * radius_sample)/wavelength)**2,
            "qback": qback,
            "g": g
        }

        scat_phase = probability_distribution * normalization_case_table[settings["normalization"]]

        color = cieobserver.wavelength_to_xyz(wavelength * 1e+9)
        xyz_phase += color * scat_phase[:, np.newaxis]
        
        for i, aux_value in enumerate(aux):
            xyz_aux[i] += color * normalization_case_table[aux_value]

    rgb_phase = cieobserver.xyz_to_rgb(xyz_phase)
    rgb_aux = cieobserver.xyz_to_rgb(xyz_aux)

    return rgb_phase, rgb_aux



def save_image(settings, mode, molecule_name, array, flavor, ior_data=None):

    script_dir = Path(__file__).resolve().parent
    output_folder = script_dir.parent / 'output' / str(molecule_name) / str(mode) / str(flavor)
    output_folder.mkdir(parents=True, exist_ok=True)  # Create directories if they don't exist

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = output_folder / f"{flavor}_{time}.exr"

    output.save_exr_image(str(path), array, settings)
    print(f"{flavor} saved to {path}")

    # IOR citations
    if ior_data is not None:
        print("Remember to include citations:")
        citations = ior_data[7].split(" | ")
        for cite in citations:
            print(cite)



def image_loop(settings, ior_data, mode, molecule_name):

    aux = np.array([item.strip() for item in settings['aux_columns'].split(',')])
    image, density_block_size = utils.build_lut(settings, aux)

    phase = np.copy(image)

    radius_array, wavelength_array, ior_array, abs_array, med_ior_array, material_data = build_arrays(settings, mode, density_block_size, ior_data) # TODO: Allow the user to enter data for medium_ior dispersion

    for i, medium_ior in enumerate(tqdm(med_ior_array, desc="Processing Medium IORs: ", position=0)):
        pool = Pool(cpu_count())
        args = None

        if mode == 'molecule_mode':
            args = [(settings, aux.tolist(), radius.tolist(), wavelength_array.tolist(), ior_array[j].tolist(), abs_array.tolist(), medium_ior) for j, radius in enumerate(radius_array)]
        else:
            args = [(settings, aux.tolist(), radius.tolist(), wavelength_array.tolist(), ior_array.tolist(), abs_array.tolist(), medium_ior) for radius in radius_array]

        results = list(tqdm(pool.imap(process_wavelengths, args), total=len(radius_array), desc="Computing molecule sizes: ", position=1, leave=False))

        pool.close()
        pool.join()

        for j, (rgb_phase, rgb_aux) in enumerate(results):
            angle_range = np.linspace(0, settings['angle_count'] - 1, settings['column_count'])
            new_rgb_phase = np.zeros((settings['column_count'], 3))

            for b in range(3):
                akima_interpolator = interpolate.Akima1DInterpolator(np.arange(settings['angle_count']), rgb_phase[:, b])
                new_rgb_phase[:, b] = akima_interpolator(angle_range)

            #TODO: To be refactored
            phase[(i * density_block_size) + j, -len(aux):, :] = rgb_aux
            phase[(i * density_block_size) + j, :-len(aux), :] = new_rgb_phase


    save_image(settings, mode, molecule_name, phase, "Phase", ior_data)

    return



def cdf_loop(flavor, mode, molecule_name, choice):

    #TODO: To be refactored
    _, custom_metadata, pdf, _, _ = utils.read_lut(molecule_name, mode, flavor, choice)

    height, width, _ = pdf.shape
    cdf = np.zeros((pdf.shape[0], pdf.shape[1], 3))
    for i in range(height):
        for c in range(3):
            row = pdf[i, :, c]
            cdf[i, :, c] = create.cdf(np.linspace(1, -1, width), row)

    custom_metadata['aux_columns'] = ""
    save_image(custom_metadata, mode, molecule_name, cdf, "CDF")

    return



def chop_loop(mode, molecule_name, choice, params):
    phase, custom_metadata, pdf, aux_columns, _ = utils.read_lut(molecule_name, mode, "Phase", choice)

    if pdf is None:
        NotImplementedError("The LUT read does not feature a valid PDF or a valid PDF could not be derived from auxiliary columns.")

    #TODO: To be refactored
    height, width, _ = phase.shape
    ratio = np.zeros((height, 1, 3))
    for i in range(height):
        row = phase[i, :, :]
        phase[i, :, :], ratio[i, 0, :] = create.chop(row, float(params[0]), float(params[1]))

    if aux_columns is None or aux_columns.ndim == 0:
        phase = np.concatenate((phase, ratio), axis=1)
    else:
        aux_columns = np.concatenate((aux_columns, ratio), axis=1)
        phase = np.concatenate((phase, aux_columns), axis=1)

    def add(string, newstring):
        if string:
            return f"{string}, {newstring}"
        else:
            return newstring
        
    custom_metadata['aux_columns'] = add(custom_metadata['aux_columns'], "density_ratio")

    save_image(custom_metadata, mode, molecule_name, phase, "Chopped_Phase")