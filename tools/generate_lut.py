from .data import read
import numpy as np
from ColorLib import cieobserver, output
from scipy import interpolate, constants
from scipy.integrate import cumulative_simpson
import miepython
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import os
from datetime import datetime

# TODO: Move everything over to a "generate.py" and two files: one for generating the mie LUT and one for the CDF
# TODO: Then, you can start on the mie chopper that chops off diffraction peaks.
# TODO: For chopper: Identify areas of signifigant spikes (difference between minimum and local maximums).
# TODO: Then, take the average rate of change of the last area of angle n, and then extrapolate that slope in a linear approximation.
# TODO: Compute the difference in the fraction of the energy between the old version and the new version. Ensure you normalize the chopped PDF.
# TODO: Prompt the user to lower their volume density by that fraction of energy. 

# More points near the end
def base_e_out(start, end, count, scale=2.5):
    return start + (end - start) * (1 - np.exp(-scale * np.linspace(0, 1, count)))



# More points near the start
def base_e_in(start, end, count, scale=2.5):
    return start + (end - start) * (np.exp(-scale * np.linspace(0, 1, count)))



def dipole_radius(new_Ndensity, ior_array, material_data): 
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
    new_ior_array = ((-1 - 2 * clausius_mossotti)/(clausius_mossotti - 1))**(1/2)

    return radius_array, new_ior_array  # (180, 1000)



def build_image(settings, aux):
    image_size = settings['column_count'] + len(aux) # One column for absorption cross-section, another for scattering cross-section.
    image_size_x = image_size
    density_block_size = int(np.floor((image_size) / settings['med_ior_rows']))

    if 'rows' in settings:
        density_block_size = settings['rows'] # Row override
        image_size_x = density_block_size

    image = np.zeros((image_size_x, image_size, 3), dtype=np.float32)
    return image, density_block_size



def build_arrays(settings, mode, density_block_size, ior_data):

    epsilon = settings['epsilon']

    wl_n, n_values, wl_k, k_values, Ndensity, pressure, temperature, citation = ior_data

    # We use in smoothing to increase the sampling of smaller sizes/densities/iors, which is typically multiple orders of magnitude smaller.
    med_ior_array = base_e_in(settings['starting_med_ior'] + epsilon, settings['ending_med_ior'], settings['med_ior_rows'], settings['med_ior_scale'])
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
        density_array = base_e_in(settings['starting_ndensity'] + epsilon, settings['ending_ndensity'], density_block_size, settings['ndensity_scale'])

        if (Ndensity is None and temperature is None and pressure is None):
            raise ValueError("The data chosen does not feature a number density, temperature, and pressure!")

        radius_array, ior_array = dipole_radius(density_array, ior_array, material_data)
    else: # Otherwise, process a range of molecule sizes rather than number densities.
        radius_array = base_e_in(settings['starting_size'] + epsilon, settings['ending_size'], density_block_size, settings['size_scale'])

    return radius_array, wavelength_array, ior_array, abs_array, med_ior_array, material_data



def invert_cdf(mu, cdf, xi):

    inverse_cdf = interpolate.interp1d(
        cdf,
        mu,
        kind='linear',
        bounds_error=False,
        fill_value=(mu[0], mu[-1])
    )

    mu_values = inverse_cdf(xi)
    return mu_values



def importance_integral(theta, pdf): # NOTE: High anisotropic functions will not have a high enough resolution to sample correctly, unless you set the image size to 18 * 10^5+
    norm_pdf = pdf / np.sum(pdf)
    sorted_mu = np.argsort(theta)
    mu_sorted = theta[sorted_mu]
    pdf_sorted = norm_pdf[sorted_mu]

    cdf_array = cumulative_simpson(pdf_sorted, x=mu_sorted, initial=0)
    cdf_array /= cdf_array[-1]

    # Pre-computing inversion means the loss of the potential for detail in areas of interest as the water phase function is
    # so anistopropic - rainbows are essentially meaningless with a pre-computed inverted CDF. Computing it on-the-fly with linear interpolation is the best option.
    #xi_array = np.linspace(0, 1, len(mu))
    #mu_values = invert_cdf(mu_sorted, cdf_array, xi_array)

    return cdf_array



def mie_theory(settings, complex_ior, med_ior, radius, wavelength):

    diameter = radius * 2
    theta = np.linspace(0, np.pi, settings['angle_count'])

    ipar, iper = miepython.ez_intensities(complex_ior, diameter, wavelength, np.cos(theta), med_ior, "one")
    probability_distribution = ((ipar + iper) / 2)

    qext, qscat, qback, g = miepython.ez_mie(complex_ior, diameter, wavelength, med_ior)
    
    return probability_distribution, theta, (qext, qscat, qback, g)


def process_wavelengths(args):
    settings, aux, radius, wavelength_array_list, ior_array_list, abs_array_list, medium_ior = args

    radius_is_list = False
    if isinstance(radius, list):
        radius_is_list = True
    
    wavelength_array = np.array(wavelength_array_list)
    ior_array = np.array(ior_array_list)
    abs_array = np.array(abs_array_list)

    xyz_pdf = np.zeros((settings['angle_count'], 3))
    xyz_phase = np.zeros((settings['angle_count'], 3))
    rgb_cdf = np.zeros((settings['angle_count'], 3))
    xyz_cross = np.zeros((len(aux), 3))
    
    for l, wavelength in enumerate(wavelength_array):
        n = ior_array[l]  # Sample the ior at the specific wavelength
        k = abs_array[l]
        # Compute phase function and cross-section using mie_theory

        radius_sample = radius
        if radius_is_list == True:
            radius_sample = radius[l]

        probability_distribution, theta, auxiliary_data = mie_theory(settings, complex(n, -k), medium_ior, radius_sample, wavelength)
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
            "wiscombe": np.pi * ((2 * np.pi * radius_sample)/wavelength)**2
        }

        scat_phase = probability_distribution * normalization_case_table[settings["normalization"]]

        color = cieobserver.wavelength_to_xyz(wavelength * 1e+9)
        xyz_phase += color * scat_phase[:, np.newaxis]
        xyz_pdf += color * probability_distribution[:, np.newaxis]
        
        for i, aux_value in enumerate(aux):
            xyz_cross[i] += color * normalization_case_table[aux_value]

    rgb_pdf = cieobserver.xyz_to_rgb(xyz_pdf)
    rgb_phase = cieobserver.xyz_to_rgb(xyz_phase)
    rgb_cross = cieobserver.xyz_to_rgb(xyz_cross)

    # Calculate the probability of light that is red, green, or blue to scatter at a certain angle.
    if settings['generate_cdf'] == "True":
        for i in range(3): # The left side of the image is xi = 0, the right side is xi = 1
            rgb_cdf[:, i] = importance_integral(np.cos(np.linspace(0, np.pi, settings['angle_count'])), rgb_pdf[:, i])

    return rgb_phase, rgb_cdf, rgb_cross


def save_image(settings, ior_data, mode, molecule_name, phase, cdf):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.abspath(os.path.join(script_dir, '..', 'output'))
    mol_folder = os.path.join(output_folder, str(molecule_name))
    mode_folder = os.path.join(mol_folder, str(mode))
    os.makedirs(mode_folder, exist_ok=True)

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    phase_path = os.path.join(mode_folder, f"Phase_{time}.exr")
    cdf_path = os.path.join(mode_folder, f"CDF_{time}.exr")

    output.save_exr_image(phase_path, phase, settings)
    print(f"Phase function saved to {phase_path}")
    
    if settings['generate_cdf'] == "True":
        output.save_exr_image(cdf_path, cdf, settings)
        print(f"CDF saved to {cdf_path}")

    print("Remember to include citations:")
    citations = ior_data[7].split(" | ")
    for cite in citations:
        print(cite)



def image_loop(settings, ior_data, mode, molecule_name):

    aux = np.array([item.strip() for item in settings['aux_columns'].split(',')])
    image, density_block_size = build_image(settings, aux)

    phase = np.copy(image)
    cdf = None
    if settings['generate_cdf'] == "True":
        cdf = np.copy(image)

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

        for j, (rgb_phase, rgb_cdf, rgb_cross) in enumerate(results):
            # Map angles onto columns
            angle_range = np.linspace(0, settings['angle_count'] - 1, settings['column_count'])
            new_rgb_phase = np.zeros((settings['column_count'], 3))
            new_rgb_cdf = np.zeros((settings['column_count'], 3))

            for b in range(3):
                akima_interpolator = interpolate.Akima1DInterpolator(np.arange(settings['angle_count']), rgb_phase[:, b])
                new_rgb_phase[:, b] = akima_interpolator(angle_range)

            for b in range(3):
                akima_interpolator = interpolate.Akima1DInterpolator(np.arange(settings['angle_count']), rgb_cdf[:, b])
                new_rgb_cdf[:, b] = akima_interpolator(angle_range)

            # Write results to image
            phase[(i * density_block_size) + j, -len(aux):, :] = rgb_cross
            phase[(i * density_block_size) + j, :-len(aux), :] = new_rgb_phase

            if settings['generate_cdf'] == "True":
                cdf[(i * density_block_size) + j, :-len(aux), :] = new_rgb_cdf


    save_image(settings, ior_data, mode, molecule_name, phase, cdf)

    # NOTE: To sample from the CDF, use the random number to sample the x axis. The resulting value of that pixel will be cos(theta)

    return

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
        if input("Would you like to evaluate molecules, rather than spheres? (Saying Y will use the Rayleigh Approximation) (Y/N): ") == "Y":
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
    image_loop(settings_data[settings_profile], ior_data, mode, molecule_name)

    


