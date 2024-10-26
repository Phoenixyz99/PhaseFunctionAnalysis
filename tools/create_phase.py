from scipy.integrate import cumulative_simpson
import miepython
from scipy import interpolate
import numpy as np
from ColorLib import output
from tools import lut_utils as utils


def invert_cdf(mu, cdf, xi):
    """Given mu, an array of all cos(theta) points in the cdf, calculate the resulting value of cos(theta) given a random number xi, between 0 and 1."""

    inverse_cdf = interpolate.interp1d(
        cdf,
        mu,
        kind='linear',
        bounds_error=False,
        fill_value=(mu[0], mu[-1])
    )

    mu_values = inverse_cdf(xi)
    return mu_values


def cdf(theta, pdf): # NOTE: High anisotropic functions will not have a high enough resolution to sample correctly, unless you set the image size to 18 * 10^5+\
    """
    mu is cos(theta)"""
    pdf = utils.normalize_phase(pdf)
    sorted_mu = np.argsort(theta)
    mu_sorted = theta[sorted_mu]
    pdf_sorted = pdf[sorted_mu]

    cdf_array = cumulative_simpson(pdf_sorted, x=mu_sorted, initial=0)
    cdf_array /= cdf_array[-1]

    # Pre-computing inversion means the loss of the potential for detail in areas of interest as the water phase function is
    # so anistopropic - rainbows are essentially meaningless with a pre-computed inverted CDF. Computing it on-the-fly with linear interpolation is the best option.
    #xi_array = np.linspace(0, 1, len(mu))
    #mu_values = invert_cdf(mu_sorted, cdf_array, xi_array)

    return cdf_array

def mie(settings, complex_ior, med_ior, radius, wavelength):

    diameter = radius * 2
    theta = np.linspace(0, np.pi, settings['angle_count'])

    ipar, iper = miepython.ez_intensities(complex_ior, diameter, wavelength, np.cos(theta), med_ior, "one")
    probability_distribution = ((ipar + iper) / 2)

    qext, qscat, qback, g = miepython.ez_mie(complex_ior, diameter, wavelength, med_ior)
    
    return probability_distribution, theta, (qext, qscat, qback, g)

def chop(phase, width=5, length=5):
    n, c = phase.shape  # Get the dimensions

    degree = int(np.floor(n // 180))  # Convert to an integer after flooring
    peak_degrees = degree * width  # Multiply number of indices by width of degrees
    deriv_degrees = degree * length  # Degrees after width to consider for extrapolation
    deriv_index = int(deriv_degrees)
    peak_index = int(peak_degrees - 1)

    prepower = sum(phase[:peak_index, :])

    if peak_index + deriv_index > n or peak_index < 0 or deriv_index <= 1:
        raise ValueError("Invalid index or n values")

    # Extract the part of the array after the peak index
    arr_after_index = phase[peak_index:peak_index + deriv_index, :]

    # Calculate the first derivative along the n dimension for each column
    first_derivative = np.gradient(arr_after_index, axis=0)  # shape (deriv_index, c)

    # Calculate the second derivative along the n dimension for each column
    second_derivative = np.gradient(first_derivative, axis=0)  # shape (deriv_index, c)

    # Compute the mean second derivative across the slice for each column
    mean_second_derivative_c = np.mean(second_derivative, axis=0)  # shape (c,)

    # Compute the first derivative at peak_index for each column
    first_derivative_at_peak = first_derivative[0, :]  # shape (c,)

    # Compute the phase at peak_index for each column
    phase_at_peak = phase[peak_index, :]  # shape (c,)

    # Replace elements before the index with extrapolated values
    for i in range(peak_index - 1, -1, -1):
        distance = peak_index - i

        # Extrapolated value for each column at position i using Taylor series expansion
        extrapolated_value_c = (
            phase_at_peak
            - first_derivative_at_peak * distance
            + 0.5 * mean_second_derivative_c * distance ** 2
        )  # shape (c,)

        # Compute the mean extrapolated value at position i across all color channels
        mean_extrapolated_value = np.mean(extrapolated_value_c)

        # Compute weight that increases towards index 0 for smooth blending
        weight = (peak_index - i) / peak_index  # Increases from 0 at peak_index to 1 at i=0

        # Blend each channel towards the uniform brightness represented by the mean extrapolated value
        phase[i, :] = (1 - weight) * extrapolated_value_c + weight * mean_extrapolated_value

    # Optionally, normalize the phase if needed (e.g., to ensure consistent brightness)
    pdf = utils.normalize_phase(phase)
    postpower = sum(pdf[:peak_index, :])

    ratio = postpower / prepower
    return pdf, ratio





def angular_integrator():

    return