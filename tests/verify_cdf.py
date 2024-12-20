import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from ColorLib import output
from pathlib import Path
import multiprocessing as mp
import sys
import pathlib

current_dir = Path(__file__).parent
tools_path = current_dir.parent / 'tools'
sys.path.append(str(tools_path))

import lut_utils as utils

# Function for parallelized sampling
def parallel_sampling(args):
    mu_arrays, cdf_normalized = args
    channel_to_visualize = 0  # Channel 0
    n_samples = 100000000 // mp.cpu_count()  # Number of samples per process

    # Rebuild the inverse CDF functions in each process
    inverse_cdf_function = {}
    for i in range(3):
        inverse_cdf_function[i] = interp1d(
            cdf_normalized[:, i],
            mu_arrays[i],
            kind='linear',
            bounds_error=False,
            fill_value=(mu_arrays[i][0], mu_arrays[i][-1])
        )

    random_samples = np.random.uniform(0, 1, size=(n_samples, 3))
    sampled_mu = np.zeros((n_samples, 3))
    for i in range(3):
        sampled_mu[:, i] = inverse_cdf_function[i](random_samples[:, channel_to_visualize])
    
    return sampled_mu

if __name__ == "__main__":

    cdf, _, _, _, _ = utils.read_lut("H2O", "droplet_mode", "CDF", 1)
    print(cdf.shape)
    print("Done")
    _, _, pdf, _, _ = utils.read_lut("H2O", "droplet_mode", "Chopped_Phase", 1)
    print(pdf.shape)

    cdf = cdf[49]
    pdf = pdf[49]

    mu = np.linspace(0, np.pi, len(cdf))  # Create mu array

    # Initialize lists to store processed data
    mu_arrays = []

    for i in range(3):
        cdf_channel = cdf[:, i]
        mu_channel = mu.copy()
        
        # Sort cdf_channel, mu_channel, and pdf_channel together
        sorted_indices = np.argsort(cdf_channel)
        mu_channel_sorted = mu_channel[sorted_indices]
        mu_arrays.append(mu_channel_sorted)

    # Prepare data to pass to parallel processes
    args = (mu_arrays, cdf)

    # Use multiprocessing to generate samples
    with mp.Pool(mp.cpu_count()) as pool:
        sampled_mu_parts = pool.map(parallel_sampling, [args] * mp.cpu_count())

    # Stack the results from all processes
    sampled_mu = np.vstack(sampled_mu_parts)

    # Ensure that the mu values are correctly paired with their corresponding pdf
    mu_selected = np.linspace(0, np.pi, pdf.shape[0])  # Original mu values, evenly spaced
    pdf_selected = pdf[:, 0]  # For channel_to_visualize = 0

    # Histogram estimation of the PDF from the sampled mu values
    bins = np.linspace(0, np.pi, len(mu_selected))
    histogram_estimated_pdf, bin_edges = np.histogram(sampled_mu[:, 0], bins=bins, density=True)

    # Normalize the histogram to match the original PDF area under the curve
    histogram_estimated_pdf *= np.sum(pdf_selected) * (bins[1] - bins[0])

    # Compute the bin centers for the histogram to align with PDF
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Flip the bin centers for the inverse CDF
    bin_centers_flipped = np.flip(bin_centers)

    # Plot the original PDF and the estimated PDF from inverse CDF sampling
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot original PDF
    ax1.plot(mu_selected, pdf_selected, label='Original PDF', color='blue', linewidth=2)  # Original PDF

    # Plot flipped inverse CDF estimated PDF
    ax1.plot(bin_centers_flipped, histogram_estimated_pdf, label='Numerical PDF', linestyle='--', color='red', linewidth=2)  # Estimated PDF

    # Set logarithmic scale on the y-axis
    ax1.set_yscale('log')

    # Add labels and legend to the plot
    ax1.set_title("Analytical PDF vs Monte Carlo of CDF-1")
    ax1.set_xlabel("Mu")
    ax1.set_ylabel("PDF (log scale)")
    ax1.legend()

    # Compute and plot the difference between the PDFs
    difference = np.abs(pdf_selected[:len(histogram_estimated_pdf)] - histogram_estimated_pdf)

    ax2.set_yscale('log')

    # Plot the difference
    ax2.plot(bin_centers_flipped, difference, label='Difference', color='green', linewidth=2)
    ax2.set_title("Difference Between Original PDF and Estimated PDF")
    ax2.set_xlabel("Mu")
    ax2.set_ylabel("Difference")
    ax2.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
