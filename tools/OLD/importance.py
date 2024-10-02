import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid

# https://math.arizona.edu/~tgk/mc/book_chap6.pdf

def compute_pdf(mu_array, p_array):

    # mu_array needs to be in ascending order
    sorted_indices = np.argsort(mu_array)
    mu_sorted = mu_array[sorted_indices]
    p_sorted = p_array[sorted_indices]

    # p(mu) = 2 * pi * p(theta)
    pdf_unnormalized = 2 * np.pi * p_sorted

    # Normalize
    integral = cumulative_trapezoid(pdf_unnormalized, mu_sorted, initial=0)
    total_integral = integral[-1]
    pdf_normalized = pdf_unnormalized / total_integral

    return mu_sorted, pdf_normalized

def compute_cdf(mu_array, pdf_array):

    # Cumulative integral
    cdf_array = cumulative_trapezoid(pdf_array, mu_array, initial=0)
    # Normalize
    cdf_array /= cdf_array[-1]
    return cdf_array

def invert_cdf(mu_array, cdf_array, xi_array):

    inverse_cdf = interp1d(
        cdf_array,
        mu_array,
        kind='linear',
        bounds_error=False,
        fill_value=(mu_array[0], mu_array[-1])
    )
    # mu for each xi
    mu_values = inverse_cdf(xi_array)
    return mu_values

def importance_sampling_lookup(mu_array, p_array, N_xi=1000):

    mu_sorted, pdf_normalized = compute_pdf(mu_array, p_array)
    
    cdf_array = compute_cdf(mu_sorted, pdf_normalized)
    
    xi_array = np.linspace(0, 1, N_xi)
    
    mu_values = invert_cdf(mu_sorted, cdf_array, xi_array)

    lookup_table = np.column_stack((xi_array, mu_values))
    return lookup_table

if __name__ == "__main__":
    def henyey_greenstein(theta, g):

        numerator = 1 - g**2
        denominator = (1 + g**2 - 2 * g * np.cos(theta))**1.5
        p_theta = numerator / (4 * np.pi * denominator)
        return p_theta

    g_param = 0.87
    num_points = 1800
    theta = np.linspace(0, np.pi, num_points)
    mu = np.cos(theta)

    phase_values = henyey_greenstein(theta, g=g_param)

    lookup_table = importance_sampling_lookup(mu, phase_values, N_xi=1800)

    xi_values = lookup_table[:, 0]
    sampled_mu = lookup_table[:, 1]
    sampled_theta = np.arccos(sampled_mu)

    def hg_importance_sampling(g, xi):
        if abs(g) < 1e-6:
            return 2 * xi - 1
        else:
            denominator = 1 - g + 2 * g * xi
            mu = (1 + g**2 - ((1 - g**2) / denominator)**2 ) / (2 * g)
            return mu

    analytical_mu = np.array([hg_importance_sampling(g_param, xi) for xi in xi_values])

    print("xi\tNumerical mu\tAnalytical mu\tDifference")
    step = 22
    for xi, num_mu, ana_mu in zip(xi_values[::step], sampled_mu[::step], analytical_mu[::step]):
        print(f"{xi:.5f}\t{num_mu:.8f}\t{ana_mu:.8f}\t{abs(num_mu - ana_mu):.2e}")
