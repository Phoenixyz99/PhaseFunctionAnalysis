import numpy as np
import math
from numba import cuda, float32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import matplotlib.pyplot as plt

MAX_BOUNCES = 256 

# https://www.pbr-book.org/3ed-2018/contents

# ==============================================================================
# GPU
# ==============================================================================

@cuda.jit(device=True, inline=True)
def henyey_greenstein_phase_function(g, cos_theta):
    denominator = (1.0 + g**2 - 2.0 * g * cos_theta)**1.5
    phase_value = (1.0 / (4.0 * math.pi)) * ((1.0 - g**2) / denominator)
    return phase_value

@cuda.jit(device=True, inline=True)
def is_point_inside_unit_sphere(point):
    return (point[0]**2 + point[1]**2 + point[2]**2) <= 1.0

@cuda.jit(device=True, inline=True)
def dot_product(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@cuda.jit(device=True, inline=True)
def normalize(vec):
    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    if norm > 0.0:
        vec[0] /= norm
        vec[1] /= norm
        vec[2] /= norm

@cuda.jit(device=True, inline=True)
def cross(a, b, result):
    result[0] = a[1] * b[2] - a[2] * b[1]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - a[1] * b[0]

@cuda.jit(device=True, inline=True)
def sample_new_direction(rng_states, idx, anisotropy, current_direction, scattered_direction):
    u = xoroshiro128p_uniform_float32(rng_states, idx)
    if abs(anisotropy) < 1e-3:
        cos_theta = 1.0 - 2.0 * u
    else:
        denom = 1.0 - anisotropy + 2.0 * anisotropy * u
        cos_theta = (1.0 + anisotropy**2 - ((1.0 - anisotropy**2) / denom)) / (2.0 * anisotropy)

    phi = 2.0 * math.pi * xoroshiro128p_uniform_float32(rng_states, idx)
    sin_theta = math.sqrt(1.0 - cos_theta**2)

    w = current_direction
    u_vec = cuda.local.array(3, dtype=float32)
    v_vec = cuda.local.array(3, dtype=float32)

    if abs(w[0]) > abs(w[1]):
        inv_len = 1.0 / math.sqrt(w[0]**2 + w[2]**2)
        u_vec[0] = -w[2] * inv_len
        u_vec[1] = 0.0
        u_vec[2] = w[0] * inv_len
    else:
        inv_len = 1.0 / math.sqrt(w[1]**2 + w[2]**2)
        u_vec[0] = 0.0
        u_vec[1] = w[2] * inv_len
        u_vec[2] = -w[1] * inv_len

    v_vec[0] = w[1] * u_vec[2] - w[2] * u_vec[1]
    v_vec[1] = w[2] * u_vec[0] - w[0] * u_vec[2]
    v_vec[2] = w[0] * u_vec[1] - w[1] * u_vec[0]

    scattered_direction[0] = sin_theta * math.cos(phi) * u_vec[0] + sin_theta * math.sin(phi) * v_vec[0] + cos_theta * w[0]
    scattered_direction[1] = sin_theta * math.cos(phi) * u_vec[1] + sin_theta * math.sin(phi) * v_vec[1] + cos_theta * w[1]
    scattered_direction[2] = sin_theta * math.cos(phi) * u_vec[2] + sin_theta * math.sin(phi) * v_vec[2] + cos_theta * w[2]

    normalize(scattered_direction)

@cuda.jit(device=True, inline=True)
def compute_ray_sphere_intersection(ray_origin, ray_direction):
    A = ray_direction[0]**2 + ray_direction[1]**2 + ray_direction[2]**2
    B = 2 * (ray_origin[0] * ray_direction[0] + ray_origin[1] * ray_direction[1] + ray_origin[2] * ray_direction[2])
    C = ray_origin[0]**2 + ray_origin[1]**2 + ray_origin[2]**2 - 1.0

    discriminant = B**2 - 4 * A * C

    if discriminant < 0:
        return -1.0

    sqrt_discriminant = discriminant**0.5
    t1 = (-B - sqrt_discriminant) / (2 * A)
    t2 = (-B + sqrt_discriminant) / (2 * A)

    if t1 > 0.0 and t2 > 0.0:
        return min(t1, t2)  # Both intersections are in front of the ray origin
    elif t1 > 0.0:
        return t1  # One intersection is in front
    elif t2 > 0.0:
        return t2  # The other intersection is in front
    else:
        return -1.0  # Both intersections are behind the ray origin

@cuda.jit(device=True, inline=True)
def path_trace_debug(rng_states, idx, sun_direction, current_position, current_direction,
                     extinction_coefficient, max_bounces, anisotropy, bounce_buffer):
    throughput = 1.0
    accumulated_radiance = 0.0

    for bounce in range(max_bounces):
        bounce_buffer[bounce] = 0.0

    for bounce in range(max_bounces):
        free_path_length = 1  # Doesn't actually matter

        current_position[0] += free_path_length * current_direction[0]
        current_position[1] += free_path_length * current_direction[1]
        current_position[2] += free_path_length * current_direction[2]

        cos_theta_sun = dot_product(current_direction, sun_direction)
        phase_value = henyey_greenstein_phase_function(anisotropy, cos_theta_sun)

        radiance_contribution = throughput * phase_value
        accumulated_radiance += radiance_contribution
        bounce_buffer[bounce] += radiance_contribution  # Store per-bounce radiance

        scattered_direction = cuda.local.array(3, dtype=float32)
        sample_new_direction(rng_states, idx, anisotropy, current_direction, scattered_direction)

        current_direction[0] = scattered_direction[0]
        current_direction[1] = scattered_direction[1]
        current_direction[2] = scattered_direction[2]

    return accumulated_radiance


@cuda.jit
def render_kernel(output_image, accumulated_radiance_image, accumulated_radiance_squared_image,
                  sample_counts_image, image_width, image_height,
                  sun_direction, rng_states,
                  extinction_coefficient, anisotropy, samples_per_frame,
                  d_bounce_buffer, max_bounces):
    x, y = cuda.grid(2)

    if x >= image_width or y >= image_height:
        return

    idx = y * image_width + x

    pixel_radiance = 0.0

    pixel_bounce_buffer = cuda.local.array(MAX_BOUNCES, dtype=float32)
    for b in range(MAX_BOUNCES):
        pixel_bounce_buffer[b] = 0.0

    camera_distance = 4.0

    angle_in_degrees = 1e-4 + (x / (image_width - 1)) * 360.0
    angle_in_radians = angle_in_degrees * (math.pi / 180)

    camera_position = cuda.local.array(3, dtype=float32)
    camera_position[0] = camera_distance * math.sin(angle_in_radians)
    camera_position[1] = 0.0
    camera_position[2] = camera_distance * math.cos(angle_in_radians)

    camera_direction = cuda.local.array(3, dtype=float32)
    camera_direction[0] = -camera_position[0]
    camera_direction[1] = -camera_position[1]
    camera_direction[2] = -camera_position[2]
    normalize(camera_direction)

    world_up = cuda.local.array(3, dtype=float32)
    world_up[0] = 0.0
    world_up[1] = 1.0
    world_up[2] = 0.0

    right = cuda.local.array(3, dtype=float32)
    cross(world_up, camera_direction, right)
    normalize(right)

    up = cuda.local.array(3, dtype=float32)
    cross(camera_direction, right, up)
    normalize(up)

    u = 0.0
    v = 0.0

    ray_origin = cuda.local.array(3, dtype=float32)
    ray_origin[0] = camera_position[0] + u * right[0] + v * up[0]
    ray_origin[1] = camera_position[1] + u * right[1] + v * up[1]
    ray_origin[2] = camera_position[2] + u * right[2] + v * up[2]

    ray_direction = cuda.local.array(3, dtype=float32)
    ray_direction[0] = camera_direction[0]
    ray_direction[1] = camera_direction[1]
    ray_direction[2] = camera_direction[2]

    for sample in range(samples_per_frame):
        sample_bounce_buffer = cuda.local.array(MAX_BOUNCES, dtype=float32)
        for b in range(MAX_BOUNCES):
            sample_bounce_buffer[b] = 0.0

        intersection_distance = compute_ray_sphere_intersection(ray_origin, ray_direction)

        if intersection_distance == -1.0:
            continue 

        current_position = cuda.local.array(3, dtype=float32)
        current_position[0] = ray_origin[0] + intersection_distance * ray_direction[0]
        current_position[1] = ray_origin[1] + intersection_distance * ray_direction[1]
        current_position[2] = ray_origin[2] + intersection_distance * ray_direction[2]

        current_direction = cuda.local.array(3, dtype=float32)
        current_direction[0] = ray_direction[0]
        current_direction[1] = ray_direction[1]
        current_direction[2] = ray_direction[2]

        sample_radiance = path_trace_debug(rng_states, idx, sun_direction, current_position, current_direction,
                                           extinction_coefficient, max_bounces, anisotropy, sample_bounce_buffer)

        pixel_radiance += sample_radiance

        for b in range(max_bounces):
            pixel_bounce_buffer[b] += sample_bounce_buffer[b]

    accumulated_radiance_image[y, x] += pixel_radiance
    accumulated_radiance_squared_image[y, x] += pixel_radiance ** 2
    sample_counts_image[y, x] += samples_per_frame

    output_image[y, x] = accumulated_radiance_image[y, x] / sample_counts_image[y, x]

    for b in range(max_bounces):
        d_bounce_buffer[y, x, b] = pixel_bounce_buffer[b] / samples_per_frame


def plot_polar_bounce_graph(bounce_data, bounce_index):
    if bounce_data.ndim != 1:
        raise ValueError("bounce_data must be a 1D array")

    epsilon = 1e-10
    energy = np.maximum(bounce_data, epsilon)
    theta = np.linspace(0, 2 * np.pi, len(bounce_data))

    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111, polar=True)

    ax.plot(theta, energy, color='b', linewidth=2)

    ax.set_title(f"Energy Distribution for Bounce {bounce_index}", va='bottom')

    plt.show()


def render_image(image_width, image_height, sun_angles,
                 extinction_coefficient, max_bounces, anisotropy):
    if max_bounces > MAX_BOUNCES:
        raise ValueError(f"max_bounces ({max_bounces}) cannot exceed MAX_BOUNCES ({MAX_BOUNCES}).")

    d_accumulated_radiance_image = cuda.device_array((image_height, image_width), dtype=np.float32)
    d_accumulated_radiance_squared_image = cuda.device_array((image_height, image_width), dtype=np.float32)
    d_sample_counts_image = cuda.device_array((image_height, image_width), dtype=np.float32)
    d_output_image = cuda.device_array((image_height, image_width), dtype=np.float32)

    d_accumulated_radiance_image[:] = 0.0
    d_accumulated_radiance_squared_image[:] = 0.0
    d_sample_counts_image[:] = 0.0

    d_bounce_buffer = cuda.device_array((image_height, image_width, MAX_BOUNCES), dtype=np.float32)
    d_bounce_buffer[:] = 0.0
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(image_width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(image_height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    num_pixels = image_width * image_height
    rng_states = create_xoroshiro128p_states(num_pixels, seed=1)

    samples_per_frame = 65536  # Number of samples per pixel per frame

    sun_theta = math.radians(sun_angles[0])
    sun_phi = math.radians(sun_angles[1])
    sun_direction = np.array([
        math.cos(sun_phi) * math.cos(sun_theta),
        math.sin(sun_phi) * math.cos(sun_theta),
        math.sin(sun_theta)
    ], dtype=np.float32)
    sun_direction /= np.linalg.norm(sun_direction)
    d_sun_direction = cuda.to_device(sun_direction)

    render_kernel[blocks_per_grid, threads_per_block](
        d_output_image, d_accumulated_radiance_image, d_accumulated_radiance_squared_image, d_sample_counts_image,
        image_width, image_height,
        d_sun_direction, rng_states,
        extinction_coefficient, anisotropy, samples_per_frame,
        d_bounce_buffer, max_bounces
    )

    output_image_host = d_output_image.copy_to_host()
    bounce_buffer_host = d_bounce_buffer.copy_to_host()

    while True:
        bounce_index = int(input(f"Enter the bounce index to visualize (0 to {max_bounces - 1}): "))

        if bounce_index < 0 or bounce_index >= max_bounces:
            raise ValueError(f"Bounce index must be between 0 and {max_bounces - 1}")

        bounce_data = bounce_buffer_host[0, :, bounce_index]

        plot_polar_bounce_graph(bounce_data, bounce_index)

# ==============================================================================
# Main
# ==============================================================================

def main():
    sun_angles = (0.0, 0.0)  # Theta (elevation), Phi (azimuth) angles in degrees

    image_width = 3600
    image_height = 1
    max_bounces = 256  # Maximum number of scattering events (must be <= MAX_BOUNCES)
    extinction_coefficient = 0.0  # NULL HERE
    anisotropy = 0.87

    render_image(
        image_width=image_width,
        image_height=image_height,
        sun_angles=sun_angles,
        extinction_coefficient=extinction_coefficient,
        max_bounces=max_bounces,
        anisotropy=anisotropy
    )

if __name__ == "__main__":
    main()
