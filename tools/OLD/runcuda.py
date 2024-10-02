import numpy as np
import math
import time
import pygame
from numba import cuda, float32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from scipy.signal import convolve2d
from ColorLib import tonemapping

pygame.init()
pygame.display.set_caption('Volumetric Path Tracer')

camera_rotation_angles = [0.0, 0.0, 0.0]
camera_position = np.array([0.0, 0.0, 3.0], dtype=np.float32) 
movement_speed = 5.0
mouse_sensitivity = 0.1
camera_moved = True 

# https://www.pbr-book.org/3ed-2018/contents

# ==============================================================================
# Device Functions for GPU Computations
# ==============================================================================

@cuda.jit(device=True, inline=True)
def henyey_greenstein_phase_function(g, cos_theta):
    denominator = (1.0 + g**2 - 2.0 * g * cos_theta)**1.5
    phase_value = (1.0 / (4.0 * math.pi)) * ((1.0 - g**2) / denominator)
    return phase_value

@cuda.jit(device=True, inline=True)
def is_point_inside_unit_cube(point):
    return (-1.0 <= point[0] <= 1.0) and (-1.0 <= point[1] <= 1.0) and (-1.0 <= point[2] <= 1.0)

@cuda.jit(device=True, inline=True)
def dot_product(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@cuda.jit(device=True, inline=True)
def sample_new_direction(rng_states, idx, anisotropy, current_direction, scattered_direction):
    u = xoroshiro128p_uniform_float32(rng_states, idx)
    if abs(anisotropy) < 1e-3:
        # Isotropic scattering
        cos_theta = 1.0 - 2.0 * u
    else:
        # Inversion Method
        denom = 1.0 - anisotropy + 2.0 * anisotropy * u
        cos_theta = (1.0 + anisotropy**2 - ((1.0 - anisotropy**2) / denom)) / (2.0 * anisotropy)

    phi = 2.0 * math.pi * xoroshiro128p_uniform_float32(rng_states, idx)
    sin_theta = math.sqrt(1.0 - cos_theta**2)

    # Orthonormal basis around current direction
    w = current_direction
    u_vec = cuda.local.array(3, dtype=float32)
    v_vec = cuda.local.array(3, dtype=float32)

    # Avoid numerical instability by choosing the largest component
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

    # Ugly ross product
    v_vec[0] = w[1] * u_vec[2] - w[2] * u_vec[1]
    v_vec[1] = w[2] * u_vec[0] - w[0] * u_vec[2]
    v_vec[2] = w[0] * u_vec[1] - w[1] * u_vec[0]

    scattered_direction[0] = sin_theta * math.cos(phi) * u_vec[0] + sin_theta * math.sin(phi) * v_vec[0] + cos_theta * w[0]
    scattered_direction[1] = sin_theta * math.cos(phi) * u_vec[1] + sin_theta * math.sin(phi) * v_vec[1] + cos_theta * w[1]
    scattered_direction[2] = sin_theta * math.cos(phi) * u_vec[2] + sin_theta * math.sin(phi) * v_vec[2] + cos_theta * w[2]

    norm = math.sqrt(scattered_direction[0]**2 + scattered_direction[1]**2 + scattered_direction[2]**2)
    scattered_direction[0] /= norm
    scattered_direction[1] /= norm
    scattered_direction[2] /= norm

@cuda.jit(device=True, inline=True)
def compute_ray_box_intersection(ray_origin, ray_direction):
    tmin = float32(-1e20)
    tmax = float32(1e20)

    # Check if the ray starts inside the cube
    if is_point_inside_unit_cube(ray_origin):
        return 0.0  # Immediate intersection when inside

    for i in range(3):
        if ray_direction[i] != 0.0:
            t1 = (-1.0 - ray_origin[i]) / ray_direction[i]
            t2 = (1.0 - ray_origin[i]) / ray_direction[i]
            t_near = min(t1, t2)
            t_far = max(t1, t2)
            tmin = max(tmin, t_near)
            tmax = min(tmax, t_far)
        else:
            # Ray is parallel to slab; no hit if origin not within slab
            if ray_origin[i] < -1.0 or ray_origin[i] > 1.0:
                return -1.0

    if tmin > tmax or tmax < 0.0:
        return -1.0  # No intersection
    return tmin if tmin >= 0.0 else tmax

@cuda.jit(device=True, inline=True)
def path_trace(rng_states, idx, sun_direction, current_position, current_direction,
               extinction_coefficient, max_bounces, anisotropy):
    throughput = 1.0 
    accumulated_radiance = 0.0

    for bounce in range(max_bounces):
        rand_val = xoroshiro128p_uniform_float32(rng_states, idx)
        free_path_length = -math.log(1.0 - rand_val) / extinction_coefficient

        # Move to the next scattering point
        current_position[0] += free_path_length * current_direction[0]
        current_position[1] += free_path_length * current_direction[1]
        current_position[2] += free_path_length * current_direction[2]

        # Check if the new position is inside the volume
        if not is_point_inside_unit_cube(current_position):
            break  # Ray has exited the volume

        transmittance = math.exp(-extinction_coefficient * free_path_length)
        throughput *= transmittance

        sun_distance = compute_ray_box_intersection(current_position, sun_direction)
        if sun_distance != -1.0:
    
            sun_transmittance = math.exp(-extinction_coefficient * sun_distance)
            cos_theta_sun = dot_product(current_direction, sun_direction)
            phase_value = henyey_greenstein_phase_function(anisotropy, cos_theta_sun)
        
            accumulated_radiance += throughput * sun_transmittance * phase_value

        scattered_direction = cuda.local.array(3, dtype=float32)
        sample_new_direction(rng_states, idx, anisotropy, current_direction, scattered_direction)

        current_direction[0] = scattered_direction[0]
        current_direction[1] = scattered_direction[1]
        current_direction[2] = scattered_direction[2]

        if throughput < 0.001:
            q = max(throughput, 0.1)
            if xoroshiro128p_uniform_float32(rng_states, idx) > q:
                break
            throughput /= q

    return accumulated_radiance

@cuda.jit(device=True, inline=True)
def apply_camera_rotation(rotation_matrix, local_direction, world_direction):
    for i in range(3):
        world_direction[i] = 0.0
        for j in range(3):
            world_direction[i] += rotation_matrix[j, i] * local_direction[j]

# ==============================================================================
# GPU Functions
# ==============================================================================

@cuda.jit
def scale_array(arr, scalar):
    x, y = cuda.grid(2)
    if x < arr.shape[1] and y < arr.shape[0]:
        arr[y, x] *= scalar

@cuda.jit
def render_kernel(output_image, accumulated_radiance_image, accumulated_radiance_squared_image,
                  sample_counts_image, image_width, image_height, half_fov_tangent, aspect_ratio,
                  camera_position, camera_rotation_matrix, sun_direction, rng_states,
                  extinction_coefficient, max_bounces, anisotropy, samples_per_frame):
    x, y = cuda.grid(2)

    if x >= image_width or y >= image_height:
        return

    idx = y * image_width + x

    pixel_radiance = 0.0

    for sample in range(samples_per_frame):
        u_rand = xoroshiro128p_uniform_float32(rng_states, idx)
        v_rand = xoroshiro128p_uniform_float32(rng_states, idx)
        u = (2.0 * ((x + u_rand) / image_width) - 1.0) * half_fov_tangent * aspect_ratio
        v = (1.0 - 2.0 * ((y + v_rand) / image_height)) * half_fov_tangent

        direction_x = u
        direction_y = v
        direction_z = -1.0
        norm = math.sqrt(direction_x**2 + direction_y**2 + direction_z**2)
        local_direction = (direction_x / norm, direction_y / norm, direction_z / norm)

        world_direction = cuda.local.array(3, dtype=float32)
        apply_camera_rotation(camera_rotation_matrix, local_direction, world_direction)

        intersection_distance = compute_ray_box_intersection(camera_position, world_direction)

        if intersection_distance == -1.0:
            continue

        current_position = cuda.local.array(3, dtype=float32)
        current_position[0] = camera_position[0] + intersection_distance * world_direction[0]
        current_position[1] = camera_position[1] + intersection_distance * world_direction[1]
        current_position[2] = camera_position[2] + intersection_distance * world_direction[2]

        current_direction = cuda.local.array(3, dtype=float32)
        current_direction[0] = world_direction[0]
        current_direction[1] = world_direction[1]
        current_direction[2] = world_direction[2]

        sample_radiance = path_trace(rng_states, idx, sun_direction, current_position, current_direction,
                                     extinction_coefficient, max_bounces, anisotropy)

        pixel_radiance += sample_radiance

    accumulated_radiance_image[y, x] += pixel_radiance
    accumulated_radiance_squared_image[y, x] += pixel_radiance ** 2
    sample_counts_image[y, x] += samples_per_frame

    output_image[y, x] = accumulated_radiance_image[y, x] / sample_counts_image[y, x]

# ==============================================================================
# Helper Functions
# ==============================================================================

def rotation_matrix_xzy(pitch, yaw, roll):

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    Ry = np.array([[np.cos(roll), 0, np.sin(roll)],
                   [0, 1, 0],
                   [-np.sin(roll), 0, np.cos(roll)]])

    rotation_matrix = Ry @ Rz @ Rx
    return rotation_matrix

# ==============================================================================
# Main Rendering Function
# ==============================================================================

def render_image(image_width, image_height, fov, sun_angles,
                 extinction_coefficient, max_bounces, anisotropy):
    global camera_rotation_angles, camera_position, movement_speed, camera_moved

    aspect_ratio = image_width / image_height
    half_fov_tangent = math.tan(math.radians(fov / 2.0))

    d_accumulated_radiance_image = cuda.device_array((image_height, image_width), dtype=np.float32)
    d_accumulated_radiance_squared_image = cuda.device_array((image_height, image_width), dtype=np.float32)
    d_sample_counts_image = cuda.device_array((image_height, image_width), dtype=np.float32)
    d_output_image = cuda.device_array((image_height, image_width), dtype=np.float32)

    d_accumulated_radiance_image[:] = 0.0
    d_accumulated_radiance_squared_image[:] = 0.0
    d_sample_counts_image[:] = 0.0

    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(image_width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(image_height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    num_pixels = image_width * image_height
    rng_states = create_xoroshiro128p_states(num_pixels, seed=1)

    samples_per_frame = 4  

    frame_count = 0

    screen = pygame.display.set_mode((image_width, image_height))
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False) 

    clock = pygame.time.Clock()

    prev_camera_rotation_angles = camera_rotation_angles.copy()
    prev_camera_position = camera_position.copy()

    while True:
        dt = clock.tick(60) / 1000.0 
        movement_distance = movement_speed * dt


        rotation_angles_rad = [math.radians(ang) for ang in camera_rotation_angles]

        rot_matrix_host = rotation_matrix_xzy(*rotation_angles_rad)
        camera_rotation_matrix = np.ascontiguousarray(rot_matrix_host, dtype=np.float32)
        camera_position_host = camera_position.copy()

        d_camera_rotation_matrix = cuda.to_device(camera_rotation_matrix)
        d_camera_position = cuda.to_device(camera_position_host)

        sun_theta = math.radians(sun_angles[0])
        sun_phi = math.radians(sun_angles[1])
        sun_direction = np.array([
            math.cos(sun_phi) * math.cos(sun_theta),
            math.sin(sun_phi) * math.cos(sun_theta),
            math.sin(sun_theta)
        ], dtype=np.float32)
        sun_direction /= np.linalg.norm(sun_direction)
        d_sun_direction = cuda.to_device(sun_direction)

        if camera_moved:
            decay = 0.5  # TAA
            scale_array[blocks_per_grid, threads_per_block](d_accumulated_radiance_image, decay)
            scale_array[blocks_per_grid, threads_per_block](d_accumulated_radiance_squared_image, decay)
            scale_array[blocks_per_grid, threads_per_block](d_sample_counts_image, decay)
            camera_moved = False

        render_kernel[blocks_per_grid, threads_per_block](
            d_output_image, d_accumulated_radiance_image, d_accumulated_radiance_squared_image, d_sample_counts_image,
            image_width, image_height, half_fov_tangent, aspect_ratio,
            d_camera_position, d_camera_rotation_matrix, d_sun_direction, rng_states,
            extinction_coefficient, max_bounces, anisotropy, samples_per_frame
        )


        output_image_host = d_output_image.copy_to_host()
        accumulated_radiance_host = d_accumulated_radiance_image.copy_to_host()
        accumulated_radiance_squared_host = d_accumulated_radiance_squared_image.copy_to_host()
        sample_counts_host = d_sample_counts_image.copy_to_host()

        test = np.stack([output_image_host] * 3, axis=-1)
        image_corrected = tonemapping.kimkautz(test)  # Should be shape (H, W, 3)

        # Convert the corrected image to 8-bit RGB
        image_8bit_rgb = np.clip(image_corrected * 255, 0, 255).astype(np.uint8)

        # Flip the image vertically to match Pygame coordinate system
        image_8bit_rgb = np.flipud(image_8bit_rgb)

        # Transpose the array to match Pygame's expected shape (width, height, 3)
        image_8bit_rgb = np.transpose(image_8bit_rgb, (1, 0, 2))

        # Create a Pygame surface from the image array
        image_surface = pygame.surfarray.make_surface(image_8bit_rgb)

        # Display the image on the screen
        screen.blit(image_surface, (0, 0))
        pygame.display.update()

        # Event handling
        camera_moved_this_frame = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_focused():
                    dx, dy = event.rel 
                    camera_rotation_angles[2] += dx * mouse_sensitivity  # Yaw 
                    camera_rotation_angles[0] -= dy * mouse_sensitivity  # Pitch 
                    camera_moved_this_frame = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    movement_speed *= 1.1
                elif event.button == 5:  # Scroll down
                    movement_speed /= 1.1

        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            return

        move_direction = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if keys[pygame.K_w]:
            # Move forward
            move_direction += np.array([0.0, 0.0, 1.0], dtype=np.float32)
            camera_moved_this_frame = True
        if keys[pygame.K_s]:
            # Move backward
            move_direction += np.array([0.0, 0.0, -1.0], dtype=np.float32)
            camera_moved_this_frame = True
        if keys[pygame.K_a]:
            # Move left
            move_direction += np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            camera_moved_this_frame = True
        if keys[pygame.K_d]:
            # Move right
            move_direction += np.array([1.0, 0.0, 0.0], dtype=np.float32)
            camera_moved_this_frame = True
        if keys[pygame.K_SPACE]:
            # Move up
            move_direction += np.array([0.0, -1.0, 0.0], dtype=np.float32)
            camera_moved_this_frame = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            # Move down
            move_direction += np.array([0.0, 1.0, 0.0], dtype=np.float32)
            camera_moved_this_frame = True
        if keys[pygame.K_q]:
            # Roll left
            camera_rotation_angles[2] -= 50 * dt
            camera_moved_this_frame = True
        if keys[pygame.K_e]:
            # Roll right
            camera_rotation_angles[2] += 50 * dt
            camera_moved_this_frame = True

        if np.linalg.norm(move_direction) > 0:
            move_direction = move_direction / np.linalg.norm(move_direction)

            rotation_angles_movement_rad = [
                math.radians(camera_rotation_angles[0]),  # Pitch
                math.radians(camera_rotation_angles[1]),  # Yaw
                0.0 
            ]
            rot_matrix_movement = rotation_matrix_xzy(*rotation_angles_movement_rad)

            movement_vector = rot_matrix_movement @ move_direction

            movement_vector[0] = -movement_vector[0]
            movement_vector[2] = -movement_vector[2]

            camera_position += movement_vector * movement_distance
            camera_moved_this_frame = True

        if camera_moved_this_frame:
            camera_moved = True

        frame_count += 1

# ==============================================================================
# Main Function
# ==============================================================================

def main():
    sun_angles = (0.0, 0.0)  # Theta (elevation), Phi (azimuth) angles in degrees

    image_width = 512
    image_height = 512
    fov = 90
    max_bounces = 30
    extinction_coefficient = 4.0
    anisotropy = 0.87

    render_image(
        image_width=image_width,
        image_height=image_height,
        fov=fov,
        sun_angles=sun_angles,
        extinction_coefficient=extinction_coefficient,
        max_bounces=max_bounces,
        anisotropy=anisotropy
    )

if __name__ == "__main__":
    main()
