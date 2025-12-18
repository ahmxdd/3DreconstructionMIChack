import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
from skimage import io, filters
from scipy.ndimage import median_filter
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.morphology import ball

from task1 import analyze_microstructure, generate_CDFs

def calculate_error(CDF_target_top, CDF_target_bottom, 
                    CDF_generated_top, CDF_generated_bottom):
    error_top = np.sum(np.abs(CDF_target_top[0] - CDF_generated_top[0]))
    error_bottom = np.sum(np.abs(CDF_target_bottom[0] - CDF_generated_bottom[0]))
    return error_top + error_bottom

def optimize(params, CDF_target_top, CDF_target_bottom, task2_generate_3d):
    structure_3d = task2_generate_3d(*params)
    random_y_index = np.random.randint(0, structure_3d.shape[1])
    slice_2d = structure_3d[:, random_y_index, :].astype(bool)
    
    combined_data, _ = analyze_microstructure(slice_2d)
    CDF_gen_top, CDF_gen_bottom = generate_CDFs(combined_data[0], combined_data[1])
    
    return calculate_error(CDF_target_top, CDF_target_bottom, CDF_gen_top, CDF_gen_bottom)

def preprocess_slice(slice_2d):
    """Convert grayscale slice to binary (True=solid, False=void)"""
    slice_norm = slice_2d.astype(float) / slice_2d.max() if slice_2d.max() > 1 else slice_2d.astype(float)
    slice_smooth = filters.gaussian(slice_norm, sigma=1)
    local_thresh = filters.threshold_local(slice_smooth, 35, offset=0.02)
    binary = median_filter((slice_smooth > local_thresh).astype(np.uint8), size=3)
    return binary.astype(bool)

def run_optimization(CDF_target_top, CDF_target_bottom, task2_generate_3d, 
                     n_reconstructions=5
                     , n_calls=50
                     ):
    space = [
        Real(1.0, 10.0, name='sigma_1'),
        Real(10.0, 30.0, name='sigma_2'),
        Real(0.0, 1.0, name='threshold'),
        Integer(1, 10, name='r_struct')
    ]
    
    results = []
    
    for i in range(n_reconstructions):
        print(f"\nReconstruction {i+1}/{n_reconstructions}")
        
        result = gp_minimize(
            func=lambda params: optimize(params, CDF_target_top, CDF_target_bottom, task2_generate_3d),
            dimensions=space,
            n_calls=n_calls,
            random_state=i,
            verbose=True
        )
        
        results.append(result)
        optimal_structure = task2_generate_3d(*result.x)
        
        np.save(f'reconstructed_structure_{i+1}.npy', optimal_structure)
        
        plt.figure(figsize=(10, 6))
        plot_convergence(result)
        plt.title(f'Convergence - Reconstruction {i+1}')
        plt.savefig(f'convergence_plot_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"σ1={result.x[0]:.3f}, σ2={result.x[1]:.3f}, thresh={result.x[2]:.3f}, r={result.x[3]}, error={result.fun:.6f}")
    
    return results




def task2_generate_3d(sigma_1, sigma_2, threshold, r_struct):
    """
    Generates a 3D binary structure representing a membrane based on the provided hyperparameters.
    
    Parameters:
    - sigma_1 (float): Standard deviation for the first Gaussian filter (smaller pores).
    - sigma_2 (float): Standard deviation for the second Gaussian filter (larger pores).
    - threshold (float): Cut-off value for binarization (0 to 1).
    - r_struct (int): Radius of the spherical structuring element for morphological closing.
    
    Returns:
    - np.ndarray: A 3D boolean numpy array (1024x899x400) where True represents solid and False represents void.
    """
    
    # Dimensions of the original FIB-SEM structure (pg 6)
    dims = (1024, 899, 400)
    
    # ==========================================
    # 1. Generate 3-D matrix (A1)
    # Create the matrix with random 0s and 1s
    # low=0 (inclusive), high=2 (exclusive)
    A1_matrix = np.random.randint(0, 2, size=dims, dtype=np.int8)

    # ==========================================
    # 2. Filter (A1) with a 3-D Gaussian smoothing kernel with standard deviation sigma_1
    A1_matrix_sigma_1 = gaussian_filter(A1_matrix.astype(np.float32), sigma=sigma_1)

    # ==========================================
    # 3. Generate second matrix (A2) to account for depth-dependent anisotropy
    A2_matrix = np.random.randint(0, 2, size=dims, dtype=np.int8)

    # ==========================================
    # 4. Filter (A2) with a 3-D Gaussian smoothing kernel with larger standard deviation sigma_2
    A2_matrix_sigma_2 = gaussian_filter(A2_matrix.astype(np.float32), sigma=sigma_2)

    # ==========================================
    # 5. Create weight matrix W
    # Weight function: W(z) = (1 - z/h)^10
    # z-direction is from the top (index 0) to bottom (index 1023)
    z_indices = np.arange(dims[0], dtype=np.float32)
    W_profile = (1.0 - z_indices / float(dims[0]))**10
    
    # Reshape to (1024, 1, 1) to broadcast across y (899) and x (400) dimensions
    W_profile_reshaped = W_profile[:, np.newaxis, np.newaxis]
    W = np.broadcast_to(W_profile_reshaped, dims)

    # ==========================================
    # 6. Combine matrices to create A_T
    # A_T = A1 .* W + A2 .* (1 - W)
    A_T = (A1_matrix_sigma_1 * W) + (A2_matrix_sigma_2 * (1.0 - W))

    # ==========================================
    # 7. Apply threshold to binarize A_T
    # If A_T(pixel) > threshold -> solid (1)
    # If A_T(pixel) <= threshold -> void (0)
    binary_mask = (A_T > threshold)
    A_T_binary = binary_mask.astype(np.int8)

    # ==========================================
    # 8. Morphological closing
    # Use spherical structuring element with radius r_struct
    # Note: ball expects an integer radius
    r_struct_int = int(r_struct)
    struct_elem = ball(r_struct_int)
    
    closed_matrix = ndimage.binary_closing(A_T_binary, structure=struct_elem)

    # Return the final 3D binary structure
    # The evaluation script expects a boolean array (True=Solid, False=Void) or (1=Solid, 0=Void)
    # The logic above sets solid=1 (True) and void=0 (False), which matches standard conventions.
    return closed_matrix

if __name__ == "__main__":
    # Load and select slice
    print("Loading 3D TIFF stack...")
    stack_3d = io.imread('originaldata.tif')
    slice_index = np.random.randint(0, stack_3d.shape[0])
    input_slice = stack_3d[slice_index, :, :]
    
    # Save slice info
    np.savez('target_slice.npz', slice=input_slice, index=slice_index)
    plt.imsave('selected_input_slice.png', input_slice, cmap='gray')
    
    # Generate target CDFs
    print(f"Processing slice {slice_index}...")
    binary_map = preprocess_slice(input_slice)
    combined_data, raw_data = analyze_microstructure(binary_map)
    CDF_target_top, CDF_target_bottom = generate_CDFs(combined_data[0], combined_data[1])
    
    np.savez('target_cdfs.npz', 
             top_probs=CDF_target_top[0], top_bins=CDF_target_top[1],
             bottom_probs=CDF_target_bottom[0], bottom_bins=CDF_target_bottom[1])
    
    # Run optimization
    print("\nStarting optimization...")
    results = run_optimization(CDF_target_top, CDF_target_bottom, task2_generate_3d)
    
    # Summary
    final_errors = [r.fun for r in results]
    optimal_params = np.array([r.x for r in results])
    
    print(f"\n{'='*60}")
    print(f"Mean error: {np.mean(final_errors):.6f} ± {np.std(final_errors):.6f}")
    print(f"Optimal parameters (mean ± std):")
    print(f"  σ1: {np.mean(optimal_params[:, 0]):.3f} ± {np.std(optimal_params[:, 0]):.3f}")
    print(f"  σ2: {np.mean(optimal_params[:, 1]):.3f} ± {np.std(optimal_params[:, 1]):.3f}")
    print(f"  threshold: {np.mean(optimal_params[:, 2]):.3f} ± {np.std(optimal_params[:, 2]):.3f}")
    print(f"  r_struct: {np.mean(optimal_params[:, 3]):.1f} ± {np.std(optimal_params[:, 3]):.1f}")