import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
from skimage import io, filters
from scipy.ndimage import median_filter

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
    """PLACEHOLDER - Replace with actual Task 2 implementation"""
    return np.random.randint(0, 2, size=(1024, 899, 400))

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