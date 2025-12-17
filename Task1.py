#variables:
#result_array is the variable that contains a 899x1024 array of 0's and 1's of the map of the image image.png
#CDF_top, CDF_bottom are tuples who have values (probability_values, bin_locations)


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage import io, color, transform, filters
from scipy.ndimage import median_filter

def process_image(image_path):
    # 1. Load the image
    # We read it in standard RGB
    image = io.imread(image_path)
    
    # 2. Convert to Grayscale
    # If image has 3 or 4 channels (RGB/RGBA), convert to gray. 
    # If it's already gray, just use it.
    if image.ndim == 3:
        if image.shape[2] == 4:
            # Convert RGBA to RGB (handles transparency)
            image = color.rgba2rgb(image)
        image_gray = color.rgb2gray(image)
    else:
        image_gray = image

    # 3. Resize to specific dimensions (899 x 1024)
    # This ensures your output array is exactly the size you asked for.
    # anti_aliasing=True keeps the image smooth during resize
    A = transform.resize(image_gray, (899, 1024), anti_aliasing=True)

    # --- MATLAB: A = imgaussfilt(A, 1); ---
    # Apply Gaussian filter with sigma=1
    A_smooth = filters.gaussian(A, sigma=1)

    # --- MATLAB: B = imbinarize(A, 'adaptive', 'sensitivity', 0.65); ---
    # Adaptive thresholding. 
    # Note: 'block_size' determines the local area size (like MATLAB's default).
    # 'offset' helps tune the sensitivity. You can tweak 'offset' to match the 0.65 feel.
    block_size = 35  # Must be an odd integer
    local_thresh = filters.threshold_local(A_smooth, block_size, offset=0.02)
    B = A_smooth > local_thresh

    # --- MATLAB: B = medfilt2(B, [3,3]); ---
    # Median filter with a 3x3 kernel
    # We cast to float/int before filtering because it expects numbers, not bools
    B_filtered = median_filter(B.astype(np.uint8), size=3)

    # Convert final result to strictly 0s and 1s
    final_array = B_filtered.astype(int)

    return final_array


def get_binary_image(image_path):
    # --- PREVIOUS STEPS (Load, Resize, Binarize) ---
    image = io.imread(image_path)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = color.rgba2rgb(image)
        image_gray = color.rgb2gray(image)
    else:
        image_gray = image

    A = transform.resize(image_gray, (899, 1024), anti_aliasing=True)
    A_smooth = filters.gaussian(A, sigma=1)
    
    block_size = 35
    local_thresh = filters.threshold_local(A_smooth, block_size, offset=0.02)
    B = A_smooth > local_thresh
    B = median_filter(B.astype(np.uint8), size=3)
    
    return B.astype(bool) # Return as boolean for easy negation

def analyze_microstructure(binary_img):
    H, W = binary_img.shape

    # 1. Calculate Distance Maps (On the FULL image to preserve boundary truth)
    # distance_transform_edt calculates distance to the nearest ZERO.
    # So, for dist_solid (1s), we calculate distance to nearest 0.
    # For dist_void (0s), we invert the image so 0s become 1s, then calc dist to nearest 0.
    
    dist_solid = distance_transform_edt(binary_img)          # Dist from solid pixel to nearest void
    dist_void = distance_transform_edt(~binary_img)          # Dist from void pixel to nearest solid

    # 2. Define Regions (Row Indices)
    # Top 20%
    limit_top = int(H * 0.20)
    # Bottom 50%
    limit_bottom_start = H - int(H * 0.50)

    # 3. Extract Data for Regions
    # We combine them into a "Signed Distance" for the CDF
    # Void = Positive values, Solid = Negative values
    
    def get_region_data(r_start, r_end):
        # Slice the distance maps
        d_solid_slice = dist_solid[r_start:r_end, :]
        d_void_slice = dist_void[r_start:r_end, :]
        
        # Flatten to 1D arrays
        flat_solid = d_solid_slice[binary_img[r_start:r_end, :] == 1]
        flat_void = d_void_slice[binary_img[r_start:r_end, :] == 0]
        
        # Combine for "Signed Distance" (Solid is negative, Void is positive)
        # This creates the continuous distribution from solid center -> interface -> pore center
        signed_dist = np.concatenate((-flat_solid, flat_void))
        
        return flat_solid, flat_void, signed_dist

    # Get data for specific regions
    top_solid, top_void, top_combined = get_region_data(0, limit_top)
    bot_solid, bot_void, bot_combined = get_region_data(limit_bottom_start, H)

    return (top_combined, bot_combined), (top_solid, top_void, bot_solid, bot_void)

def plot_results(top_data, bot_data):
    top_combined, bot_combined = top_data
    
    plt.figure(figsize=(12, 10))

    # --- Plot 1: Top 20% Distribution ---
    plt.subplot(2, 1, 1)
    # Plot histogram (Normalized PDF)
    plt.hist(top_combined, bins=100, density=True, color='skyblue', alpha=0.7, edgecolor='black', label='Distribution')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Phase Boundary')
    plt.title("Top 20% (Skin Layer): Combined Distance Function")
    plt.xlabel("Distance (Pixels) \n <--- Solid Interior | Void Interior --->")
    plt.ylabel("Normalized Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Plot 2: Bottom 50% Distribution ---
    plt.subplot(2, 1, 2)
    plt.hist(bot_combined, bins=100, density=True, color='salmon', alpha=0.7, edgecolor='black', label='Distribution')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Phase Boundary')
    plt.title("Bottom 50% (Substructure): Combined Distance Function")
    plt.xlabel("Distance (Pixels) \n <--- Solid Interior | Void Interior --->")
    plt.ylabel("Normalized Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def generate_CDFs(top_data, bot_data, bins=100):
    """
    Converts raw signed-distance arrays into normalized histogram targets.
    
    Parameters:
        top_data (np.array): Raw signed distances from the top 20% (from previous step).
        bot_data (np.array): Raw signed distances from the bottom 50% (from previous step).
        bins (int): Resolution of the target distribution (default 100).
        
    Returns:
        CDF_top (tuple): (probability_values, bin_locations)
        CDF_bottom (tuple): (probability_values, bin_locations)
    """
    
    def compute_target_hist(data, n_bins):
        # density=True is CRITICAL for optimization targets.
        # It ensures the area under the curve is 1, so you match SHAPE, not pixel counts.
        counts, bin_edges = np.histogram(data, bins=n_bins, density=True)
        
        # Convert edges to centers (standard x-axis for curve fitting)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return counts, bin_centers

    # Generate the optimization targets
    # These variables contain the Y-values (counts) and X-values (distances)
    probs_top, centers_top = compute_target_hist(top_data, bins)
    probs_bot, centers_bot = compute_target_hist(bot_data, bins)
    
    return (probs_top, centers_top), (probs_bot, centers_bot)
# --- EXECUTION ---
if __name__ == "__main__":
    image_path = 'image.png' # <--- INPUT IMAGE
    
    try:
        
        #get result_array
        result_array = process_image(image_path)
        print(f"Output Shape: {result_array.shape}") # Should be (899, 1024)
        print(f"Unique values: {np.unique(result_array)}") # Should be [0 1]
        
        # Get Binary
        binary_map = get_binary_image(image_path)
        
        # Analyze
        combined_data, raw_data = analyze_microstructure(binary_map)
        
        CDF_top, CDF_bottom = generate_CDFs(combined_data[0], combined_data[1])
    
        # Plot
        plot_results(combined_data, raw_data)
        
        print("Analysis Complete.")
        print(f"Top 20% Mean Pore Size: {np.mean(raw_data[1]):.2f} pixels")
        print(f"Bottom 50% Mean Pore Size: {np.mean(raw_data[3]):.2f} pixels")


 
        # 1. Top Target
        target_y_top = CDF_top[0]  # (Probability)
        target_x_top = CDF_top[1]  # The distance values (x-axis)

        # 2. Bottom Target
        target_y_bot = CDF_bottom[0] 
        target_x_bot = CDF_bottom[1]
        print(f"Optimization Target Generated: Top CDF has {len(target_y_top)} points.")
        print(f"Optimization Target Generated: Bottom CDF has {len(target_y_bot)} points.")
    except Exception as e:
        print(f"Error: {e}")